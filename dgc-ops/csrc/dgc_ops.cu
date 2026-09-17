#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr int kHistogramBlocks = 2048;

__global__ void dgc_update_fp32_kernel(
    const float* __restrict__ grad,
    float* __restrict__ velocity,
    float* __restrict__ residual,
    float momentum,
    int64_t numel) {
    const int64_t index =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) {
        const float next_velocity =
            momentum * velocity[index] + grad[index];
        velocity[index] = next_velocity;
        residual[index] += next_velocity;
    }
}

__global__ void dgc_gather_reset_fp32_kernel(
    float* __restrict__ residual,
    float* __restrict__ velocity,
    const int32_t* __restrict__ indices,
    float* __restrict__ values,
    int64_t count) {
    const int64_t selected =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (selected < count) {
        const int32_t index = indices[selected];
        values[selected] = residual[index];
        residual[index] = 0.0f;
        if (velocity != nullptr) {
            velocity[index] = 0.0f;
        }
    }
}

__global__ void dgc_reconstruct_fp32_kernel(
    const int32_t* __restrict__ indices,
    const float* __restrict__ values,
    float* __restrict__ output,
    int64_t count) {
    const int64_t selected =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (selected < count) {
        atomicAdd(output + indices[selected], values[selected]);
    }
}

__global__ void dgc_radix_histogram_fp32_kernel(
    const float* __restrict__ residual,
    int64_t numel,
    uint32_t prefix,
    uint32_t prefix_mask,
    int shift,
    uint32_t* __restrict__ histogram) {
    __shared__ uint32_t local_histogram[256];
    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x) {
        local_histogram[bin] = 0;
    }
    __syncthreads();

    const int64_t start =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = start; index < numel; index += stride) {
        const uint32_t bits = __float_as_uint(fabsf(residual[index]));
        if ((bits & prefix_mask) == prefix) {
            const uint32_t digit = (bits >> shift) & 0xffu;
            atomicAdd(local_histogram + digit, 1u);
        }
    }
    __syncthreads();

    for (int bin = threadIdx.x; bin < 256; bin += blockDim.x) {
        const uint32_t count = local_histogram[bin];
        if (count != 0) {
            atomicAdd(histogram + bin, count);
        }
    }
}

__global__ void dgc_emit_selected_fp32_kernel(
    float* __restrict__ residual,
    float* __restrict__ velocity,
    int64_t numel,
    uint32_t threshold_bits,
    uint32_t greater_total,
    uint32_t ties_needed,
    uint32_t* __restrict__ counters,
    int32_t* __restrict__ output_indices,
    float* __restrict__ output_values) {
    const int64_t start =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride =
        static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t index = start; index < numel; index += stride) {
        const uint32_t bits = __float_as_uint(fabsf(residual[index]));
        uint32_t output_position = 0;
        bool emit = false;
        if (bits > threshold_bits) {
            output_position = atomicAdd(counters, 1u);
            emit = output_position < greater_total;
        } else if (bits == threshold_bits) {
            const uint32_t tie_position = atomicAdd(counters + 1, 1u);
            if (tie_position < ties_needed) {
                output_position = greater_total + tie_position;
                emit = true;
            }
        }

        if (emit) {
            output_indices[output_position] = static_cast<int32_t>(index);
            output_values[output_position] = residual[index];
            residual[index] = 0.0f;
            if (velocity != nullptr) {
                velocity[index] = 0.0f;
            }
        }
    }
}

void check_fp32_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kFloat32,
        name,
        " must have dtype torch.float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_int32_cuda_contiguous(
    const torch::Tensor& tensor,
    const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(
        tensor.scalar_type() == torch::kInt32,
        name,
        " must have dtype torch.int32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

void dgc_update_cuda(
    torch::Tensor grad,
    torch::Tensor velocity,
    torch::Tensor residual,
    double momentum) {
    check_fp32_cuda_contiguous(grad, "grad");
    check_fp32_cuda_contiguous(velocity, "velocity");
    check_fp32_cuda_contiguous(residual, "residual");
    TORCH_CHECK(grad.device() == velocity.device(), "device mismatch");
    TORCH_CHECK(grad.device() == residual.device(), "device mismatch");
    TORCH_CHECK(grad.numel() == velocity.numel(), "size mismatch");
    TORCH_CHECK(grad.numel() == residual.numel(), "size mismatch");
    TORCH_CHECK(momentum >= 0.0 && momentum < 1.0, "invalid momentum");

    const c10::cuda::CUDAGuard device_guard(grad.device());
    const int64_t numel = grad.numel();
    const int blocks = static_cast<int>((numel + kThreads - 1) / kThreads);
    const auto stream = at::cuda::getCurrentCUDAStream();

    dgc_update_fp32_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
        grad.data_ptr<float>(),
        velocity.data_ptr<float>(),
        residual.data_ptr<float>(),
        static_cast<float>(momentum),
        numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor dgc_gather_reset_cuda(
    torch::Tensor residual,
    torch::Tensor velocity,
    torch::Tensor indices) {
    check_fp32_cuda_contiguous(residual, "residual");
    check_int32_cuda_contiguous(indices, "indices");
    TORCH_CHECK(indices.device() == residual.device(), "device mismatch");
    TORCH_CHECK(residual.numel() <= INT32_MAX, "residual exceeds int32 index range");

    float* velocity_ptr = nullptr;
    if (velocity.defined() && velocity.numel() > 0) {
        check_fp32_cuda_contiguous(velocity, "velocity");
        TORCH_CHECK(velocity.device() == residual.device(), "device mismatch");
        TORCH_CHECK(velocity.numel() == residual.numel(), "size mismatch");
        velocity_ptr = velocity.data_ptr<float>();
    }

    const c10::cuda::CUDAGuard device_guard(residual.device());
    auto values = torch::empty({indices.numel()}, residual.options());
    const int64_t count = indices.numel();
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    const auto stream = at::cuda::getCurrentCUDAStream();

    dgc_gather_reset_fp32_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
        residual.data_ptr<float>(),
        velocity_ptr,
        indices.data_ptr<int32_t>(),
        values.data_ptr<float>(),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return values;
}

void dgc_reconstruct_cuda(
    torch::Tensor indices,
    torch::Tensor values,
    torch::Tensor output) {
    check_int32_cuda_contiguous(indices, "indices");
    check_fp32_cuda_contiguous(values, "values");
    check_fp32_cuda_contiguous(output, "output");
    TORCH_CHECK(indices.device() == values.device(), "device mismatch");
    TORCH_CHECK(indices.device() == output.device(), "device mismatch");
    TORCH_CHECK(indices.numel() == values.numel(), "size mismatch");
    TORCH_CHECK(output.numel() <= INT32_MAX, "output exceeds int32 index range");

    const c10::cuda::CUDAGuard device_guard(output.device());
    const int64_t count = indices.numel();
    const int blocks = static_cast<int>((count + kThreads - 1) / kThreads);
    const auto stream = at::cuda::getCurrentCUDAStream();

    dgc_reconstruct_fp32_kernel<<<blocks, kThreads, 0, stream.stream()>>>(
        indices.data_ptr<int32_t>(),
        values.data_ptr<float>(),
        output.data_ptr<float>(),
        count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<torch::Tensor> dgc_select_reset_cuda(
    torch::Tensor residual,
    torch::Tensor velocity,
    int64_t k) {
    check_fp32_cuda_contiguous(residual, "residual");
    TORCH_CHECK(residual.numel() <= INT32_MAX, "residual exceeds int32 index range");
    TORCH_CHECK(k >= 1 && k <= residual.numel(), "k must be in [1, numel]");

    float* velocity_ptr = nullptr;
    if (velocity.defined() && velocity.numel() > 0) {
        check_fp32_cuda_contiguous(velocity, "velocity");
        TORCH_CHECK(velocity.device() == residual.device(), "device mismatch");
        TORCH_CHECK(velocity.numel() == residual.numel(), "size mismatch");
        velocity_ptr = velocity.data_ptr<float>();
    }

    const c10::cuda::CUDAGuard device_guard(residual.device());
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto int_options = residual.options().dtype(torch::kInt32);
    auto histogram = torch::empty({256}, int_options);
    auto counters = torch::zeros({2}, int_options);
    auto output_indices = torch::empty({k}, int_options);
    auto output_values = torch::empty({k}, residual.options());

    std::array<uint32_t, 256> host_histogram{};
    uint32_t prefix = 0;
    uint32_t prefix_mask = 0;
    int64_t remaining = k;

    for (int shift = 24; shift >= 0; shift -= 8) {
        cudaMemsetAsync(
            histogram.data_ptr<int32_t>(),
            0,
            256 * sizeof(uint32_t),
            stream.stream());
        dgc_radix_histogram_fp32_kernel<<<
            kHistogramBlocks, kThreads, 0, stream.stream()>>>(
            residual.data_ptr<float>(),
            residual.numel(),
            prefix,
            prefix_mask,
            shift,
            reinterpret_cast<uint32_t*>(histogram.data_ptr<int32_t>()));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        cudaMemcpyAsync(
            host_histogram.data(),
            histogram.data_ptr<int32_t>(),
            256 * sizeof(uint32_t),
            cudaMemcpyDeviceToHost,
            stream.stream());
        C10_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));

        uint64_t higher_count = 0;
        int chosen_digit = -1;
        for (int digit = 255; digit >= 0; --digit) {
            const uint64_t bin_count = host_histogram[digit];
            if (higher_count + bin_count >= static_cast<uint64_t>(remaining)) {
                chosen_digit = digit;
                remaining -= static_cast<int64_t>(higher_count);
                break;
            }
            higher_count += bin_count;
        }
        TORCH_CHECK(chosen_digit >= 0, "radix selection failed to find threshold digit");
        prefix |= static_cast<uint32_t>(chosen_digit) << shift;
        prefix_mask |= 0xffu << shift;
    }

    const uint32_t ties_needed = static_cast<uint32_t>(remaining);
    const uint32_t greater_total = static_cast<uint32_t>(k - remaining);
    dgc_emit_selected_fp32_kernel<<<
        kHistogramBlocks, kThreads, 0, stream.stream()>>>(
        residual.data_ptr<float>(),
        velocity_ptr,
        residual.numel(),
        prefix,
        greater_total,
        ties_needed,
        reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>()),
        output_indices.data_ptr<int32_t>(),
        output_values.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_indices, output_values};
}
