#include <torch/extension.h>
#include <vector>

void dgc_update_cuda(
    torch::Tensor grad,
    torch::Tensor velocity,
    torch::Tensor residual,
    double momentum);

torch::Tensor dgc_gather_reset_cuda(
    torch::Tensor residual,
    torch::Tensor velocity,
    torch::Tensor indices);

void dgc_reconstruct_cuda(
    torch::Tensor indices,
    torch::Tensor values,
    torch::Tensor output);

std::vector<torch::Tensor> dgc_select_reset_cuda(
    torch::Tensor residual,
    torch::Tensor velocity,
    int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def(
        "dgc_update",
        &dgc_update_cuda,
        "Fused DGC velocity/residual update (CoreX CUDA)");
    module.def(
        "dgc_gather_reset",
        &dgc_gather_reset_cuda,
        "Gather selected residual values and reset residual/velocity (CoreX CUDA)");
    module.def(
        "dgc_reconstruct",
        &dgc_reconstruct_cuda,
        "Reconstruct a dense gradient from int32 sparse indices and FP32 values (CoreX CUDA)");
    module.def(
        "dgc_select_reset",
        &dgc_select_reset_cuda,
        "Exact low-workspace FP32 magnitude top-k with int32 output and state reset (CoreX CUDA)");
}
