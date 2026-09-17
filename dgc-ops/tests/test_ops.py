#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgc_ops_corex


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        max_error = (actual - expected).abs().max().item()
        raise AssertionError(f"{name} mismatch; max_abs_error={max_error}")


def test_update() -> None:
    torch.manual_seed(1234)
    numel = 1_000_003
    momentum = 0.9
    grad = torch.randn(numel, device="cuda", dtype=torch.float32)
    velocity = torch.randn_like(grad)
    residual = torch.randn_like(grad)
    expected_velocity = velocity.clone()
    expected_residual = residual.clone()

    expected_velocity.mul_(momentum).add_(grad)
    expected_residual.add_(expected_velocity)
    dgc_ops_corex.dgc_update(grad, velocity, residual, momentum)
    torch.cuda.synchronize()

    assert_close("velocity", velocity, expected_velocity)
    assert_close("residual", residual, expected_residual)


def test_gather_reset() -> None:
    residual = torch.arange(16, device="cuda", dtype=torch.float32)
    velocity = residual.clone().add_(100)
    indices = torch.tensor([1, 4, 9, 15], device="cuda", dtype=torch.int32)
    expected_values = residual[indices.to(torch.int64)].clone()

    values = dgc_ops_corex.dgc_gather_reset(residual, velocity, indices)
    torch.cuda.synchronize()

    assert_close("gathered values", values, expected_values)
    selected = indices.to(torch.int64)
    assert_close("residual reset", residual[selected], torch.zeros_like(values))
    assert_close("velocity reset", velocity[selected], torch.zeros_like(values))


def test_reconstruct() -> None:
    indices = torch.tensor([1, 3, 3, 7], device="cuda", dtype=torch.int32)
    values = torch.tensor([2.0, 4.0, 8.0, 16.0], device="cuda")
    output = torch.zeros(10, device="cuda", dtype=torch.float32)
    expected = output.clone()
    expected.scatter_add_(0, indices.to(torch.int64), values)

    dgc_ops_corex.dgc_reconstruct(indices, values, output)
    torch.cuda.synchronize()
    assert_close("reconstruct", output, expected)


def test_select_reset() -> None:
    torch.manual_seed(4321)
    numel = 1_000_003
    k = 10_003
    residual = torch.randn(numel, device="cuda", dtype=torch.float32)
    velocity = torch.randn_like(residual)
    residual_before = residual.clone()
    expected_values, expected_indices = torch.topk(
        residual_before.abs(), k, sorted=False
    )
    del expected_values

    indices32, values = dgc_ops_corex.dgc_select_reset(residual, velocity, k)
    torch.cuda.synchronize()
    indices64 = indices32.to(torch.int64)

    actual_sorted = torch.sort(indices64).values
    expected_sorted = torch.sort(expected_indices).values
    if not torch.equal(actual_sorted, expected_sorted):
        raise AssertionError("select_reset indices differ from exact torch.topk")
    assert_close("select_reset values", values, residual_before[indices64])
    assert_close("select_reset residual", residual[indices64], torch.zeros_like(values))
    assert_close("select_reset velocity", velocity[indices64], torch.zeros_like(values))


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/CoreX device is unavailable")
    test_update()
    test_gather_reset()
    test_reconstruct()
    test_select_reset()
    print("PASS: all CoreX CUDA operator correctness tests")
