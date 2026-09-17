# Copyright (c) 2026. Local addition (not part of upstream NVIDIA Megatron-Core).

"""
Deep Gradient Compression (DGC) sparse gradient synchronization.

Implements the top-k sparsification with momentum correction and error-feedback
described in "Deep Gradient Compression: Reducing the Communication Bandwidth for
Distributed Training" (Lin et al., 2017, https://arxiv.org/abs/1712.01887), adapted to
Megatron-Core's bucketed gradient buffers (see `_ParamAndGradBucketGroup` in
param_and_grad_buffer.py for the integration point).

See megatron/core/distributed/DGC_IMPLEMENTATION.md for full design notes, the
communication-cost trade-off, and known limitations (e.g. distributed-optimizer /
overlap_grad_reduce are not yet supported by this code path).
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch


@dataclass
class DGCConfig:
    """Validated runtime parameters for the DGC compressor.

    Mirrors the `dgc_*` fields of `DistributedDataParallelConfig`; kept as a separate
    small dataclass so `DGCCompressor` has no dependency on the (much larger) DDP config.
    """

    density: float = 1.0
    """Fraction of gradient elements retained and communicated per bucket, in (0, 1].
    E.g. 0.01 means 1% of the gradient is communicated each step (99% sparsity)."""

    momentum: float = 0.9
    """Momentum correction factor for the local velocity (error-feedback) buffer."""

    min_numel_to_compress: int = 16384
    """Buckets with fewer elements than this fall back to dense all-reduce: for small
    buckets the index/value pair overhead outweighs any bandwidth savings."""

    implementation: str = "torch"
    """Local compression implementation: reference PyTorch or CoreX CUDA extension."""

    def __post_init__(self):
        assert 0.0 < self.density <= 1.0, "dgc_density must be in (0, 1]"
        assert 0.0 <= self.momentum < 1.0, "dgc_momentum must be in [0, 1)"
        assert self.min_numel_to_compress >= 1, "dgc_min_numel_to_compress must be >= 1"
        assert self.implementation in ("torch", "corex"), (
            "dgc_impl must be one of: torch, corex"
        )


class _DGCBucketState:
    """Per-bucket error-feedback state: DGC velocity (momentum) and residual buffers.

    Both buffers are the same size as the bucket's gradient buffer and persist for the
    entire run, so enabling DGC costs an extra 1x (momentum disabled) or 2x (momentum
    enabled) the bucket's grad memory, per rank, on top of what dense training already
    uses -- see DGCCompressor.state_bytes_per_rank() and DGC_IMPLEMENTATION.md section 8
    (out-of-memory troubleshooting).
    """

    __slots__ = ("velocity", "residual")

    def __init__(self, numel: int, device: torch.device, dtype: torch.dtype, use_velocity: bool):
        self.velocity = (
            torch.zeros(numel, device=device, dtype=dtype) if use_velocity else None
        )
        self.residual = torch.zeros(numel, device=device, dtype=dtype)


class DGCCompressor:
    """
    Computes the local top-k sparsified gradient for a bucket, maintaining the
    momentum-corrected error-feedback buffers (velocity, residual) that recover, over
    time, the gradient information dropped by sparsification in earlier steps.

    One instance is owned per `_ParamAndGradBucketGroup`. State is keyed by the Python
    `id()` of each bucket's `grad_data` tensor (not `bucket.bucket_id`, which is only
    unique within a single underlying `_ParamAndGradBuffer` and can collide across
    buffers, e.g. the fp8/bf16 buffers grouped together for fp8 models).
    """

    def __init__(self, config: DGCConfig):
        self.config = config
        self._state: Dict[int, _DGCBucketState] = {}
        self._corex_ops = None
        if config.implementation == "corex":
            try:
                import dgc_ops_corex
            except ImportError as error:
                raise RuntimeError(
                    "--dgc-impl corex requires the compiled dgc_ops_corex extension "
                    "on PYTHONPATH"
                ) from error
            self._corex_ops = dgc_ops_corex

    def should_compress(self, numel: int) -> bool:
        """Whether a bucket of this size is worth sparsifying."""
        return numel >= self.config.min_numel_to_compress

    def topk_count(self, numel: int) -> int:
        """Number of elements retained/communicated for a bucket of the given size."""
        return max(1, int(numel * self.config.density))

    def state_bytes_per_rank(self, numel: int, dtype: torch.dtype) -> int:
        """Extra GPU memory (bytes), per rank, the error-feedback state for a bucket of
        this size will occupy: one `residual` buffer always, plus one `velocity` buffer
        when `config.momentum > 0`."""
        element_size = torch.empty((), dtype=dtype).element_size()
        multiplier = 2 if self.config.momentum > 0.0 else 1
        return numel * element_size * multiplier

    def _get_state(self, key: int, grad: torch.Tensor) -> _DGCBucketState:
        state = self._state.get(key)
        if state is None:
            state = _DGCBucketState(
                grad.numel(), grad.device, grad.dtype, use_velocity=self.config.momentum > 0.0
            )
            self._state[key] = state
        return state

    def preallocate(self, grad: torch.Tensor) -> None:
        """Eagerly allocate this bucket's error-feedback state (instead of lazily on the
        first `compress()` call), so an out-of-memory failure -- if the state genuinely
        does not fit -- happens immediately at DDP construction time with a clear cause,
        rather than deep inside a later backward pass."""
        self._get_state(id(grad), grad)

    def compress(self, grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select the top-k elements of `grad` (by magnitude) after momentum correction
        (when `config.momentum > 0`) and error-feedback accumulation.

        Does NOT modify `grad` itself; the caller is responsible for replacing its
        contents with the globally-reduced sparse result (see
        `_ParamAndGradBucketGroup._start_grad_sync_dgc`). Returns `(indices, values)`,
        both flat 1-D tensors of length `topk_count(grad.numel())` on `grad`'s device.
        The reference implementation returns int64 indices; the CoreX implementation
        returns int32 indices and uses the matching custom reconstruction kernel.
        """
        key = id(grad)
        state = self._get_state(key, grad)

        if self.config.implementation == "corex":
            if grad.dtype != torch.float32 or not grad.is_contiguous():
                raise RuntimeError(
                    "CoreX DGC operators currently require contiguous FP32 grad buffers"
                )
            if grad.numel() > 2**31 - 1:
                raise RuntimeError("CoreX DGC int32 indices require numel <= INT32_MAX")
            if state.velocity is not None:
                self._corex_ops.dgc_update(
                    grad, state.velocity, state.residual, self.config.momentum
                )
                velocity = state.velocity
            else:
                state.residual.add_(grad)
                velocity = torch.empty(0, device=grad.device, dtype=grad.dtype)
            return tuple(
                self._corex_ops.dgc_select_reset(
                    state.residual, velocity, self.topk_count(grad.numel())
                )
            )

        # DGC momentum correction + local gradient accumulation (Lin et al., eq. 1-2).
        # When momentum is 0, the velocity buffer is skipped entirely (it would be a
        # pure passthrough of grad anyway) to halve the error-feedback memory cost.
        if state.velocity is not None:
            state.velocity.mul_(self.config.momentum).add_(grad)
            state.residual.add_(state.velocity)
        else:
            state.residual.add_(grad)

        k = self.topk_count(grad.numel())
        _, indices = torch.topk(state.residual.abs(), k, sorted=False)
        values = state.residual[indices].clone()

        # Error feedback: communicated positions restart accumulation from zero; all
        # other positions keep accumulating toward a future communication round.
        if state.velocity is not None:
            state.velocity[indices] = 0.0
        state.residual[indices] = 0.0

        return indices, values

    def reconstruct(
        self,
        gathered_indices: Iterable[torch.Tensor],
        gathered_values: Iterable[torch.Tensor],
        output: torch.Tensor,
    ) -> None:
        """Reconstruct the globally summed sparse gradient into ``output``."""
        output.zero_()
        if self.config.implementation == "corex":
            for rank_indices, rank_values in zip(gathered_indices, gathered_values):
                self._corex_ops.dgc_reconstruct(rank_indices, rank_values, output)
            return
        for rank_indices, rank_values in zip(gathered_indices, gathered_values):
            output.scatter_add_(0, rank_indices, rank_values)
