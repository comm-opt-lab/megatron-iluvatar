# TG200 DGC CoreX operator results

Date: 2026-08-08

## Scope

The vendor baseline `/home/yaowenxuan/ff/tg200/ixmegatron` was not modified.
Operator development lives in `/home/yaowenxuan/ff/tg200/dgc-ops`. Integration is
guarded by `--dgc-impl {torch,corex}` in `ixmegatron-speedup`, with `torch` remaining
the default fallback.

## Implemented operators

- Fused FP32 velocity/residual update.
- Int32 gather/reset for already-selected indices.
- FP32 atomic sparse reconstruction from int32 indices.
- Exact FP32 magnitude top-k based on four 8-bit radix histogram passes plus one
  compact/reset pass. It does not materialize `residual.abs()` and does not sort the
  full bucket.

## Correctness

`tests/test_ops.py` passes all checks:

- fused state update versus PyTorch;
- gather/reset values and state mutation;
- reconstruct with duplicate indices;
- exact selected index set versus `torch.topk` for 1,000,003 random FP32 values.

The 3-step eight-GPU smoke run completed without NaN, skipped iterations, OOM, or
runtime errors:

```text
/home/yaowenxuan/ff/pretrain/minicpm/runs/
20260808_034914-dgc-corex-density0.001
```

## Microbenchmarks

### Fused state update

| Elements | PyTorch | CoreX | Speedup |
|---:|---:|---:|---:|
| 16,777,216 | 0.459 ms | 0.286 ms | 1.60x |
| 67,108,864 | 1.765 ms | 1.079 ms | 1.64x |
| 403,723,264 | 10.532 ms | 6.469 ms | 1.63x |
| 737,956,864 | 19.260 ms | 11.839 ms | 1.63x |

IXKN reports about 90-93% achieved occupancy, about 98% LSU utilization, and roughly
1150-1200 GB/s aggregate memory access throughput for the fused update kernel.

### Sparse helper operators

| Elements | Gather/reset | Speedup | Reconstruct | Speedup |
|---:|---:|---:|---:|---:|
| 403,723,264 | 0.377 -> 0.294 ms | 1.28x | 1.367 -> 1.351 ms | 1.01x |
| 737,956,864 | 0.627 -> 0.527 ms | 1.19x | 2.489 -> 2.471 ms | 1.01x |

Reconstruction is dominated by zeroing the full dense output; optimizing only the
atomic scatter is therefore not a current priority.

### Exact selection

| Elements | k | PyTorch | CoreX | Speedup | Extra peak memory |
|---:|---:|---:|---:|---:|---:|
| 67,108,864 | 67,108 | 15.436 ms | 7.669 ms | 2.01x | 2.251 GiB -> 0.001 GiB |
| 403,723,264 | 403,723 | 89.234 ms | 45.388 ms | 1.97x | 13.541 GiB -> 0.004 GiB |
| 737,956,864 | 737,956 | 162.490 ms | 82.827 ms | 1.96x | 24.753 GiB -> 0.006 GiB |

The first small 16M PyTorch measurement contained one-time warmup and is intentionally
not used as a stable speedup claim.

### Complete local compression pipeline

| Elements | PyTorch | CoreX | Speedup | Extra peak memory |
|---:|---:|---:|---:|---:|
| 403,723,264 | 172.439 ms | 56.577 ms | 3.05x | 13.541 GiB -> 0.004 GiB |
| 737,956,864 | 182.685 ms | 102.404 ms | 1.78x | 24.753 GiB -> 0.006 GiB |

## End-to-end no-profiler A/B

All runs used 8 GPUs, TP1/PP4/DP2, 16 layers, sequence length 64, global batch 32,
10 iterations, and profiler disabled. Mean below is iterations 2-10.

| Variant | Run | Mean iteration | Relative result |
|---|---|---:|---:|
| Dense baseline | `20260808_035338` | 1390.17 ms | reference |
| PyTorch DGC | `20260808_035421-dgc-density0.001` | 1517.57 ms | 9.16% slower than dense |
| CoreX DGC | `20260808_035504-dgc-corex-density0.001` | 1451.50 ms | 4.35% faster than PyTorch DGC; 4.41% slower than dense |

For iterations 3-10, CoreX DGC is 5.58% faster than PyTorch DGC and 3.21% slower
than dense.

## Profiler caveat

The ten-step PyTorch-profiler run completed and generated eight traces:

```text
20260808_035022-dgc-corex-density0.001
```

Its `nccl:all_gather` annotation time was unexpectedly inflated. A standalone two-GPU
collective benchmark showed int32+FP32 all-gather at about 1.002 ms versus int64+FP32
at about 0.883 ms for k=737,956, so dtype does not explain the multi-fold profiler
increase. End-to-end claims therefore use the same-session profiler-off A/B above.

IXKN hardware counter replay intentionally makes timed iterations hundreds or thousands
of times slower and must not be interpreted as normal kernel latency. IXKN also emitted
an internal `std::out_of_range` after exporting the histogram profile, but the profile
remained importable and the target application did not fail.

## Next optimization priorities

1. Replace the first radix histogram's single shared 256-bin table with warp/private
   histograms and hierarchical reduction. The first pass is about 5.66 ms at 67M
   elements because magnitude exponents are concentrated into a small number of bins.
2. Move radix digit selection fully onto the device to remove four host synchronizations.
3. Investigate avoiding full dense `output.zero_()` during reconstruction, for example
   generation tags or sparse optimizer consumption. Optimizing atomic scatter alone has
   negligible benefit.
4. Add a profiler methodology that correctly attributes custom extension kernels and
   NCCL wait time before using trace-level ALL_COMM as an acceptance gate.
5. Run longer repeated A/B tests and convergence checks before changing the default from
   `torch` to `corex`.

## Reproduction

```bash
cd /home/yaowenxuan/ff/tg200/dgc-ops
export CUDA_HOME=/usr/local/corex-1.2.0
python3 setup.py build_ext --inplace
CUDA_VISIBLE_DEVICES=6 python3 tests/test_ops.py
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_ops.py
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_sparse.py
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_select.py
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_pipeline.py
```
