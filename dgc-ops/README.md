# DGC CoreX CUDA operators

Standalone CoreX CUDA operator experiments for the TG200 DGC path. Lives in
`dgc-ops/` so the vendor Megatron tree (`Megatron-LM/`) stays unchanged.
Built artifacts (`.so`, `build/`) are not committed.

## Build

```bash
export CUDA_HOME=/usr/local/corex-1.2.0
export PATH="${CUDA_HOME}/bin:${PATH}"
python3 setup.py build_ext --inplace
```

## Test

```bash
CUDA_VISIBLE_DEVICES=6 python3 tests/test_ops.py
```

## Benchmark

```bash
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_ops.py
```

Sparse gather/reset and reconstruction benchmark:

```bash
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_sparse.py
```

Exact low-workspace magnitude selection benchmark:

```bash
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_select.py
```

Complete local compression pipeline benchmark:

```bash
CUDA_VISIBLE_DEVICES=6 python3 benchmarks/bench_pipeline.py
```
