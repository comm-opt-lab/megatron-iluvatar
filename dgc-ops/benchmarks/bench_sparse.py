#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgc_ops_corex


def elapsed_ms(fn, warmup: int, repeat: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def benchmark_size(
    numel: int,
    density: float,
    dp_size: int,
    warmup: int,
    repeat: int,
) -> dict:
    torch.cuda.empty_cache()
    selected = max(1, int(numel * density))
    stride = max(1, numel // selected)
    indices64 = torch.arange(selected, device="cuda", dtype=torch.int64) * stride
    indices32 = indices64.to(torch.int32)
    residual = torch.empty(numel, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    velocity = torch.empty_like(residual).uniform_(-1, 1)

    def torch_gather_reset() -> None:
        residual[indices64].clone()
        velocity[indices64] = 0.0
        residual[indices64] = 0.0

    torch_gather_ms = elapsed_ms(torch_gather_reset, warmup, repeat)
    residual.uniform_(-1, 1)
    velocity.uniform_(-1, 1)

    def corex_gather_reset() -> None:
        dgc_ops_corex.dgc_gather_reset(residual, velocity, indices32)

    corex_gather_ms = elapsed_ms(corex_gather_reset, warmup, repeat)

    del residual
    del velocity
    output = torch.empty(numel, device="cuda", dtype=torch.float32)
    rank_indices64 = [indices64 for _ in range(dp_size)]
    flat_indices32 = indices32.repeat(dp_size)
    rank_values = [torch.randn(selected, device="cuda") for _ in range(dp_size)]
    flat_values = torch.cat(rank_values)

    def torch_reconstruct() -> None:
        output.zero_()
        for rank_index, rank_value in zip(rank_indices64, rank_values):
            output.scatter_add_(0, rank_index, rank_value)

    torch_reconstruct_ms = elapsed_ms(torch_reconstruct, warmup, repeat)

    def corex_reconstruct() -> None:
        output.zero_()
        dgc_ops_corex.dgc_reconstruct(flat_indices32, flat_values, output)

    corex_reconstruct_ms = elapsed_ms(corex_reconstruct, warmup, repeat)
    return {
        "numel": numel,
        "density": density,
        "selected_per_rank": selected,
        "dp_size": dp_size,
        "torch_gather_reset_ms": torch_gather_ms,
        "corex_gather_reset_ms": corex_gather_ms,
        "gather_reset_speedup": torch_gather_ms / corex_gather_ms,
        "torch_reconstruct_ms": torch_reconstruct_ms,
        "corex_reconstruct_ms": corex_reconstruct_ms,
        "reconstruct_speedup": torch_reconstruct_ms / corex_reconstruct_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="403723264,737956864")
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--output", default="results/sparse_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for numel in [int(value) for value in args.sizes.split(",") if value]:
        row = benchmark_size(
            numel,
            args.density,
            args.dp_size,
            args.warmup,
            args.repeat,
        )
        rows.append(row)
        print(
            f"numel={numel:,} k={row['selected_per_rank']:,} "
            f"gather_reset={row['torch_gather_reset_ms']:.3f}->{row['corex_gather_reset_ms']:.3f} ms "
            f"({row['gather_reset_speedup']:.3f}x) "
            f"reconstruct={row['torch_reconstruct_ms']:.3f}->{row['corex_reconstruct_ms']:.3f} ms "
            f"({row['reconstruct_speedup']:.3f}x)"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
