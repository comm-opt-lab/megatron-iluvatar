#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dgc_ops_corex


def timed_once(fn):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)
    extra_peak = torch.cuda.max_memory_allocated() - base_allocated
    return elapsed, extra_peak / (1024**3), result


def benchmark_size(numel: int, density: float) -> dict:
    k = max(1, int(numel * density))
    original = torch.empty(numel, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    residual = original.clone()
    velocity = torch.zeros_like(residual)

    def torch_select():
        _, indices = torch.topk(residual.abs(), k, sorted=False)
        values = residual[indices].clone()
        velocity[indices] = 0.0
        residual[indices] = 0.0
        return indices, values

    torch_ms, torch_peak_gib, _ = timed_once(torch_select)
    residual.copy_(original)
    velocity.zero_()

    def corex_select():
        return dgc_ops_corex.dgc_select_reset(residual, velocity, k)

    corex_ms, corex_peak_gib, _ = timed_once(corex_select)
    del original, residual, velocity
    torch.cuda.empty_cache()
    return {
        "numel": numel,
        "density": density,
        "k": k,
        "torch_ms": torch_ms,
        "corex_ms": corex_ms,
        "speedup": torch_ms / corex_ms,
        "torch_extra_peak_gib": torch_peak_gib,
        "corex_extra_peak_gib": corex_peak_gib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="16777216,67108864,403723264,737956864")
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--output", default="results/select_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for numel in [int(value) for value in args.sizes.split(",") if value]:
        try:
            row = benchmark_size(numel, args.density)
        except torch.OutOfMemoryError as error:
            print(f"OOM numel={numel:,}: {error}")
            torch.cuda.empty_cache()
            continue
        rows.append(row)
        print(
            f"numel={numel:,} k={row['k']:,} "
            f"time={row['torch_ms']:.3f}->{row['corex_ms']:.3f} ms "
            f"speedup={row['speedup']:.3f}x "
            f"extra_peak={row['torch_extra_peak_gib']:.3f}->{row['corex_extra_peak_gib']:.3f} GiB"
        )

    if rows:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
