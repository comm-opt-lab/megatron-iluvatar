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
    return (
        start.elapsed_time(end),
        (torch.cuda.max_memory_allocated() - base_allocated) / (1024**3),
        result,
    )


def run_size(numel: int, density: float, momentum: float) -> dict:
    k = max(1, int(numel * density))
    grad = torch.empty(numel, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    initial_velocity = torch.empty_like(grad).uniform_(-1, 1)
    initial_residual = torch.empty_like(grad).uniform_(-1, 1)
    velocity = initial_velocity.clone()
    residual = initial_residual.clone()

    def torch_pipeline():
        velocity.mul_(momentum).add_(grad)
        residual.add_(velocity)
        _, indices = torch.topk(residual.abs(), k, sorted=False)
        values = residual[indices].clone()
        velocity[indices] = 0.0
        residual[indices] = 0.0
        return indices, values

    torch_ms, torch_peak_gib, _ = timed_once(torch_pipeline)
    velocity.copy_(initial_velocity)
    residual.copy_(initial_residual)

    def corex_pipeline():
        dgc_ops_corex.dgc_update(grad, velocity, residual, momentum)
        return dgc_ops_corex.dgc_select_reset(residual, velocity, k)

    corex_ms, corex_peak_gib, _ = timed_once(corex_pipeline)
    return {
        "numel": numel,
        "density": density,
        "momentum": momentum,
        "k": k,
        "torch_ms": torch_ms,
        "corex_ms": corex_ms,
        "speedup": torch_ms / corex_ms,
        "torch_extra_peak_gib": torch_peak_gib,
        "corex_extra_peak_gib": corex_peak_gib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="403723264,737956864")
    parser.add_argument("--density", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output", default="results/pipeline_benchmark.csv")
    args = parser.parse_args()

    rows = []
    for numel in [int(value) for value in args.sizes.split(",") if value]:
        row = run_size(numel, args.density, args.momentum)
        rows.append(row)
        print(
            f"numel={numel:,} k={row['k']:,} "
            f"pipeline={row['torch_ms']:.3f}->{row['corex_ms']:.3f} ms "
            f"speedup={row['speedup']:.3f}x "
            f"extra_peak={row['torch_extra_peak_gib']:.3f}->{row['corex_extra_peak_gib']:.3f} GiB"
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
