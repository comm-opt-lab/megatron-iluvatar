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


def benchmark_size(numel: int, momentum: float, warmup: int, repeat: int) -> dict:
    torch.cuda.empty_cache()
    grad = torch.empty(numel, device="cuda", dtype=torch.float32).uniform_(-1, 1)
    velocity = torch.zeros_like(grad)
    residual = torch.zeros_like(grad)
    torch.cuda.synchronize()

    def torch_update() -> None:
        velocity.mul_(momentum).add_(grad)
        residual.add_(velocity)

    torch_ms = elapsed_ms(torch_update, warmup, repeat)
    velocity.zero_()
    residual.zero_()
    torch.cuda.synchronize()

    def corex_update() -> None:
        dgc_ops_corex.dgc_update(grad, velocity, residual, momentum)

    corex_ms = elapsed_ms(corex_update, warmup, repeat)
    allocated_gib = torch.cuda.memory_allocated() / (1024**3)
    reserved_gib = torch.cuda.memory_reserved() / (1024**3)
    speedup = torch_ms / corex_ms
    return {
        "numel": numel,
        "state_gib": numel * 4 * 3 / (1024**3),
        "torch_ms": torch_ms,
        "corex_ms": corex_ms,
        "speedup": speedup,
        "allocated_gib": allocated_gib,
        "reserved_gib": reserved_gib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="16777216,67108864,403723264,737956864",
        help="Comma-separated FP32 element counts",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", default="results/update_benchmark.csv")
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",") if value]
    rows = []
    for numel in sizes:
        row = benchmark_size(numel, args.momentum, args.warmup, args.repeat)
        rows.append(row)
        print(
            f"numel={numel:,} torch={row['torch_ms']:.3f} ms "
            f"corex={row['corex_ms']:.3f} ms speedup={row['speedup']:.3f}x "
            f"allocated={row['allocated_gib']:.2f} GiB"
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
