#!/usr/bin/env python3
import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist


def measure(tensor: torch.Tensor, warmup: int, repeat: int) -> float:
    outputs = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    for _ in range(warmup):
        dist.all_gather(outputs, tensor)
    torch.cuda.synchronize()
    dist.barrier()
    samples = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter()
        dist.all_gather(outputs, tensor)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    dist.barrier()
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="403723,737956")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    for count in [int(value) for value in args.counts.split(",") if value]:
        index64 = torch.arange(count, device="cuda", dtype=torch.int64)
        index32 = index64.to(torch.int32)
        values = torch.randn(count, device="cuda", dtype=torch.float32)
        int64_ms = measure(index64, args.warmup, args.repeat)
        int32_ms = measure(index32, args.warmup, args.repeat)
        values_ms = measure(values, args.warmup, args.repeat)
        if rank == 0:
            print(
                f"count={count:,} int64={int64_ms:.3f} ms "
                f"int32={int32_ms:.3f} ms fp32={values_ms:.3f} ms "
                f"pair64={int64_ms + values_ms:.3f} ms "
                f"pair32={int32_ms + values_ms:.3f} ms"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
