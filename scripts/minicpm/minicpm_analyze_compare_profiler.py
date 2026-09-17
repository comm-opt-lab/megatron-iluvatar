#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare TG200 MiniCPM baseline and DGC PyTorch profiler traces.

Default behavior:
  1. Use the completed 16-layer baseline/DGC run profile paths.
  2. Write minicpm_base_profiler.log and minicpm_dgc_profiler.log by default.
  3. Print only ALL_COMM summary.
  4. Write only minicpm-dgc-result.csv.

Skip profiler log output:
  python3 analyze_compare_profiler_v3.py --no-write-profiler-logs
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable



DEFAULT_BASE_PROFILE_DIR = (
    "/home/yaowenxuan/ff/pretrain/minicpm/runs/20260806_101518/profile"
)

DEFAULT_OPT_PROFILE_DIR = (
    "/home/yaowenxuan/ff/pretrain/minicpm/runs/20260806_101622-dgc-density0.001/profile"
)

DEFAULT_RESULT_CSV = "minicpm-dgc-result.csv"
DEFAULT_BASE_LOG = "minicpm_base_profiler.log"
DEFAULT_OPT_LOG = "minicpm_dgc_profiler.log"


@dataclass
class Stat:
    count: int = 0
    dur_us: float = 0.0

    def add(self, dur_us: float) -> None:
        self.count += 1
        self.dur_us += dur_us

    def merge(self, other: "Stat") -> None:
        self.count += other.count
        self.dur_us += other.dur_us


@dataclass
class AnalysisResult:
    by_op: dict[str, Stat]
    total_comm_us: float
    files_found: int
    files_scanned: int
    comm_events: int
    profile_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base/optimized Megatron profiler communication time."
    )
    parser.add_argument(
        "--base-profile-dir",
        default=DEFAULT_BASE_PROFILE_DIR,
        help=f"Base profiler directory. Default: {DEFAULT_BASE_PROFILE_DIR}",
    )
    parser.add_argument(
        "--opt-profile-dir",
        default=DEFAULT_OPT_PROFILE_DIR,
        help=f"Optimized profiler directory. Default: {DEFAULT_OPT_PROFILE_DIR}",
    )
    parser.add_argument(
        "--result-csv",
        default=DEFAULT_RESULT_CSV,
        help=f"Output CSV path. Default: {DEFAULT_RESULT_CSV}",
    )
    parser.add_argument(
        "--no-write-profiler-logs",
        action="store_true",
        default=False,
        help="Skip writing base_profiler.log and opt_profiler.log.",
    )
    parser.add_argument(
        "--base-log",
        default=DEFAULT_BASE_LOG,
        help=f"Base profiler log path. Only used when profiler logs are written (default). Default: {DEFAULT_BASE_LOG}",
    )
    parser.add_argument(
        "--opt-log",
        default=DEFAULT_OPT_LOG,
        help=f"Optimized profiler log path. Only used when profiler logs are written (default). Default: {DEFAULT_OPT_LOG}",
    )
    parser.add_argument(
        "--event-source",
        choices=("comm", "nccl", "c10d", "hccl-kernel", "all"),
        default="nccl",
        help=(
            "Which trace events to count. "
            "Default nccl counts TG200 profiler events whose names begin with nccl:."
        ),
    )
    parser.add_argument(
        "--include-coalesced",
        action="store_true",
        help="Include nccl:coalesced wrapper events.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="Only scan first N json files. 0 means all.",
    )
    parser.add_argument(
        "--top-ops",
        type=int,
        default=20,
        help="Number of Top Communication Operators written into profiler logs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker processes. Default: 4.",
    )
    return parser.parse_args()


def iter_trace_files(profile_dir: Path) -> Iterable[Path]:
    return sorted(p for p in profile_dir.rglob("*.json") if p.is_file())


def is_counted_comm_event(name: str, event_source: str) -> bool:
    low = name.lower()

    is_nccl = low.startswith("nccl:")
    is_c10d = low.startswith("c10d::")
    is_hccl_kernel = low.startswith("hcclkernel_")
    is_c10d_p2p = is_c10d and any(token in low for token in ("::send", "::recv"))

    if event_source == "comm":
        return is_hccl_kernel or is_c10d_p2p
    if event_source == "nccl":
        return is_nccl
    if event_source == "c10d":
        return is_c10d
    if event_source == "hccl-kernel":
        return is_hccl_kernel
    return is_hccl_kernel or is_nccl or is_c10d


def read_events(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    events = data.get("traceEvents", [])
    return events if isinstance(events, list) else []


def analyze_one_file(
    path_str: str,
    event_source: str,
    include_coalesced: bool,
) -> dict:
    path = Path(path_str)

    by_op: dict[str, Stat] = defaultdict(Stat)
    events_scanned = 0
    comm_events = 0

    try:
        events = read_events(path)
    except Exception as exc:
        return {
            "ok": False,
            "path": path_str,
            "error": repr(exc),
            "events_scanned": 0,
            "comm_events": 0,
            "by_op": {},
        }

    for event in events:
        events_scanned += 1

        name = str(event.get("name", ""))
        if not is_counted_comm_event(name, event_source):
            continue

        if not include_coalesced and name.lower() == "nccl:coalesced":
            continue

        dur = event.get("dur")
        if dur is None:
            continue

        try:
            dur_us = float(dur)
        except (TypeError, ValueError):
            continue

        by_op[name].add(dur_us)
        comm_events += 1

    return {
        "ok": True,
        "path": path_str,
        "error": "",
        "events_scanned": events_scanned,
        "comm_events": comm_events,
        "by_op": dict(by_op),
    }


def merge_stat_dict(dst: dict[str, Stat], src: dict[str, Stat]) -> None:
    for key, stat in src.items():
        dst[key].merge(stat)


def run_profiler_analysis(
    profile_dir: str | Path,
    *,
    event_source: str,
    include_coalesced: bool,
    max_files: int,
    workers: int,
) -> AnalysisResult:
    profile_dir = Path(profile_dir)

    if not profile_dir.exists():
        raise SystemExit(f"ERROR: profile_dir does not exist: {profile_dir}")

    files = list(iter_trace_files(profile_dir))
    if max_files > 0:
        files = files[:max_files]

    if not files:
        raise SystemExit(f"ERROR: no json trace files found in: {profile_dir}")

    by_op: dict[str, Stat] = defaultdict(Stat)
    files_scanned = 0
    comm_events = 0
    failed_files: list[tuple[str, str]] = []

    workers = max(workers, 1)

    if workers == 1:
        for path in files:
            result = analyze_one_file(str(path), event_source, include_coalesced)

            if not result["ok"]:
                failed_files.append((result["path"], result["error"]))
                continue

            files_scanned += 1
            comm_events += result["comm_events"]
            merge_stat_dict(by_op, result["by_op"])
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    analyze_one_file,
                    str(path),
                    event_source,
                    include_coalesced,
                )
                for path in files
            ]

            for future in as_completed(futures):
                result = future.result()

                if not result["ok"]:
                    failed_files.append((result["path"], result["error"]))
                    continue

                files_scanned += 1
                comm_events += result["comm_events"]
                merge_stat_dict(by_op, result["by_op"])

    if failed_files:
        print(f"WARNING: failed files: {len(failed_files)}")
        for failed_path, error in failed_files[:5]:
            print(f"  - {failed_path}: {error}")

    total_comm_us = sum(stat.dur_us for stat in by_op.values())

    return AnalysisResult(
        by_op=dict(by_op),
        total_comm_us=total_comm_us,
        files_found=len(files),
        files_scanned=files_scanned,
        comm_events=comm_events,
        profile_dir=str(profile_dir),
    )


def safe_speedup(base_ms: float, opt_ms: float) -> float:
    if opt_ms == 0:
        return float("inf")
    return base_ms / opt_ms


def safe_decrease_pct(base_ms: float, opt_ms: float) -> float:
    if base_ms <= 0:
        return 0.0
    return (base_ms - opt_ms) / base_ms * 100.0


def pct(value: float, total: float) -> str:
    if total <= 0:
        return "  0.00%"
    return f"{(value / total * 100):6.2f}%"


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    print(title)
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    print("   ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * (sum(widths) + 3 * (len(widths) - 1)))
    for row in rows:
        print(
            "   ".join(
                row[i].rjust(widths[i]) if i else row[i].ljust(widths[i])
                for i in range(len(row))
            )
        )
    print()


def build_profiler_log(result: AnalysisResult, *, event_source: str, workers: int, top_ops: int) -> str:
    total_comm_us = result.total_comm_us

    op_rows: list[list[str]] = []
    for name, stat in sorted(
        result.by_op.items(),
        key=lambda item: item[1].dur_us,
        reverse=True,
    )[:top_ops]:
        op_rows.append(
            [
                name,
                f"{stat.count:,}",
                f"{stat.dur_us / 1000.0:,.2f}",
                f"{stat.dur_us / max(stat.count, 1) / 1000.0:,.2f}",
                pct(stat.dur_us, total_comm_us),
            ]
        )

    buf = io.StringIO()
    with redirect_stdout(buf):
        print()
        print("PyTorch Profiler Communication Summary")
        print("=" * 45)
        print(f"Profile dir       : {result.profile_dir}")
        print(f"Event source      : {event_source}")
        print(f"Workers           : {workers}")
        print(f"Files found       : {result.files_found}")
        print(f"Files scanned     : {result.files_scanned}")
        print(f"Comm events       : {result.comm_events}")
        print()
        print_table(
            f"Top {min(top_ops, len(result.by_op))} Communication Operators",
            ["Operator", "Count", "Total ms", "Avg ms", "% all comm"],
            op_rows,
        )

    return buf.getvalue()


def write_result_csv(
    base_ms: float,
    opt_ms: float,
    result_csv: str | Path,
) -> None:
    delta_ms = base_ms - opt_ms
    decrease_pct = safe_decrease_pct(base_ms, opt_ms)
    verdict = "PASS" if decrease_pct > 10.0 else "FAILED"

    with Path(result_csv).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item",
                "baseline_ms",
                "optimized_ms",
                "delta_ms",
                "decrease_pct",
                "verdict",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "item": "ALL_COMM",
                "baseline_ms": f"{base_ms:.2f}",
                "optimized_ms": f"{opt_ms:.2f}",
                "delta_ms": f"{delta_ms:.2f}",
                "decrease_pct": f"{decrease_pct:.2f}",
                "verdict": verdict,
            }
        )


def print_summary(
    base_path: str,
    opt_path: str,
    base_ms: float,
    opt_ms: float,
) -> None:
    delta_ms = base_ms - opt_ms
    decrease_pct = safe_decrease_pct(base_ms, opt_ms)
    # speedup = safe_speedup(base_ms, opt_ms)
    verdict = "PASS" if decrease_pct > 10.0 else "FAILED"

    print()
    print("Communication Operator Speedup")
    print("=" * 130)
    print(f"Baseline : {base_path}")
    print(f"Optimized: {opt_path}")
    print()
    print("ALL_COMM based on Top Communication Operators")
    print("-" * 130)
    print(
        f"{'Item':<32} "
        f"{'Baseline ms':>18} "
        f"{'Optimized ms':>18} "
        f"{'Delta ms':>18} "
        f"{'Decrease':>12} "
        f"{'Verdict':>8}"
    )
    print("-" * 130)

    print(
        f"{'ALL_COMM':<32} "
        f"{base_ms:>18,.2f} "
        f"{opt_ms:>18,.2f} "
        f"{delta_ms:>18,.2f} "
        f"{decrease_pct:>11.2f}% "
        f"{verdict:>8}"
    )

    print("Notes:")
    print("  decrease = (baseline_total_ms - optimized_total_ms) / baseline_total_ms")
    print("  verdict  = PASS if decrease > 10%, else FAILED")
    print("  ALL_COMM = sum(Total ms) of parsed Top Communication Operators")


def main() -> int:
    args = parse_args()

    base_result = run_profiler_analysis(
        args.base_profile_dir,
        event_source=args.event_source,
        include_coalesced=args.include_coalesced,
        max_files=args.max_files,
        workers=args.workers,
    )

    opt_result = run_profiler_analysis(
        args.opt_profile_dir,
        event_source=args.event_source,
        include_coalesced=args.include_coalesced,
        max_files=args.max_files,
        workers=args.workers,
    )

    base_ms = base_result.total_comm_us / 1000.0
    opt_ms = opt_result.total_comm_us / 1000.0

    write_result_csv(base_ms, opt_ms, args.result_csv)

    if not args.no_write_profiler_logs:
        Path(args.base_log).write_text(
            build_profiler_log(
                base_result,
                event_source=args.event_source,
                workers=args.workers,
                top_ops=args.top_ops,
            ),
            encoding="utf-8",
        )
        Path(args.opt_log).write_text(
            build_profiler_log(
                opt_result,
                event_source=args.event_source,
                workers=args.workers,
                top_ops=args.top_ops,
            ),
            encoding="utf-8",
        )

    print_summary(
        str(args.base_profile_dir),
        str(args.opt_profile_dir),
        base_ms,
        opt_ms,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
