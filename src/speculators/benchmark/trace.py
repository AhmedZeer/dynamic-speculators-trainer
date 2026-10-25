"""Summarize bounded Chrome/Kineto traces without treating them as benchmarks."""

import csv
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

from speculators.bank.artifacts import write_json

CPU_CATEGORIES = {
    "cpu_op",
    "user_annotation",
    "python_function",
    "cuda_runtime",
    "cuda_driver",
}
GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
ITERATION = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def union_duration(intervals):
    total, end = 0.0, None
    for start, stop in sorted(intervals):
        if end is None or start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _cpu_rows(events):  # noqa: C901 -- nested interval accounting and launch IDs
    """Exclusive CPU time subtracts the union of contained child intervals."""
    threads = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") in CPU_CATEGORIES:
            threads[(event.get("pid"), event.get("tid"))].append(event)
    rows, external, correlations = [], {}, {}
    for thread in threads.values():
        stack = []
        for event in sorted(thread, key=lambda e: (e["ts"], -e["dur"])):
            start, end = event["ts"], event["ts"] + event["dur"]
            while stack and (start >= stack[-1]["end"] or end > stack[-1]["end"]):
                stack.pop()
            scope = stack[-1]["scope"] if stack else None
            iteration = ITERATION.fullmatch(event["name"])
            if iteration:
                context, _, generation, _ = map(int, iteration.groups())
                phase = (
                    "mixed"
                    if context and generation
                    else "prefill"
                    if context
                    else "decode"
                )
                scope = f"vllm.execute_model.{phase}"
            if event["name"].startswith("speculators."):
                scope = event["name"]
            row = {
                "event": event,
                "start": start,
                "end": end,
                "scope": scope,
                "children": [],
            }
            if stack:
                stack[-1]["children"].append((start, end))
            stack.append(row)
            rows.append(row)
            args = event.get("args", {})
            if "External id" in args:
                external[args["External id"]] = scope
            if "correlation" in args:
                correlations[args["correlation"]] = scope
    for row in rows:
        row["self_us"] = max(0.0, row["event"]["dur"] - union_duration(row["children"]))
    return rows, external, correlations


def _write_csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_traces(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cpu, kernels, scopes = (
        defaultdict(lambda: [0, 0.0, 0.0]),
        defaultdict(lambda: [0, 0.0]),
        defaultdict(lambda: [0, 0.0]),
    )
    trace_summaries, iterations = [], []
    for trace_path in paths:
        path = Path(trace_path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            trace = json.load(handle)
        events = trace.get("traceEvents", [])
        rows, external, correlations = _cpu_rows(events)
        for row in rows:
            event = row["event"]
            entry = cpu[(event.get("cat"), event["name"])]
            entry[0] += 1
            entry[1] += event["dur"] / 1000
            entry[2] += row["self_us"] / 1000
            match = ITERATION.fullmatch(event["name"])
            if match:
                context, ctx_tokens, generation, gen_tokens = map(int, match.groups())
                iterations.append(
                    {
                        "context_requests": context,
                        "generation_requests": generation,
                        "scheduled_tokens": ctx_tokens + gen_tokens,
                        "cpu_inclusive_ms": event["dur"] / 1000,
                    }
                )
        gpu = [
            e for e in events if e.get("ph") == "X" and e.get("cat") in GPU_CATEGORIES
        ]
        gpu_intervals, kernel_intervals = [], []
        attributed = unattributed = 0.0
        for event in gpu:
            duration = event["dur"] / 1000
            entry = kernels[(event["cat"], event["name"])]
            entry[0] += 1
            entry[1] += duration
            interval = (event["ts"], event["ts"] + event["dur"])
            gpu_intervals.append(interval)
            if event["cat"] == "kernel":
                kernel_intervals.append(interval)
            args = event.get("args", {})
            scope = correlations.get(args.get("correlation")) or external.get(
                args.get("External id")
            )
            if scope:
                scopes[scope][0] += 1
                scopes[scope][1] += duration
                attributed += duration
            else:
                unattributed += duration
        span = (
            (
                max(end for _, end in gpu_intervals)
                - min(start for start, _ in gpu_intervals)
            )
            / 1000
            if gpu_intervals
            else None
        )
        busy = union_duration(gpu_intervals) / 1000 if gpu_intervals else None
        trace_summaries.append(
            {
                "file": str(path.relative_to(output))
                if path.is_relative_to(output)
                else str(path),
                "cuda_activity_span_ms": span,
                "cuda_activity_union_ms": busy,
                "cuda_kernel_union_ms": union_duration(kernel_intervals) / 1000
                if gpu_intervals
                else None,
                "gaps_within_cuda_activity_span_ms": max(0.0, span - busy)
                if span is not None
                else None,
                "attributed_cuda_activity_ms": attributed,
                "unattributed_cuda_activity_ms": unattributed,
            }
        )
    cpu_rows = [
        {
            "category": category,
            "name": name,
            "calls": v[0],
            "cpu_total_ms": v[1],
            "cpu_self_ms": v[2],
        }
        for (category, name), v in cpu.items()
    ]
    cpu_rows.sort(key=lambda r: r["cpu_self_ms"], reverse=True)
    kernel_rows = [
        {
            "category": category,
            "name": name,
            "calls": v[0],
            "cuda_total_ms": v[1],
            "cuda_mean_us": 1000 * v[1] / v[0],
        }
        for (category, name), v in kernels.items()
    ]
    kernel_rows.sort(key=lambda r: r["cuda_total_ms"], reverse=True)
    count = len(iterations)
    summary = {
        "measurement_kind": "profiled_diagnostic",
        "traces": trace_summaries,
        "captured_worker_iterations": count,
        "mean_scheduled_requests_per_captured_iteration": sum(
            r["context_requests"] + r["generation_requests"] for r in iterations
        )
        / count
        if count
        else None,
        "iteration_phases": {
            "prefill": sum(
                r["context_requests"] > 0 and r["generation_requests"] == 0
                for r in iterations
            ),
            "decode": sum(
                r["context_requests"] == 0 and r["generation_requests"] > 0
                for r in iterations
            ),
            "mixed": sum(
                r["context_requests"] > 0 and r["generation_requests"] > 0
                for r in iterations
            ),
        },
        "cpu_operations": cpu_rows,
        "cuda_operations": kernel_rows,
        "cuda_attribution_to_ranges": [
            {"name": name, "activity_calls": v[0], "cuda_activity_ms": v[1]}
            for name, v in scopes.items()
        ],
        "synchronization": [r for r in cpu_rows if "synchroniz" in r["name"].lower()],
        "limitations": [
            "CPU self time excludes recorded children. Inclusive totals overlap.",
            "CUDA durations can overlap. Union time is not SM utilization.",
            "CUDA gaps cover only the first-to-last recorded CUDA activity.",
            "CUDA attribution uses launch IDs. Unattributed work stays unknown.",
            "Token counters cover the whole workload, not the trace window.",
            "Use the unprofiled pass for throughput and seconds per output token.",
        ],
    }
    write_json(output / "profile_summary.json", summary)
    _write_csv(
        output / "cpu_operations.csv",
        cpu_rows,
        ["category", "name", "calls", "cpu_total_ms", "cpu_self_ms"],
    )
    _write_csv(
        output / "kernel_summary.csv",
        kernel_rows,
        ["category", "name", "calls", "cuda_total_ms", "cuda_mean_us"],
    )
    lines = [
        "# Runtime profile (diagnostic)",
        "",
        f"Captured worker iteration annotations: {count}.",
        "",
        "Instrumentation affects timing. Use the separate unprofiled benchmark.",
        "",
        "| CPU operation | Calls | Self CPU ms | Inclusive CPU ms |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['name'].replace('|', '/')} | {r['calls']} | "
        f"{r['cpu_self_ms']:.3f} | {r['cpu_total_ms']:.3f} |"
        for r in cpu_rows[:20]
    ]
    lines += [
        "",
        "| CUDA operation | Calls | Summed CUDA ms | Mean µs |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [
        f"| {r['name'].replace('|', '/')} | {r['calls']} | "
        f"{r['cuda_total_ms']:.3f} | {r['cuda_mean_us']:.3f} |"
        for r in kernel_rows[:20]
    ]
    lines += ["", *[f"- {note}" for note in summary["limitations"]]]
    (output / "profile_summary.md").write_text("\n".join(lines) + "\n")
    return summary
