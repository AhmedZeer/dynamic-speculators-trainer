"""Bounded native worker traces, kept separate from unprofiled measurements."""

import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

from speculators.bank.artifacts import file_digest, write_json
from speculators.benchmark.config import Profiling, Settings, Suite
from speculators.generator.serving.endpoint import endpoint_root


def profiler_config(settings, provenance):
    options = settings.profiling or Profiling()
    return {
        "profiler": "torch",
        "torch_profiler_dir": str((Path(provenance) / "traces").resolve()),
        "torch_profiler_activities": ["CPU", "CUDA"],
        "torch_profiler_with_stack": options.with_stack,
        "torch_profiler_record_shapes": options.record_shapes,
        "torch_profiler_with_memory": options.with_memory,
        "torch_profiler_use_gzip": True,
        "ignore_frontend": True,
        "max_iterations": options.max_iterations,
    }


def validate_profiler_config(config):
    expected = set(profiler_config(Settings(), Path()))
    if (
        not isinstance(config, dict)
        or set(config) != expected
        or config["profiler"] != "torch"
        or not Path(config["torch_profiler_dir"]).is_absolute()
        or config["torch_profiler_activities"] != ["CPU", "CUDA"]
        or config["ignore_frontend"] is not True
        or config["torch_profiler_use_gzip"] is not True
    ):
        raise ValueError(
            "Expected bounded CPU/CUDA profiler; absolute path; ignore_frontend=true"
        )
    Profiling(
        max_iterations=config["max_iterations"],
        with_stack=config["torch_profiler_with_stack"],
        record_shapes=config["torch_profiler_record_shapes"],
        with_memory=config["torch_profiler_with_memory"],
    )


def profile_request(target, action):
    """vLLM 0.31.0 profile endpoints return empty 200 responses, not JSON."""
    request = Request(  # noqa: S310 -- explicit serving endpoint
        endpoint_root(target) + f"/{action}_profile", data=b"", method="POST"
    )
    with urlopen(request, timeout=1800) as response:  # noqa: S310
        response.read()


def trace_snapshot(root):
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.stat().st_size)
        for path in Path(root).rglob("*")
        if path.is_file()
    }


@contextmanager
def capture_profile(target, destination, config):
    """Always stop/flush after the client, including client failures/cancellation."""
    from speculators.benchmark.trace import summarize_traces  # noqa: PLC0415

    root = Path(config["torch_profiler_dir"])
    destination = Path(destination)
    before = trace_snapshot(root)
    profile_request(target, "start")
    try:
        yield
    finally:
        profile_request(target, "stop")
        after = trace_snapshot(root)
        saved = []
        for relative, stamp in after.items():
            if before.get(relative) == stamp:
                continue
            source = root / relative
            target_path = destination / "traces" / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_path)
            saved.append(target_path)
        traces = [p for p in saved if p.name.endswith((".json", ".json.gz"))]
        if not traces:
            raise RuntimeError(
                "Worker emitted no new JSON trace; profiling did not capture execution"
            )
        write_json(
            destination / "profile_capture.json",
            {
                "measurement_kind": "profiled_diagnostic",
                "configuration": config,
                "files": {
                    str(p.relative_to(destination)): file_digest(p) for p in saved
                },
                "scope": "Bounded window; token counters cover the full workload.",
            },
        )
        summary = summarize_traces(traces, destination)
        if not summary["cuda_operations"] or not summary["captured_worker_iterations"]:
            raise RuntimeError(
                "Trace lacks worker iteration annotations or CUDA activity"
            )


def profile_artifacts(output):
    output = Path(output)
    manifest = json.loads((output / "profile_capture.json").read_text())
    paths = [output / name for name in manifest["files"]]
    for path in paths:
        if file_digest(path) != manifest["files"][str(path.relative_to(output))]:
            raise ValueError(f"Changed or missing profile artifact: {path}")
    return [
        output / name
        for name in (
            "profile_capture.json",
            "profile_summary.json",
            "profile_summary.md",
            "kernel_summary.csv",
            "cpu_operations.csv",
        )
    ] + paths


def run_profile(config_path, output_root, *, resume=False, dry_run=False):
    from speculators.benchmark.suite import run_suite  # noqa: PLC0415

    config = Suite.load(config_path)
    root = Path(output_root).resolve()
    if dry_run:
        return {
            "dry_run": True,
            "configuration": config.model_dump(mode="json"),
            "phases": [
                "benchmark (profiler disabled)",
                "profiled (bounded CPU/CUDA traces)",
            ],
            "note": "No downloads or processes started",
        }
    root.mkdir(parents=True, exist_ok=True)
    request = config.model_dump(mode="json")
    plan = root / "profile_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != request:
        raise ValueError("Profiling configuration changed; choose a new output root")
    write_json(plan, request)
    benchmark = run_suite(
        config_path, root / "benchmark", resume=resume, profile_mode=False
    )
    if benchmark.get("failed", 0):
        return {"failed": benchmark["failed"], "benchmark": benchmark}
    # Both phases use the exact resolved checkpoints and frozen prompt exports.
    diagnostic_root = root / "profiled"
    diagnostic_root.mkdir(exist_ok=True)
    for name in ("resolved_models.json", "workloads"):
        source, destination = root / "benchmark" / name, diagnostic_root / name
        if source.exists() and not destination.exists():
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
    diagnostic = run_suite(
        config_path, root / "profiled", resume=resume, profile_mode=True
    )
    (root / "README.md").write_text(
        "# Inference profiling run\n\n"
        "Use benchmark/summary.md for unprofiled throughput. Profiling runs live under "
        "profiled/; their timing is diagnostic.\n\n"
        "Each completed profiled measurement contains profile_summary.md/json, "
        "kernel_summary.csv, cpu_operations.csv, native traces and capture metadata. "
        "The trace window is bounded independently of whole-workload counters.\n"
    )
    return {
        "failed": diagnostic.get("failed", 0),
        "benchmark": benchmark,
        "profiled": diagnostic,
        "directory": str(root),
    }


def write_profile_summary(output, plan, state):
    reference_state_path = output.parent / "benchmark" / "state.json"
    reference_state = (
        json.loads(reference_state_path.read_text())
        if reference_state_path.exists()
        else {"tasks": {}}
    )
    rows = []
    for key, task in state["tasks"].items():
        summary = None
        if task.get("status") == "completed":
            summary = json.loads(
                (output / task["output"] / "profile_summary.json").read_text()
            )
        reference = reference_state["tasks"].get(key, {})
        reference_metrics = None
        if reference.get("status") == "completed":
            result = json.loads(
                (
                    reference_state_path.parent / reference["output"] / "results.json"
                ).read_text()
            )
            measured = result["groups"][task["category"]]["performance"][0]
            tokens = measured.get("total_output_tokens", 0)
            reference_metrics = {
                "output_tokens_per_second": measured["output_tokens_per_second"],
                "aggregate_seconds_per_output_token": measured["measured_seconds"]
                / tokens
                if tokens
                else None,
            }
        rows.append(
            {
                "measurement": key,
                "status": task["status"],
                "captured_iterations": summary["captured_worker_iterations"]
                if summary
                else None,
                "profile_summary": str(Path(task["output"]) / "profile_summary.md")
                if summary
                else None,
                "unprofiled_reference": reference_metrics,
            }
        )
    lines = [
        "# Profiling suite (diagnostic)",
        "",
        "Use ../benchmark/summary.md for throughput. Traces cover bounded windows.",
        "",
        "| Measurement | Status | Captured iterations | Details |",
        "| --- | --- | ---: | --- |",
    ]
    for row in rows:
        link = f"[profile]({row['profile_summary']})" if row["profile_summary"] else ""
        lines.append(
            f"| {row['measurement']} | {row['status']} | "
            f"{row['captured_iterations']} | {link} |"
        )
    for name in ("summary.md", "profile_summary.md"):
        (output / name).write_text("\n".join(lines) + "\n")
    write_json(
        output / "summary.json",
        {
            "measurement_kind": "profiled_diagnostic",
            "suite_identity": plan["identity"],
            "measurements": rows,
        },
    )
    write_json(
        output / "failures.json",
        {
            key: value
            for key, value in state["tasks"].items()
            if value.get("status") != "completed"
        },
    )
