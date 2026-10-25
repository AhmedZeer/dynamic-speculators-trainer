"""Runtime capture control, trace accounting and separation from benchmarks."""

import gzip
import json
from pathlib import Path

import pytest
import torch

from speculators.bank.artifacts import file_digest, write_json
from speculators.benchmark import profiling, suite
from speculators.benchmark.config import Suite
from speculators.benchmark.trace import summarize_traces, union_duration
from speculators.generator.serving import guidellm, ranges


def event(name, category, start, duration, **args):
    return {
        "name": name,
        "cat": category,
        "ts": start,
        "dur": duration,
        "pid": 1,
        "tid": 2,
        "ph": "X",
        "args": args,
    }


def trace_events():
    return [
        event("execute_context_0(0)_generation_4(16)", "user_annotation", 0, 100),
        event("speculators.token_mapping", "user_annotation", 10, 60),
        event("aten::sort", "cpu_op", 20, 20, **{"External id": 5}),
        event("cudaLaunchKernel", "cuda_runtime", 25, 5, correlation=8),
        event("cudaStreamSynchronize", "cuda_runtime", 45, 10),
        event("lora_shrink", "kernel", 30, 20, correlation=8),
        event("lora_expand", "kernel", 40, 20, **{"External id": 5}),
        event("Memcpy HtoD", "gpu_memcpy", 70, 5),
        event("target_gemm", "kernel", 80, 10),
    ]


def write_trace(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as handle:
        json.dump({"traceEvents": trace_events()}, handle)


def test_trace_intervals_self_time_cuda_overlap_and_attribution(tmp_path):
    trace = tmp_path / "worker.pt.trace.json.gz"
    write_trace(trace)
    result = summarize_traces([trace], tmp_path)
    assert union_duration([(1, 5), (3, 6), (8, 9)]) == 6
    cpu = {row["name"]: row for row in result["cpu_operations"]}
    assert cpu["speculators.token_mapping"]["cpu_total_ms"] == 0.06
    assert cpu["speculators.token_mapping"]["cpu_self_ms"] == 0.03
    assert cpu["execute_context_0(0)_generation_4(16)"]["cpu_self_ms"] == 0.04
    observed = result["traces"][0]
    assert observed["cuda_activity_span_ms"] == 0.06
    assert observed["cuda_activity_union_ms"] == 0.045
    assert observed["cuda_kernel_union_ms"] == 0.04
    assert observed["gaps_within_cuda_activity_span_ms"] == pytest.approx(0.015)
    assert observed["attributed_cuda_activity_ms"] == 0.04
    assert result["captured_worker_iterations"] == 1
    assert result["mean_scheduled_requests_per_captured_iteration"] == 4
    assert result["iteration_phases"] == {"prefill": 0, "decode": 1, "mixed": 0}
    assert result["synchronization"][0]["name"] == "cudaStreamSynchronize"
    assert (tmp_path / "kernel_summary.csv").is_file()
    assert "profiled_diagnostic" in (tmp_path / "profile_summary.json").read_text()


def test_cpu_torch_profiler_records_our_ranges_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setattr(ranges, "ENABLED", True)

    @ranges.traced("speculators.generator")
    def operation():
        return torch.ones(3, 3) @ torch.ones(3, 3)

    with (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profiler,
        torch.profiler.record_function("execute_context_1(3)_generation_0(0)"),
    ):
        operation()
    path = tmp_path / "cpu.trace.json"
    profiler.export_chrome_trace(str(path))
    summary = summarize_traces([path], tmp_path)
    assert any(
        row["name"] == "speculators.generator" for row in summary["cpu_operations"]
    )
    assert summary["iteration_phases"]["prefill"] == 1
    assert summary["traces"][0]["cuda_activity_union_ms"] is None
    assert not summary["cuda_operations"]


@pytest.mark.parametrize("fail_client", [False, True])
def test_capture_stops_on_success_or_failure_and_copies_only_new_files(
    tmp_path, monkeypatch, fail_client
):
    root = tmp_path / "native"
    old = root / "old.pt.trace.json.gz"
    write_trace(old)
    calls = []

    def request(target, action):
        calls.append(action)
        if action == "stop":
            write_trace(root / "new.pt.trace.json.gz")
            (root / "profiler_out_0.txt").write_text("operator table")

    monkeypatch.setattr(profiling, "profile_request", request)
    destination = tmp_path / "measurement"

    def execute():
        with profiling.capture_profile(
            "http://localhost:8000/v1", destination, {"torch_profiler_dir": str(root)}
        ):
            if fail_client:
                raise ValueError("client failed")

    if fail_client:
        with pytest.raises(ValueError, match="client failed"):
            execute()
    else:
        execute()
    assert calls == ["start", "stop"]
    assert not (destination / "traces/old.pt.trace.json.gz").exists()
    artifacts = profiling.profile_artifacts(destination)
    assert destination / "traces/new.pt.trace.json.gz" in artifacts
    assert (destination / "profile_summary.md").exists()
    (destination / "traces/new.pt.trace.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed"):
        profiling.profile_artifacts(destination)


def test_capture_rejects_missing_or_non_worker_traces(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "profile_request", lambda *_: None)
    with (
        pytest.raises(RuntimeError, match="no new JSON trace"),
        profiling.capture_profile(
            "http://localhost",
            tmp_path / "out",
            {"torch_profiler_dir": str(tmp_path / "empty")},
        ),
    ):
        pass

    def emit_invalid_trace(target, action):
        if action == "stop":
            write_json(tmp_path / "empty" / "invalid.trace.json", {"traceEvents": []})

    monkeypatch.setattr(profiling, "profile_request", emit_invalid_trace)
    with (
        pytest.raises(RuntimeError, match="lacks worker"),
        profiling.capture_profile(
            "http://localhost",
            tmp_path / "invalid",
            {"torch_profiler_dir": str(tmp_path / "empty")},
        ),
    ):
        pass


def test_native_profile_http_accepts_empty_response(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b""

    def open_url(request, timeout):
        calls.append((request.full_url, request.method, timeout))
        return Response()

    monkeypatch.setattr(profiling, "urlopen", open_url)
    profiling.profile_request("http://localhost:8000/v1", "start")
    assert calls[0][:2] == ("http://localhost:8000/start_profile", "POST")


def test_configs_and_commands_select_serial_control_and_bounded_capture(tmp_path):
    config = Suite.load("examples/eagle3-lora/profile_generator_serial.yaml")
    assert len(config.models) == 3
    assert config.models[1].batching_policy == "fixed_shared_adapter"
    settings = config.settings
    generated = {
        "type": "generator",
        "path": "/bundle",
        "adapter_policy": "per_request",
    }
    command = suite.server_command(Path("/repo"), generated, {}, settings, tmp_path)
    native = json.loads(command[command.index("--profiler-config") + 1])
    profiling.validate_profiler_config(native)
    assert native["max_iterations"] == 30
    assert native["ignore_frontend"] is True
    client = suite.client_command(
        Path("/repo"),
        {"category": "nemo-math"},
        {"path": "manifest"},
        settings,
        Path("/tokens"),
        tmp_path,
        tmp_path / "out",
    )
    assert "--capture-profile" in client
    bad = {**native, "max_iterations": 0}
    with pytest.raises(ValueError):
        profiling.validate_profiler_config(bad)
    settings.profiling = None
    command = suite.server_command(Path("/repo"), generated, {}, settings, tmp_path)
    assert "--profiler-config" not in command
    with pytest.raises(ValueError, match="benchmark profile"):
        suite.run_suite(
            "examples/eagle3-lora/profile_generator_serial.yaml",
            tmp_path / "unused",
            dry_run=True,
        )


def test_profile_orchestrator_runs_unprofiled_first_freezes_inputs_and_resumes(
    tmp_path, monkeypatch
):
    calls = []

    def run(config, output, **kwargs):
        calls.append((Path(output).name, kwargs))
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        if not kwargs["profile_mode"]:
            write_json(output / "resolved_models.json", {"immutable_sha": "a" * 40})
            write_json(output / "workloads/manifest.json", {"identity": "frozen"})
        else:
            assert (
                json.loads((output / "resolved_models.json").read_text())[
                    "immutable_sha"
                ]
                == "a" * 40
            )
            assert (output / "workloads/manifest.json").is_file()
        return {"failed": 0, "completed": 6}

    monkeypatch.setattr(suite, "run_suite", run)
    config = "examples/eagle3-lora/profile_generator_serial.yaml"
    profiling.run_profile(config, tmp_path / "run", resume=True)
    assert [phase for phase, _ in calls] == ["benchmark", "profiled"]
    assert [kw["profile_mode"] for _, kw in calls] == [False, True]
    assert all(kw["resume"] for _, kw in calls)
    assert (tmp_path / "run/README.md").is_file()


def test_profile_summary_never_presents_diagnostic_throughput_as_benchmark(tmp_path):
    output = tmp_path / "profiled"
    output.mkdir()
    write_json(output / "task/profile_summary.json", {"captured_worker_iterations": 30})
    state = {
        "tasks": {
            "model/math": {"status": "completed", "output": "task", "category": "math"}
        }
    }
    profiling.write_profile_summary(output, {"identity": "suite"}, state)
    assert "tokens/s" not in (output / "summary.md").read_text()
    result = json.loads((output / "summary.json").read_text())
    assert result["measurement_kind"] == "profiled_diagnostic"
    assert result["measurements"][0]["captured_iterations"] == 30


def test_server_provenance_does_not_duplicate_packages_or_previous_traces(tmp_path):
    source = tmp_path / "server"
    source.mkdir()
    original = source / "vanilla-vllm/vllm/v1/eagle.py"
    original.parent.mkdir(parents=True)
    original.write_text("native")
    (source / "vanilla-vllm/huge-library.so").write_bytes(b"not provenance")
    write_json(
        source / "vanilla_source.json",
        {"original_sha256": {"vllm/v1/eagle.py": file_digest(original)}},
    )
    write_json(source / "traces/previous.json", {"large": "trace"})
    destination = tmp_path / "copy"
    guidellm.copy_server_provenance(source, destination)
    assert (destination / "vanilla-vllm/vllm/v1/eagle.py").read_text() == "native"
    assert not (destination / "vanilla-vllm/huge-library.so").exists()
    assert not (destination / "traces").exists()


def test_native_worker_scope_attributes_target_execution_without_patching_it(tmp_path):
    path = tmp_path / "native.trace.json"
    write_json(
        path,
        {
            "traceEvents": [
                event(
                    "execute_context_0(0)_generation_4(16)", "user_annotation", 0, 100
                ),
                event("aten::mm", "cpu_op", 10, 20, **{"External id": 10}),
                event("gemm", "kernel", 15, 30, **{"External id": 10}),
            ]
        },
    )
    result = summarize_traces([path], tmp_path)
    assert result["cuda_attribution_to_ranges"] == [
        {
            "name": "vllm.execute_model.decode",
            "activity_calls": 1,
            "cuda_activity_ms": 0.03,
        }
    ]
