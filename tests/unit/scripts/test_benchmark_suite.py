"""Suite orchestration with real CPU subprocesses and frozen miniature workloads."""

import json
import os
import signal
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import yaml
from pydantic import ValidationError

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.benchmark import prepare, suite
from speculators.benchmark.config import Benchmark, Model, Settings, Suite
from speculators.benchmark.processes import (
    ensure_port_free,
    owned_process,
    termination_cleanup,
)
from speculators.benchmark.progress import report_failure, stage_progress
from speculators.generator.serving.endpoint import validate_status
from speculators.generator.serving.validation import load_validation
from tests.unit.generator.test_serving import write_bundle


def workload(output, categories=("nemo-math", "nemo-code")):
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "models": {"target": suite.DEFAULT_TARGET},
        "groups": {},
    }
    for category in categories:
        path = output / f"{category}.jsonl"
        row = {"prompt_token_ids": [1, 2], "output_tokens_count": 4}
        path.write_text(json.dumps(row) + "\n")
        manifest["groups"][category] = {
            "file": path.name,
            "sha256": file_digest(path),
            "rows": 1,
        }
    manifest["identity"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return {
        "path": str(output / "manifest.json"),
        "identity": manifest["identity"],
        "models": manifest["models"],
        "groups": manifest["groups"],
    }


def test_config_and_dry_run(tmp_path):
    config = {
        "models": [{"name": "base", "type": "drafter", "model": "owner/model"}],
        "benchmarks": [{"dataset": "nemotron", "categories": "math, code"}],
    }
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(config))
    result = suite.run_suite(path, tmp_path / "output", dry_run=True)
    assert result["configuration"]["benchmarks"][0]["categories"] == ["math", "code"]
    assert not (tmp_path / "output").exists()
    config["models"] *= 2
    with pytest.raises(ValidationError, match="unique"):
        Suite.model_validate(config)
    with pytest.raises(ValidationError, match="subfolder"):
        Model(name="bad", type="lora", model="a/b", subfolder="../escape")
    with pytest.raises(ValidationError):
        Settings(concurrency=9)
    with pytest.raises(ValidationError, match="only to generators"):
        Model(name="bad", type="lora", model="owner/repo", conditioning_examples=2)


def test_benchmark_diagnostics_include_failure_excerpt_and_heartbeats(tmp_path, capsys):
    log = tmp_path / "client.log"
    log.write_text("\n".join(["old detail"] * 20 + ["ValueError: failed prompt check"]))
    report_failure("base/nemo-math", RuntimeError("client exited 1"), log)
    with stage_progress("Waiting for server", interval=0.01):
        time.sleep(0.035)
    text = capsys.readouterr().err
    assert "[benchmark pid=" in text
    assert "lora-bank" not in text
    assert "FAILED base/nemo-math: RuntimeError: client exited 1" in text
    assert "ValueError: failed prompt check" in text
    assert str(log) in text
    assert text.count("old detail") == 11
    assert "Waiting for server: still running" in text
    assert "Waiting for server: completed" in text


def test_hub_bundle_resolution_pins_and_excludes_optimizer(tmp_path, monkeypatch):
    bundle = tmp_path / "snapshot" / "step-1000"
    bundle.parent.mkdir()
    write_bundle(bundle, "static_lora")
    downloads, revisions = [], []
    monkeypatch.setattr(
        prepare,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *a, **kw: (
                revisions.append(kw) or SimpleNamespace(sha="a" * 40)
            )
        ),
    )
    monkeypatch.setattr(
        prepare,
        "snapshot_download",
        lambda *a, **kw: downloads.append(kw) or str(bundle.parent),
    )
    spec = Model(name="lora", type="lora", model="owner/repo", subfolder=bundle.name)
    model = prepare.resolve_model(spec, tmp_path, tmp_path / "staged")
    assert model["revision"] == "a" * 40
    assert revisions == [{"revision": "main"}]
    assert downloads[0]["allow_patterns"][0] == "step-1000/bundle.json"
    assert "*/checkpoint.pt" in downloads[0]["ignore_patterns"]
    prepare.resolve_model(spec, tmp_path, tmp_path / "staged", pinned=model["revision"])
    assert len(revisions) == 1  # Resume keeps its immutable commit without main lookup.
    with pytest.raises(ValueError, match="declared generator"):
        prepare.resolve_model(
            Model(name="wrong", type="generator", model=str(bundle)),
            tmp_path,
            tmp_path / "staged",
        )


def test_local_drafter_hashes_and_staging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    model = Model(name="draft", type="drafter", model=str(source), revision="a" * 40)
    resolved = prepare.resolve_model(model, tmp_path, tmp_path / "staged")
    assert Path(resolved["path"]).is_dir()
    assert Path(resolved["path"]).name == "a" * 40
    assert resolved["local_files_sha256"] == {
        "config.json": file_digest(source / "config.json")
    }
    with pytest.raises(ValueError, match="outside"):
        prepare.resolve_model(model, tmp_path, source / "inside")


def test_shared_workload_retains_reference_and_strict_standalone_checks(
    tmp_path, monkeypatch
):
    def export(cfg, output, **kwargs):
        result = workload(output)
        manifest = json.loads(Path(result["path"]).read_text())
        manifest["models"]["drafter"] = {"id": "original", "revision": "a" * 40}
        manifest.pop("identity")
        manifest["identity"] = digest(manifest)
        write_json(Path(result["path"]), manifest)

    monkeypatch.setattr(prepare, "ensure_inputs", lambda *a, **kw: None)
    monkeypatch.setattr(prepare, "export_validation", export)
    path = tmp_path / "validation"
    info = prepare.prepare_workload(
        Benchmark(dataset="nemotron"), None, path, Settings(), tmp_path
    )
    shared, _ = load_validation(Path(info["path"]))
    original, _ = load_validation(path / "source_manifest.json")
    status = {
        "configuration": {
            "kind": "eagle3_base",
            "models": {"target": suite.DEFAULT_TARGET, "drafter": {"id": "different"}},
        },
        "workers": [{"use_v2_model_runner": False}],
    }
    validate_status(status, shared)
    with pytest.raises(ValueError, match="drafter revision"):
        validate_status(status, original)
    assert (
        prepare.prepare_workload(
            Benchmark(dataset="nemotron"), None, path, Settings(), tmp_path
        )
        == info
    )


def test_command_builders_keep_provenance_and_skip_correctness(tmp_path):
    model = {"type": "drafter", "path": "owner/drafter", "revision": "a" * 40}
    server = suite.server_command(
        tmp_path, model, suite.DEFAULT_TARGET, Settings(), tmp_path / "server"
    )
    assert "--provenance-dir" in server
    assert "--drafter-revision" in server
    model = {"type": "lora", "path": "/bundle"}
    assert "--adapter-bundle" in suite.server_command(
        tmp_path, model, {}, Settings(), tmp_path
    )
    client = suite.client_command(
        tmp_path,
        {"category": "nemo-math"},
        {"path": "manifest"},
        Settings(),
        tmp_path,
        tmp_path,
        tmp_path,
    )
    assert "--correctness-report" not in client
    assert "--server-provenance" in client
    assert client[client.index("--max-requests") + 1] == "0"
    generator = {"type": "generator", "path": "/bundle", "conditioning_examples": 2}
    command = suite.server_command(
        tmp_path, generator, {}, Settings(concurrency=8), tmp_path
    )
    assert command[command.index("--concurrency") + 1] == "8"
    assert command[command.index("--conditioning-examples") + 1] == "2"


def test_fetches_only_metadata_and_selected_validation(tmp_path, monkeypatch):
    bank = SimpleNamespace(
        hub=SimpleNamespace(
            release="r0001",
            trajectories_repo="owner/trajectories",
            trajectories_revision="a" * 40,
        )
    )
    monkeypatch.setattr(prepare, "campaign_banks", lambda cfg: {"math": bank})
    monkeypatch.setattr(
        prepare, "jobs", lambda cfg: [{"validation_subsets": ["math-00000"]}]
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {"manifest": {"subsets": [{"id": "math-00000"}, {"id": "math-00001"}]}}
        )
    )
    requests = []
    monkeypatch.setattr(
        prepare,
        "hf_hub_download",
        lambda repo, name, **kw: requests.append(name) or str(source),
    )
    cfg = SimpleNamespace(trajectories_path=tmp_path / "trajectories")
    prepare.ensure_inputs(cfg, nemotron=True)
    assert requests == [
        "releases/r0001/math/data.json",
        "releases/r0001/math/math-00000/validation.parquet",
    ]
    prepare.ensure_inputs(cfg, nemotron=True)
    assert len(requests) == 2


def test_owned_process_cleanup_and_occupied_port(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        with pytest.raises(RuntimeError, match="occupied"):
            ensure_port_free(port)
        assert listener.fileno() >= 0
    ensure_port_free(port)
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    with (
        pytest.raises(KeyboardInterrupt),
        owned_process(
            command, tmp_path / "process.log", cwd=tmp_path, env=os.environ, timeout=1
        ) as child,
    ):
        raise KeyboardInterrupt
    assert child.poll() is not None
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(KeyboardInterrupt), termination_cleanup():
        os.kill(os.getpid(), signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) == previous


def test_cleanup_reaps_workers_even_when_parent_exits(tmp_path):
    pid_file = tmp_path / "worker.pid"
    code = (
        "import subprocess, sys; from pathlib import Path; "
        "child = subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(60)']); "
        "Path(sys.argv[1]).write_text(str(child.pid))"
    )
    with owned_process(
        [sys.executable, "-c", code, str(pid_file)],
        tmp_path / "parent.log",
        cwd=tmp_path,
        env=os.environ,
        timeout=1,
    ) as parent:
        assert parent.wait(timeout=5) == 0
        worker = psutil.Process(int(pid_file.read_text()))
    deadline = time.monotonic() + 2
    while (
        worker.is_running()
        and worker.status() != psutil.STATUS_ZOMBIE
        and time.monotonic() < deadline
    ):
        time.sleep(0.01)
    assert not worker.is_running() or worker.status() == psutil.STATUS_ZOMBIE


SERVER = """
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
status = json.loads(sys.argv[2])
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        body = status if self.path.endswith("status") else {}
        self.wfile.write(json.dumps(body).encode())
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""

CLIENT = """
import json, pathlib, sys
output, category, workload, model, marker, fail = sys.argv[1:]
output, marker = pathlib.Path(output), pathlib.Path(marker)
with marker.open("a") as handle:
    handle.write(model + "/" + category + "\\n")
if fail == "True":
    sys.exit(2)
workload = json.loads(workload)
directory = output / category
directory.mkdir(parents=True)
result = {"groups": {category: {
    "dataset_sha256": workload["groups"][category]["sha256"],
    "performance": [{"successful_requests": 1, "output_tokens_per_second": 12.5}]
}}}
(output / "results.json").write_text(json.dumps(result))
(output / "benchmark_identity.json").write_text(json.dumps({
    "validation": workload["identity"], "concurrency": workload.get("concurrency", 1),
    "server": {"kind": "eagle3_base"}}))
(directory / "guidellm.json").write_text("{}")
"""


@pytest.fixture
def fake_suite(tmp_path, monkeypatch):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    config = tmp_path / "suite.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"name": name, "type": "drafter", "model": "owner/" + name}
                    for name in ("base", "other")
                ],
                "benchmarks": [{"dataset": "nemotron", "categories": ["math", "code"]}],
                "settings": {
                    "port": port,
                    "tokenizer_path": str(tokenizer),
                    "shutdown_timeout_seconds": 1,
                    "startup_timeout_seconds": 5,
                },
            }
        )
    )
    monkeypatch.setattr(suite, "save_provenance", lambda output: None)
    monkeypatch.setattr(
        suite,
        "resolve_model",
        lambda spec, *a, **kw: {
            "name": spec.name,
            "type": spec.type,
            "path": spec.model,
            "model": spec.model,
            "revision": "a" * 40,
        },
    )

    def prepare_workload(benchmark, cfg, output, settings, tokenizer):
        return workload(output)

    monkeypatch.setattr(suite, "prepare_workload", prepare_workload)
    server_script, client_script = tmp_path / "server.py", tmp_path / "client.py"
    server_script.write_text(SERVER)
    client_script.write_text(CLIENT)
    launches, children = [], []

    def server_command(root, model, target, settings, provenance):
        launches.append(model["name"])
        status = {
            "configuration": {
                "kind": "eagle3_base",
                "max_num_seqs": settings.concurrency,
                "models": {
                    "target": target,
                    "drafter": {"id": model["path"], "revision": model["revision"]},
                },
            },
            "workers": [{"use_v2_model_runner": False}],
        }
        return [sys.executable, str(server_script), str(port), json.dumps(status)]

    marker = tmp_path / "clients.txt"
    failures = set()

    def client_command(  # noqa: PLR0917 -- mirrors production builder
        root, task, data, settings, tokenizer, provenance, output
    ):
        fail = task["key"] in failures
        failures.discard(task["key"])
        return [
            sys.executable,
            str(client_script),
            str(output),
            task["category"],
            json.dumps({**data, "concurrency": settings.concurrency}),
            task["model"],
            str(marker),
            str(fail),
        ]

    original = suite.owned_process

    from contextlib import contextmanager  # noqa: PLC0415

    @contextmanager
    def track(*args, **kwargs):
        with original(*args, **kwargs) as process:
            children.append(process)
            yield process

    monkeypatch.setattr(suite, "server_command", server_command)
    monkeypatch.setattr(suite, "client_command", client_command)
    monkeypatch.setattr(suite, "owned_process", track)
    return SimpleNamespace(
        config=config,
        output=tmp_path / "output",
        launches=launches,
        marker=marker,
        failures=failures,
        children=children,
        port=port,
    )


def test_suite_matrix_resume_and_corrupt_result_replay(fake_suite):
    case = fake_suite
    result = suite.run_suite(case.config, case.output)
    assert result["completed"] == 4
    assert result["failed"] == 0
    assert case.launches == ["base", "other"]
    assert all(child.poll() is not None for child in case.children)
    ensure_port_free(case.port)
    assert "12.50" in (case.output / "summary.md").read_text()
    suite.run_suite(case.config, case.output, resume=True)
    assert len(case.marker.read_text().splitlines()) == 4
    assert len(case.launches) == 2
    state = json.loads((case.output / "state.json").read_text())
    entry = state["tasks"]["base/nemotron/nemo-math"]
    assert entry["wall_seconds"] > 0
    native = case.output / entry["output"] / "nemo-math/guidellm.json"
    native.write_text("corrupt")
    suite.run_suite(case.config, case.output, resume=True)
    assert len(case.marker.read_text().splitlines()) == 5
    assert len(case.launches) == 3
    with pytest.raises(FileExistsError, match="resume"):
        suite.run_suite(case.config, case.output)


def test_suite_concurrency_sweep_separates_results_and_resumes(fake_suite):
    case = fake_suite
    config = yaml.safe_load(case.config.read_text())
    config["settings"]["concurrencies"] = [1, 2]
    case.config.write_text(yaml.safe_dump(config))
    result = suite.run_suite(case.config, case.output)
    assert result["completed"] == 8
    assert case.launches == ["base-c1", "base-c2", "other-c1", "other-c2"]
    summary = json.loads((case.output / "summary.json").read_text())
    assert [row["concurrency"] for row in summary["rows"]] == [1, 2, 1, 2]
    suite.run_suite(case.config, case.output, resume=True)
    assert len(case.launches) == 4
    assert len(case.marker.read_text().splitlines()) == 8


def test_add_concurrency_reuses_serial_results_without_launching_serial_servers(
    fake_suite,
):
    case = fake_suite
    suite.run_suite(case.config, case.output)
    # Simulate artifacts created before concurrency-sweep fields existed.
    saved_request = json.loads((case.output / "suite_request.json").read_text())
    saved_request["settings"].pop("concurrencies")
    saved_request["settings"].pop("batch_wait_ms")
    write_json(case.output / "suite_request.json", saved_request)
    saved_plan = json.loads((case.output / "suite.json").read_text())
    for model in saved_plan["models"]:
        model.pop("base_name")
        model.pop("concurrency")
    write_json(case.output / "suite.json", saved_plan)
    original = json.loads((case.output / "state.json").read_text())
    entry = original["tasks"]["base/nemotron/nemo-math"]
    paths = {name: (case.output / name).read_bytes() for name in entry["checksums"]}
    config = yaml.safe_load(case.config.read_text())
    config["settings"]["concurrencies"] = [1, 2]
    case.config.write_text(yaml.safe_dump(config))
    result = suite.run_suite(case.config, case.output, resume=True)
    assert result["completed"] == 8
    assert result["failed"] == 0
    assert case.launches == ["base", "other", "base-c2", "other-c2"]
    assert len(case.marker.read_text().splitlines()) == 8
    state = json.loads((case.output / "state.json").read_text())
    reused = state["tasks"]["base-c1/nemotron/nemo-math"]
    assert "base/nemotron/nemo-math" not in state["tasks"]
    assert reused["output"] == entry["output"]
    assert reused["checksums"] == entry["checksums"]
    assert reused["reused_from_task"] == "base/nemotron/nemo-math"
    assert all(
        (case.output / name).read_bytes() == contents
        for name, contents in paths.items()
    )
    archives = list((case.output / "suite_attempts").glob("*/previous_state.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == original
    suite.run_suite(case.config, case.output, resume=True)
    assert len(case.launches) == 4
    config["settings"]["max_tokens"] = 128
    case.config.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="identical models, datasets"):
        suite.run_suite(case.config, case.output, resume=True)


def test_resume_rejects_removing_concurrency():
    previous = {
        "models": [{"name": "base", "type": "drafter", "model": "owner/model"}],
        "benchmarks": [{"dataset": "nemotron"}],
        "settings": {"concurrencies": [1, 2]},
    }
    config = Suite.model_validate(previous)
    config.settings.concurrencies = [1]
    with pytest.raises(ValueError, match="adding concurrency"):
        suite.validate_resume_request(previous, config)


def test_suite_client_failure_continues_then_resume_retries_only_failure(
    fake_suite, monkeypatch, capsys
):
    case = fake_suite
    case.failures.add("base/nemotron/nemo-math")
    result = suite.run_suite(case.config, case.output)
    assert result["completed"] == 3
    assert result["failed"] == 1
    logs = capsys.readouterr().err
    assert "FAILED base/nemotron/nemo-math" in logs
    assert "Recycling server base after client failure" in logs
    assert "COMPLETED other/nemotron/nemo-math: 12.50 tok/s" in logs
    assert all(child.poll() is not None for child in case.children)
    original = suite.server_command
    monkeypatch.setattr(
        suite,
        "server_command",
        lambda *args: [sys.executable, "-c", "raise SystemExit(2)"],
    )
    assert suite.run_suite(case.config, case.output, resume=True)["failed"] == 1
    state = json.loads((case.output / "state.json").read_text())
    assert state["tasks"]["base/nemotron/nemo-math"]["attempt"] == 1
    monkeypatch.setattr(suite, "server_command", original)
    result = suite.run_suite(case.config, case.output, resume=True)
    assert result["completed"] == 4
    assert result["failed"] == 0
    assert len(case.marker.read_text().splitlines()) == 5
    state = json.loads((case.output / "state.json").read_text())
    assert state["tasks"]["base/nemotron/nemo-math"]["attempt"] == 2
    ensure_port_free(case.port)


def test_startup_failure_is_recorded_and_resume_can_recover(fake_suite, monkeypatch):
    case = fake_suite
    original = suite.server_command
    monkeypatch.setattr(
        suite,
        "server_command",
        lambda *args: [sys.executable, "-c", "raise SystemExit(2)"],
    )
    result = suite.run_suite(case.config, case.output)
    assert result["failed"] == 4
    assert not case.marker.exists()
    failures = json.loads((case.output / "failures.json").read_text())
    assert all(entry["phase"] == "server_start" for entry in failures.values())
    monkeypatch.setattr(suite, "server_command", original)
    assert suite.run_suite(case.config, case.output, resume=True)["completed"] == 4
    ensure_port_free(case.port)
