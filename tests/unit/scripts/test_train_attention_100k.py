import json
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from scripts import train_attention_100k as launch
from speculators.generator import epoch_training as training
from speculators.generator.campaign_config import CampaignConfig


def records():
    return {
        domain: {
            "manifest": {
                "subsets": [
                    {
                        "id": f"{domain}-{i:05d}",
                        "train_indices": list(range(1000)),
                        "validation_indices": list(range(100)),
                    }
                    for i in range(25)
                ],
                "revisions": {
                    "target_sha": "a" * 40,
                    "drafter_sha": "b" * 40,
                    "dataset_sha": "d" * 40,
                },
                "cache_spec": {
                    "models": {
                        "target": "Qwen/Qwen3-8B",
                        "drafter": "RedHatAI/Qwen3-8B-speculator.eagle3",
                    }
                },
            },
            "artifacts": [],
        }
        for domain in launch.DOMAINS
    }


def args(tmp_path):
    return SimpleNamespace(
        output_root=tmp_path / "run",
        hub_repo="azeeer/test-100k",
        trajectories_revision="main",
        port=18000,
        gpu_memory_utilization=0.4,
        prepare_only=False,
        startup_timeout=60,
    )


def test_prepare_pins_release_on_resume_and_matches_reference(tmp_path, monkeypatch):
    request = args(tmp_path)
    info_calls = []
    snapshots = []
    monkeypatch.setattr(
        launch,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info=lambda *a, **kw: (
                info_calls.append(kw) or SimpleNamespace(sha="c" * 40)
            )
        ),
    )
    monkeypatch.setattr(launch, "release_records", lambda *a: records())
    monkeypatch.setattr(
        launch,
        "snapshot_download",
        lambda repo, **kw: (
            snapshots.append((repo, kw)) or str(tmp_path / "models" / repo)
        ),
    )
    cfg, path, provenance = launch.prepare(request)
    loaded = CampaignConfig.load(path)
    assert loaded.validation_examples_per_category == 250
    assert loaded.max_training_updates == 2000
    assert len(loaded.experiment_subsets) == 100
    assert loaded.optimization.pretraining_validation_interval == 250
    assert loaded.optimization.checkpoint_interval == 500
    recipe = yaml.safe_load(Path(loaded.training_recipe).read_text())
    assert recipe["hub"]["trajectories_revision"] == "c" * 40
    assert recipe["hub"]["release"] == "r0002-100k"
    reference = CampaignConfig.load(
        launch.REPO / "examples/eagle3-lora/attention_24k_6k_v1/train.yaml"
    )
    assert loaded.architecture == reference.architecture
    assert loaded.target_modules == reference.target_modules
    assert loaded.context == reference.context
    assert loaded.optimization.loss_fn == reference.optimization.loss_fn
    server, train = launch.commands(request, cfg, path, provenance)
    assert server[server.index("--gpu-memory-utilization") + 1] == "0.4"
    assert (
        server[server.index("--hidden-states-path") + 1] == cfg["online"]["shared_dir"]
    )
    assert server[server.index("--provenance-dir") + 1] == str(provenance)
    assert train[train.index("--max-updates") + 1] == "2000"
    assert train[train.index("--hub-upload-interval") + 1] == "500"
    launch.prepare(request)
    assert len(info_calls) == 1  # Moving main is never resolved again on resume.
    assert snapshots[0][1]["revision"] == snapshots[3][1]["revision"] == "c" * 40
    # A run prepared by the pre-cap launcher can adopt the smaller validation
    # set while retaining the resolved dataset SHA and cached downloads.
    state_path = request.output_root / "launcher/run.json"
    state = json.loads(state_path.read_text())
    state.pop("training_signature")
    state.pop("validation_examples_per_category")
    import hashlib  # noqa: PLC0415

    legacy = (
        (launch.EXAMPLE / "train.yaml")
        .read_bytes()
        .split(b"\n# Runtime validation only;", 1)[0]
    )
    state["template_sha256"] = hashlib.sha256(
        legacy + (launch.EXAMPLE / "training_recipe.yaml").read_bytes()
    ).hexdigest()
    state_path.write_text(json.dumps(state))
    launch.prepare(request)
    assert json.loads(state_path.read_text())["validation_examples_per_category"] == 250
    assert len(info_calls) == 1
    request.trajectories_revision = "e" * 40
    with pytest.raises(ValueError, match="retain the pinned"):
        launch.prepare(request)


def test_incomplete_release_fails_before_model_download(tmp_path, monkeypatch):
    data = records()
    data["chat"]["manifest"]["subsets"].pop()

    def download(repo, path, **kw):
        domain = path.split("/")[2]
        file = tmp_path / f"{domain}.json"
        file.write_text(json.dumps(data[domain]))
        return file

    monkeypatch.setattr(launch, "hf_hub_download", download)
    with pytest.raises(ValueError, match="25 complete subsets in chat"):
        launch.release_records("repo", "c" * 40)


def test_prepare_only_does_not_launch_gpu_processes(tmp_path, monkeypatch):
    request = args(tmp_path)
    request.prepare_only = True  # Avoid checking a real port in this unit test.
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    monkeypatch.setattr(
        launch, "prepare", lambda _: ({}, tmp_path / "train.yaml", provenance)
    )
    monkeypatch.setattr(launch, "commands", lambda *a: (["server"], ["train"]))
    monkeypatch.setattr(CampaignConfig, "load", lambda _: None)
    monkeypatch.setattr(
        training,
        "train_full_data",
        lambda *a, **kw: {
            "job": {
                "training_examples": 100000,
                "updates": 2000,
                "validation_examples": 1000,
            }
        },
    )
    # Prepare-only verifies data without ever launching a subprocess.
    monkeypatch.setattr(
        launch.subprocess,
        "Popen",
        lambda *a, **kw: pytest.fail("unexpected GPU launch"),
    )
    assert launch.run(request) == 0


def test_stop_only_owned_process_group_even_if_leader_exited(monkeypatch):
    calls = []
    child = SimpleNamespace(pid=123456, wait=lambda **kw: calls.append(("wait", kw)))
    monkeypatch.setattr(launch.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    launch.stop_owned(child)
    assert calls[0] == (123456, signal.SIGTERM)


def test_readiness_detects_early_server_exit():
    with pytest.raises(RuntimeError, match="exited during startup"):
        launch.wait_ready(SimpleNamespace(poll=lambda: 1), 18000, 60)


@pytest.mark.parametrize("startup_failure", [False, True])
def test_managed_run_stops_owned_processes(tmp_path, monkeypatch, startup_failure):
    request = args(tmp_path)
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    monkeypatch.setattr(
        launch, "prepare", lambda _: ({}, tmp_path / "train.yaml", provenance)
    )
    monkeypatch.setattr(launch, "commands", lambda *a: (["server"], ["trainer"]))
    monkeypatch.setattr(CampaignConfig, "load", lambda _: None)
    monkeypatch.setattr(
        training,
        "train_full_data",
        lambda *a, **kw: {
            "job": {
                "training_examples": 100000,
                "updates": 2000,
                "validation_examples": 1000,
            }
        },
    )

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def connect_ex(self, *a):
            return 111

    monkeypatch.setattr(launch.socket, "socket", Probe)
    server = SimpleNamespace(poll=lambda: None)
    trainer = SimpleNamespace(poll=lambda: 0, returncode=0)
    started, stopped = [], []

    def popen(command, **kw):
        assert kw["start_new_session"] is True
        started.append(command)
        return server if command == ["server"] else trainer

    def ready(*a):
        if startup_failure:
            raise RuntimeError("startup failed")

    monkeypatch.setattr(launch.subprocess, "Popen", popen)
    monkeypatch.setattr(launch, "wait_ready", ready)
    monkeypatch.setattr(launch, "stop_owned", stopped.append)
    if startup_failure:
        with pytest.raises(RuntimeError, match="startup failed"):
            launch.run(request)
        assert started == [["server"]]
        assert stopped == [None, server]
    else:
        assert launch.run(request) == 0
        assert started == [["server"], ["trainer"]]
        assert stopped == [trainer, server]
