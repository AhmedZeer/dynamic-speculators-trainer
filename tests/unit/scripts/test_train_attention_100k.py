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
    for key in (
        "ram_cache_gib",
        "concurrency",
        "buffered_examples",
        "lookahead_updates",
        "reserve_gib",
    ):
        assert getattr(loaded.online, key) == getattr(reference.online, key)
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
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
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
        expected_gpu = "2" if command == ["server"] else "3"
        assert kw["env"]["CUDA_VISIBLE_DEVICES"] == expected_gpu
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


def test_pilot_prepares_small_config_and_isolates_mode(tmp_path, monkeypatch):
    request = args(tmp_path)
    request.pilot = True
    request.output_root = tmp_path / "pilot"
    request.hub_repo = "azeeer/test-pilot"
    monkeypatch.setattr(
        launch,
        "HfApi",
        lambda: SimpleNamespace(
            repo_info=lambda *a, **kw: SimpleNamespace(sha="c" * 40)
        ),
    )
    monkeypatch.setattr(launch, "release_records", lambda *a: records())
    monkeypatch.setattr(
        launch, "snapshot_download", lambda repo, **kw: str(tmp_path / "models" / repo)
    )
    cfg, path, provenance = launch.prepare(request)
    loaded = CampaignConfig.load(path)
    assert loaded.training_examples_per_category == 250
    assert loaded.validation_examples_per_category == 250
    assert loaded.optimization.pretraining_updates == 10
    assert loaded.optimization.pretraining_validation_interval == 10
    assert loaded.optimization.checkpoint_interval == 10
    assert loaded.hub_upload_interval == 10
    assert loaded.wandb.group.endswith("-pilot")
    assert "pilot" in str(loaded.input_root)
    _, command = launch.commands(request, cfg, path, provenance)
    assert command[command.index("--max-updates") + 1] == "10"
    assert command[command.index("--hub-upload-interval") + 1] == "10"
    request.pilot = False
    with pytest.raises(ValueError, match="separate --output-root"):
        launch.prepare(request)


@pytest.mark.parametrize("pilot", [False, True])
def test_cli_pilot_defaults_are_separate(tmp_path, monkeypatch, pilot):
    captured = []
    argv = ["train_attention_100k.py"] + (["--pilot"] if pilot else [])
    monkeypatch.setattr(launch.sys, "argv", argv)
    monkeypatch.setattr(launch, "run", lambda request: captured.append(request) or 0)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.setattr(launch.signal, "signal", lambda *a: None)
    assert launch.main() == 0
    request = captured[0]
    assert request.gpu_memory_utilization == 0.40
    assert request.max_num_seqs == 128
    assert request.output_root.name.endswith("-pilot") == pilot
    assert request.hub_repo.endswith("-pilot") == pilot


def test_cli_10k_uses_separate_ttt7_destinations(monkeypatch):
    captured = []
    monkeypatch.setattr(launch.sys, "argv", ["train_attention_100k.py", "--variant", "10k"])
    monkeypatch.setattr(launch, "run", lambda request: captured.append(request) or 0)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.setattr(launch.signal, "signal", lambda *a: None)
    assert launch.main() == 0
    assert captured[0].output_root.name == "dyn-spec-attention-100k-10k-v1-ttt7"
    assert captured[0].hub_repo == "azeeer/eagle3-generator-attention-100k-10k-v1-ttt7"


def test_cli_ttt3_continuation_retains_original_repo(monkeypatch):
    captured = []
    monkeypatch.setattr(
        launch.sys, "argv", ["train_attention_100k.py", "--variant", "10k-ttt3"]
    )
    monkeypatch.setattr(launch, "run", lambda request: captured.append(request) or 0)
    monkeypatch.setattr(launch.os, "chdir", lambda _: None)
    monkeypatch.setattr(launch.signal, "signal", lambda *a: None)
    assert launch.main() == 0
    request = captured[0]
    assert request.output_root.name == "dyn-spec-attention-100k-10k-ttt3-v1"
    assert request.hub_repo == "azeeer/eagle3-generator-attention-100k-2k-v1"
    assert launch.limits(request) == (100000, 10000, 1000)


def test_ttt3_continuation_uses_original_dataset_without_resolving_main(
    tmp_path, monkeypatch
):
    request = args(tmp_path)
    request.variant = "10k-ttt3"
    monkeypatch.setattr(launch, "HfApi", lambda: None)
    revisions = []
    monkeypatch.setattr(
        launch, "release_records", lambda repo, revision: revisions.append(revision) or records()
    )
    monkeypatch.setattr(
        launch, "snapshot_download", lambda repo, **kw: str(tmp_path / "models" / repo)
    )
    cfg, path, _ = launch.prepare(request)
    assert revisions == ["93750ccc4ba1701c2ec6b75ca5e4c1a5d867daa6"]
    recipe = yaml.safe_load(Path(CampaignConfig.load(path).training_recipe).read_text())
    assert recipe["training"]["ttt_steps"] == 3
    assert cfg["max_training_updates"] == 10000
    assert cfg["optimization"]["checkpoint_interval"] == 1000


def test_gpu_headroom_options_are_forwarded(tmp_path):
    request = args(tmp_path)
    request.max_num_seqs = 64
    request.gpu_memory_utilization = 0.35
    cfg = {"online": {"target_model_path": "/model", "shared_dir": "/shared"}}
    server, _ = launch.commands(
        request, cfg, tmp_path / "train.yaml", tmp_path / "provenance"
    )
    assert server[server.index("--max-num-seqs") + 1] == "64"
    assert server[server.index("--gpu-memory-utilization") + 1] == "0.35"
    assert server[server.index("--max-num-batched-tokens") + 1] == "8193"


def test_10k_variant_configuration_and_schedule(tmp_path, monkeypatch):
    request = args(tmp_path)
    request.variant = "10k"
    monkeypatch.setattr(
        launch, "HfApi", lambda: SimpleNamespace(
            repo_info=lambda *a, **kw: SimpleNamespace(sha="c" * 40)
        )
    )
    monkeypatch.setattr(launch, "release_records", lambda *a: records())
    monkeypatch.setattr(
        launch, "snapshot_download", lambda repo, **kw: str(tmp_path / "models" / repo)
    )
    cfg, path, provenance = launch.prepare(request)
    loaded = CampaignConfig.load(path)
    assert loaded.max_training_updates == 10000
    assert loaded.optimization.pretraining_updates == 10000
    assert loaded.wandb.group == "attention-100k-10k-v1-ttt7"
    recipe = yaml.safe_load(Path(loaded.training_recipe).read_text())
    assert recipe["training"]["ttt_steps"] == 7
    assert yaml.safe_load((launch.EXAMPLE / "training_recipe.yaml").read_text())[
        "training"
    ]["ttt_steps"] == 3
    assert loaded.optimization.checkpoint_interval == 1000
    assert loaded.hub_upload_interval == 1000
    _, command = launch.commands(request, cfg, path, provenance)
    assert command[command.index("--max-updates") + 1] == "10000"
    assert command[command.index("--hub-upload-interval") + 1] == "1000"
    corpus = SimpleNamespace(indices=lambda _: range(1000))
    schedule = training.EpochSchedule(loaded, corpus, loaded.experiment_subsets, 10)
    assert len(schedule.updates) >= 10000
    request.variant = "2k"
    with pytest.raises(ValueError, match="separate --output-root"):
        launch.prepare(request)


def test_prepared_run_accepts_only_resource_sizing_changes(tmp_path):
    import hashlib  # noqa: PLC0415

    template = yaml.safe_load((launch.EXAMPLE / "train.yaml").read_text())
    recipe = yaml.safe_load((launch.EXAMPLE / "training_recipe.yaml").read_text())
    template.pop("validation_examples_per_category")
    previous = {**template, "online": dict(template["online"])}
    previous["online"].update(ram_cache_gib=96, concurrency=512, buffered_examples=1024)
    state = {
        "training_signature": hashlib.sha256(
            json.dumps(
                {"training": previous, "recipe": recipe}, sort_keys=True
            ).encode()
        ).hexdigest()
    }
    (tmp_path / "train.yaml").write_text(yaml.safe_dump(previous))
    assert launch.resource_change_matches(state, template, recipe, tmp_path)
    assert template["online"]["concurrency"] == 128
    changed = {**template, "optimization": {**template["optimization"], "lr": 0.5}}
    assert not launch.resource_change_matches(state, changed, recipe, tmp_path)


@pytest.mark.parametrize("visible,expected", [
    ("2,3", ("2", "3")),
    ("3", ("3", "3")),
    ("GPU-first,GPU-second", ("GPU-first", "GPU-second")),
    ("2,3,5", ("2", "3")),
])
def test_gpu_assignment_isolates_subprocesses(monkeypatch, visible, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setenv("CLUSTER_JOB_MARKER", "preserved")
    server, trainer = launch.process_environments()
    assert (server["CUDA_VISIBLE_DEVICES"], trainer["CUDA_VISIBLE_DEVICES"]) == expected
    assert server["CLUSTER_JOB_MARKER"] == trainer["CLUSTER_JOB_MARKER"] == "preserved"
    assert launch.os.environ["CUDA_VISIBLE_DEVICES"] == visible


def test_gpu_assignment_without_explicit_visibility(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    server, trainer = launch.process_environments()
    assert "CUDA_VISIBLE_DEVICES" not in server
    assert "CUDA_VISIBLE_DEVICES" not in trainer


@pytest.mark.parametrize("visible", ["", "-1", "2,2", "2,,3"])
def test_invalid_gpu_assignment_fails_before_launch(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    with pytest.raises(ValueError, match="distinct GPU"):
        launch.process_environments()


def test_parallelism_overrides_and_checkpoint_cadence_can_resume(tmp_path, monkeypatch):
    request = args(tmp_path)
    request.variant = "10k"
    monkeypatch.setattr(launch, "HfApi", lambda: SimpleNamespace(
        repo_info=lambda *a, **kw: SimpleNamespace(sha="c" * 40)
    ))
    monkeypatch.setattr(launch, "release_records", lambda *a: records())
    monkeypatch.setattr(
        launch, "snapshot_download", lambda repo, **kw: str(tmp_path / repo)
    )
    cfg, path, provenance = launch.prepare(request)
    # Simulate the previously published 10K configuration's 500-step cadence.
    import hashlib  # noqa: PLC0415

    previous = yaml.safe_load((launch.EXAMPLE / "train_10k.yaml").read_text())
    previous.pop("validation_examples_per_category")
    previous["optimization"]["checkpoint_interval"] = 500
    previous["hub_upload_interval"] = 500
    saved = yaml.safe_load(path.read_text())
    saved["optimization"]["checkpoint_interval"] = 500
    saved["hub_upload_interval"] = 500
    path.write_text(yaml.safe_dump(saved))
    state_path = request.output_root / "launcher/run.json"
    state = json.loads(state_path.read_text())
    recipe = yaml.safe_load((launch.EXAMPLE / "training_recipe_ttt7.yaml").read_text())
    state["template_sha256"] = "old-template"
    state["training_signature"] = hashlib.sha256(json.dumps(
        {"training": previous, "recipe": recipe}, sort_keys=True
    ).encode()).hexdigest()
    state_path.write_text(json.dumps(state))
    request.extraction_concurrency = 64
    request.buffered_examples = 128
    request.lookahead_updates = 8
    request.max_num_batched_tokens = 16386
    cfg, path, provenance = launch.prepare(request)
    assert cfg["online"]["concurrency"] == 64
    assert cfg["online"]["buffered_examples"] == 128
    assert cfg["online"]["lookahead_updates"] == 8
    assert cfg["optimization"]["checkpoint_interval"] == 1000
    server, _ = launch.commands(request, cfg, path, provenance)
    assert server[server.index("--max-num-batched-tokens") + 1] == "16386"
