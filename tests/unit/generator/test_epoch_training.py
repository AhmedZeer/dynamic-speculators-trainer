"""Exact data passes, epoch bundles, and recovery after a failed Hub upload."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.generator import epoch_training, publish
from speculators.generator.serving.bundle import read_bundle
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401


def test_schedule_visits_every_row_once_and_preserves_boundaries():
    cfg = SimpleNamespace(
        seed=42,
        context=SimpleNamespace(min_examples=1, max_examples=8),
        optimization=SimpleNamespace(episodes_per_update=4),
    )
    members = {"math-00000": list(range(23)), "code-00000": list(range(23, 50))}
    corpus = SimpleNamespace(indices=lambda name: members[name])
    schedule = epoch_training.EpochSchedule(cfg, corpus, list(members), 10)
    previous = 0
    for epoch, end in enumerate(schedule.epoch_ends, start=1):
        seen = []
        for step in range(previous, end):
            episodes = schedule.update(step)
            assert 1 <= len(episodes) <= 4
            for episode in episodes:
                assert set(episode["indices"]) <= set(members[episode["subset"]])
                seen.extend(episode["indices"])
        assert sorted(seen) == list(range(50))
        assert schedule.completed_epochs(end) == epoch
        assert schedule.completed_epochs(end - 1) == epoch - 1
        previous = end
    repeated = epoch_training.EpochSchedule(cfg, corpus, list(members), 10)
    assert repeated.updates == schedule.updates


def test_conditioning_configs_share_exact_24k_schedule_and_1000_step_budget():
    from speculators.generator.campaign_config import DOMAINS, CampaignConfig  # noqa: PLC0415
    from speculators.generator.campaign_train import conditioning_subset  # noqa: PLC0415

    root = Path(__file__).resolve().parents[3] / "examples/eagle3-lora"
    members = {}
    for domain in DOMAINS:
        for number in range(6):
            start = len(members) * 1000
            members[f"{domain}-{number:05d}"] = list(range(start, start + 1000))
    corpus = SimpleNamespace(indices=lambda name: members[name])
    schedules = []
    for name in [f"generator_conditioning_24k_1k_k{k}.yaml" for k in [1, 2, 4, 8]] + [
        "lora_conditioning_24k_1k.yaml"
    ]:
        cfg = CampaignConfig.load(root / name)
        schedule = epoch_training.EpochSchedule(
            cfg, corpus, sorted(members), 2, cohort_size=cfg.training_cohort_size
        )
        assert schedule.epoch_ends == [750, 1500]
        assert (
            sum(
                len(item["indices"])
                for update in schedule.updates[:1000]
                for item in update
            )
            == 32000
        )
        cohort = schedule.update(0)[0]["indices"]
        if cfg.training_conditioning_examples:
            selected = conditioning_subset(
                cohort, cfg.training_conditioning_examples, cfg.seed
            )
            assert len(selected) == cfg.training_conditioning_examples
            assert set(selected) <= set(cohort)
        schedules.append(schedule.updates[:1000])
    assert all(schedule == schedules[0] for schedule in schedules)


def test_capped_full_data_generator_and_static_lora_use_same_loss_schedule(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg, _, _ = online_campaign
    cfg.training_cohort_size = 8
    cfg.hub_upload_interval = 1000
    cfg.optimization.pretraining_validation_interval = 500
    provenance = tmp_path / "extractor"
    provenance.mkdir()
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text("test provenance")
    cfg.online.drafter_model_path = tmp_path / "drafter"
    cfg.online.drafter_model_path.mkdir()
    (cfg.online.drafter_model_path / "config.json").write_text("{}")
    uploads = []

    class API:
        def whoami(self):
            return {"name": "test"}

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            metadata = read_bundle(folder)
            saved = torch.load(folder / "checkpoint.pt", weights_only=True)
            assert saved["progress"] == 2
            assert saved["optimizer"]["state"]
            uploads.append(metadata)
            return SimpleNamespace(
                oid="test-commit", commit_url="https://hf.test/commit"
            )

    monkeypatch.setattr(epoch_training, "HfApi", API)
    monkeypatch.setattr(publish, "HfApi", API)
    timings = []
    for arm, k in [("generator", 1), ("generator", 8), ("lora", None)]:
        cfg.training_conditioning_examples = k
        root = tmp_path / f"{arm}-{k}"
        plan = epoch_training.train_full_data(
            cfg,
            root,
            f"example/{arm}-{k}",
            provenance,
            epochs=2,
            max_updates=2,
            arm=arm,
            dry_run=True,
        )["job"]
        assert plan["training_cohort_size"] == 8
        assert plan["updates"] == 2
        assert plan["arm"] == ("direct" if arm == "generator" else "fresh_lora")
        result = epoch_training.train_full_data(
            cfg,
            root,
            f"example/{arm}-{k}",
            provenance,
            epochs=2,
            max_updates=2,
            arm=arm,
        )
        assert result["completed_updates"] == 2
        assert result["completed_epochs"] == 0
        if k is not None:
            assert result["validation_conditioning_examples"] == min(4, k)
        timing = json.loads(
            (root / "pretraining" / plan["label"] / "timing.json").read_text()
        )
        assert [row["train/examples"] for row in timing] == [32, 32]
        assert [row["train/conditioning_size_mean"] for row in timing] == [
            k or 0,
            k or 0,
        ]
        timings.append([row["train/response_tokens"] for row in timing])
    assert timings[0] == timings[1] == timings[2]
    assert [metadata["kind"] for metadata in uploads] == [
        "generator",
        "generator",
        "static_lora",
    ]
    assert [
        metadata["source_job"].get("training_conditioning_examples")
        for metadata in uploads
    ] == [1, 8, None]


def test_full_data_cli_accepts_capped_static_arm(tmp_path, monkeypatch):
    from typer.testing import CliRunner  # noqa: PLC0415

    from speculators.cli import app  # noqa: PLC0415

    captured = {}

    def run(*args, **kwargs):
        captured.update(kwargs)
        return {"dry_run": kwargs["dry_run"], "arm": kwargs["arm"]}

    monkeypatch.setattr(epoch_training, "train_full_data", run)
    config = (
        Path(__file__).resolve().parents[3]
        / "examples/eagle3-lora/lora_conditioning_24k_1k.yaml"
    )
    result = CliRunner().invoke(
        app,
        [
            "generator",
            "train-full-data",
            "--config",
            str(config),
            "--output-root",
            str(tmp_path / "run"),
            "--hub-repo",
            "example/lora",
            "--vllm-provenance-dir",
            str(tmp_path / "provenance"),
            "--arm",
            "lora",
            "--max-updates",
            "1000",
            "--epochs",
            "2",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["max_updates"] == 1000
    assert captured["arm"] == "lora"


def test_hub_failure_resumes_boundary_and_uploads_portable_checkpoints(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg, _, _ = online_campaign
    cfg.context.min_examples = cfg.context.max_examples = 4
    cfg.optimization.pretraining_validation_interval = 1000
    cfg.hub_upload_interval = 10
    root = tmp_path / "epoch-run"
    provenance = tmp_path / "vllm"
    provenance.mkdir()
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text("original " + name)
    drafter = tmp_path / "local-drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text("{}")
    cfg.online.drafter_model_path = drafter
    fail = [True]
    uploaded = []

    class API:
        def whoami(self):
            return {"name": "test"}

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            folder = Path(kwargs["folder_path"])
            read_bundle(folder)
            saved = torch.load(folder / "checkpoint.pt", weights_only=True)
            assert saved["optimizer"]["state"]
            assert saved["torch_rng"].numel()
            assert (folder / "train_command.txt").exists()
            assert (folder / "speculators.patch").exists()
            assert (folder / "provenance/vllm/vllm_command.txt").exists()
            if fail[0]:
                fail[0] = False
                raise RuntimeError("simulated upload failure")
            uploaded.append((kwargs["path_in_repo"], saved["progress"]))
            return SimpleNamespace(
                oid="test-commit", commit_url="https://hf.test/commit"
            )

    monkeypatch.setattr(epoch_training, "HfApi", API)
    monkeypatch.setattr(publish, "HfApi", API)

    def unsupported_link(*_):
        raise OSError("hard links unavailable on mounted filesystem")

    monkeypatch.setattr(Path, "hardlink_to", unsupported_link)
    plan = epoch_training.train_full_data(
        cfg,
        root,
        "example/full-data",
        provenance,
        epochs=4,
        save_every_epochs=2,
        dry_run=True,
    )["job"]
    assert len(plan["subsets"]) == 24
    assert plan["training_examples"] == 192
    with pytest.raises(RuntimeError, match="upload failure"):
        epoch_training.train_full_data(
            cfg, root, "example/full-data", provenance, epochs=4, save_every_epochs=2
        )
    run = root / "pretraining/all-data-4epochs"
    failed = torch.load(run / "checkpoint.pt", weights_only=True)
    assert failed["progress"] == 10
    saved_identity = failed["identity"]
    assert [row["update"] for row in failed["history"]] == [0]
    cfg.optimization.pretraining_validation_interval = 2
    result = epoch_training.train_full_data(
        cfg, root, "example/full-data", provenance, epochs=4, save_every_epochs=2
    )
    assert result["completed_updates"] == plan["updates"]
    restored = torch.load(run / "checkpoint.pt", weights_only=True)
    assert restored["identity"] == saved_identity
    assert list((run / "provenance").glob("validation-change-*/configuration.json"))
    assert [row["update"] for row in result["history"]] == [
        0,
        *range(10, plan["updates"] + 1, 2),
    ]
    assert uploaded == [(f"step-{step:08d}", step) for step in [10, 20, 30, 40, 48]]
    ledger = json.loads((run / "hub_checkpoints.json").read_text())
    assert set(ledger) == {f"step-{step:08d}" for step in [10, 20, 30, 40, 48]}
    timing = json.loads((run / "timing.json").read_text())
    assert timing[-1]["train/examples_seen"] == 768
    assert timing[-1]["train/epoch"] == 4


def test_epoch_upload_identity_migrates_to_step_uploads(tmp_path):
    from copy import deepcopy  # noqa: PLC0415

    from speculators.generator.campaign_resume import resume_identity  # noqa: PLC0415

    saved = {
        "configuration": {
            "output_root": str(tmp_path),
            "optimization": {"pretraining_updates": 13369, "lr": 0.0001},
        },
        "job": {"updates": 13369, "epochs": 10, "save_every_epochs": 2},
    }
    (tmp_path / "resolved_experiment.json").write_text(json.dumps(saved))
    current = deepcopy(saved)
    current["job"].pop("save_every_epochs")
    current["job"]["hub_upload_interval"] = 1000
    assert resume_identity(tmp_path, current) == saved
    assert list((tmp_path / "provenance").glob("resume-*/configuration.json"))
    current["configuration"]["optimization"]["lr"] *= 2
    assert resume_identity(tmp_path, current) == current
