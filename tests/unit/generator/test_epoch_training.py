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


def test_hub_failure_resumes_boundary_and_uploads_portable_checkpoints(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    cfg, _, _ = online_campaign
    cfg.context.min_examples = cfg.context.max_examples = 4
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
    assert failed["progress"] == plan["epoch_ends"][1]
    result = epoch_training.train_full_data(
        cfg, root, "example/full-data", provenance, epochs=4, save_every_epochs=2
    )
    assert result["completed_updates"] == plan["updates"]
    assert uploaded == list(
        zip(["epoch-0002", "epoch-0004"], plan["epoch_ends"][1::2], strict=True)
    )
    ledger = json.loads((run / "hub_checkpoints.json").read_text())
    assert set(ledger) == {"epoch-0002", "epoch-0004"}
    timing = json.loads((run / "timing.json").read_text())
    assert timing[-1]["train/examples_seen"] == 768
    assert timing[-1]["train/epoch"] == 4
