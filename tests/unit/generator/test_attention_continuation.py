"""Portable recovery keeps bytes/state intact and rejects incompatible training."""

import copy
import json

import pytest

from speculators.generator import attention_continuation as recovery
from speculators.generator.campaign_resume import resume_identity


@pytest.fixture
def source(tmp_path, monkeypatch):
    archive = tmp_path / "source"
    archive.mkdir()
    (archive / "checkpoint.pt").write_bytes(b"model-optimizer-step2000-rng")
    identity = {
        "configuration": {
            "output_root": "/old/output", "bank_config": "/old/math.json",
            "additional_bank_configs": ["/old/code.json"],
            "optimization": {"pretraining_updates": 2000, "checkpoint_interval": 500},
            "architecture": {"width": 512},
        },
        "drafter_training": {"ttt_steps": 3},
        "bank_data": {"membership": "same-pinned-data"},
        "job": {
            "updates": 2000, "epoch_ends": [5552, 11150],
            "label": "all-data-generator-2000steps", "max_updates": 2000,
            "vllm_provenance_dir": "/old/provenance", "drafter_model_path": "/old/model",
            "hub_repo": recovery.SOURCE_REPO, "hub_upload_interval": 500,
        },
    }
    result = {
        "completed_updates": 2000,
        "checkpoint_sha256": recovery.sha256(archive / "checkpoint.pt"),
    }
    for name, value in (
        ("resolved_experiment.json", identity), ("result.json", result),
        ("configuration.json", {"optimization": {"pretraining_validation_interval": 250}}),
    ):
        (archive / name).write_text(json.dumps(value))
    for name in ("train_command.txt", "speculators.patch", "checkpoint_sha256.txt"):
        (archive / name).write_text("provenance")
    for name in ("vllm_command.txt", "vllm.patch"):
        (archive / name).write_text("source extraction provenance")
    (archive / "run.json").write_text(json.dumps({
        "trajectories_revision": recovery.TRAJECTORY_REVISION,
    }))

    def download(repo, name, revision):
        assert repo == recovery.SOURCE_REPO
        assert revision == recovery.SOURCE_REVISION
        return str(archive / name.split("/")[-1])

    monkeypatch.setattr(recovery, "hf_hub_download", download)
    current = copy.deepcopy(identity)
    current["configuration"].update(
        output_root="/new/output", bank_config="/new/math.json",
        additional_bank_configs=["/new/code.json"],
    )
    current["configuration"]["optimization"].update(
        pretraining_updates=10000, checkpoint_interval=1000,
    )
    current["job"].update(
        updates=10000, max_updates=10000, label="all-data-generator-10000steps",
        vllm_provenance_dir="/new/provenance", drafter_model_path="/new/model",
        hub_upload_interval=1000,
    )
    return archive, identity, current


def test_restore_continues_without_overwriting_local_progress(tmp_path, source):
    archive, saved, current = source
    root = tmp_path / "run"
    recovery.restore(root, {"job": current["job"]})
    output = root / "pretraining" / current["job"]["label"]
    checkpoint = output / "checkpoint.pt"
    assert checkpoint.read_bytes() == (archive / "checkpoint.pt").read_bytes()
    assert not (output / "result.json").exists()  # Source completion cannot skip training.
    assert resume_identity(output, current) == saved
    assert json.loads((output / "hub_checkpoints.json").read_text())[recovery.SOURCE_FOLDER]
    for key in ("architecture",):
        changed = copy.deepcopy(current)
        changed["configuration"][key]["width"] = 1024
        assert resume_identity(output, changed) == changed
    changed = copy.deepcopy(current)
    changed["drafter_training"]["ttt_steps"] = 7
    assert resume_identity(output, changed) == changed
    changed = copy.deepcopy(current)
    changed["bank_data"]["membership"] = "different-data"
    assert resume_identity(output, changed) == changed
    checkpoint.write_bytes(b"newer-local-progress")
    recovery.restore(root, {"job": current["job"]})
    assert checkpoint.read_bytes() == b"newer-local-progress"


def test_bad_source_checksum_never_installs_checkpoint(tmp_path, source):
    archive, _, current = source
    (archive / "checkpoint.pt").write_bytes(b"corrupt")
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="checksum"):
        recovery.restore(root, {"job": current["job"]})
    assert not list(root.glob("pretraining/**/checkpoint.pt"))


def test_schedule_must_match_before_restoring(tmp_path, source):
    _, _, current = source
    current["job"]["epoch_ends"] = [6000, 12000]
    with pytest.raises(ValueError, match="schedule"):
        recovery.restore(tmp_path / "run", {"job": current["job"]})
