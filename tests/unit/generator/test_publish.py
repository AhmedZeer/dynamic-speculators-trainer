"""Portable campaign exports, provenance inclusion, and explicit Hub uploads."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from speculators.bank.artifacts import file_digest, write_json
from speculators.cli import app
from speculators.generator import campaign, publish
from speculators.generator.serving.bundle import bundle_state, load_generator
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401


@pytest.fixture(params=["generator", "static_lora"])
def completed_run(online_campaign, tmp_path, request):  # noqa: F811
    cfg, _, requests = online_campaign
    job = campaign.jobs(cfg)[0]
    if request.param == "static_lora":
        job = {**job, "arm": "fresh_lora", "label": "test-fresh-lora"}
    result = campaign.run_one(cfg, job)
    run = Path(result["checkpoint_path"]).parent
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    write_json(drafter / "config.json", {})
    provenance = cfg.output_root / "provenance" / "vllm"
    provenance.mkdir(parents=True)
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text(f"original {name}\n")
    (provenance / "unrelated.bin").write_bytes(b"do not upload")
    return cfg, run, drafter, provenance, requests


def test_push_exports_loadable_weights_and_provenance(completed_run, monkeypatch):
    cfg, run, drafter, provenance, requests = completed_run
    before = len(requests)
    calls = []

    class API:
        def create_repo(self, **kwargs):
            calls.append(kwargs)

        def upload_folder(self, **kwargs):
            calls.append(kwargs)
            folder = kwargs["folder_path"]
            metadata, state = bundle_state(folder)
            if metadata["kind"] == "generator":
                load_generator(metadata, state)
            else:
                assert set(state) == {"o_proj.A", "o_proj.B", "v_proj.A", "v_proj.B"}
            assert (
                metadata["source_job"]
                == json.loads((run / "resolved_experiment.json").read_text())["job"]
            )
            assert not (folder / "checkpoint.pt").exists()
            assert (folder / "train_command.txt").is_file()
            assert (folder / "speculators.patch").is_file()
            assert (folder / "timing.json").is_file()
            assert not (folder / "provenance/vllm/unrelated.bin").exists()
            for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
                assert (folder / "provenance/vllm" / name).read_bytes() == (
                    provenance / name
                ).read_bytes()
            manifest = json.loads((folder / "upload_manifest.json").read_text())
            for name, entry in manifest["files"].items():
                assert file_digest(folder / name) == entry["sha256"]
            return SimpleNamespace(
                oid="commit-sha", commit_url="https://hf.test/commit"
            )

    monkeypatch.setattr(publish, "HfApi", API)
    report = publish.push_run(
        run, "example/direct-4k", drafter_model_path=drafter, private=True
    )
    assert report["commit_sha"] == "commit-sha"
    assert calls[0] == {
        "repo_id": "example/direct-4k",
        "repo_type": "model",
        "private": True,
        "exist_ok": True,
    }
    assert calls[1]["repo_id"] == "example/direct-4k"
    assert not Path(calls[1]["folder_path"]).exists()
    assert len(requests) == before
    assert cfg.wandb.enabled is False


def test_cli_dry_run_and_optional_training_checkpoint(completed_run, monkeypatch):
    _, run, drafter, _, _ = completed_run
    monkeypatch.setattr(publish, "HfApi", lambda: pytest.fail("dry run contacted Hub"))
    result = CliRunner().invoke(
        app,
        [
            "generator",
            "push-to-hub",
            "--run-dir",
            str(run),
            "--repo-id",
            "example/dry-run",
            "--drafter-model-path",
            str(drafter),
            "--include-training-checkpoint",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["dry_run"]
    assert "checkpoint.pt" in report["files"]
    assert report["bytes"] > (run / "checkpoint.pt").stat().st_size


def test_bad_checkpoint_and_missing_provenance_fail_before_upload(
    completed_run, monkeypatch
):
    _, run, drafter, provenance, _ = completed_run
    monkeypatch.setattr(
        publish, "HfApi", lambda: pytest.fail("invalid upload contacted Hub")
    )
    identity_path = run / "resolved_experiment.json"
    identity = json.loads(identity_path.read_text())
    changed = {**identity, "job": {**identity["job"], "label": "changed"}}
    write_json(identity_path, changed)
    with pytest.raises(ValueError, match="identities differ"):
        publish.push_run(run, "example/identity", drafter_model_path=drafter)
    write_json(identity_path, identity)
    (provenance / "vllm_command.txt").unlink()
    with pytest.raises(FileNotFoundError, match="--provenance-dir"):
        publish.push_run(run, "example/missing", drafter_model_path=drafter)
    with (run / "checkpoint.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        publish.push_run(run, "example/changed", drafter_model_path=drafter)


def test_colocated_extraction_provenance(tmp_path):
    run = tmp_path / "pretraining" / "run"
    run.mkdir(parents=True)
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (tmp_path / name).write_text(name)
    assert publish.extraction_provenance(run) == tmp_path
