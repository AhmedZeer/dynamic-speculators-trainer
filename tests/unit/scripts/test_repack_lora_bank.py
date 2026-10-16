"""Standalone bank packing preserves weights/provenance and isolates uploads."""

import importlib.util
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/repack_lora_bank.py"
spec = importlib.util.spec_from_file_location("repack_lora_bank", SCRIPT)
repack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repack)


@pytest.fixture
def downloaded_bank(tmp_path):
    local = tmp_path / "download"
    root = local / "releases/b0002/math/math-00000/seed-47"
    run = root / "runs/math-00000/seed-47"
    for step, epoch in ((10, 1), (20, 1), (30, 2)):
        snapshot = run / "snapshots" / f"step-{step:08d}"
        adapter = snapshot / "adapter"
        adapter.mkdir(parents=True)
        save_file(
            {
                "base.o_proj.lora_A.weight": torch.arange(4).reshape(2, 2).float()
                + step,
                "base.o_proj.lora_B.weight": torch.ones(2, 2, dtype=torch.bfloat16)
                * step,
            },
            adapter / "adapter_model.safetensors",
        )
        repack.write_json(adapter / "adapter_config.json", {"r": 2, "lora_alpha": 4})
        repack.write_json(
            snapshot / "entry.json", {"epoch": epoch, "collection_step": step}
        )
        (snapshot / "train_command.txt").write_text("original training command\n")
        (snapshot / "speculators.patch").write_text("original source patch\n")
    recovery = run / "recovery/2"
    recovery.mkdir(parents=True)
    (recovery / "optimizer_state_dict.pt").write_bytes(b"original optimizer bytes")
    server = root / "provenance/server"
    server.mkdir(parents=True)
    (server / "vllm_command.txt").write_text("original vllm command\n")
    (server / "checkpoint_sha256.txt").write_text("original model hash\n")
    files = {
        p.relative_to(root).as_posix(): {
            "bytes": p.stat().st_size,
            "sha256": repack.checksum(p),
        }
        for p in root.rglob("*")
        if p.is_file()
    }
    repack.write_json(
        root / "completion.json",
        {
            "candidate_count": 3,
            "files": files,
            "identity": {"training": {"warmup_epochs": 1}},
        },
    )
    return local, root


def test_pack_preserves_exact_weights_and_all_original_artifacts(
    downloaded_bank, tmp_path
):
    local, root = downloaded_bank
    before = {
        p.relative_to(local).as_posix(): repack.checksum(p)
        for p in local.rglob("*")
        if p.is_file()
    }
    output = tmp_path / "packed"
    catalog = repack.pack_bank(local, output)
    assert catalog["release"] == catalog["source_release"] == "b0002"
    assert catalog["runs"][0]["snapshot_count"] == 3
    packed = output / "releases/b0002/math/math-00000/seed-47"
    manifest = json.loads((packed / "manifest.json").read_text())
    assert len(manifest["epochs"]) == 2
    for epoch in manifest["epochs"]:
        weights = load_file(packed / epoch["file"])
        assert repack.checksum(packed / epoch["file"]) == epoch["sha256"]
        for snapshot in epoch["snapshots"]:
            original = load_file(
                root
                / "runs/math-00000/seed-47/snapshots"
                / snapshot["snapshot_id"]
                / "adapter/adapter_model.safetensors"
            )
            for key, packed_key in snapshot["tensor_keys"].items():
                assert weights[packed_key].dtype == original[key].dtype
                torch.testing.assert_close(
                    weights[packed_key], original[key], rtol=0, atol=0
                )
    archived = {}
    for filename in manifest["archives"]:
        with tarfile.open(packed / filename) as archive:
            for member in archive.getmembers():
                archived[member.name] = archive.extractfile(member).read()
    record = json.loads((root / "completion.json").read_text())
    for name in record["files"]:
        if "/snapshots/" not in name or not name.endswith("adapter_model.safetensors"):
            assert archived[name] == (root / name).read_bytes()
    assert archived["completion.json"] == (root / "completion.json").read_bytes()
    assert before == {
        p.relative_to(local).as_posix(): repack.checksum(p)
        for p in local.rglob("*")
        if p.is_file()
    }
    assert (output / "releases/b0002/repack-provenance/repack_lora_bank.py").is_file()


@pytest.mark.parametrize("parent", [None, "a" * 40])
def test_upload_uses_only_separate_repo_and_preserves_release(
    downloaded_bank, tmp_path, parent
):
    local, _ = downloaded_bank
    output = tmp_path / "packed"
    repack.pack_bank(local, output)
    calls = []
    commit = SimpleNamespace(commit_url="https://example.test/commit", oid="b" * 40)
    api = SimpleNamespace(
        create_repo=lambda *args, **kwargs: calls.append(("create", args, kwargs)),
        repo_info=lambda *args, **kwargs: SimpleNamespace(sha=parent),
        list_repo_files=lambda *args, **kwargs: [],
        create_commit=lambda **kwargs: (calls.append(("commit", kwargs)), commit)[1],
    )
    assert repack.publish_bank(output, repack.DEST_REPO, api=api) == commit
    uploaded = calls[-1][1]
    assert uploaded["repo_id"] == repack.DEST_REPO != repack.SOURCE_REPO
    assert uploaded["parent_commit"] == parent
    assert all(
        op.path_in_repo.startswith("releases/b0002/") for op in uploaded["operations"]
    )
    with pytest.raises(ValueError, match="different repository"):
        repack.publish_bank(output, repack.SOURCE_REPO, api=api)
    api.list_repo_files = lambda *args, **kwargs: ["releases/b0002/catalog.json"]
    api.repo_info = lambda *args, **kwargs: SimpleNamespace(sha="c" * 40)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        repack.publish_bank(output, repack.DEST_REPO, api=api)


def test_missing_source_file_fails_without_partial_release(downloaded_bank, tmp_path):
    local, root = downloaded_bank
    next(
        root.glob("runs/*/seed-*/snapshots/*/adapter/adapter_model.safetensors")
    ).unlink()
    output = tmp_path / "packed"
    with pytest.raises(ValueError, match="Missing or incomplete"):
        repack.pack_bank(local, output)
    assert not (output / "releases/b0002").exists()


def test_cli_no_push_builds_from_download_directory(downloaded_bank, tmp_path):
    local, _ = downloaded_bank
    output = tmp_path / "packed"
    result = subprocess.run(  # noqa: S603 -- fixed local script and pytest-owned paths
        [
            sys.executable,
            str(SCRIPT),
            "--local-dir",
            str(local),
            "--output-dir",
            str(output),
            "--no-push",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "releases/b0002/catalog.json").is_file()
    assert "Packing bank runs" in result.stderr
    assert "Snapshots" in result.stderr
