"""A downloaded Hub layout is a direct training input, without network reads."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from safetensors.torch import save_file

from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.staging import stage_hidden
from speculators.bank.transfer import BankFileTransfer
from speculators.bank.workflow import train_config
from speculators.train.config.schema import TrainConfig


@pytest.mark.parametrize("raw", [True, False])
def test_snapshot_layout_reads_without_fetch_or_cache_copy(tmp_path, monkeypatch, raw):
    local = tmp_path / "download"
    relative = "releases/r0001/stem/stem-00000/raw/sample.safetensors"
    path = local / relative
    path.parent.mkdir(parents=True)
    states = torch.arange(48).reshape(6, 4, 2).to(torch.bfloat16)
    payload = {"hidden_states": states, "token_ids": torch.arange(6)}
    save_file(payload if raw else {f"sample/{k}": v for k, v in payload.items()}, path)
    ref = {"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)}
    hidden = tmp_path / "hidden"
    write_json(
        hidden / "index.json",
        {
            "cache_gib": 1,
            "reserve_gib": 0,
            "rows": {
                "7": {
                    "primary_id": "sample",
                    "shard": ref,
                    "format": "raw" if raw else "sharded",
                    "prompt_length": 3,
                }
            },
        },
    )

    def forbidden(*args, **kwargs):
        pytest.fail("A staged file must never be downloaded again")

    monkeypatch.setattr("speculators.bank.hub_cache.fetch", forbidden)
    transfer = BankFileTransfer(hidden, local)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(transfer.get_cached, [7, 7]))
    for result in results:
        torch.testing.assert_close(result["hidden_states"], states)
    torch.testing.assert_close(transfer.get_prompt_states(7, 3), states[:3])
    if raw:
        torch.testing.assert_close(
            transfer.hub_store.read(7, summary=True), states[2, :-1].flatten().float()
        )
    assert not list((hidden / ".shards").glob("*.safetensors"))


@pytest.mark.parametrize("include_validation", [True, False])
def test_parallel_staging_writes_consumable_subset_config(
    tmp_path, monkeypatch, include_validation
):
    root, local = tmp_path / "bank", tmp_path / "download"
    references, rows = {}, {}
    for index in range(2):
        relative = f"releases/r0001/stem/stem-00000/raw/{index}.safetensors"
        path = tmp_path / "remote" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {"token_ids": torch.arange(3), "hidden_states": torch.zeros(3, 4, 2)}, path
        )
        references[relative] = {
            "path": relative,
            "repo": "test/activations",
            "revision": "a" * 40,
            "bytes": path.stat().st_size,
            "sha256": file_digest(path),
        }
        rows[str(index)] = {"shard": references[relative], "format": "raw"}
    manifest = {
        "root": str(root),
        "data_path": str(root / "data"),
        "hidden_states_path": str(root / "hidden"),
        "subsets": [
            {"id": "stem-00000", "train_indices": [0], "validation_indices": [1]}
        ],
    }
    write_json(root / "manifest.json", manifest)
    write_json(root / "hidden" / "index.json", {"rows": rows})
    barrier = Barrier(2 if include_validation else 1)

    def download(**kwargs):
        assert kwargs["revision"] == "a" * 40
        barrier.wait(timeout=5)
        path = local / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path / "remote" / kwargs["filename"], path)
        return path

    monkeypatch.setattr("speculators.bank.staging.hf_hub_download", download)
    bank = BankConfig(output_root=root)
    generated = BankConfig.load(
        stage_hidden(bank, local, ["stem-00000"], 2, include_validation)
    )
    assert (local / list(references)[1]).exists() == include_validation
    assert generated.execution.training_subsets == ["stem-00000"]
    values = train_config(generated, manifest, "stem-00000", 42, "target", "draft")
    resolved = TrainConfig.model_validate(values["train"])
    assert resolved.bank.bank_activation_local_dir == str(local.resolve())
    assert json.loads((root / "manifest.json").read_text()) == manifest

    def unexpected_hash(path):
        pytest.fail("Verified unchanged files should not be rehashed")

    monkeypatch.setattr("speculators.bank.staging.file_digest", unexpected_hash)
    stage_hidden(bank, local, ["stem-00000"], 2, include_validation)
    monkeypatch.setattr("speculators.bank.staging.file_digest", file_digest)
    path = local / next(iter(references))
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="checksum mismatch"):
        stage_hidden(bank, local, ["stem-00000"], 2, include_validation)


@pytest.mark.parametrize("legacy_boundary", [None, 5])
def test_raw_summary_uses_explicit_trajectory_boundary(tmp_path, legacy_boundary):
    from speculators.bank.hub_cache import HubActivationStore  # noqa: PLC0415

    states = torch.arange(48).reshape(6, 4, 2).to(torch.bfloat16)
    path = tmp_path / "raw.safetensors"
    save_file({"hidden_states": states, "token_ids": torch.arange(6)}, path)
    row = {"format": "raw", "primary_id": "sample"}
    if legacy_boundary is not None:
        row["prompt_length"] = legacy_boundary
    result = HubActivationStore.read_tensors(path, row, prompt_length=3, summary=True)
    torch.testing.assert_close(result, states[2, :-1].flatten().float())
    # Explicit boundaries must not be written back into immutable source indexes.
    assert row.get("prompt_length") == legacy_boundary


@pytest.mark.parametrize("boundary", [None, 0, 7])
def test_raw_summary_rejects_missing_or_invalid_boundary(tmp_path, boundary):
    from speculators.bank.hub_cache import HubActivationStore  # noqa: PLC0415

    path = tmp_path / "raw.safetensors"
    save_file({"hidden_states": torch.zeros(6, 4, 2)}, path)
    row = {"format": "raw", "primary_id": "sample"}
    with pytest.raises(ValueError, match="valid trajectory prompt_length"):
        HubActivationStore.read_tensors(path, row, prompt_length=boundary, summary=True)
