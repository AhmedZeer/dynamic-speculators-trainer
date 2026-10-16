"""Local packed inputs and online extraction exercised with real tiny training."""

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import load_from_disk
from safetensors.torch import load_file, save_file

from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.generator import campaign, online_activations
from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.online_activations import OnlineActivations
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401


@pytest.fixture
def online_campaign(tiny_campaign, tmp_path, monkeypatch):  # noqa: F811
    cfg, downloads = tiny_campaign
    cfg.activation_source = "vllm"
    cfg.data_path = tmp_path / "packed"
    cfg.trajectories_path = tmp_path / "trajectories"
    cfg.online.shared_dir = tmp_path / "shared"
    cfg.online.reserve_gib = 0
    cfg.pilot.updates = 2
    catalog = {
        "format": "speculators-packed-lora-bank",
        "version": 1,
        "release": "b0002",
        "runs": [],
    }
    for bank_path in [cfg.bank_config, *cfg.additional_bank_configs]:
        bank = BankConfig.load(bank_path)
        bank.hidden_states.dtype = "float32"
        write_json(bank_path, bank.model_dump(mode="json"))
        domain = bank.dataset.category
        manifest = json.loads((bank.output_root / "manifest.json").read_text())
        manifest["cache_fingerprint"] = domain + "-cache"
        manifest["cache_spec"] = {
            "hidden_states": bank.hidden_states.model_dump(mode="json"),
            "models": {
                role: getattr(bank.models, role) for role in ("target", "drafter")
            },
        }
        rows = load_from_disk(manifest["data_path"])
        # Distinct train/validation prompts exercise prompt-only extraction;
        # the underlying shared fixture otherwise uses identical prompts.
        rows = rows.map(
            lambda row, i: {
                "input_ids": [1, 3 if i % 12 < 8 else 2, *row["input_ids"][2:]]
            },
            with_indices=True,
        )
        artifacts = []
        for subset in manifest["subsets"]:
            for split in ("train", "validation"):
                indices = subset[f"{split}_indices"]
                subset[f"{split}_ids"] = list(rows.select(indices)["primary_id"])
                relative = (
                    f"releases/{bank.hub.release}/{domain}/"
                    f"{subset['id']}/{split}.parquet"
                )
                path = cfg.trajectories_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                rows.select(indices).to_parquet(path)
                artifacts.append({"path": relative, "bytes": path.stat().st_size})
            snapshots, tensors = [], {}
            if int(subset["id"].rsplit("-", 1)[1]) >= 4:
                continue
            for step in range(1, 11):
                sid = f"step-{step:08d}"
                path = (
                    bank.output_root
                    / "runs"
                    / subset["id"]
                    / "seed-42"
                    / "snapshots"
                    / sid
                )
                original = load_file(path / "adapter" / "adapter_model.safetensors")
                keys = {key: f"{sid}.{key}" for key in original}
                tensors.update({keys[k]: v for k, v in original.items()})
                snapshots.append(
                    {
                        "snapshot_id": sid,
                        "tensor_keys": keys,
                        "entry": {"epoch": 1, "collection_step": step},
                    }
                )
            relative = f"{domain}/{subset['id']}/seed-42"
            output = cfg.data_path / "releases/b0002" / relative
            output.mkdir(parents=True)
            save_file(tensors, output / "epoch-01.safetensors")
            packed = {
                "source_identity": {
                    k: manifest[k]
                    for k in ("prepared_fingerprint", "cache_fingerprint", "revisions")
                },
                "epochs": [
                    {
                        "collection_epoch": 1,
                        "file": "epoch-01.safetensors",
                        "bytes": (output / "epoch-01.safetensors").stat().st_size,
                        "snapshots": snapshots,
                    }
                ],
            }
            packed["source_identity"].update(
                subset=subset, seed=42, lora=bank.lora.model_dump(mode="json")
            )
            write_json(output / "manifest.json", packed)
            catalog["runs"].append(
                {
                    "path": relative,
                    "manifest_sha256": file_digest(output / "manifest.json"),
                }
            )
        write_json(
            cfg.trajectories_path / f"releases/{bank.hub.release}/{domain}/data.json",
            {"manifest": manifest, "artifacts": artifacts},
        )
    write_json(cfg.data_path / "releases/b0002/catalog.json", catalog)
    requests = []

    class Client:
        models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=bank.models.target)])
        )

        def close(self):
            pass

    def extract(client, model, data, **kwargs):
        tokens = data["input_ids"]
        requests.append(list(tokens))
        path = cfg.online.shared_dir / f"{uuid.uuid4()}.safetensors"
        save_file(
            {
                "token_ids": torch.tensor(tokens),
                "hidden_states": torch.arange(
                    len(tokens) * 32, dtype=torch.float32
                ).reshape(-1, 4, 8)
                / 100,
            },
            path,
        )
        return str(path)

    monkeypatch.setattr(online_activations.openai, "OpenAI", lambda **_: Client())
    monkeypatch.setattr(online_activations, "generate_hidden_states", extract)
    return cfg, downloads, requests


def test_local_online_ten_jobs_and_pilot(online_campaign):
    cfg, downloads, requests = online_campaign
    result = campaign.run_campaign(cfg, "all")
    assert result["runs"] == 10
    assert not downloads
    assert requests
    assert any(len(r) == 2 for r in requests)
    assert any(len(r) > 2 for r in requests)
    assert not list(cfg.online.shared_dir.iterdir())
    report = json.loads((cfg.output_root / "pilot/report.json").read_text())
    assert report["passed"]
    assert len(report["results"]) == 3
    assert all(r["cache_end"]["temporary_bytes"] == 0 for r in report["results"])
    before = len(requests)
    assert campaign.run_campaign(cfg, "all")["runs"] == 10
    assert len(requests) == before


def test_online_full_prefetch_supplies_summary_without_second_request(online_campaign):
    cfg, downloads, requests = online_campaign
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    try:
        cache.prefetch([{"index": 0, "summary": False}])
        summary = cache.get_summary(0)
        payload = cache.get_full(0)
        torch.testing.assert_close(summary, payload["hidden_states"][1, :-1].flatten())
        assert len(requests) == 1
        assert not downloads
        assert not list(cfg.online.shared_dir.iterdir())
    finally:
        cache.close()


def test_online_rejects_wrong_tokens_and_cleans_file(online_campaign, monkeypatch):
    cfg, _, _ = online_campaign
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])

    def bad(*args, **kwargs):
        path = cfg.online.shared_dir / "bad.safetensors"
        save_file(
            {"token_ids": torch.tensor([99]), "hidden_states": torch.zeros(1, 4, 8)},
            path,
        )
        return str(path)

    monkeypatch.setattr(online_activations, "generate_hidden_states", bad)
    try:
        with pytest.raises(ValueError, match="Token ids"):
            cache.get_full(0)
        assert not list(cfg.online.shared_dir.iterdir())
    finally:
        cache.close()


def test_local_input_rejects_changed_membership(online_campaign):
    cfg, _, _ = online_campaign
    path = cfg.trajectories_path / "releases/r0001/math/data.json"
    record = json.loads(path.read_text())
    record["manifest"]["subsets"][0]["train_ids"][0] = "different"
    write_json(path, record)
    with pytest.raises(ValueError, match="identity/order"):
        load_local_inputs(cfg)


def test_online_configuration_dry_run():
    cfg = CampaignConfig.load(
        Path("examples/eagle3-lora/generator_campaign_online.yaml")
    )
    result = campaign.run_campaign(cfg, dry_run=True)
    assert len(result["jobs"]) == 10
    assert cfg.online.buffered_examples == 8
