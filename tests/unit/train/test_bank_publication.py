"""Completed bank publication, pinned reads and cloud subset scheduling."""

import hashlib
import json
import shutil

import pytest
import torch
from huggingface_hub import CommitOperationAdd
from safetensors.torch import save_file

from speculators.bank import cloud, hub, publication
from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.publication import BankPublisher
from speculators.bank.workflow import train_config
from speculators.train.config.schema import TrainConfig
from tests.unit.train.test_bank_hub import FakeHub


@pytest.fixture
def bank_setup(tmp_path, monkeypatch):
    remote = FakeHub(tmp_path / "remote")
    monkeypatch.setattr(publication, "HfApi", lambda: remote)
    monkeypatch.setattr(publication, "hf_hub_download", remote.download)
    monkeypatch.setattr(hub, "hf_hub_download", remote.download)
    cfg = BankConfig(
        output_root=tmp_path / "scratch",
        lora={"rank": 2},
        hub={"mode": "consumer", "trajectories_revision": "a" * 40},
        training={"seeds": [42, 43]},
    )
    manifest = {
        "root": str(cfg.output_root),
        "data_path": str(cfg.output_root / "data"),
        "hidden_states_path": str(cfg.output_root / "hidden"),
        "prepared_fingerprint": "prepared",
        "cache_fingerprint": "cache",
        "revisions": {"target_sha": "target", "drafter_sha": "draft"},
        "subsets": [
            {"id": "math-00000", "train_indices": [0], "validation_indices": [1]},
            {"id": "math-00001", "train_indices": [2], "validation_indices": [3]},
            {
                "id": "math-00005",
                "role": "adaptation",
                "train_indices": [4],
                "validation_indices": [5],
            },
        ],
    }
    write_json(cfg.output_root / "manifest.json", manifest)
    return cfg, manifest, remote


def completed_run(cfg, manifest, subset, seed):
    run = cfg.output_root / "runs" / subset["id"] / f"seed-{seed}"
    snapshot = run / "snapshots" / "step-00000010"
    adapter = snapshot / "adapter"
    adapter.mkdir(parents=True)
    save_file(
        {
            "base.o_proj.lora_A.weight": torch.zeros(2, 4096),
            "base.o_proj.lora_B.weight": torch.zeros(4096, 2),
            "base.v_proj.lora_A.weight": torch.zeros(2, 8192),
            "base.v_proj.lora_B.weight": torch.zeros(1024, 2),
        },
        adapter / "adapter_model.safetensors",
    )
    write_json(
        adapter / "adapter_config.json",
        {
            "r": 2,
            "lora_alpha": cfg.lora.alpha,
            "target_modules": cfg.lora.target_modules,
        },
    )
    context = {
        "prepared_fingerprint": manifest["prepared_fingerprint"],
        "cache_fingerprint": manifest["cache_fingerprint"],
        "subset_id": subset["id"],
        "seed": seed,
    }
    write_json(
        snapshot / "entry.json",
        {
            **context,
            "epoch": 1,
            "collection_step": 10,
            "global_step": 20,
            "adapter_sha256": file_digest(adapter / "adapter_model.safetensors"),
            "adapter_config_sha256": file_digest(adapter / "adapter_config.json"),
        },
    )
    write_json(run / "epochs" / "1.json", {"loss": 0.5})
    recovery = run / "recovery" / "1"
    shutil.copytree(adapter, recovery / "adapter")
    write_json(
        recovery / "training_state.json",
        {
            "phase": "done",
            "global_step": 20,
            "collection_step": 10,
        },
    )
    torch.save({}, recovery / "optimizer_state_dict.pt")
    torch.save([], recovery / "rng.pt")
    write_json(run / "bank_context.json", context)
    settings = train_config(cfg, manifest, subset["id"], seed, "target", "draft")
    write_json(
        run / "resolved_train.json",
        TrainConfig.model_validate(settings["train"]).flatten(),
    )
    (run / "train_command.txt").write_text("Training command and git SHA\n")
    (run / "speculators.patch").write_text("Training source patch\n")
    return run


def test_complete_bank_is_one_commit_and_pinned_restore(bank_setup):
    cfg, manifest, remote = bank_setup
    subset = manifest["subsets"][0]
    run = completed_run(cfg, manifest, subset, 42)
    publisher = BankPublisher(cfg)
    record, commit = publisher.publish_run(manifest, subset, 42)
    assert remote.commits == 1
    assert record["candidate_count"] == 1
    assert "runs/math-00000/seed-42/train_command.txt" in record["files"]
    assert (
        "runs/math-00000/seed-42/recovery/1/optimizer_state_dict.pt" in record["files"]
    )
    prefix = publisher.prefix(subset, 42)
    paths = remote.list_repo_files(cfg.hub.banks_repo)
    assert prefix + "/completion.json" in paths
    publisher.publish_run(manifest, subset, 42)
    assert remote.commits == 1
    copy_cfg = cfg.model_copy(update={"output_root": cfg.output_root.parent / "reader"})
    consumer = BankPublisher(copy_cfg, create=False)
    restored = consumer.restore_run(manifest, subset, 42, revision=commit)
    assert restored == record
    restored_run = copy_cfg.output_root / "runs" / subset["id"] / "seed-42"
    assert file_digest(
        restored_run / "snapshots/step-00000010/adapter/adapter_model.safetensors"
    ) == file_digest(run / "snapshots/step-00000010/adapter/adapter_model.safetensors")
    assert remote.commits == 1
    changed = cfg.model_copy(deep=True)
    changed.training.warmup_lr = 0.0002
    with pytest.raises(ValueError, match="settings differ"):
        BankPublisher(changed).completed(manifest, subset, 42)


def test_cloud_stages_one_subset_and_publishes_each_seed(bank_setup, monkeypatch):
    cfg, manifest, remote = bank_setup
    cfg.execution.prefetch_next_subset = False
    monkeypatch.setattr(cloud, "prepare", lambda config, stage: manifest)
    staged, trained = [], []

    def stage(config, local_dir, subsets, workers):
        assert workers == 8
        local_dir.mkdir(parents=True)
        (local_dir / "activation").write_bytes(b"downloaded input")
        staged.append((subsets[0], local_dir))

    def train(config):
        subset_id = config.execution.training_subsets[0]
        seed = config.training.seeds[0]
        # Prior subset inputs have been released before staging the next subset.
        assert len(list((cfg.output_root / ".cloud-activations").iterdir())) == 1
        subset = next(s for s in manifest["subsets"] if s["id"] == subset_id)
        completed_run(config, manifest, subset, seed)
        trained.append((subset_id, seed))

    monkeypatch.setattr(cloud, "stage_hidden", stage)
    monkeypatch.setattr(cloud.workflow, "train", train)
    cloud.train_cloud(cfg)
    assert trained == [
        ("math-00000", 42),
        ("math-00000", 43),
        ("math-00001", 42),
        ("math-00001", 43),
    ]
    assert len(staged) == 2
    assert all(not path.exists() for _, path in staged)
    assert remote.commits == 4
    cloud.train_cloud(cfg)
    assert len(staged) == 2
    assert remote.commits == 4


def test_producer_provenance_is_restored_and_carried_into_bank(bank_setup):
    cfg, manifest, remote = bank_setup
    repo = cfg.hub.trajectories_repo
    remote.create_repo(repo, private=False)
    command = b"Pinned extraction command and target checkpoint hash\n"
    prompts = b"{}\n"
    commit = f"{1:040x}"

    def reference(path, content):
        return {
            "repo": repo,
            "path": path,
            "revision": commit,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
        }

    provenance_path = "provenance/producer/vllm_command.txt"
    selection = {
        "manifest": manifest,
        "artifacts": [reference("prompts.jsonl", prompts)],
        "provenance": [reference(provenance_path, command)],
    }
    remote.create_commit(
        repo,
        parent_commit="0" * 40,
        operations=[
            CommitOperationAdd(
                path_in_repo="releases/r0001/math/select.json",
                path_or_fileobj=json.dumps(selection).encode(),
            ),
            CommitOperationAdd(path_in_repo="prompts.jsonl", path_or_fileobj=prompts),
            CommitOperationAdd(path_in_repo=provenance_path, path_or_fileobj=command),
        ],
    )
    cfg.hub.trajectories_revision = commit
    restored_manifest = hub.restore(cfg, "select")
    restored = list((cfg.output_root / "provenance").rglob("vllm_command.txt"))
    assert len(restored) == 1
    assert restored[0].read_bytes() == command
    subset = restored_manifest["subsets"][0]
    completed_run(cfg, restored_manifest, subset, 42)
    record, _ = BankPublisher(cfg).publish_run(restored_manifest, subset, 42)
    assert any(name.endswith("vllm_command.txt") for name in record["files"])
