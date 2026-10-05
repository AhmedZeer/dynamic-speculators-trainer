"""Bank data pairing, update cadence, and real PEFT recovery tests."""

# Exercise legacy global NumPy state used by existing training augmentation.
# ruff: noqa: NPY002

import json
import random
import shutil

import numpy as np
import pytest
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from torch import nn

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.inspect import factors, update_distance
from speculators.bank.workflow import (
    cache_spec,
    partition_rows,
    prepare_arrow,
    prepare_rows,
    train_config,
    validate_cache,
)
from speculators.train.bank import BankSchedule, BankTrainer, capture_rng, restore_rng
from speculators.train.config import TrainConfig
from speculators.train.data import ArrowDataset
from speculators.train.dataloader import _setup_dataloader
from speculators.train.lora import load_adapter_checkpoint
from speculators.train.trainer import TrainerConfig


def small_config():
    return BankConfig.model_validate(
        {
            "partition": {
                "train_examples": 2,
                "validation_examples": 1,
                "subset_ids": [0, 1],
            },
            "conditioning": {"max_examples": 2},
        }
    )


def source_rows():
    return [
        {
            "uuid": str(i),
            "messages": [
                {"role": "user", "content": f"Problem {i}"},
                {"role": "assistant", "content": "Original answer"},
            ],
        }
        for i in range(8)
    ]


def test_partition_is_stable_disjoint_and_deduplicates():
    rows = source_rows()
    rows.append({**rows[0], "uuid": "alias"})
    selected, subsets, stats = partition_rows(rows, small_config())
    reordered, _, _ = partition_rows(list(reversed(rows)), small_config())
    assert [r["id"] for r in selected] == [r["id"] for r in reordered]
    assert stats["unique_prompts"] == 8
    assert stats["incomplete_group_examples"] == 2
    all_indices = []
    for subset in subsets:
        assert not set(subset["train_ids"]) & set(subset["validation_ids"])
        all_indices.extend(subset["train_indices"] + subset["validation_indices"])
    assert sorted(all_indices) == list(range(6))


def test_preparation_reorders_and_preserves_prompt_boundary(tmp_path):
    selected, subsets, _ = partition_rows(source_rows(), small_config())
    (tmp_path / "prompts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in selected)
    )
    responses = [
        {
            "primary_id": row["id"],
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 0, 1],
            "metadata": {"sampling_params": {}},
        }
        for row in reversed(selected)
    ]
    rows = prepare_rows(responses, {"root": str(tmp_path), "category": "math"}, 8)
    assert [r["primary_id"] for r in rows] == [r["id"] for r in selected]
    assert all(r["prompt_length"] == 2 for r in rows)
    with pytest.raises(ValueError, match="identities"):
        prepare_rows(responses[:-1], {"root": str(tmp_path), "category": "math"}, 8)
    responses[0]["loss_mask"] = [0, 1, 0]
    with pytest.raises(ValueError, match="prefix"):
        prepare_rows(responses, {"root": str(tmp_path), "category": "math"}, 8)


def test_explicit_selection_uses_original_hidden_state_indices(tmp_path):
    rows = [
        {"input_ids": [i, i + 1], "loss_mask": [0, 1], "seq_len": 2} for i in range(5)
    ]
    Dataset.from_list(rows).with_format("torch").save_to_disk(str(tmp_path / "data"))
    cache = tmp_path / "data" / "hidden_states"
    cache.mkdir()
    for i, row in enumerate(rows):
        save_file(
            {
                "token_ids": torch.tensor(row["input_ids"]),
                "hidden_states": torch.full((2, 4, 3), float(i)),
            },
            str(cache / f"hs_{i}.safetensors"),
        )
    dataset = ArrowDataset(8, tmp_path / "data", row_indices=[4, 1], on_missing="raise")
    assert dataset._map_to_file_idx(0) == 4
    assert dataset._map_to_file_idx(1) == 1
    assert dataset._get_raw_data(0)["hidden_states"][0, 0] == 4
    assert dataset._get_raw_data(1)["input_ids"].tolist() == [1, 2]
    loader = _setup_dataloader(
        dataset, total_seq_len=8, hidden_size=3, num_workers=0, sampler_seed=42
    )
    rng_before = torch.get_rng_state().clone()
    iterator = iter(loader)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert iterator is not None
    save_file(
        {"token_ids": torch.tensor([99, 99]), "hidden_states": torch.ones(2, 4, 3)},
        str(cache / "hs_1.safetensors"),
    )
    with pytest.raises(ValueError, match="token ids"):
        dataset._get_raw_data(1)


def test_generated_train_configuration_roundtrips(tmp_path):
    cfg = BankConfig()
    manifest = {
        "root": str(tmp_path),
        "data_path": str(tmp_path / "data"),
        "hidden_states_path": str(tmp_path / "hidden_states"),
    }
    settings = train_config(cfg, manifest, "math-00000", 42, "target", "draft")

    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(settings))
    resolved = TrainConfig.resolve(["--config", str(path)])
    assert resolved.bank.bank_save_interval == 10
    assert resolved.trainer.epochs == 5
    assert resolved.data.num_workers == 0
    assert resolved.lora.lora_target_modules == ["o_proj", "v_proj"]
    with pytest.raises(SystemExit):
        TrainConfig.resolve(["--config", str(path), "--max-steps", "10"])


def test_effective_update_distance_is_factorization_invariant():
    a, b = torch.randn(2, 5), torch.randn(4, 2)
    assert update_distance({"v": (a, b)}, {"v": (a / 2, b * 2)}, 2) == pytest.approx(
        0, abs=1e-5
    )
    c, d = torch.randn(2, 5), torch.randn(4, 2)
    expected = torch.linalg.vector_norm(b @ a - d @ c).item() * 2
    assert update_distance({"v": (a, b)}, {"v": (c, d)}, 2) == pytest.approx(
        expected, rel=1e-5
    )
    with pytest.raises(ValueError, match="Non-finite"):
        factors(
            {"v.lora_A.weight": torch.full((2, 5), float("nan")), "v.lora_B.weight": b},
            2,
        )


def test_rng_state_roundtrip():
    before = capture_rng()
    values = (random.random(), np.random.rand(), torch.rand(3))
    restore_rng(before)
    assert random.random() == values[0]
    assert np.random.rand() == values[1]
    assert torch.equal(torch.rand(3), values[2])


class TinyDrafter(nn.Module):
    def __init__(self):
        super().__init__()
        self.o_proj = nn.Linear(3, 3, bias=False)
        self.v_proj = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.o_proj(x) + self.v_proj(x)


def tiny_lora():
    torch.manual_seed(12)
    random.seed(12)
    np.random.seed(12)
    model = get_peft_model(
        TinyDrafter(),
        LoraConfig(
            r=2, lora_alpha=4, target_modules=["o_proj", "v_proj"], lora_dropout=0.1
        ),
    )
    model._speculators_lora_enabled = True
    model._speculators_lora_save_merged = False
    return model


class PackedSampler:
    def __init__(self):
        self._cached_generated_batches = None
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._cached_generated_batches = None

    def _generate_batches(self, epoch):
        if (
            self._cached_generated_batches
            and self._cached_generated_batches[0] == epoch
        ):
            return self._cached_generated_batches[1]
        return list(range([3, 7, 8, 6, 9][epoch]))


class TinyLoader:
    num_workers = 0

    def __init__(self):
        self.batch_sampler = PackedSampler()

    def __len__(self):
        return len(self.batch_sampler._generate_batches(self.batch_sampler.epoch))

    def __iter__(self):
        return iter(self.batch_sampler._generate_batches(self.batch_sampler.epoch))


class CpuBankTrainer(BankTrainer):
    def setup_model(self):
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            load_adapter_checkpoint(self.model, self.checkpointer.prev_path)

    def train_epoch(self, epoch):
        self.model.train()
        self.train_loader.batch_sampler.set_epoch(epoch)
        skip = self._prepare_resume_skip(epoch)
        for local_step, _batch in enumerate(self.train_loader, 1 + skip):
            x = torch.randn(2, 3) + random.random() + np.random.rand()
            loss = self.model(x).square().mean()
            self._optimizers_zero_grad()
            loss.backward()
            self._optimizers_step()
            self.global_step += 1
            self.updates.append((epoch, self.optimizers[0].param_groups[0]["lr"]))
            self.after_optimizer_step(epoch, local_step)
            if self.stop_at == self.global_step:
                raise RuntimeError("simulated process failure")

    def val_epoch(self, epoch):
        self.evaluations.append(epoch)
        return {"loss_epoch": 1 / (epoch + 1)}


def cpu_trainer(path, *, resume=False, stop_at=None):
    trainer = CpuBankTrainer(
        tiny_lora(),
        TrainerConfig(
            lr=1e-3,
            num_epochs=5,
            save_path=str(path),
            scheduler_type="none",
            resume_from_checkpoint=resume,
        ),
        TinyLoader(),
        TinyLoader(),
        schedule=BankSchedule(1, 4, 1e-5, 10),
        context={"seed": 12},
    )
    trainer.stop_at = stop_at
    trainer.updates, trainer.evaluations = [], []
    return trainer


def test_epoch_schedule_and_continuous_collection_cadence(tmp_path):
    trainer = cpu_trainer(tmp_path)
    trainer.run_training()
    assert trainer.completed
    assert trainer.evaluations == [0, 1, 2, 3, 4]
    assert [lr for epoch, lr in trainer.updates if epoch == 0] == [1e-3] * 3
    assert all(lr == 1e-5 for epoch, lr in trainer.updates if epoch > 0)
    assert trainer.collection_step == 30
    paths = sorted((tmp_path / "snapshots").glob("step-*"))
    assert [p.name for p in paths] == [
        "step-00000010",
        "step-00000020",
        "step-00000030",
    ]
    assert [json.loads((p / "entry.json").read_text())["epoch"] for p in paths] == [
        2,
        3,
        4,
    ]
    assert len(list((tmp_path / "recovery").iterdir())) == 2
    resumed = cpu_trainer(tmp_path, resume=True)
    resumed.run_training()
    assert resumed.updates == []
    assert resumed.evaluations == []


def test_mid_epoch_resume_replays_to_identical_adapter(tmp_path):
    uninterrupted = cpu_trainer(tmp_path / "full")
    uninterrupted.run_training()
    interrupted = cpu_trainer(tmp_path / "resumed", stop_at=17)
    with pytest.raises(RuntimeError, match="simulated"):
        interrupted.run_training()
    resumed = cpu_trainer(tmp_path / "resumed", resume=True)
    resumed.run_training()
    assert resumed.collection_step == 30
    for name, value in uninterrupted.model.state_dict().items():
        assert torch.equal(value, resumed.model.state_dict()[name]), name
    for full_path in (tmp_path / "full" / "snapshots").glob("step-*"):
        resumed_path = tmp_path / "resumed" / "snapshots" / full_path.name
        full = load_file(str(full_path / "adapter/adapter_model.safetensors"))
        other = load_file(str(resumed_path / "adapter/adapter_model.safetensors"))
        assert all(torch.equal(value, other[name]) for name, value in full.items())
    optimizer = torch.load(
        resumed.checkpointer.prev_path / "optimizer_state_dict.pt", weights_only=True
    )
    assert all(
        state["exp_avg"].dtype == torch.float32 for state in optimizer["state"].values()
    )


def test_resume_repairs_unpublished_snapshot(tmp_path):
    trainer = cpu_trainer(tmp_path, stop_at=13)
    with pytest.raises(RuntimeError, match="simulated"):
        trainer.run_training()

    shutil.rmtree(tmp_path / "snapshots" / "step-00000010")
    resumed = cpu_trainer(tmp_path, resume=True)
    assert (tmp_path / "snapshots" / "step-00000010" / "entry.json").exists()
    resumed.run_training()
    assert len(list((tmp_path / "snapshots").glob("step-*"))) == 3


def test_prepared_cache_validates_membership_and_rejects_corruption(
    tmp_path, monkeypatch
):
    cfg = small_config()
    selected, subsets, _ = partition_rows(source_rows(), cfg)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in selected))
    responses = [
        {
            "primary_id": row["id"],
            "input_ids": [1, 2, 3],
            "loss_mask": [0, 0, 1],
            "metadata": {},
        }
        for row in reversed(selected)
    ]
    (tmp_path / "responses.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in responses)
    )
    revisions = {
        "target_sha": "target",
        "drafter_sha": "drafter",
        "dataset_sha": "data",
    }
    spec = cache_spec(cfg, revisions)
    manifest = {
        "root": str(tmp_path),
        "category": "math",
        "data_path": str(tmp_path / "data"),
        "hidden_states_path": str(tmp_path / "hidden_states"),
        "subsets": subsets,
        "prompts_sha256": file_digest(prompts),
        "revisions": revisions,
        "cache_spec": spec,
        "cache_fingerprint": digest(spec),
    }
    prepare_arrow(cfg, manifest)
    prepare_arrow(cfg, manifest)  # Idempotent, including the persisted Torch format.
    cache = tmp_path / "hidden_states"
    cache.mkdir()
    for i in range(len(selected)):
        save_file(
            {
                "token_ids": torch.tensor([1, 2, 3]),
                "hidden_states": torch.zeros(3, 4, 4096, dtype=torch.bfloat16),
            },
            str(cache / f"hs_{i}.safetensors"),
        )
    validate_cache(manifest, cfg)
    receipt = tmp_path / "hidden_states_validation.json"
    # A clean interruption resumes at the first unvalidated row, not row zero.
    from speculators.data_generation import offline  # noqa: PLC0415

    receipt.unlink()
    checker = offline.check_hidden_state_file
    visited = []

    def interrupted(path, *args):
        visited.append(path.name)
        if path.name == "hs_3.safetensors":
            raise InterruptedError("interrupted validation")
        return checker(path, *args)

    monkeypatch.setattr(offline, "check_hidden_state_file", interrupted)
    with pytest.raises(InterruptedError, match="interrupted validation"):
        validate_cache(manifest, cfg)
    assert json.loads(receipt.read_text())["validated_files"] == 3
    visited.clear()

    def resumed(path, *args):
        visited.append(path.name)
        return checker(path, *args)

    monkeypatch.setattr(offline, "check_hidden_state_file", resumed)
    validate_cache(manifest, cfg)
    assert visited == [f"hs_{i}.safetensors" for i in range(3, 6)]
    receipt_time = receipt.stat().st_mtime_ns
    validate_cache(manifest, cfg)
    assert receipt.stat().st_mtime_ns == receipt_time
    # A prior validation receipt must never permit training with missing rows.
    backup = tmp_path / "cache-backup"
    cache.rename(backup)
    with pytest.raises(FileNotFoundError, match="0/6 required files present"):
        validate_cache(manifest, cfg)
    backup.rename(cache)
    missing_paths = [cache / "hs_2.safetensors", cache / "hs_4.safetensors"]
    saved = [path.read_bytes() for path in missing_paths]
    for path in missing_paths:
        path.unlink()
    with pytest.raises(FileNotFoundError, match="4/6 required files present") as error:
        validate_cache(manifest, cfg)
    assert "First missing indices: 2, 4" in str(error.value)
    assert "--stage hidden" in str(error.value)
    assert receipt.stat().st_mtime_ns == receipt_time
    for path, payload in zip(missing_paths, saved, strict=True):
        path.write_bytes(payload)
    manifest["subsets"][0]["train_ids"].reverse()
    with pytest.raises(ValueError, match="membership"):
        validate_cache(manifest, cfg)
    manifest["subsets"][0]["train_ids"].reverse()
    write_json(tmp_path / "manifest.json", manifest)
    save_file(
        {
            "token_ids": torch.tensor([99, 2, 3]),
            "hidden_states": torch.zeros(3, 4, 4096, dtype=torch.bfloat16),
        },
        str(cache / "hs_0.safetensors"),
    )
    with pytest.raises(ValueError, match="Token ids"):
        validate_cache(manifest, cfg)
