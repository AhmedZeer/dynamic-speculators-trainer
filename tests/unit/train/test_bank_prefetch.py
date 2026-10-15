"""Prefetched activation batches preserve noise and subset scheduling."""

import pickle
from threading import Event

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file

from speculators.bank import cloud
from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.hub_cache import HubActivationStore
from speculators.bank.transfer import BankFileTransfer
from speculators.train.data import ArrowDataset
from speculators.train.dataloader import _setup_dataloader
from speculators.train.noise_transforms import IndexedUniformNoise
from tests.unit.train.test_bank_publication import (  # noqa: F401
    bank_setup,
    completed_run,
)


def test_spawned_workers_preserve_indexed_noise_and_epoch_changes(tmp_path):
    rows = [
        {"input_ids": [i, i + 1], "loss_mask": [0, 1], "seq_len": 2} for i in range(6)
    ]
    data = tmp_path / "data"
    Dataset.from_list(rows).with_format("torch").save_to_disk(str(data))
    hidden = data / "hidden_states"
    hidden.mkdir()
    for i, row in enumerate(rows):
        save_file(
            {
                "token_ids": torch.tensor(row["input_ids"]),
                "hidden_states": torch.full((2, 4, 3), float(i)),
            },
            hidden / f"hs_{i}.safetensors",
        )
    transform = IndexedUniformNoise(0.05, 42)
    dataset = ArrowDataset(
        8,
        data,
        transfer=BankFileTransfer(hidden),
        row_indices=list(range(6)),
        on_missing="raise",
        transform=transform,
    )
    serial = _setup_dataloader(dataset, 8, 3, num_workers=0, sampler_seed=42)
    parallel = _setup_dataloader(
        dataset, 8, 3, num_workers=2, prefetch_factor=2, sampler_seed=42
    )
    before = torch.get_rng_state().clone()
    expected = list(serial)
    observed = list(parallel)
    assert torch.equal(before, torch.get_rng_state())
    for actual, wanted in zip(observed, expected, strict=True):
        for key in wanted:
            if isinstance(wanted[key], torch.Tensor):
                torch.testing.assert_close(actual[key], wanted[key], rtol=0, atol=0)
    transform.set_epoch(1)
    changed = list(parallel)
    assert not torch.equal(changed[0]["hidden_states"], expected[0]["hidden_states"])
    restarted = IndexedUniformNoise(0.05, 42)
    restarted.set_epoch(1)
    raw = dataset._get_raw_data(3)
    torch.testing.assert_close(
        restarted.apply_indexed(dict(raw), 3)["hidden_states"],
        dataset[3]["hidden_states"],
        rtol=0,
        atol=0,
    )


def test_activation_cache_reuses_owned_payloads_and_evicts(tmp_path, monkeypatch):
    local, hidden = tmp_path / "local", tmp_path / "hidden"
    local.mkdir()
    rows = {}
    for i in range(2):
        path = local / f"{i}.safetensors"
        save_file(
            {
                "hidden_states": torch.full((2, 4, 3), float(i)),
                "token_ids": torch.arange(2),
            },
            path,
        )
        rows[str(i)] = {
            "format": "raw",
            "primary_id": str(i),
            "shard": {
                "path": path.name,
                "sha256": file_digest(path),
                "bytes": path.stat().st_size,
            },
        }
    write_json(hidden / "index.json", {"rows": rows, "cache_gib": 1, "reserve_gib": 0})
    store = HubActivationStore(hidden, local, memory_cache_gib=112 / 2**30)
    first = store.read(0)
    first["hidden_states"] = torch.ones_like(first["hidden_states"])
    original = store.read_tensors
    calls = []

    def read(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(store, "read_tensors", read)
    store.read(0)
    assert not calls
    assert torch.count_nonzero(store.read(0)["hidden_states"]) == 0
    store.read(1)
    store.read(0)
    assert len(calls) == 2
    assert store.memory_bytes <= store.memory_capacity
    monkeypatch.delattr(store, "read_tensors")
    spawned = pickle.loads(pickle.dumps(store))  # noqa: S301 -- locally created object
    assert spawned.memory_bytes == 0
    assert not spawned.memory


@pytest.mark.parametrize("space_available", [True, False])
def test_cloud_overlaps_or_defers_next_download(
    bank_setup,  # noqa: F811 -- imported pytest fixture
    monkeypatch,
    space_available,
):
    cfg, manifest, remote = bank_setup
    cfg.training.seeds = [42]
    monkeypatch.setattr(cloud, "prepare", lambda *args: manifest)
    monkeypatch.setattr(cloud, "can_stage", lambda *args: space_available)
    next_started, release_next = Event(), Event()
    paths = []

    def stage(config, local_dir, subsets, workers, include_validation=True):
        assert not include_validation
        local_dir.mkdir(parents=True)
        paths.append(local_dir)
        if subsets == ["math-00001"]:
            next_started.set()
            if space_available:
                assert release_next.wait(5)

    def train(config):
        name = config.execution.training_subsets[0]
        if name == "math-00000":
            if space_available:
                assert next_started.wait(5)
                release_next.set()
            else:
                assert not next_started.is_set()
        subset = next(s for s in manifest["subsets"] if s["id"] == name)
        completed_run(config, manifest, subset, 42)

    monkeypatch.setattr(cloud, "stage_hidden", stage)
    monkeypatch.setattr(cloud.workflow, "train", train)
    cloud.train_cloud(cfg)
    assert remote.commits == 2
    assert len(paths) == 2
    assert all(not p.exists() for p in paths)
