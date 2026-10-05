"""Raw prefetch overlaps reads without changing augmentation or resume order."""

import threading

import pytest
import torch
from datasets import Dataset
from hs_connectors.transfer import FileTransfer
from safetensors.torch import save_file

from speculators.train.data import ArrowDataset
from speculators.train.dataloader import _setup_dataloader
from speculators.train.noise_transforms import AddUniformNoise
from speculators.train.prefetch import PrefetchBatchSampler, RawPrefetchTransfer


def make_data(tmp_path, count=6):
    data = tmp_path / "data"
    cache = tmp_path / "cache"
    cache.mkdir()
    Dataset.from_list(
        [
            {"input_ids": [i, i + 1, i + 2], "loss_mask": [0, 1, 1], "seq_len": 3}
            for i in range(count)
        ]
    ).with_format("torch").save_to_disk(data)
    for i in range(count):
        save_file(
            {
                "token_ids": torch.tensor([i, i + 1, i + 2]),
                "hidden_states": torch.full((3, 4, 2), float(i)),
            },
            str(cache / f"hs_{i}.safetensors"),
        )
    return data, cache


def loader(data, transfer, indices):
    dataset = ArrowDataset(
        datapath=str(data),
        max_len=6,
        transfer=transfer,
        on_missing="raise",
        row_indices=indices,
        transform=AddUniformNoise(0.05),
    )
    return _setup_dataloader(dataset, 6, 2, num_workers=0, sampler_seed=17)


def test_prefetch_reads_future_samples_before_consumer_and_preserves_rng():
    entered, release = threading.Event(), threading.Event()
    owner = threading.get_ident()
    readers = []

    class Source:
        def get_cached(self, index):
            readers.append(threading.get_ident())
            entered.set()
            assert release.wait(2)
            return {"hidden_states": torch.ones(4), "token_ids": torch.tensor([index])}

    transfer = RawPrefetchTransfer(Source(), 1024, 2)
    rng = torch.get_rng_state().clone()
    try:
        transfer.prefetch([5])
        assert entered.wait(2)
        assert torch.equal(rng, torch.get_rng_state())
        release.set()
        assert transfer.get_cached(5)["token_ids"].item() == 5
        assert transfer.get_cached(5)["token_ids"].item() == 5
        assert len(readers) == 1
        assert readers[0] != owner
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        release.set()
        transfer.close()


def test_real_loader_prefetch_preserves_packing_noise_and_subset_ids(tmp_path):
    data, cache = make_data(tmp_path)
    indices = [4, 1, 5, 0]
    torch.manual_seed(49)
    reference = list(loader(data, FileTransfer(cache), indices))
    reference_rng = torch.get_rng_state().clone()
    transfer = RawPrefetchTransfer(FileTransfer(cache), 1024 * 1024, 2)
    try:
        torch.manual_seed(49)
        actual = list(loader(data, transfer, indices))
        assert torch.equal(reference_rng, torch.get_rng_state())
        assert len(actual) == len(reference)
        for expected, batch in zip(reference, actual, strict=True):
            for key in expected:
                if isinstance(expected[key], torch.Tensor):
                    assert torch.equal(expected[key], batch[key]), key
                else:
                    assert expected[key] == batch[key], key
        # The second epoch reuses raw data rather than re-reading the files.
        reads = transfer.stats()["file_reads"]
        list(loader(data, transfer, indices))
        assert transfer.stats()["file_reads"] == reads
        assert transfer.stats()["cache_bytes"] <= 1024 * 1024
    finally:
        transfer.close()


def test_resume_sampler_skips_io_for_already_consumed_batches(tmp_path):
    data, cache = make_data(tmp_path)
    transfer = RawPrefetchTransfer(FileTransfer(cache), 1024 * 1024, 2)
    try:
        batches = loader(data, transfer, [4, 1, 5, 0])
        sampler = batches.batch_sampler
        assert isinstance(sampler, PrefetchBatchSampler)
        full = sampler._generate_batches(0)
        sampler._cached_generated_batches = (0, full[1:])
        wanted = {
            batches.dataset.file_indices[int(i)] for batch in full[1:] for i in batch
        }
        list(batches)
        assert transfer.stats()["file_reads"] == len(wanted)
    finally:
        transfer.close()


def test_prefetch_budget_eviction_and_read_errors():
    class Source:
        def get_cached(self, index):
            if index == 9:
                raise OSError("Drive disconnected")
            return {"hidden_states": torch.ones(8), "token_ids": torch.tensor([index])}

    transfer = RawPrefetchTransfer(Source(), 40, 1)
    try:
        for i in range(5):
            transfer.get_cached(i)
            assert transfer.stats()["cache_bytes"] <= 40
        with pytest.raises(OSError, match="Drive disconnected"):
            transfer.get_cached(9)
    finally:
        transfer.close()
    assert transfer.stats()["cache_bytes"] == 0


def test_prefetch_training_resume_matches_synchronous_adapter(tmp_path):
    from tests.unit.train.test_lora_bank import cpu_trainer  # noqa: PLC0415

    data, cache = make_data(tmp_path, count=8)
    indices = list(range(8))
    reference = cpu_trainer(tmp_path / "full")
    reference.train_loader = loader(data, FileTransfer(cache), indices)
    reference.run_training()

    transfer = RawPrefetchTransfer(FileTransfer(cache), 1024 * 1024, 2)
    try:
        interrupted = cpu_trainer(tmp_path / "resumed", stop_at=16)
        interrupted.train_loader = loader(data, transfer, indices)
        with pytest.raises(RuntimeError, match="simulated"):
            interrupted.run_training()
    finally:
        transfer.close()
    transfer = RawPrefetchTransfer(FileTransfer(cache), 1024 * 1024, 2)
    try:
        resumed = cpu_trainer(tmp_path / "resumed", resume=True)
        resumed.train_loader = loader(data, transfer, indices)
        resumed.run_training()
        assert resumed.global_step == reference.global_step
        for key, value in reference.model.state_dict().items():
            assert torch.equal(value, resumed.model.state_dict()[key]), key
    finally:
        transfer.close()
