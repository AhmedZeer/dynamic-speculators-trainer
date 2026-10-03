from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import os

import torch
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    CollateFn,
)
from speculators.train.distributed import get_dp_rank, get_dp_size
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    torch.set_num_threads(1)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.accelerator.is_available():
        torch.accelerator.set_device_index(local_rank)


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    max_batches: int | None = None,
    sampler_seed: int | None = None,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=get_dp_size(),
        rank=get_dp_rank(),
        max_batches=max_batches,
        seed=sampler_seed or 0,
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=True,
        collate_fn=CollateFn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
        # Iterator creation draws a worker seed even with zero workers. Keep that
        # draw out of training's global RNG so mid-epoch resume preserves dropout.
        generator=(
            torch.Generator().manual_seed(sampler_seed)
            if sampler_seed is not None
            else None
        ),
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    generation_validation_retries: int,
    max_consecutive_generation_failures: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    train_data_ratio: float = 0.9,
    max_train_samples: int | None = None,
    max_train_batches: int | None = None,
    train_indices: list[int] | None = None,
    val_indices: list[int] | None = None,
    sampler_seed: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.
    """
    _limit_worker_threads()
    noise_transform = AddUniformNoise(std=noise_std)

    if (train_indices is None) != (val_indices is None):
        raise ValueError("Explicit training and validation selections must be paired")
    if train_indices is not None and set(train_indices) & set(val_indices or []):
        raise ValueError("Training and validation selections overlap")
    if train_indices is None and not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    train_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        transform=noise_transform,
        train_ratio=train_data_ratio,
        max_train_samples=max_train_samples,
        split="train",
        row_indices=train_indices,
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
        generation_validation_retries=generation_validation_retries,
        max_consecutive_generation_failures=max_consecutive_generation_failures,
    )
    val_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        train_ratio=train_data_ratio,
        max_train_samples=max_train_samples,
        split="val",
        row_indices=val_indices,
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
        generation_validation_retries=generation_validation_retries,
        max_consecutive_generation_failures=max_consecutive_generation_failures,
    )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        max_batches=max_train_batches,
        sampler_seed=sampler_seed,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        sampler_seed=sampler_seed,
    )

    return train_loader, val_loader
