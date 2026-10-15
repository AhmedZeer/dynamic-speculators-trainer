"""Unit tests for training dataloader setup."""

from unittest.mock import Mock

from speculators.train import dataloader


def test_worker_init_binds_to_local_rank(monkeypatch):
    set_num_threads = Mock()
    is_available = Mock(return_value=True)
    set_device_index = Mock()

    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(dataloader.torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(dataloader.torch.accelerator, "is_available", is_available)
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0)

    set_num_threads.assert_called_once_with(1)
    is_available.assert_called_once_with()
    set_device_index.assert_called_once_with(3)


def test_worker_init_skips_device_binding_without_accelerator(monkeypatch):
    set_num_threads = Mock()
    is_available = Mock(return_value=False)
    set_device_index = Mock()

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setattr(dataloader.torch, "set_num_threads", set_num_threads)
    monkeypatch.setattr(dataloader.torch.accelerator, "is_available", is_available)
    monkeypatch.setattr(
        dataloader.torch.accelerator, "set_device_index", set_device_index
    )

    dataloader._worker_init_fn(worker_id=0)

    set_num_threads.assert_called_once_with(1)
    is_available.assert_called_once_with()
    set_device_index.assert_not_called()


def test_train_only_loader_never_constructs_validation_dataset(monkeypatch):
    dataset = Mock()
    arrow = Mock(return_value=dataset)
    setup = Mock(return_value="training-loader")
    monkeypatch.setattr(dataloader, "ArrowDataset", arrow)
    monkeypatch.setattr(dataloader, "_setup_dataloader", setup)
    train, validation = dataloader.create_train_val_loaders(
        data_path="unused",
        total_seq_len=8192,
        hidden_states_dtype=dataloader.torch.bfloat16,
        noise_std=0.05,
        vllm_endpoint="unused",
        on_missing="raise",
        on_generate="cache",
        verifier_name_or_path="target",
        request_timeout=600,
        max_retries=3,
        generation_validation_retries=2,
        max_consecutive_generation_failures=20,
        hidden_size=4096,
        num_target_layers=3,
        num_workers=4,
        prefetch_factor=2,
        preprocess=None,
        train_indices=[0],
        val_indices=[1],
        sampler_seed=42,
        include_validation=False,
    )
    assert train == "training-loader"
    assert validation is None
    assert arrow.call_count == setup.call_count == 1
    assert arrow.call_args.kwargs["split"] == "train"
