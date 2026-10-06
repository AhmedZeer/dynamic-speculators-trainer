"""Drive-safe bank reads own their tensors and preserve native crash diagnostics."""

import os
import signal
import subprocess

import pytest
import torch
from safetensors import SafetensorError
from safetensors.torch import save_file

from hs_connectors import FileTransfer
from speculators.bank import workflow
from speculators.bank.config import BankConfig
from speculators.bank.transfer import BankFileTransfer


def test_bank_reads_owned_tensors_without_mmap(tmp_path, monkeypatch, capsys):
    path = tmp_path / "hs_3.safetensors"
    expected = {
        "hidden_states": torch.arange(24, dtype=torch.bfloat16).reshape(3, 4, 2),
        "token_ids": torch.tensor([2, 3, 4]),
    }
    save_file(expected, path)
    # The standard connector keeps its existing default behavior.
    ordinary = FileTransfer(tmp_path).get_cached(3)
    assert all(torch.equal(ordinary[key], value) for key, value in expected.items())
    del ordinary

    def forbidden(*args, **kwargs):
        pytest.fail("Bank hidden states must not use mmap-backed load_file")

    monkeypatch.setattr("hs_connectors.transfer.load_file", forbidden)
    transfer = BankFileTransfer(tmp_path)
    loaded = transfer.get_cached(3)
    # A later truncate must not invalidate the already-loaded tensor storage.
    path.write_bytes(b"")
    assert all(torch.equal(loaded[key], value) for key, value in expected.items())
    assert transfer.get_cached(9) is None
    assert not capsys.readouterr().err


def test_bank_read_errors_include_file_path(tmp_path):
    path = tmp_path / "hs_3.safetensors"
    path.write_bytes(b"truncated")
    with pytest.raises(SafetensorError) as error:
        BankFileTransfer(tmp_path).get_cached(3)
    assert str(path) in "\n".join(error.value.__notes__)


def test_native_crash_enables_tracebacks_and_identifies_run(
    tmp_path, monkeypatch, capsys
):
    cfg = BankConfig()
    cfg.training.seeds = [43]
    manifest = {
        "root": str(tmp_path),
        "data_path": str(tmp_path / "data"),
        "hidden_states_path": str(tmp_path / "hidden_states"),
        "revisions": {"target_sha": "target", "drafter_sha": "draft"},
        "subsets": [
            {"id": "math-00000", "train_indices": [0], "validation_indices": [1]}
        ],
    }
    monkeypatch.setenv("PYTHONFAULTHANDLER", "0")
    monkeypatch.setattr(workflow, "select_data", lambda cfg: manifest)
    monkeypatch.setattr(workflow, "validate_cache", lambda *args: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda model, revision: revision
    )

    def crash(command, *, check, env):
        assert check
        assert env["PYTHONFAULTHANDLER"] == "1"
        assert os.environ["PYTHONFAULTHANDLER"] == "0"
        raise subprocess.CalledProcessError(-signal.SIGBUS, command)

    monkeypatch.setattr(workflow.subprocess, "run", crash)
    with pytest.raises(subprocess.CalledProcessError) as error:
        workflow.train(cfg)
    assert error.value.returncode == -signal.SIGBUS
    output = capsys.readouterr().err
    assert "terminated by SIGBUS" in output
    assert "seed=43" in output
    assert "last committed recovery checkpoint" in output
