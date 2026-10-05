"""Bank diagnostics are visible before extraction and ignore root log filters."""

import logging
import time
from importlib import import_module

import pytest

from speculators.bank import progress, workflow
from speculators.bank.config import BankConfig


@pytest.mark.parametrize("root_level", [logging.WARNING, logging.CRITICAL])
def test_stage_logs_blocking_work_and_memory_without_root_logger(capsys, root_level):
    root = logging.getLogger()
    previous = root.level
    root.setLevel(root_level)
    try:
        with progress.stage_progress("Manifest loading", interval=0.01):
            time.sleep(0.045)
    finally:
        root.setLevel(previous)
    output = capsys.readouterr().err
    assert "Manifest loading: starting" in output
    assert "Manifest loading: still running" in output
    assert "Manifest loading: completed" in output
    assert "process RSS=" in output


def test_stage_failure_stops_heartbeat_and_preserves_error(capsys):
    def fail():
        with progress.stage_progress("Loading", interval=0.01):
            raise OSError("Drive unavailable")

    with pytest.raises(OSError, match="Drive unavailable"):
        fail()
    assert "Loading: stopped" in capsys.readouterr().err
    time.sleep(0.03)
    assert not capsys.readouterr().err


def test_prepare_reports_source_and_manifest_stage_before_extraction(
    tmp_path, monkeypatch, capsys
):
    cli = import_module("speculators.cli.lora_bank")
    config = tmp_path / "bank.yaml"
    config.write_text("{}")
    (tmp_path / "manifest.json").write_text("{}")
    cfg = BankConfig.model_validate({"output_root": str(tmp_path)})
    monkeypatch.setattr(cli.BankConfig, "load", lambda path: cfg)
    manifest = {"root": str(tmp_path), "subsets": []}
    observed = []

    def select(settings):
        output = capsys.readouterr().err
        assert f"code={workflow.__file__}" in output
        assert "Loading bank configuration: completed" in output
        assert "Loading or selecting bank manifest: starting" in output
        return manifest

    def extract(settings, selected):
        observed.append(selected)
        assert "Prepare stage hidden: starting" in capsys.readouterr().err

    monkeypatch.setattr(workflow, "select_data", select)
    monkeypatch.setattr(workflow, "extract", extract)
    cli.prepare(config, stage="hidden")
    assert observed == [manifest]
    assert "Prepare stage hidden: completed" in capsys.readouterr().err


def test_hidden_stage_missing_manifest_fails_without_loading_source(
    tmp_path, monkeypatch, capsys
):
    cfg = BankConfig.model_validate({"output_root": str(tmp_path)})

    def forbidden(settings):
        pytest.fail("Hidden stage must not select the full source dataset")

    monkeypatch.setattr(workflow, "select_data", forbidden)
    with pytest.raises(
        FileNotFoundError, match="Check output_root and the Drive mount"
    ):
        workflow.prepare(cfg, "hidden")
    assert str(tmp_path) in capsys.readouterr().err


@pytest.mark.parametrize("failed", [True, False])
def test_training_startup_stack_watchdog_stops_on_readiness_or_failure(
    monkeypatch, failed
):
    from types import SimpleNamespace  # noqa: PLC0415

    cli = import_module("speculators.train.cli")
    calls = []
    monkeypatch.setattr(
        progress.faulthandler,
        "dump_traceback_later",
        lambda timeout, **kwargs: calls.append(timeout),
    )
    monkeypatch.setattr(
        progress.faulthandler,
        "cancel_dump_traceback_later",
        lambda: calls.append("cancel"),
    )

    def run(cfg, startup_ready=None):
        if failed:
            raise OSError("startup blocked")
        assert calls == [60]
        startup_ready()
        assert calls == [60, "cancel"]
        return "completed"

    monkeypatch.setattr(cli, "_run_training", run)
    cfg = SimpleNamespace(bank=SimpleNamespace(bank_manifest="manifest.json"))
    if failed:
        with pytest.raises(OSError, match="startup blocked"):
            cli.main(cfg)
    else:
        assert cli.main(cfg) == "completed"
    assert calls == [60, "cancel"]
