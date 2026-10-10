"""Modal control-plane tests: no authentication or paid cloud execution."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("modal")
REPO = Path(__file__).resolve().parents[3]


def app_module():
    spec = importlib.util.spec_from_file_location(
        "modal_prepare", REPO / "scripts/modal/app.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runner(tmp_path, monkeypatch, *, complete=False, seconds=0):
    module = app_module()
    config = tmp_path / "prepare.yaml"
    config.write_text(yaml.safe_dump({"run_id": "test", "gpu_hours_limit": 1}))
    monkeypatch.setattr(module, "initialize", Mock())
    monkeypatch.setattr(module, "prepare", Mock())
    monkeypatch.setattr(module, "source_provenance", lambda: {"git_sha": "test"})
    report = {
        "banks": [
            {"domain": d, "subset": f"{d}-00000", "complete": complete}
            for d in ("chat", "math", "code", "stem")
        ],
        "gpu_runtime": {"seconds": seconds},
    }
    monkeypatch.setattr(module, "status", Mock())
    module.status.remote.return_value = report
    module.prepare.remote.return_value = []
    return module, config


def test_completed_corpus_does_not_allocate_gpu(tmp_path, monkeypatch):
    module, config = runner(tmp_path, monkeypatch, complete=True)
    module.main(command="prepare", config=str(config))
    module.initialize.remote.assert_called_once()
    module.prepare.remote.assert_not_called()


def test_runtime_guard_checked_before_gpu_request(tmp_path, monkeypatch):
    module, config = runner(tmp_path, monkeypatch, seconds=3600)
    with pytest.raises(RuntimeError, match="no GPU was requested"):
        module.main(command="prepare", config=str(config))
    module.prepare.remote.assert_not_called()


def test_smoke_separate_identity_and_all_domains(tmp_path, monkeypatch):
    module, config = runner(tmp_path, monkeypatch)
    module.main(command="prepare", config=str(config), smoke=True)
    values, _ = module.initialize.remote.call_args.args
    assert values["run_id"] == "test-smoke"
    assert values["bank"]["partition"] == {
        "train_examples": 2,
        "validation_examples": 1,
    }
    assert values["subset_ids"] == [0]
    assert values["bank"]["responses"]["max_tokens"] == 32
    module.prepare.remote.assert_called_once()


def test_default_command_is_status_without_gpu(tmp_path, monkeypatch):
    module, config = runner(tmp_path, monkeypatch)
    module.main(config=str(config))
    module.status.remote.assert_called_once_with("test")
    module.initialize.remote.assert_not_called()
    module.prepare.remote.assert_not_called()


def test_gpu_worker_commits_and_records_cumulative_time(tmp_path, monkeypatch):
    from speculators.data_generation import pipeline  # noqa: PLC0415

    module = app_module()
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module, "artifacts", Mock())
    monkeypatch.setattr(module, "models", Mock())
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *_args, **_kwargs: "test==1\n"
    )
    cfg = pipeline.PrepareConfig(run_id="test")
    root = tmp_path / cfg.run_id
    root.mkdir()
    (root / "selection.json").write_text(
        json.dumps(
            {
                "identity": cfg.identity(),
                "model_path": "/models/target",
                "banks": [
                    {"domain": "chat", "subset": 0, "root": "chat"},
                    {"domain": "math", "subset": 0, "root": "math"},
                ],
            }
        )
    )
    (root / "gpu_runtime.json").write_text('{"seconds": 60, "attempts": 1}')
    monkeypatch.setattr(pipeline, "bank_status", lambda _root: {"complete": False})
    calls = []

    def prepare(_cfg, directory, *_args, commit):
        calls.append(directory)
        commit()
        return {"complete": True}

    monkeypatch.setattr(pipeline, "prepare_subset", prepare)
    result = module.prepare.local(
        cfg.model_dump(mode="json"), [], [], {"git_sha": "test"}
    )
    assert calls == ["chat", "math"]
    assert len(result) == 2
    module.artifacts.reload.assert_called_once()
    assert module.artifacts.commit.call_count >= 4
    ledger = json.loads((root / "gpu_runtime.json").read_text())
    assert ledger["seconds"] >= 60
    assert ledger["attempts"] == 2
    assert (root / "provenance/attempt-2/config.json").exists()
    assert (root / "provenance/attempt-2/requirements.txt").exists()
