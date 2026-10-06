"""Exercise worker limits, labelled logs, and cancellation with real CPU processes."""

import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from speculators.bank import workflow
from speculators.bank.config import BankConfig
from speculators.bank.runner import TrainingRun, run_parallel


def job(tmp_path, number, code):
    return TrainingRun(
        number,
        5,
        "math-00000",
        40 + number,
        1000,
        100,
        tmp_path / f"seed-{40 + number}",
        [sys.executable, "-c", code, str(tmp_path), str(number)],
    )


def test_worker_limit_and_all_runs_complete(tmp_path, capsys):
    code = """
import json, sys, time
from pathlib import Path
root, number = Path(sys.argv[1]), int(sys.argv[2])
started = time.monotonic()
print("training output", flush=True)
time.sleep(0.2)
(root / f"{number}.json").write_text(json.dumps([started, time.monotonic()]))
"""
    previous_handler = signal.getsignal(signal.SIGTERM)
    run_parallel([job(tmp_path, i, code) for i in range(5)], 2)
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    events = []
    for i in range(5):
        start, end = json.loads((tmp_path / f"{i}.json").read_text())
        events.extend([(start, 1), (end, -1)])
    active = maximum = 0
    for _, delta in sorted(events):
        active += delta
        maximum = max(maximum, active)
    assert maximum == 2
    assert active == 0
    output = capsys.readouterr().err
    assert "[subset=math-00000, seed=40] training output" in output
    assert "[subset=math-00000, seed=44] training output" in output


def test_failure_stops_active_run_and_does_not_start_queue(tmp_path, capsys):
    waiting = """
import signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
def stop(signum, frame):
    (root / "terminated").touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
child_code = (
    "import signal, sys, time; from pathlib import Path; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "Path(sys.argv[1]).touch(); time.sleep(60)"
)
child = subprocess.Popen([sys.executable, "-c", child_code, str(root / "child-ready")])
(root / "child-pid").write_text(str(child.pid))
deadline = time.monotonic() + 5
while not (root / "child-ready").exists() and time.monotonic() < deadline:
    time.sleep(0.01)
(root / "active").touch()
while True: time.sleep(0.1)
"""
    failing = """
import sys, time
from pathlib import Path
root = Path(sys.argv[1])
deadline = time.monotonic() + 5
while not (root / "active").exists() and time.monotonic() < deadline: time.sleep(0.01)
sys.exit(7)
"""
    queued = """
import sys
from pathlib import Path
(Path(sys.argv[1]) / "queued").touch()
"""
    runs = [
        job(tmp_path, 0, waiting),
        job(tmp_path, 1, failing),
        job(tmp_path, 2, queued),
    ]
    previous_handler = signal.getsignal(signal.SIGTERM)
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_parallel(runs, 2)
    assert error.value.returncode == 7
    assert (tmp_path / "terminated").exists()
    assert not (tmp_path / "queued").exists()
    # A torchrun-like descendant that ignores SIGTERM must also be stopped.
    pid = int((tmp_path / "child-pid").read_text())
    state = Path(f"/proc/{pid}/stat")
    assert not state.exists() or state.read_text().split()[2] == "Z"
    assert signal.getsignal(signal.SIGTERM) == previous_handler
    assert "seed=41" in capsys.readouterr().err


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_interruption_stops_workers(tmp_path, signum):
    code = """
import os, signal, sys, time
from pathlib import Path
root = Path(sys.argv[1])
def stop(signum, frame):
    (root / "terminated").touch()
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
(root / "active").touch()
os.kill(os.getppid(), int(sys.argv[2]))
while True: time.sleep(0.1)
"""
    previous_handler = signal.getsignal(signal.SIGTERM)
    run = job(tmp_path, 0, code)
    run.command[-1] = str(int(signum))
    with pytest.raises(KeyboardInterrupt):
        run_parallel([run], 2)
    assert (tmp_path / "terminated").exists()
    assert signal.getsignal(signal.SIGTERM) == previous_handler


def test_parallel_setting_is_operational_and_index_waits_for_completion(
    tmp_path, monkeypatch
):
    cfg = BankConfig.model_validate({"execution": {"n_workers": 2}})
    manifest = {
        "root": str(tmp_path),
        "data_path": str(tmp_path / "data"),
        "hidden_states_path": str(tmp_path / "hidden_states"),
        "revisions": {"target_sha": "target", "drafter_sha": "draft"},
        "subsets": [
            {"id": "math-00000", "train_indices": [0], "validation_indices": [1]}
        ],
    }
    before = workflow.train_config(cfg, manifest, "math-00000", 42, "target", "draft")
    cfg.execution.n_workers = 3
    assert (
        workflow.train_config(cfg, manifest, "math-00000", 42, "target", "draft")
        == before
    )
    events = []
    monkeypatch.setattr(workflow, "select_data", lambda cfg: manifest)
    monkeypatch.setattr(workflow, "validate_cache", lambda *args: None)
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download", lambda model, revision: revision
    )

    def launch(runs, workers):
        events.append("train")
        assert workers == 3
        assert [run.seed for run in runs] == [42, 43, 44]
        assert len({run.output for run in runs}) == 3
        assert all((run.output / "bank_train.yaml").is_file() for run in runs)

    def inspect(cfg):
        assert events == ["train"]
        events.append("inspect")
        return {"candidate_count": 1}

    monkeypatch.setattr(workflow, "run_parallel", launch)
    monkeypatch.setattr(workflow, "inspect_bank", inspect)
    assert workflow.train(cfg) == {"candidate_count": 1}
    assert events == ["train", "inspect"]
    with pytest.raises(ValidationError):
        BankConfig.model_validate({"execution": {"n_workers": 0}})
