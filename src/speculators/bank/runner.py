"""Bounded subprocess scheduling for independent bank subset/seed runs."""

import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from speculators.bank.progress import report


@dataclass
class TrainingRun:
    number: int
    total: int
    subset_id: str
    seed: int
    train_examples: int
    validation_examples: int
    output: Path
    command: list[str]

    @property
    def label(self):
        return f"subset={self.subset_id}, seed={self.seed}"

    def starting(self):
        report(
            f"Starting training run {self.number}/{self.total}: {self.label}, "
            f"train examples={self.train_examples}, "
            f"validation examples={self.validation_examples}, output={self.output}"
        )

    def completed(self, started):
        report(
            f"Training run {self.number}/{self.total} completed in "
            f"{time.monotonic() - started:.1f}s: {self.label}"
        )

    def failed(self, returncode):
        cause = (
            signal.Signals(-returncode).name
            if returncode < 0
            else f"exit code {returncode}"
        )
        report(
            f"Training subprocess terminated by {cause}: {self.label}, "
            f"output={self.output}. Check the child log above for the failing "
            "operation. Rerunning resumes from the last committed recovery checkpoint."
        )


@dataclass
class ActiveRun:
    run: TrainingRun
    process: subprocess.Popen
    reader: threading.Thread
    started: float


def _relay_output(process, label):
    for line in process.stdout:
        sys.stderr.write(f"[{label}] {line.rstrip()}\n")
        sys.stderr.flush()


def _stop_active(active):
    if not active:
        return
    report(f"Stopping {len(active)} active training runs; queued runs will not start")
    for worker in active:
        with suppress(ProcessLookupError):
            os.killpg(worker.process.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    for worker in active:
        with suppress(subprocess.TimeoutExpired):
            worker.process.wait(timeout=max(0, deadline - time.monotonic()))
    # Also stop descendants (e.g. torchrun ranks) if their leader exited first.
    for worker in active:
        with suppress(ProcessLookupError):
            os.killpg(worker.process.pid, signal.SIGKILL)
        worker.process.wait()
        worker.reader.join()
        worker.process.stdout.close()


def run_parallel(runs: list[TrainingRun], n_workers: int):  # noqa: C901
    """Refill free slots and clean up whole subprocess groups on failure or signal."""
    if n_workers < 1:
        raise ValueError("n_workers must be at least one")
    pending = iter(runs)
    active = []
    exhausted = False
    env = {**os.environ, "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"}
    previous_sigterm = None

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        previous_sigterm = signal.signal(signal.SIGTERM, interrupted)
    try:
        while active or not exhausted:
            # Check every active run before launching any replacement.
            for worker in list(active):
                returncode = worker.process.poll()
                if returncode is None:
                    continue
                if returncode:
                    worker.run.failed(returncode)
                    raise subprocess.CalledProcessError(returncode, worker.run.command)
                worker.reader.join()
                worker.process.stdout.close()
                active.remove(worker)
                worker.run.completed(worker.started)
            while not exhausted and len(active) < n_workers:
                run = next(pending, None)
                if run is None:
                    exhausted = True
                    break
                run.starting()
                started = time.monotonic()
                process = subprocess.Popen(  # noqa: S603 -- argv, never a shell
                    run.command,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    bufsize=1,
                    start_new_session=True,
                )
                reader = threading.Thread(
                    target=_relay_output, args=(process, run.label), daemon=True
                )
                active.append(ActiveRun(run, process, reader, started))
                reader.start()
            if active:
                time.sleep(0.1)
    finally:
        try:
            _stop_active(active)
        finally:
            if previous_sigterm is not None:
                signal.signal(signal.SIGTERM, previous_sigterm)
