"""Console stage diagnostics independent of training logger configuration."""

import os
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

import typer


def process_memory() -> str:
    """Read this process's resident memory, rather than total machine RAM."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        gib = pages * os.sysconf("SC_PAGE_SIZE") / (1024**3)
        return f"process RSS={gib:.2f} GiB"
    except (OSError, ValueError, IndexError):
        return "process RSS=unavailable"


def report(message: str):
    typer.echo(
        f"[lora-bank pid={os.getpid()}] {message} | {process_memory()}", err=True
    )


@contextmanager
def stage_progress(
    label: str, interval: float = 10.0, details: Callable[[], str] | None = None
):
    """Report stages immediately and during blocking I/O, without root filters."""
    started = time.monotonic()
    stopped = threading.Event()

    def describe():
        return f"{label}; {details()}" if details else label

    def heartbeat():
        while not stopped.wait(interval):
            report(f"{describe()}: still running ({time.monotonic() - started:.1f}s)")

    report(f"{describe()}: starting")
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    outcome = "completed"
    try:
        yield
    except BaseException:
        outcome = "stopped"
        raise
    finally:
        stopped.set()
        thread.join()
        report(f"{describe()}: {outcome} ({time.monotonic() - started:.1f}s)")
