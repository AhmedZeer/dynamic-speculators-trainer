"""Console stage diagnostics independent of training logger configuration."""

import faulthandler
import os
import sys
import threading
import time
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
def stage_progress(label: str, interval: float = 10.0):
    """Report stages immediately and during blocking I/O, without root filters."""
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(interval):
            report(f"{label}: still running ({time.monotonic() - started:.1f}s)")

    report(f"{label}: starting")
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
        report(f"{label}: {outcome} ({time.monotonic() - started:.1f}s)")


class StartupWatchdog:
    """Obtain thread stacks even when Python logging or its event loop is blocked."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.armed = False

    def __enter__(self):
        try:
            faulthandler.dump_traceback_later(
                self.timeout, file=sys.stderr, repeat=False
            )
            self.armed = True
        except (OSError, RuntimeError, ValueError):
            report("Startup stack diagnostics unavailable on this stderr stream")
        return self

    def finish(self):
        if self.armed:
            faulthandler.cancel_dump_traceback_later()
            self.armed = False

    def __exit__(self, *args):
        self.finish()
