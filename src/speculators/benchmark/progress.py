"""Benchmark diagnostics, including bounded excerpts from subprocess failures."""

from functools import partial

from speculators.bank.progress import report as shared_report
from speculators.bank.progress import stage_progress as shared_stage

report = partial(shared_report, component="benchmark")
stage_progress = partial(shared_stage, interval=30, component="benchmark")


def report_failure(label, error, log):
    report(f"FAILED {label}: {type(error).__name__}: {error}; log={log}")
    try:
        with log.open("rb") as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 8192))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return
    for line in lines[-12:]:
        report(f"{label} log | {line}")
