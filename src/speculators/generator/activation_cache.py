"""Shared, bounded local-disk cache for immutable prepared bank activations."""

import shutil
import threading
from pathlib import Path

from filelock import FileLock, Timeout
from hs_connectors.transfer import wait_for_lock
from safetensors.torch import load

from speculators.bank.progress import report, stage_progress


class ActivationCache:
    def __init__(self, transfer, root, capacity_gib, reserve_gib):
        self.transfer, self.root = transfer, root
        self.capacity = int(capacity_gib * 2**30)
        self.reserve = int(reserve_gib * 2**30)
        self.hits = self.misses = self.bypasses = 0
        self.stats_lock = threading.Lock()
        self.warned = False
        if self.capacity:
            root.mkdir(parents=True, exist_ok=True)
            # A killed worker can leave an unfinished copy and reservation.
            # Never remove one still owned by another experiment process.
            for reservation in root.glob("*.reservation"):
                try:
                    with (
                        FileLock(
                            str(reservation.with_suffix(".safetensors")) + ".lock",
                            timeout=0,
                        ),
                        FileLock(str(root / ".cache.lock")),
                    ):
                        reservation.with_suffix(".pending").unlink(missing_ok=True)
                        reservation.unlink(missing_ok=True)
                except Timeout:
                    continue

    def count(self, field):
        with self.stats_lock:
            setattr(self, field, getattr(self, field) + 1)

    def get_cached(self, index):
        if not self.capacity:
            return self.transfer.get_cached(index)
        target = self.root / f"hs_{index}.safetensors"
        # Per-row locks prevent multiple experiment processes copying the same
        # file; the short directory lock protects reads, reservations and LRU.
        with (
            stage_progress(f"Activation cache row={index}", quiet=True),
            FileLock(str(target) + ".lock"),
            FileLock(str(self.root / ".cache.lock")),
        ):
            if target.exists():
                payload = load(target.read_bytes())
                target.touch()
                self.count("hits")
                return payload
        with FileLock(str(target) + ".lock"):
            # Another process may have finished while we released the locks.
            with FileLock(str(self.root / ".cache.lock")):
                if target.exists():
                    payload = load(target.read_bytes())
                    target.touch()
                    self.count("hits")
                    return payload
            source = self.transfer.hidden_states_path / target.name
            if Path(str(source) + ".lock").exists():
                wait_for_lock(str(source) + ".lock")
            if not source.exists():
                return None
            size = source.stat().st_size
            reservation = target.with_suffix(".reservation")
            pending = target.with_suffix(".pending")
            with FileLock(str(self.root / ".cache.lock")):
                cached = sorted(
                    self.root.glob("*.safetensors"), key=lambda p: p.stat().st_mtime
                )
                occupied = sum(p.stat().st_size for p in cached)
                reserved = sum(
                    int(p.read_text()) for p in self.root.glob("*.reservation")
                )
                occupied += reserved
                pending_bytes = sum(
                    p.stat().st_size for p in self.root.glob("*.pending")
                )
                free = shutil.disk_usage(self.root).free - max(
                    0, reserved - pending_bytes
                )
                while (
                    size <= self.capacity
                    and cached
                    and (occupied + size > self.capacity or free - size < self.reserve)
                ):
                    victim = cached.pop(0)
                    freed = victim.stat().st_size
                    victim.unlink()
                    occupied -= freed
                    free += freed
                fits = occupied + size <= self.capacity and free - size >= self.reserve
                if fits:
                    reservation.write_text(str(size))
            if not fits:
                self.count("bypasses")
                if not self.warned:
                    report(
                        "Activation cache cannot fit a payload within its "
                        "capacity/free-space reserve; using direct buffered reads"
                    )
                    self.warned = True
                return self.transfer.get_cached(index)
            try:
                with stage_progress(
                    f"Staging activation row={index}, {size / 2**20:.1f} MiB",
                    quiet=True,
                ):
                    with source.open("rb") as reader, pending.open("wb") as writer:
                        shutil.copyfileobj(reader, writer, length=4 * 2**20)
                    with FileLock(str(self.root / ".cache.lock")):
                        pending.replace(target)
                        reservation.unlink(missing_ok=True)
                        payload = load(target.read_bytes())
                        target.touch()
                self.count("misses")
                return payload
            finally:
                with FileLock(str(self.root / ".cache.lock")):
                    pending.unlink(missing_ok=True)
                    reservation.unlink(missing_ok=True)
