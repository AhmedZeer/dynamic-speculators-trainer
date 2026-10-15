"""Concurrent immutable Hub downloads, short quota locks, and pinned LRU reads."""

import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

from filelock import FileLock, Timeout

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.staging import _download_with_retries


class CampaignCache:
    """One cache across categories and jobs; returned tensors must own their data.

    File locks cover downloads, checksum verification, and active reads. The
    accounting lock covers reservations and eviction only, never network IO.
    Prefetch futures retain paths rather than decoded tensors.
    """

    def __init__(self, root, settings):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = int(settings.capacity_gib * 2**30)
        self.reserve = int(settings.reserve_gib * 2**30)
        self.pool = ThreadPoolExecutor(max_workers=settings.workers)
        self.verifiers = ThreadPoolExecutor(max_workers=settings.verification_workers)
        self.mutex = Lock()
        self.futures = {}
        self.references = {}
        self.protected = set()
        self.stats = dict.fromkeys(
            (
                "download_bytes",
                "download_seconds",
                "download_retries",
                "verification_seconds",
                "hits",
                "misses",
                "evictions",
                "evicted_bytes",
                "peak_cache_bytes",
            ),
            0,
        )

    def key(self, ref):
        return digest({k: ref[k] for k in ("repo", "revision", "path", "sha256")})

    def path(self, ref):
        return self.root / (self.key(ref) + ".safetensors")

    def add(self, **values):
        with self.mutex:
            for key, value in values.items():
                if key == "peak_cache_bytes":
                    self.stats[key] = max(self.stats[key], value)
                else:
                    self.stats[key] += value

    @contextmanager
    def accounting(self):
        with FileLock(str(self.root / ".accounting.lock")):
            yield

    def _reserve(self, ref):
        """Called under the file lock. Remove stale reservations without races."""
        key, required = self.key(ref), ref["bytes"]
        if required > self.capacity:
            raise OSError("Activation file exceeds campaign cache capacity")
        with self.accounting():
            for path in self.root.glob("*.reservation"):
                if path.stem == key:
                    path.unlink()
                    continue
                try:
                    with FileLock(str(self.root / (path.stem + ".lock")), timeout=0):
                        path.unlink(missing_ok=True)
                        shutil.rmtree(
                            self.root / (path.stem + ".download"), ignore_errors=True
                        )
                except Timeout:
                    continue
            files = sorted(
                self.root.glob("*.safetensors"),
                key=lambda p: (
                    p.with_suffix(".access").stat().st_mtime_ns
                    if p.with_suffix(".access").exists()
                    else p.stat().st_mtime_ns
                ),
            )
            occupied = sum(p.stat().st_size for p in files)
            reserved = sum(int(p.read_text()) for p in self.root.glob("*.reservation"))
            # Partial downloads already consume free disk. Subtracting their
            # reservations again is conservative and keeps the free-space floor.
            for victim in files:
                if occupied + reserved + required <= self.capacity and (
                    shutil.disk_usage(self.root).free
                    >= reserved + required + self.reserve
                ):
                    break
                with self.mutex:
                    protected = victim.stem in self.protected
                if protected:
                    continue
                try:
                    with FileLock(str(victim.with_suffix(".lock")), timeout=0):
                        size = victim.stat().st_size
                        victim.unlink()
                        victim.with_suffix(".verified.json").unlink(missing_ok=True)
                        victim.with_suffix(".access").unlink(missing_ok=True)
                        occupied -= size
                        self.add(evictions=1, evicted_bytes=size)
                except Timeout:
                    continue
            if occupied + reserved + required > self.capacity or (
                shutil.disk_usage(self.root).free < reserved + required + self.reserve
            ):
                raise OSError(
                    "Campaign cache cannot fit the protected prefetch window; "
                    "reduce downloads.lookahead_updates or increase disk capacity"
                )
            (self.root / (key + ".reservation")).write_text(str(required))
            self.add(peak_cache_bytes=occupied + reserved + required)

    def _identity(self, path, ref):
        stat = path.stat()
        return {
            "reference": ref,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
        }

    def _verified(self, path, ref):
        if not path.exists():
            return False
        before = self._identity(path, ref)
        if before["size"] != ref["bytes"]:
            raise ValueError(f"Activation size mismatch: {ref['path']}")
        receipt = path.with_suffix(".verified.json")
        try:
            if json.loads(receipt.read_text()) == before:
                return True
        except (OSError, ValueError):
            pass
        started = time.perf_counter()
        if file_digest(path) != ref["sha256"]:
            raise ValueError(f"Activation checksum mismatch: {ref['path']}")
        if self._identity(path, ref) != before:
            raise ValueError(f"Activation changed during verification: {path}")
        write_json(receipt, before)
        self.add(verification_seconds=time.perf_counter() - started)
        return True

    def _ensure(self, ref):
        path = self.path(ref)
        with FileLock(str(path.with_suffix(".lock"))):
            if self.verifiers.submit(self._verified, path, ref).result():
                self.add(hits=1)
                return path
            self._reserve(ref)
            scratch = self.root / (self.key(ref) + ".download")
            try:
                started = time.perf_counter()
                source = _download_with_retries(
                    ref, scratch, on_retry=lambda: self.add(download_retries=1)
                )
                self.add(
                    download_bytes=ref["bytes"],
                    misses=1,
                    download_seconds=time.perf_counter() - started,
                )
                self.verifiers.submit(self._verified, source, ref).result()
                with self.accounting():
                    source.replace(path)
                    # Publish and release the reservation atomically, so other
                    # workers never count both the file and its reservation.
                    write_json(
                        path.with_suffix(".verified.json"), self._identity(path, ref)
                    )
                    (self.root / (self.key(ref) + ".reservation")).unlink(
                        missing_ok=True
                    )
                return path
            finally:
                # Keep resumable partial downloads after a failed transfer.
                if path.exists():
                    with self.accounting():
                        (self.root / (self.key(ref) + ".reservation")).unlink(
                            missing_ok=True
                        )
                    shutil.rmtree(scratch, ignore_errors=True)

    def prefetch(self, refs):
        for ref in refs:
            key = self.key(ref)
            with self.mutex:
                self.references[key] = ref
                future = self.futures.get(key)
                if future is None:
                    self.futures[key] = self.pool.submit(self._ensure, ref)

    def protect(self, refs):
        refs = list(refs)
        with self.mutex:
            self.protected = {self.key(ref) for ref in refs}
            for key in list(self.futures):
                if key not in self.protected and self.futures[key].done():
                    del self.futures[key]
                    self.references.pop(key, None)

    @contextmanager
    def open(self, ref):
        self.prefetch([ref])
        with self.mutex:
            future = self.futures[self.key(ref)]
        path = future.result()
        # Eviction can race between a finished prefetch and this read. Recheck
        # under the same per-file lock used by the eviction process.
        while True:
            with FileLock(str(path.with_suffix(".lock"))):
                if path.exists():
                    self.verifiers.submit(self._verified, path, ref).result()
                    path.with_suffix(".access").touch()
                    yield path
                    return
            self._ensure(ref)

    def snapshot(self):
        with self.mutex:
            values = dict(self.stats)
        with self.accounting():
            values["cache_bytes"] = sum(
                p.stat().st_size for p in self.root.glob("*.safetensors")
            )
        values["disk_free_bytes"] = shutil.disk_usage(self.root).free
        return values

    @contextmanager
    def reserve_output(self, bytes_needed):
        """Keep temporary checkpoint writes within the same free-space floor."""
        ref = {
            "repo": "local",
            "revision": "checkpoint",
            "path": "checkpoint",
            "sha256": "output-reservation",
            "bytes": bytes_needed,
        }
        key = self.key(ref)
        with FileLock(str(self.root / (key + ".lock"))):
            self._reserve(ref)
            try:
                yield
            finally:
                with self.accounting():
                    (self.root / (key + ".reservation")).unlink(missing_ok=True)

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.verifiers.shutdown(wait=True)
