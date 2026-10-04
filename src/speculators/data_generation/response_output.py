"""Local append-only response staging with bounded, atomic remote snapshots."""

import asyncio
import hashlib
import logging
import os
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

COPY_CHUNK_SIZE = 1024 * 1024


def _matching_prefix(left: Path, right: Path) -> bool:
    """Compare the shared prefix without loading either file into memory."""
    remaining = min(left.stat().st_size, right.stat().st_size)
    with left.open("rb") as a, right.open("rb") as b:
        while remaining:
            size = min(remaining, COPY_CHUNK_SIZE)
            if a.read(size) != b.read(size):
                return False
            remaining -= size
    return True


def _trim_partial_line(path: Path) -> None:
    """Discard only an incomplete trailing write left by a killed process."""
    with path.open("r+b") as stream:
        size = stream.seek(0, os.SEEK_END)
        if not size:
            return
        stream.seek(size - 1)
        if stream.read(1) == b"\n":
            return
        end = size
        while end:
            start = max(0, end - COPY_CHUNK_SIZE)
            stream.seek(start)
            block = stream.read(end - start)
            newline = block.rfind(b"\n")
            if newline >= 0:
                stream.truncate(start + newline + 1)
                return
            end = start
        stream.truncate(0)


def _copy_snapshot(source: Path, destination: Path, size: int) -> None:
    """Copy exactly the flushed prefix, even if workers keep appending locally."""
    pending = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.pending")
    try:
        with source.open("rb") as src, pending.open("wb") as dst:
            remaining = size
            while remaining:
                block = src.read(min(remaining, COPY_CHUNK_SIZE))
                if not block:
                    raise OSError("Local response file shrank during synchronization")
                dst.write(block)
                remaining -= len(block)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)


class ResponseOutput:
    """One writer process; flush locally per example, sync remotely per interval."""

    def __init__(
        self, output: Path, errors: Path, staging_dir: Path | None, interval: int
    ):
        if interval < 1:
            raise ValueError("Response sync interval must be positive")
        self.destinations = (output.resolve(), errors.resolve())
        self.staging_dir = staging_dir
        self.interval = interval
        self.completed = 0
        self._lock = asyncio.Lock()
        self.files = []
        if staging_dir is None:
            self.paths = self.destinations
        else:
            identity = hashlib.sha256(str(self.destinations[0]).encode()).hexdigest()
            root = staging_dir.resolve() / identity
            self.paths = (root / output.name, root / errors.name)
            if any(
                path == dest
                for path, dest in zip(self.paths, self.destinations, strict=True)
            ):
                raise ValueError(
                    "Response staging must be separate from its destination"
                )

    def _prepare(self):
        for local, destination in zip(self.paths, self.destinations, strict=True):
            destination.parent.mkdir(parents=True, exist_ok=True)
            local.parent.mkdir(parents=True, exist_ok=True)
            if self.staging_dir is None:
                continue
            if local.exists():
                _trim_partial_line(local)
            if destination.exists():
                if local.exists() and not _matching_prefix(local, destination):
                    raise ValueError(
                        f"Local staging and destination diverged: {destination}. "
                        "Use a new staging directory or reconcile the files."
                    )
                if (
                    not local.exists()
                    or local.stat().st_size < destination.stat().st_size
                ):
                    _copy_snapshot(destination, local, destination.stat().st_size)
                    _trim_partial_line(local)
            else:
                local.touch(exist_ok=True)

    async def __aenter__(self):
        # Drive reads and initial copies must not run on the event-loop thread.
        await asyncio.to_thread(self._prepare)
        self.files = [path.open("a", encoding="utf-8") for path in self.paths]
        return self

    async def sync(self):
        if self.staging_dir is None:
            return
        async with self._lock:
            sizes = []
            for stream, path in zip(self.files, self.paths, strict=True):
                stream.flush()
                sizes.append(path.stat().st_size)
            copy = asyncio.create_task(asyncio.to_thread(self._sync_files, sizes))
            try:
                await asyncio.shield(copy)
            except asyncio.CancelledError:
                # A thread cannot be cancelled; finish it before another sync or close.
                await copy
                raise

    def _sync_files(self, sizes):
        for local, destination, size in zip(
            self.paths, self.destinations, sizes, strict=True
        ):
            _copy_snapshot(local, destination, size)

    async def example_completed(self):
        self.completed += 1
        if self.completed % self.interval == 0:
            try:
                await self.sync()
            except OSError:
                # Keep generating locally during a transient Drive outage. Final
                # synchronization must succeed, or the command fails visibly.
                logger.exception(
                    "Response sync failed; local files retained. "
                    "Retrying at the next interval or on exit"
                )

    async def __aexit__(self, exc_type, exc, traceback):
        try:
            await self.sync()
        finally:
            for stream in self.files:
                stream.close()
