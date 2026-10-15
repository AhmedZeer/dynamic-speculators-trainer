"""Bounded shared shard cache; returned tensors own their memory."""

import json
import shutil
from collections import OrderedDict
from pathlib import Path

from filelock import FileLock
from safetensors import safe_open

from speculators.bank.hub import fetch
from speculators.bank.progress import report


class HubActivationStore:
    def __init__(self, directory, local_dir=None, memory_cache_gib=0):
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "index.json").read_text())
        self.cache = self.directory / ".shards"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.capacity = int(self.index["cache_gib"] * 2**30)
        self.reserve = int(self.index["reserve_gib"] * 2**30)
        self.local_dir = Path(local_dir).resolve() if local_dir else None
        self.memory_capacity = int(memory_cache_gib * 2**30)
        self.memory = OrderedDict()
        self.memory_bytes = 0

    def __getstate__(self):
        # Spawned workers own separate bounded caches, never serialized payloads.
        return {**self.__dict__, "memory": OrderedDict(), "memory_bytes": 0}

    def read_local(self, target, row, prompt_length, summary):
        """Cached tensors are read-only; return a fresh mapping for transforms."""
        if not self.memory_capacity or prompt_length is not None or summary:
            return self.read_tensors(target, row, prompt_length, summary)
        key = (row["shard"]["sha256"], row["primary_id"])
        if key in self.memory:
            value, size = self.memory.pop(key)
            self.memory[key] = (value, size)
            return dict(value)
        value = self.read_tensors(target, row, prompt_length, summary)
        size = sum(t.numel() * t.element_size() for t in value.values())
        if size <= self.memory_capacity:
            while self.memory_bytes + size > self.memory_capacity:
                _, (_, removed) = self.memory.popitem(last=False)
                self.memory_bytes -= removed
            self.memory[key] = (value, size)
            self.memory_bytes += size
        return dict(value)

    def read(self, index, prompt_length=None, summary=False):  # noqa: C901 -- local/shared/Hub backends
        row = self.index["rows"].get(str(index))
        if row is None:
            return None
        is_raw = row.get("format") == "raw"
        reference = row["summary" if summary and not is_raw else "shard"]
        shared_cache = getattr(self, "shared_cache", None)
        if shared_cache is not None:
            with shared_cache.open(reference) as target:
                return self.read_tensors(target, row, prompt_length, summary)
        if self.local_dir is not None:
            target = (self.local_dir / reference["path"]).resolve()
            if not target.is_relative_to(self.local_dir):
                raise ValueError(
                    "Activation reference escapes the local download folder"
                )
            if not target.is_file():
                raise FileNotFoundError(
                    f"Stage the selected subset before training; missing {target}"
                )
            if target.stat().st_size != reference["bytes"]:
                raise ValueError(f"Incomplete local activation file: {target}")
            # An explicitly staged snapshot is immutable and user-owned. It
            # needs neither eviction nor the private cache's global read lock.
            return self.read_local(target, row, prompt_length, summary)
        target = self.cache / (reference["sha256"] + ".safetensors")
        # Reads and eviction share a lock so no mmap references outlive eviction.
        with FileLock(str(self.cache / ".lock")):
            if not target.exists():
                required = reference["bytes"]
                if required > self.capacity:
                    raise ValueError("A shard exceeds configured Hub cache capacity")
                files = sorted(
                    self.cache.glob("*.safetensors"), key=lambda p: p.stat().st_mtime
                )
                total = sum(p.stat().st_size for p in files)
                while files and (
                    total + required > self.capacity
                    or shutil.disk_usage(self.cache).free < required + self.reserve
                ):
                    oldest = files.pop(0)
                    total -= oldest.stat().st_size
                    oldest.unlink()
                if shutil.disk_usage(self.cache).free < required + self.reserve:
                    raise OSError(
                        "Insufficient local disk for activation shard and reserve"
                    )
                report(f"Downloading activation shard: {reference['path']}")
                fetch(reference, target)
            target.touch()
            return self.read_tensors(target, row, prompt_length, summary)

    @staticmethod
    def read_tensors(target, row, prompt_length, summary):
        is_raw = row.get("format") == "raw"
        with safe_open(target, framework="pt", device="cpu") as tensors:
            primary = row["primary_id"]
            if is_raw:
                hidden = tensors.get_slice("hidden_states")
                if summary:
                    boundary = (
                        prompt_length
                        if prompt_length is not None
                        else row.get("prompt_length")
                    )
                    if boundary is None or not 1 <= boundary <= hidden.get_shape()[0]:
                        raise ValueError(
                            "Raw activation summaries require a valid trajectory "
                            "prompt_length"
                        )
                    return hidden[boundary - 1, :-1].flatten().float().clone()
                if prompt_length is not None:
                    return hidden[:prompt_length].clone()
                return {
                    name: tensors.get_tensor(name).clone()
                    for name in tensors.keys()  # noqa: SIM118
                }
            if summary:
                return tensors.get_tensor(primary).clone()
            if prompt_length is not None:
                return tensors.get_slice(f"{primary}/hidden_states")[
                    :prompt_length
                ].clone()
            return {
                name.split("/", 1)[1]: tensors.get_tensor(name).clone()
                for name in tensors.keys()  # noqa: SIM118 -- safetensors is not iterable
                if name.startswith(primary + "/")
            }
