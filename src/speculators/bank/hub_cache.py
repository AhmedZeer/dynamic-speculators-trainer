"""Bounded shared shard cache; returned tensors own their memory."""

import json
import shutil
from pathlib import Path

from filelock import FileLock
from safetensors import safe_open

from speculators.bank.hub import fetch
from speculators.bank.progress import report


class HubActivationStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.index = json.loads((self.directory / "index.json").read_text())
        self.cache = self.directory / ".shards"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.capacity = int(self.index["cache_gib"] * 2**30)
        self.reserve = int(self.index["reserve_gib"] * 2**30)

    def read(self, index, prompt_length=None, summary=False):  # noqa: C901, PLR0911 -- two supported storage layouts
        row = self.index["rows"].get(str(index))
        if row is None:
            return None
        is_raw = row.get("format") == "raw"
        reference = row["summary" if summary and not is_raw else "shard"]
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
            with safe_open(target, framework="pt", device="cpu") as tensors:
                primary = row["primary_id"]
                if is_raw:
                    hidden = tensors.get_slice("hidden_states")
                    if summary:
                        return (
                            hidden[row["prompt_length"] - 1, :-1]
                            .flatten()
                            .float()
                            .clone()
                        )
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
