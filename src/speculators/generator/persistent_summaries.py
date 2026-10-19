"""Restartable per-subset prompt summary matrices, independent of generator weights."""

import time

import torch
from filelock import FileLock
from safetensors.torch import load, save_file

from speculators.bank.artifacts import digest, write_json
from speculators.bank.progress import report


class PersistentSummaries:
    def __init__(self, cfg, bank, inputs):
        self.rows, self.subsets = inputs.rows, inputs.subsets
        self.identity = {
            "version": 1,
            "sources": inputs.identity["sources"],
            "revisions": inputs.identity["revisions"],
            "models": {
                role: getattr(bank.models, role) for role in ("target", "drafter")
            },
            "layers": bank.hidden_states.layer_ids,
            "dtype": bank.hidden_states.dtype,
            "condition": "last",
        }
        self.root = cfg.online.summary_cache_dir / digest(self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "identity.json", self.identity)
        self.legacy_root = cfg.staging_dir / "online-summaries"
        self.lengths = torch.tensor(list(self.rows["prompt_length"]), dtype=torch.int64)
        self.loaded = set()

    def legacy_path(self, index):
        row = self.rows[index]
        identity = {
            "version": 1,
            "models": self.identity["revisions"],
            "layers": self.identity["layers"],
            "tokens": row["input_ids"][: row["prompt_length"]],
            "prompt_length": row["prompt_length"],
            "condition": "last",
        }
        return self.legacy_root / (digest(identity) + ".pt")

    def prepare(self, names, cache, width):
        started = time.perf_counter()
        stats = dict.fromkeys(
            (
                "loaded_examples",
                "generated_examples",
                "legacy_examples",
                "memory_examples",
                "shards_written",
            ),
            0,
        )
        for name in sorted(set(names)):
            subset = self.subsets[name]
            indices = subset["train_indices"] + subset["validation_indices"]
            if name in self.loaded:
                stats["memory_examples"] += len(indices)
                continue
            path = self.root / f"{name}.safetensors"
            with FileLock(str(path) + ".lock"):
                if path.is_file():
                    payload = load(path.read_bytes())
                    stats["loaded_examples"] += len(indices)
                else:
                    payload = self.build(name, indices, cache, width, stats)
                    pending = path.with_suffix(".pending")
                    try:
                        save_file(payload, pending)
                        pending.replace(path)
                    finally:
                        pending.unlink(missing_ok=True)
                    stats["shards_written"] += 1
                self.validate(payload, indices, width)
                with cache.lock:
                    cache.summaries.update(zip(indices, payload["values"], strict=True))
                self.loaded.add(name)
            report(
                f"Prompt summaries {name}: ready; examples={len(indices)}; "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )
        return {**stats, "seconds": time.perf_counter() - started}

    def build(self, name, indices, cache, width, stats):
        started = time.perf_counter()
        # Import old small files once, then retain contiguous subset matrices.
        missing = []
        for index in indices:
            if index in cache.summaries:
                stats["memory_examples"] += 1
                continue
            legacy = self.legacy_path(index)
            if legacy.is_file():
                value = torch.load(legacy, map_location="cpu", weights_only=True)
                if value.shape != (width,) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid legacy prompt summary: {legacy}")
                cache.summaries[index] = value.float()
                stats["legacy_examples"] += 1
            else:
                missing.append(index)
        cache.protect([])
        limit = cache.settings.buffered_examples
        for start in range(0, len(missing), limit):
            chunk = missing[start : start + limit]
            refs = [{"index": i, "summary": True} for i in chunk]
            cache.protect(refs)
            cache.prefetch(refs)
            for index in chunk:
                cache.get_summary(index)
                stats["generated_examples"] += 1
            completed = start + len(chunk)
            if (
                start == 0
                or completed // 128 > start // 128
                or completed == len(missing)
            ):
                report(
                    f"Prompt summaries {name}: generated={completed}/{len(missing)}; "
                    f"elapsed={time.perf_counter() - started:.2f}s"
                )
        cache.protect([])
        payload = {
            "indices": torch.tensor(indices, dtype=torch.int64),
            "lengths": self.lengths[indices],
            "values": torch.stack([cache.summaries[i] for i in indices]).float(),
        }
        self.validate(payload, indices, width)
        return payload

    def validate(self, payload, indices, width):
        values = payload["values"]
        if (
            not torch.equal(payload["indices"], torch.tensor(indices))
            or not torch.equal(payload["lengths"], self.lengths[indices])
            or values.shape != (len(indices), width)
            or values.dtype != torch.float32
        ):
            raise ValueError("Persistent prompt summary shape or membership differs")
        if any(not torch.isfinite(chunk).all() for chunk in values.split(256)):
            raise ValueError("Non-finite persistent prompt summaries")
