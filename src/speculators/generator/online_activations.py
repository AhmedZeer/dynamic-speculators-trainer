"""Bounded vLLM extraction with transient shared files, never Hub activations."""

import shutil
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import RLock

import openai
import torch
from safetensors.torch import load

from speculators.bank.artifacts import digest
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    generate_hidden_states,
    wait_for_lock,
)
from speculators.generator.campaign_data import CampaignCorpus
from speculators.generator.persistent_summaries import PersistentSummaries

HIDDEN_STATE_NDIM = 3


class ActivationRAM:
    """Campaign-owned CPU cache, shared by sequential jobs on the same rows."""

    def __init__(self, capacity_gib, validation_indices=()):
        self.limit = int(capacity_gib * 2**30)
        self.validation_indices = set(validation_indices)
        self.lock = RLock()
        self.full = OrderedDict()
        self.summaries = OrderedDict()
        self.conditioning = OrderedDict()
        self.summary_store = None
        self.bytes = self.peak_bytes = self.evictions = 0

    def put(self, index, payload):
        size = sum(t.numel() * t.element_size() for t in payload.values())
        with self.lock:
            if size > self.limit or index in self.full:
                return
            while self.full and self.bytes + size > self.limit:
                victim = next(
                    (i for i in self.full if i not in self.validation_indices), None
                )
                if victim is None:
                    if index not in self.validation_indices:
                        return
                    victim = next(iter(self.full))
                old = self.full.pop(victim)
                self.bytes -= sum(t.numel() * t.element_size() for t in old.values())
                self.evictions += 1
            self.full[index] = payload
            self.bytes += size
            self.peak_bytes = max(self.peak_bytes, self.bytes)


class OnlineActivations:
    def __init__(self, cfg, rows, bank, *, client=None, ram=None):
        self.cfg, self.rows, self.bank = cfg, rows, bank
        self.settings = cfg.online
        self.root = self.settings.shared_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root = cfg.output_root
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=self.settings.concurrency)
        self.ram = ram if ram is not None else ActivationRAM(cfg.online.ram_cache_gib)
        self.lock = self.ram.lock
        self.futures = OrderedDict()
        self.window = []
        self.consumed = set()
        self.summaries = self.ram.summaries
        self.stats = dict.fromkeys(
            (
                "requests",
                "prompt_requests",
                "full_requests",
                "extraction_seconds",
                "extracted_tokens",
                "extracted_bytes",
                "peak_temporary_bytes",
                "temporary_bytes",
                "buffer_hits",
                "hits",
                "misses",
            ),
            0,
        )
        self.client = client or openai.OpenAI(
            base_url=self.settings.endpoint, api_key="EMPTY", max_retries=0
        )

    def check_server(self):
        models = {m.id for m in self.client.models.list().data}
        if self.bank.models.target not in models:
            raise ValueError(
                f"vLLM must serve {self.bank.models.target}; got {sorted(models)}"
            )

    def _add(self, **values):
        with self.lock:
            for key, value in values.items():
                self.stats[key] += value
            self.stats["peak_temporary_bytes"] = max(
                self.stats["peak_temporary_bytes"], self.stats["temporary_bytes"]
            )

    def _generate(self, index, summary):
        row = self.rows[index]
        tokens = (
            row["input_ids"][: row["prompt_length"]] if summary else row["input_ids"]
        )
        path = None
        size = 0
        accounted = False
        started = time.perf_counter()
        try:
            handle = generate_hidden_states(
                self.client,
                self.bank.models.target,
                {"input_ids": list(tokens)},
                timeout=self.settings.request_timeout,
                max_retries=self.settings.max_retries,
            )
            path = Path(handle).resolve()
            if not path.is_relative_to(self.root):
                path = None
                raise ValueError("vLLM returned a file outside online.shared_dir")
            lock = Path(str(path) + ".lock")
            if lock.exists():
                wait_for_lock(str(lock), timeout=self.settings.request_timeout)
            size = path.stat().st_size
            self._add(temporary_bytes=size)
            accounted = True
            # Buffered deserialization owns tensor storage before file cleanup.
            payload = load(path.read_bytes())
            check_hidden_states(payload, list(tokens))
            states = payload["hidden_states"]
            if states.ndim != HIDDEN_STATE_NDIM or states.shape[1] != len(
                self.bank.hidden_states.layer_ids
            ):
                raise ValueError("vLLM hidden-state layer shape differs from the bank")
            if hasattr(self, "hidden_width") and states.shape[2] != self.hidden_width:
                raise ValueError("vLLM hidden-state width differs from the drafter")
            if states.dtype != getattr(torch, self.bank.hidden_states.dtype):
                raise ValueError("vLLM hidden-state dtype differs from the bank")
            boundary = row["prompt_length"]
            vector = states[boundary - 1, :-1].flatten().float().clone()
            with self.lock:
                self.summaries.setdefault(index, vector)
                self.summaries.move_to_end(index)
            if not summary:
                self.ram.put(index, payload)
            self._add(
                requests=1,
                prompt_requests=int(summary),
                full_requests=int(not summary),
                extraction_seconds=time.perf_counter() - started,
                extracted_tokens=len(tokens),
                extracted_bytes=size,
            )
            return vector if summary else payload
        finally:
            if path is not None:
                path.unlink(missing_ok=True)
            if accounted:
                self._add(temporary_bytes=-size)

    def prefetch(self, refs):
        # Never enqueue the entire lookahead: a bounded number of futures also
        # bounds already-decoded CPU tensors waiting for training to consume them.
        with self.lock:
            self.window = list(
                dict.fromkeys(
                    self.window + [(ref["index"], ref["summary"]) for ref in refs]
                )
            )
            self._refill()

    def _refill(self):
        """Caller holds the lock. Queue only the bounded front of the lookahead."""
        for key in self.window:
            if key in self.consumed or key in self.futures:
                continue
            if not key[1] and key[0] in self.ram.full:
                continue
            if key[1] and (key[0] in self.summaries or (key[0], False) in self.futures):
                continue
            if len(self.futures) >= self.settings.buffered_examples:
                break
            self.futures[key] = self.pool.submit(self._generate, *key)

    def protect(self, refs):
        # Consumed results are removed below. Retain only the requested window;
        # uncancellable in-flight jobs are drained before removing their results.
        wanted = {(r["index"], r["summary"]) for r in refs}
        discarded = []
        with self.lock:
            self.window = list(dict.fromkeys((r["index"], r["summary"]) for r in refs))
            self.consumed.clear()
            for key in list(self.futures):
                future = self.futures[key]
                if key not in wanted:
                    discarded.append(future)
                    del self.futures[key]
        # Cancel every queued request before waiting on active workers, so those
        # workers cannot start obsolete queued work while we drain them.
        running = [future for future in discarded if not future.cancel()]
        for future in running:
            future.result()

    def _future(self, index, summary):
        key = (index, summary)
        while True:
            with self.lock:
                if key in self.futures:
                    self.stats["buffer_hits"] += 1
                    return self.futures[key]
                if len(self.futures) < self.settings.buffered_examples:
                    future = self.pool.submit(self._generate, *key)
                    self.futures[key] = future
                    return future
                # Demand takes precedence over the farthest speculative request.
                _, discarded = self.futures.popitem(last=True)
            if not discarded.cancel():
                discarded.result()

    def get_full(self, index):
        with self.lock:
            if index in self.ram.full and (index, False) not in self.futures:
                self.stats["hits"] += 1
                self.ram.full.move_to_end(index)
                self.consumed.add((index, False))
                self._refill()
                return self.ram.full[index]
            self.stats["misses"] += 1
        key = (index, False)
        future = self._future(index, False)
        result = future.result()
        with self.lock:
            if self.futures.get(key) is future:
                del self.futures[key]
            self.consumed.add(key)
            self._refill()
        self.ram.put(index, result)
        return result

    def get_summary(self, index):
        with self.lock:
            pending = self.futures.get((index, True))
            if index in self.summaries and pending is None:
                self.summaries.move_to_end(index)
                return self.summaries[index]
            full = self.futures.get((index, False))
        if full is not None:
            payload = full.result()
            return (
                payload["hidden_states"][self.rows[index]["prompt_length"] - 1, :-1]
                .flatten()
                .float()
                .clone()
            )
        key = (index, True)
        future = self._future(index, True)
        result = future.result()
        with self.lock:
            if self.futures.get(key) is future:
                del self.futures[key]
            self.consumed.add(key)
            self._refill()
        return result

    def snapshot(self):
        with self.lock:
            stats = dict(self.stats)
            stats["buffered_examples"] = len(self.futures)
        return {
            **stats,
            "cache_bytes": self.ram.bytes,
            "peak_cache_bytes": self.ram.peak_bytes,
            "evictions": self.ram.evictions,
            "hit_rate": stats["hits"] / max(1, stats["hits"] + stats["misses"]),
            "summary_examples": len(self.summaries),
            "disk_free_bytes": shutil.disk_usage(self.output_root).free,
            "temporary_disk_free_bytes": shutil.disk_usage(self.root).free,
        }

    @contextmanager
    def reserve_output(self, bytes_needed):
        if (
            shutil.disk_usage(self.output_root).free
            < bytes_needed + self.settings.reserve_gib * 2**30
        ):
            raise OSError("Insufficient free disk for an online training checkpoint")
        yield

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.futures.clear()
        self.window.clear()
        self.client.close()


class OnlineTransfer:
    def __init__(self, cache):
        self.cache = cache

    def get_cached(self, index):
        return self.cache.get_full(index)

    def read(self, index, prompt_length=None, summary=False):  # noqa: ARG002
        return self.cache.get_summary(index) if summary else self.cache.get_full(index)


class OnlineCorpus(CampaignCorpus):
    """Reuse packing/collation and lazy summaries without constructing a Hub store."""

    def __init__(self, cfg, bank, cache, inputs):
        self.cfg, self.bank, self.cache = cfg, bank, cache
        self.manifest = inputs.manifests["math"]
        self.rows = inputs.rows
        self.subsets = {s: inputs.subsets[s] for s in cfg.experiment_subsets}
        self.identity = {
            **inputs.identity,
            "subsets": self.subsets,
            "activation_source": "vllm",
        }
        self.cache_key = digest(self.identity)
        self.local = cfg.staging_dir / self.cache_key
        self.summary_root = cfg.staging_dir / "online-summaries"
        self.summary_root.mkdir(parents=True, exist_ok=True)
        self.summary_memory = cache.ram.conditioning
        self.summary_memory_limit = len(self.rows)
        self.summary_hits = self.summary_misses = 0
        self.summary_seconds = self.last_batch_data_seconds = 0.0
        self.summaries = None
        self.transfer = OnlineTransfer(cache)
        self.activations = self.transfer
        if cache.ram.summary_store is None:
            cache.ram.summary_store = PersistentSummaries(cache.cfg, bank, inputs)
        self.summary_store = cache.ram.summary_store
        self.training_window = []

    def prepare(self, model):
        super().prepare(model)
        self.cache.hidden_width = model.fc.weight.shape[1] // (
            len(self.bank.hidden_states.layer_ids) - 1
        )
        self.summary_width = model.fc.weight.shape[1]

    def prepare_summaries(self, names):
        return self.summary_store.prepare(names, self.cache, self.summary_width)

    def context(self, indices, condition, device):
        if condition != "last":
            raise ValueError("Online campaign requires last conditioning")
        started = time.perf_counter()
        with self.cache.lock:
            values = [self.cache.summaries.get(index) for index in indices]
        if any(value is None for value in values):
            return super().context(indices, condition, device)
        result = (
            torch.stack(values).to(device),
            self.summary_store.lengths[indices].to(device),
        )
        self.summary_hits += len(indices)
        self.summary_seconds += time.perf_counter() - started
        return result

    def set_training_window(self, episodes, *, full):
        self.training_window = list(episodes)
        self._window_refs(full)

    def _window_refs(self, full):
        indices = [i for episode in self.training_window for i in episode]
        refs = self.references(indices, full=full)
        self.cache.protect(refs)
        self.cache.prefetch(refs)

    def prepare_episode(self, indices, *, full):
        if self.training_window and self.training_window[0] == indices:
            self._window_refs(full)
            self.training_window.pop(0)
            return
        refs = self.references(indices, full=full)
        self.cache.protect(refs)
        self.cache.prefetch(refs)

    def summary_path(self, index):
        row = self.rows[index]
        identity = {
            "version": 1,
            "models": self.manifest["revisions"],
            "layers": self.bank.hidden_states.layer_ids,
            "tokens": row["input_ids"][: row["prompt_length"]],
            "prompt_length": row["prompt_length"],
            "condition": "last",
        }
        return self.summary_root / (digest(identity) + ".pt")

    def references(self, indices, *, full):
        refs = []
        for index in dict.fromkeys(indices):
            if full:
                refs.append({"index": index, "summary": False})
            elif (
                index not in self.cache.summaries
                and not self.summary_path(index).is_file()
            ):
                refs.append({"index": index, "summary": True})
        return refs

    def activation_chunks(self, chunks):
        # Each chunk is within token_budget. Results are released as each batch
        # leaves the generator; no persistent full-activation cache is retained.
        for position, chunk in enumerate(chunks):
            upcoming = [i for group in chunks[position : position + 2] for i in group]
            self.cache.prefetch(self.references(upcoming, full=True))
            started = time.perf_counter()
            payloads = [self.activations.get_cached(index) for index in chunk]
            self.last_batch_data_seconds = time.perf_counter() - started
            yield chunk, payloads
            del payloads
