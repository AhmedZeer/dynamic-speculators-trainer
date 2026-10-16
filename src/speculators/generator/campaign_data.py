"""Lazy conditioning summaries and deterministic campaign episode schedules."""

import random
import time
from collections import OrderedDict

import torch
from filelock import FileLock

from speculators.bank.artifacts import digest
from speculators.generator.data import Corpus, atomic_torch_save

SUMMARY_MEMORY_ENTRIES = 1024


class EpisodeSchedule:
    """Index-addressable randomness: loss evaluation never changes data sampling."""

    def __init__(self, cfg, corpus, subsets, *, long_first=False):
        self.cfg, self.corpus = cfg, corpus
        self.subsets = sorted(subsets)
        self.domains = sorted({name.rsplit("-", 1)[0] for name in subsets})
        self.long_first = long_first

    def update(self, step):
        episodes = []
        for episode in range(self.cfg.optimization.episodes_per_update):
            rng = random.Random(self.cfg.seed + step * 100003 + episode * 1009)
            # Category RNG and context-size RNG are independent of the number
            # of subsets, preserving a nested design across data scales.
            domain = rng.choice(self.domains)
            size = rng.randint(
                self.cfg.context.min_examples, self.cfg.context.max_examples
            )
            candidates = [s for s in self.subsets if s.rsplit("-", 1)[0] == domain]
            subset = rng.choice(candidates)
            indices = self.corpus.indices(subset)
            cohort = rng.sample(indices, size)
            if self.long_first and step == 0:
                # The pilot exercises long trajectories from every category.
                domain = self.domains[episode % len(self.domains)]
                subset = next(s for s in self.subsets if s.rsplit("-", 1)[0] == domain)
                cohort = sorted(
                    self.corpus.indices(subset),
                    key=lambda i: self.corpus.rows[i]["seq_len"],
                    reverse=True,
                )[:size]
            episodes.append({"subset": subset, "indices": cohort})
        return episodes


class CampaignCorpus(Corpus):
    def __init__(self, cfg, bank, cache):
        super().__init__(cfg, bank)
        self.cache = cache
        self.summary_seconds = 0.0
        self.summary_hits = self.summary_misses = 0
        self.summary_memory = OrderedDict()
        self.summary_root = cfg.staging_dir / "summaries"
        self.summary_root.mkdir(parents=True, exist_ok=True)
        for transfer in self.transfer.transfers:
            if transfer.hub_store is None:
                raise ValueError("Campaign requires pinned Hub activation indexes")
            transfer.hub_store.shared_cache = cache
            transfer.hub_store.local_dir = None

    def prepare(self, model):
        # Runtime.new_generator only needs the last dimension to construct G.
        # No activation is fetched until a scheduled context requests it.
        self.summaries = {"last": torch.empty(0, model.fc.weight.shape[1])}

    def reference(self, index, *, summary=False):
        transfer, local = self.transfer.locate(index)
        row = transfer.hub_store.index["rows"][str(local)]
        kind = "summary" if summary and row.get("format") != "raw" else "shard"
        return row[kind], row

    def summary_path(self, index):
        ref, row = self.reference(index, summary=True)
        identity = {
            "version": 1,
            "reference": ref,
            "primary_id": row["primary_id"],
            # Older published indexes omit boundaries. The prepared Arrow row
            # owns the prompt boundary regardless of index format/version.
            "prompt_length": self.rows[index]["prompt_length"],
            "condition": "last",
            "models": self.manifest["revisions"],
        }
        return self.summary_root / (digest(identity) + ".pt")

    def references(self, indices, *, full):
        result = {}
        for index in indices:
            if full:
                ref, _ = self.reference(index)
                result[self.cache.key(ref)] = ref
            if not self.summary_path(index).is_file():
                ref, _ = self.reference(index, summary=True)
                result[self.cache.key(ref)] = ref
        return list(result.values())

    def context(self, indices, condition, device):
        if condition != "last":
            raise ValueError("Campaign summaries support last conditioning")
        started = time.perf_counter()
        values = []
        for index in indices:
            path = self.summary_path(index)
            if path in self.summary_memory:
                values.append(self.summary_memory[path])
                self.summary_memory.move_to_end(path)
                self.summary_hits += 1
                continue
            with FileLock(str(path) + ".lock"):
                if path.is_file():
                    value = torch.load(path, map_location="cpu", weights_only=True)
                    self.summary_hits += 1
                else:
                    value = self.transfer.read(
                        index,
                        prompt_length=self.rows[index]["prompt_length"],
                        summary=True,
                    )
                    if value is None or not torch.isfinite(value).all():
                        raise ValueError(f"Invalid prompt summary: row={index}")
                    atomic_torch_save(path, value)
                    self.summary_misses += 1
            values.append(value)
            self.summary_memory[path] = value
            if len(self.summary_memory) > getattr(
                self, "summary_memory_limit", SUMMARY_MEMORY_ENTRIES
            ):
                self.summary_memory.popitem(last=False)
        self.summary_seconds += time.perf_counter() - started
        return (
            torch.stack(values).to(device),
            torch.tensor(
                [self.rows[i]["prompt_length"] for i in indices], device=device
            ),
        )

    def prefetch(self, indices, *, full):
        refs = self.references(indices, full=full)
        self.cache.prefetch(refs)
        return refs
