"""Bounded background reads of immutable hidden states, before augmentation."""

from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock

from hs_connectors import HiddenStatesTransfer


class RawPrefetchTransfer(HiddenStatesTransfer):
    """One I/O thread and an LRU cache; futures never retain tensor payloads."""

    def __init__(self, source, max_bytes: int, batches: int):
        if max_bytes <= 0 or batches <= 0:
            raise ValueError("Prefetch cache size and batch lookahead must be positive")
        self.source = source
        self.max_bytes = max_bytes
        self.batches = batches
        self._cache = OrderedDict()
        self._pending: dict[int, Future] = {}
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="hidden-state-read"
        )
        self._bytes = 0
        self._reads = self._hits = self._waits = 0
        self._closed = False

    def _load_source(self, index):
        with self._lock:
            self._reads += 1
        return self.source.get_cached(index)

    def _read(self, index):
        payload = self._load_source(index)
        if payload is None:
            raise FileNotFoundError(f"Missing saved hidden states for row {index}")
        # safetensors uses mmap. Copy to owned CPU memory here so prefetch
        # actually reads the bytes, and future GPU consumption avoids page faults.
        size = sum(value.numel() * value.element_size() for value in payload.values())
        if size > self.max_bytes:
            return  # Leave oversized samples on the ordinary consumer path.
        payload = {key: value.clone() for key, value in payload.items()}
        with self._lock:
            while self._cache and self._bytes + size > self.max_bytes:
                _, (_, removed) = self._cache.popitem(last=False)
                self._bytes -= removed
            self._cache[index] = (payload, size)
            self._bytes += size

    def prefetch(self, indices):
        with self._lock:
            if self._closed:
                raise RuntimeError("Hidden-state prefetcher is closed")
            for requested in indices:
                index = int(requested)
                if index in self._cache:
                    continue
                future = self._pending.get(index)
                if future is not None and (
                    not future.done() or future.exception() is not None
                ):
                    continue
                self._pending[index] = self._executor.submit(self._read, index)

    def get_cached(self, file_idx):
        with self._lock:
            cached = self._cache.get(file_idx)
            if cached is not None:
                self._hits += 1
                self._cache.move_to_end(file_idx)
                self._pending.pop(file_idx, None)
                return cached[0]
        self.prefetch([file_idx])
        with self._lock:
            future = self._pending[file_idx]
            if not future.done():
                self._waits += 1
        future.result()  # Propagate background I/O failures on the consumer thread.
        with self._lock:
            self._pending.pop(file_idx, None)
            cached = self._cache.get(file_idx)
            if cached is not None:
                self._cache.move_to_end(file_idx)
                return cached[0]
        # A sample can exceed the budget or be evicted by later lookahead reads.
        return self._load_source(file_idx)

    def stats(self):
        with self._lock:
            return {
                "cache_bytes": self._bytes,
                "cache_samples": len(self._cache),
                "file_reads": self._reads,
                "cache_hits": self._hits,
                "consumer_waits": self._waits,
            }

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._lock:
            self._pending.clear()
            self._cache.clear()
            self._bytes = 0

    def setup(self):
        return self.source.setup()

    def get_generated(self, handle):
        return self.source.get_generated(handle)

    def cache(self, handle, file_idx):
        return self.source.cache(handle, file_idx)

    def delete(self, handle):
        return self.source.delete(handle)


class PrefetchBatchSampler:
    """Look ahead in the actual sampler order without invoking Dataset.__getitem__."""

    def __init__(self, sampler, transfer, file_index):
        self.sampler = sampler
        self.transfer = transfer
        self.file_index = file_index

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    @property
    def _cached_generated_batches(self):
        return self.sampler._cached_generated_batches  # noqa: SLF001

    @_cached_generated_batches.setter
    def _cached_generated_batches(self, value):
        # Trainer's existing fast-skip writes through to the original sampler.
        self.sampler._cached_generated_batches = value  # noqa: SLF001

    def __len__(self):
        return len(self.sampler)

    def __iter__(self):
        batches = iter(self.sampler)
        upcoming = deque()
        for _ in range(self.transfer.batches + 1):
            batch = next(batches, None)
            if batch is None:
                break
            upcoming.append(batch)
        self.transfer.prefetch(self.file_index(i) for batch in upcoming for i in batch)
        while upcoming:
            current = upcoming.popleft()
            yield current
            batch = next(batches, None)
            if batch is not None:
                upcoming.append(batch)
                self.transfer.prefetch(self.file_index(i) for i in batch)
