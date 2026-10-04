"""Single-pass JSONL ingestion and bounded RAM prefetch for generation workers."""

import asyncio
import json
from itertools import islice
from pathlib import Path

READ_BUFFER_BYTES = 1024 * 1024


def jsonl_rows(path: Path):
    """Open once and read buffered blocks; never reopen per request."""
    with path.open(encoding="utf-8", buffering=READ_BUFFER_BYTES) as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}, line {line_number}")
            yield row


async def buffered_prompts(rows, chunk_size: int):
    """Yield indexed rows from RAM while reading the next chunk off-thread."""
    if chunk_size < 1:
        raise ValueError("Prompt chunk size must be positive")
    iterator = enumerate(rows)
    pending = None

    def read_chunk():
        return list(islice(iterator, chunk_size))

    try:
        pending = asyncio.create_task(asyncio.to_thread(read_chunk))
        while True:
            chunk = await asyncio.shield(pending)
            if not chunk:
                break
            pending = asyncio.create_task(asyncio.to_thread(read_chunk))
            for item in chunk:
                yield item
    finally:
        # Threads cannot be cancelled. Wait before closing the underlying file,
        # including when the producer reaches --limit or is interrupted.
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)
        close = getattr(rows, "close", None)
        if close is not None:
            await asyncio.to_thread(close)
