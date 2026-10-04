"""Prompt chunk order, bounded read-ahead, and source cleanup."""

import asyncio
import json
import threading
from contextlib import aclosing

import pytest

from speculators.data_generation.prompt_buffer import buffered_prompts, jsonl_rows


def test_jsonl_chunks_preserve_order_and_skip_blank_lines(tmp_path):
    source = tmp_path / "prompts.jsonl"
    source.write_text("\n".join(json.dumps({"id": i}) for i in range(7)) + "\n\n")

    async def run():
        async with aclosing(buffered_prompts(jsonl_rows(source), 3)) as prompts:
            return [item async for item in prompts]

    assert asyncio.run(run()) == [(i, {"id": i}) for i in range(7)]


def test_read_ahead_is_bounded_off_thread_and_closes_on_early_stop():
    main_thread = threading.get_ident()
    read, closed = [], []

    def source():
        try:
            for i in range(100):
                assert threading.get_ident() != main_thread
                read.append(i)
                yield {"id": i}
        finally:
            closed.append(True)

    async def run():
        async with aclosing(buffered_prompts(source(), 3)) as prompts:
            async for index, row in prompts:
                assert (index, row) == (0, {"id": 0})
                break

    asyncio.run(run())
    assert read == list(range(6))  # Current chunk and one prefetch, never all rows.
    assert closed == [True]


@pytest.mark.parametrize("content", ['{"broken":\n', "[1, 2]\n"])
def test_bad_jsonl_fails_with_line_number(tmp_path, content):
    source = tmp_path / "prompts.jsonl"
    source.write_text(content)
    with pytest.raises(ValueError, match="line 1"):
        list(jsonl_rows(source))


def test_cancelled_consumer_closes_source():
    closed = []

    def source():
        try:
            yield from [{"id": i} for i in range(10)]
        finally:
            closed.append(True)

    async def run():
        async with aclosing(buffered_prompts(source(), 2)) as prompts:
            await anext(prompts)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run())
    assert closed == [True]
