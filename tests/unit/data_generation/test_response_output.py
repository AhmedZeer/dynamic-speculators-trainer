"""Local response staging, remote publication, and interruption behavior."""

import asyncio
import json
import threading

import pytest

from speculators.data_generation import response_output
from speculators.data_generation.response_output import ResponseOutput


def write_row(manager, value, stream=0):
    manager.files[stream].write(json.dumps({"primary_id": value}) + "\n")
    manager.files[stream].flush()


def rows(path):
    return [json.loads(line)["primary_id"] for line in path.read_text().splitlines()]


def manager_at(root, interval=2):
    return ResponseOutput(
        root / "drive/out.jsonl", root / "drive/errors.jsonl", root / "local", interval
    )


def test_interval_and_final_sync_include_errors(tmp_path):
    async def run():
        manager = manager_at(tmp_path)
        async with manager:
            write_row(manager, "first")
            await manager.example_completed()
            assert not manager.destinations[0].exists()
            write_row(manager, "failed", stream=1)
            await manager.example_completed()
            assert rows(manager.destinations[0]) == ["first"]
            assert rows(manager.destinations[1]) == ["failed"]
            write_row(manager, "last")
            await manager.example_completed()
            assert rows(manager.destinations[0]) == ["first"]
        assert rows(manager.destinations[0]) == ["first", "last"]

    asyncio.run(run())


def test_existing_drive_file_and_unsynced_local_rows_resume(tmp_path):
    async def run():
        manager = manager_at(tmp_path)
        manager.destinations[0].parent.mkdir(parents=True)
        manager.destinations[0].write_text('{"primary_id":"old"}\n')
        async with manager:
            assert rows(manager.paths[0]) == ["old"]
        with manager.paths[0].open("a") as stream:
            stream.write('{"primary_id":"unsynced"}\n{"unfinished":')
        restarted = manager_at(tmp_path)
        async with restarted:
            assert rows(restarted.paths[0]) == ["old", "unsynced"]
        assert rows(restarted.destinations[0]) == ["old", "unsynced"]

    asyncio.run(run())


def test_destination_ahead_of_local_is_reused(tmp_path):
    async def run():
        manager = manager_at(tmp_path)
        async with manager:
            write_row(manager, "first")
        with manager.destinations[0].open("a") as stream:
            stream.write('{"primary_id":"newer"}\n')
        async with manager_at(tmp_path) as restarted:
            assert rows(restarted.paths[0]) == ["first", "newer"]

    asyncio.run(run())


def test_divergent_files_fail_without_overwriting(tmp_path):
    async def run():
        manager = manager_at(tmp_path)
        async with manager:
            write_row(manager, "local")
        manager.destinations[0].write_text('{"primary_id":"different"}\n')
        with pytest.raises(ValueError, match="diverged"):
            async with manager_at(tmp_path):
                pytest.fail("Divergent staging must not open")
        assert rows(manager.destinations[0]) == ["different"]
        assert rows(manager.paths[0]) == ["local"]

    asyncio.run(run())


def test_slow_copy_allows_appends_and_copies_only_captured_prefix(
    tmp_path, monkeypatch
):
    original = response_output._copy_snapshot
    entered, release = threading.Event(), threading.Event()

    def slow_copy(source, destination, size):
        if destination.name == "out.jsonl":
            entered.set()
            assert release.wait(timeout=5)
        original(source, destination, size)

    async def run():
        manager = manager_at(tmp_path)
        async with manager:
            monkeypatch.setattr(response_output, "_copy_snapshot", slow_copy)
            write_row(manager, "first")
            sync = asyncio.create_task(manager.sync())
            assert await asyncio.to_thread(entered.wait, 5)
            try:
                write_row(manager, "during-copy")
            finally:
                release.set()
            await sync
            assert rows(manager.destinations[0]) == ["first"]
        assert rows(manager.destinations[0]) == ["first", "during-copy"]

    asyncio.run(run())


def test_failed_replace_preserves_remote_and_retains_local(tmp_path, monkeypatch):
    async def run():
        manager = manager_at(tmp_path, interval=1)
        async with manager:
            write_row(manager, "first")
            await manager.sync()
            original = response_output.Path.replace

            def fail_replace(path, target):
                if target == manager.destinations[0]:
                    raise OSError("Drive unavailable")
                return original(path, target)

            monkeypatch.setattr(response_output.Path, "replace", fail_replace)
            write_row(manager, "second")
            await manager.example_completed()
            assert rows(manager.destinations[0]) == ["first"]
            assert rows(manager.paths[0]) == ["first", "second"]
            assert not list(manager.destinations[0].parent.glob("*.pending"))
            monkeypatch.setattr(response_output.Path, "replace", original)
        assert rows(manager.destinations[0]) == ["first", "second"]

    asyncio.run(run())


def test_cancellation_runs_final_sync(tmp_path):
    async def interrupted():
        async with manager_at(tmp_path) as manager:
            write_row(manager, "completed")
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(interrupted())
    assert rows(tmp_path / "drive/out.jsonl") == ["completed"]


def test_direct_mode_preserves_append_behavior(tmp_path):
    async def run():
        for value in ["first", "second"]:
            async with ResponseOutput(
                tmp_path / "out.jsonl", tmp_path / "errors.jsonl", None, 1000
            ) as manager:
                write_row(manager, value)
                await manager.example_completed()
        assert rows(tmp_path / "out.jsonl") == ["first", "second"]

    asyncio.run(run())


def test_failed_final_sync_reports_failure_and_keeps_local(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OSError("Drive unavailable")

    async def run():
        manager = manager_at(tmp_path, interval=1)
        monkeypatch.setattr(response_output, "_copy_snapshot", unavailable)

        async def generate():
            async with manager:
                write_row(manager, "retained")
                await manager.example_completed()

        with pytest.raises(OSError, match="Drive unavailable"):
            await generate()
        assert rows(manager.paths[0]) == ["retained"]
        assert all(stream.closed for stream in manager.files)

    asyncio.run(run())
