"""Extraction publication, resume integrity, and graceful fail-fast behavior."""

import asyncio
import threading
from importlib import import_module
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file

from speculators.bank import progress
from speculators.bank.config import BankConfig
from speculators.data_generation import offline
from speculators.data_generation.offline import publish_hidden_states

extraction = import_module("speculators.cli.generate_offline_data")


def make_states(path, tokens=(1, 2), *, finite=True):
    hidden = torch.ones(len(tokens), 4, 3)
    if not finite:
        hidden[0, 0, 0] = float("nan")
    save_file({"token_ids": torch.tensor(tokens), "hidden_states": hidden}, str(path))


def test_only_validated_complete_files_receive_final_name(tmp_path, monkeypatch):
    source, target = tmp_path / "request.safetensors", tmp_path / "hs_0.safetensors"
    make_states(source)
    move = offline.shutil.move

    def observe_copy(src, pending):
        assert not target.exists()
        assert pending.endswith(".pending")
        return move(src, pending)

    monkeypatch.setattr(offline.shutil, "move", observe_copy)
    publish_hidden_states(source, target, [1, 2], validate=True)
    assert load_file(target)["token_ids"].tolist() == [1, 2]
    assert not source.exists()
    assert not list(tmp_path.glob("*.pending"))


@pytest.mark.parametrize("finite", [True, False])
def test_bad_source_is_not_published(tmp_path, finite):
    source, target = tmp_path / "request.safetensors", tmp_path / "hs_0.safetensors"
    make_states(source, finite=finite)
    tokens = [99, 2] if finite else [1, 2]
    with pytest.raises(ValueError):
        publish_hidden_states(source, target, tokens, validate=True)
    assert source.exists()
    assert not target.exists()


def test_failed_copy_keeps_old_target_and_cleans_pending(tmp_path, monkeypatch):
    source, target = tmp_path / "request.safetensors", tmp_path / "hs_0.safetensors"
    make_states(source)
    target.write_bytes(b"previous destination")

    def fail_copy(src, pending):
        offline.Path(pending).write_bytes(b"partial")
        raise OSError("Drive disconnected")

    monkeypatch.setattr(offline.shutil, "move", fail_copy)
    with pytest.raises(OSError, match="Drive disconnected"):
        publish_hidden_states(source, target, [1, 2], validate=True)
    assert target.read_bytes() == b"previous destination"
    assert not list(tmp_path.glob("*.pending"))


def test_resume_requeues_incomplete_or_wrong_token_files(tmp_path):
    make_states(tmp_path / "hs_0.safetensors")
    (tmp_path / "hs_1.safetensors").write_bytes(b"partial")
    make_states(tmp_path / "hs_2.safetensors", tokens=(99, 2))
    (tmp_path / ".hs_3.safetensors.pending").write_bytes(b"in-progress")
    dataset = Dataset.from_list([{"input_ids": [1, 2]} for _ in range(4)])
    assert extraction._verified_existing_indices(tmp_path, dataset, True) == [0]
    # The invalid originals remain until a validated replacement is ready.
    assert (tmp_path / "hs_1.safetensors").exists()


class Progress:
    def update(self, n):
        pass

    def set_postfix(self, values, refresh=False):
        pass


def stats():
    return {
        "ok": 0,
        "errors": 0,
        "total_vllm_s": 0.0,
        "total_write_s": 0.0,
        "start_time": extraction.time.perf_counter(),
    }


def worker(queue, destination, cancelled, counters, *, request_slots=None, **kwargs):
    return extraction._worker(
        client=None,
        model="target",
        queue=queue,
        pbar=Progress(),
        vllm_semaphore=request_slots or asyncio.Semaphore(2),
        write_semaphore=asyncio.Semaphore(1),
        hidden_states_output_dir=destination,
        validate_outputs=True,
        request_timeout=600,
        max_retries=0,
        fail_on_error=True,
        skipped_indices=[],
        cancel_event=cancelled,
        failure_tracker=None,
        stats=counters,
        **kwargs,
    )


def test_failure_stops_new_jobs_but_finishes_other_inflight_save(tmp_path, monkeypatch):
    async def run():
        source = tmp_path / "request.safetensors"
        destination = tmp_path / "cache"
        destination.mkdir()
        make_states(source)
        queue = asyncio.Queue(maxsize=4)
        for index in range(4):
            queue.put_nowait({"idx": index, "input_ids": [1, 2]})
        started, cancelled = asyncio.Event(), asyncio.Event()
        counters = stats()

        async def generate(client, model, item, **kw):
            if item["idx"] == 0:
                await started.wait()
                raise TimeoutError("request exhausted retries")
            assert item["idx"] == 1
            started.set()
            await cancelled.wait()
            return str(source)

        monkeypatch.setattr(extraction, "generate_hidden_states_async", generate)
        workers = [
            asyncio.create_task(worker(queue, destination, cancelled, counters))
            for _ in range(2)
        ]
        await cancelled.wait()
        with pytest.raises(RuntimeError, match="row 0 during request"):
            await extraction._shutdown_workers(workers, queue, cancelled)
        assert counters["ok"] == 1
        assert counters["errors"] == 1
        assert (destination / "hs_1.safetensors").is_file()
        assert not (destination / "hs_0.safetensors").exists()
        assert not (destination / "hs_2.safetensors").exists()

    asyncio.run(run())


def test_full_queue_with_no_live_workers_cannot_deadlock():
    async def run():
        async def failed():
            raise RuntimeError("worker failed")

        task = asyncio.create_task(failed())
        await asyncio.gather(task, return_exceptions=True)
        queue = asyncio.Queue(maxsize=1)
        queue.put_nowait({"idx": 1})
        cancelled = asyncio.Event()
        cancelled.set()
        with pytest.raises(RuntimeError, match="worker failed"):
            await asyncio.wait_for(
                extraction._shutdown_workers([task], queue, cancelled), timeout=1
            )
        assert queue.empty()

    asyncio.run(run())


def test_lock_wait_uses_configured_timeout(tmp_path, monkeypatch):
    async def run():
        source = tmp_path / "request.safetensors"
        make_states(source)
        source.with_suffix(".safetensors.lock").touch()
        queue = asyncio.Queue()
        queue.put_nowait({"idx": 0, "input_ids": [1, 2]})
        queue.put_nowait(None)
        waits = []

        async def generate(*args, **kw):
            return str(source)

        async def wait(path, timeout):
            waits.append(timeout)

        monkeypatch.setattr(extraction, "generate_hidden_states_async", generate)
        monkeypatch.setattr(extraction, "wait_for_lock_async", wait)
        await worker(queue, tmp_path, asyncio.Event(), stats())
        assert waits == [600]
        assert (tmp_path / "hs_0.safetensors").exists()

    asyncio.run(run())


def test_bank_uses_separate_extraction_concurrency(monkeypatch):
    workflow = import_module("speculators.bank.workflow")
    captured = []
    monkeypatch.setattr(
        extraction, "generate_offline_data", lambda **kw: captured.append(kw)
    )
    monkeypatch.setattr(workflow, "validate_cache", lambda *args: None)
    manifest = {"prepared": True, "data_path": "data", "hidden_states_path": "states"}
    cfg = BankConfig.model_validate(
        {"execution": {"concurrency": 256, "hidden_state_log_interval": 3}}
    )
    workflow.extract(cfg, manifest)
    assert captured[0]["concurrency"] == 16
    assert captured[0]["write_concurrency"] == 2
    assert captured[0]["progress_log_interval"] == 3
    assert captured[0]["trust_existing_outputs"] is True
    assert captured[0]["validate_outputs"] is True


def test_waiting_request_is_not_started_after_failure(tmp_path, monkeypatch):
    async def run():
        queue = asyncio.Queue()
        queue.put_nowait({"idx": 0, "input_ids": [1, 2]})
        queue.put_nowait(None)
        cancelled = asyncio.Event()
        slots = asyncio.Semaphore(0)

        async def unexpected_request(*args, **kwargs):
            pytest.fail("An unstarted request must not launch after cancellation")

        monkeypatch.setattr(
            extraction, "generate_hidden_states_async", unexpected_request
        )
        task = asyncio.create_task(
            worker(queue, tmp_path, cancelled, stats(), request_slots=slots)
        )
        await asyncio.sleep(0)
        cancelled.set()
        slots.release()
        await task
        assert not list(tmp_path.glob("hs_*.safetensors"))

    asyncio.run(run())


def test_cancelled_worker_finishes_started_publication(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = extraction.publish_hidden_states

    def slow_publish(*args):
        entered.set()
        assert release.wait(timeout=5)
        original(*args)

    async def run():
        source = tmp_path / "request.safetensors"
        make_states(source)
        queue = asyncio.Queue()
        queue.put_nowait({"idx": 0, "input_ids": [1, 2]})
        queue.put_nowait(None)

        async def generate(*args, **kwargs):
            return str(source)

        monkeypatch.setattr(extraction, "generate_hidden_states_async", generate)
        monkeypatch.setattr(extraction, "publish_hidden_states", slow_publish)
        task = asyncio.create_task(worker(queue, tmp_path, asyncio.Event(), stats()))
        assert await asyncio.to_thread(entered.wait, 5)
        try:
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert load_file(tmp_path / "hs_0.safetensors")["token_ids"].tolist() == [1, 2]
        assert not list(tmp_path.glob("*.pending"))

    asyncio.run(run())


def test_heartbeat_reports_stalled_work_and_final_error(caplog):
    async def run():
        counters = stats() | {"reused": 7, "total": 10}
        async with extraction._ProgressLogger(counters, 0.01):
            await asyncio.sleep(0.035)
            counters["ok"] = 1
            counters["errors"] = 1
            raise RuntimeError("server stopped")

    caplog.set_level("INFO", logger=extraction.__name__)
    with pytest.raises(RuntimeError, match="server stopped"):
        asyncio.run(run())
    assert (
        "running: 7/10 complete | reused=7 saved=0 failed=0 remaining=3" in caplog.text
    )
    assert (
        "stopped: 8/10 complete | reused=7 saved=1 failed=1 remaining=2" in caplog.text
    )


def test_resume_progress_includes_existing_rows_and_only_published_files(  # noqa: C901
    tmp_path, monkeypatch, caplog
):
    data, cache = tmp_path / "data", tmp_path / "cache"
    cache.mkdir()
    Dataset.from_list([{"input_ids": [1, 2]} for _ in range(4)]).save_to_disk(data)
    make_states(cache / "hs_0.safetensors")
    make_states(cache / "hs_2.safetensors")
    bars = []
    requested = []
    startup_complete = []

    class Bar(Progress):
        def __init__(self, **kwargs):
            self.options = kwargs
            self.advanced = 0
            bars.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n):
            self.advanced += n

    class Client:
        def __init__(self, **kwargs):
            self.models = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def list(self):
            return SimpleNamespace(data=[SimpleNamespace(id="target")])

    async def generate(client, model, item, **kwargs):
        assert startup_complete == [True]
        requested.append(item["idx"])
        if item["idx"] == 3:
            raise TimeoutError("unavailable")
        source = tmp_path / "request.safetensors"
        make_states(source)
        return str(source)

    monkeypatch.setattr(extraction, "tqdm", Bar)
    monkeypatch.setattr(extraction.openai, "AsyncOpenAI", Client)
    monkeypatch.setattr(extraction, "build_client_item", lambda row: row)
    monkeypatch.setattr(extraction, "generate_hidden_states_async", generate)
    caplog.set_level("INFO", logger=extraction.__name__)
    asyncio.run(
        extraction._generate_and_save_hidden_states(
            model="target",
            endpoint="http://unused",
            preprocessed_data=str(data),
            output=str(cache),
            max_samples=None,
            concurrency=1,
            validate_outputs=True,
            request_timeout=600,
            max_retries=0,
            fail_on_error=False,
            max_consecutive_errors=10,
            world_size=1,
            rank=0,
            on_startup_complete=lambda: startup_complete.append(True),
        )
    )
    assert requested == [1, 3]
    assert bars[0].options["initial"] == 2
    assert bars[0].options["total"] == 4
    assert bars[0].advanced == 1
    assert (
        "finished: 3/4 complete | reused=2 saved=1 failed=1 remaining=1" in caplog.text
    )


def test_slow_startup_stage_logs_before_completion(caplog):
    started, release = threading.Event(), threading.Event()

    def slow_load():
        started.set()
        assert release.wait(timeout=2)
        return "dataset"

    async def run():
        task = asyncio.create_task(
            extraction._run_logged_stage(
                "Loading dataset",
                slow_load,
                interval=0.01,
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 1)
            await asyncio.sleep(0.035)
            assert "Loading dataset: starting" in caplog.text
            assert "Loading dataset: still running" in caplog.text
            assert not task.done()
        finally:
            release.set()
        assert await task == "dataset"

    caplog.set_level("INFO", logger=extraction.__name__)
    asyncio.run(run())
    assert "Loading dataset: completed" in caplog.text


def test_startup_failure_is_logged_and_propagated(caplog):
    def fail_load():
        raise OSError("Drive unavailable")

    caplog.set_level("INFO", logger=extraction.__name__)
    with pytest.raises(OSError, match="Drive unavailable"):
        asyncio.run(
            extraction._run_logged_stage(
                "Loading dataset",
                fail_load,
                interval=0.01,
            )
        )
    assert "Loading dataset: stopped" in caplog.text


def test_extraction_console_bypasses_root_handlers(monkeypatch, capsys):
    monkeypatch.setattr(extraction.logger, "handlers", [])
    monkeypatch.setattr(extraction.logger, "propagate", True)
    monkeypatch.setattr(extraction.logger, "level", extraction.logging.WARNING)

    class BrokenHandler(extraction.logging.Handler):
        def emit(self, record):
            pytest.fail("Extraction records must not enter root/Rich handlers")

    monkeypatch.setattr(extraction.logging.getLogger(), "handlers", [BrokenHandler()])
    extraction._configure_extraction_logger()
    extraction._configure_extraction_logger()
    assert len(extraction.logger.handlers) == 1
    extraction.logger.info("Loading dataset immediately")
    assert "Loading dataset immediately" in capsys.readouterr().err


@pytest.mark.parametrize("failed", [True, False])
def test_startup_watchdog_always_cancelled(monkeypatch, failed):
    calls = []
    monkeypatch.setattr(
        progress.faulthandler,
        "dump_traceback_later",
        lambda timeout, **kw: calls.append(timeout),
    )
    monkeypatch.setattr(
        progress.faulthandler,
        "cancel_dump_traceback_later",
        lambda: calls.append("cancelled"),
    )

    def run():
        with extraction._StartupWatchdog() as watchdog:
            if failed:
                raise OSError("startup failed")
            watchdog.finish()
            watchdog.finish()

    if failed:
        with pytest.raises(OSError, match="startup failed"):
            run()
    else:
        run()
    assert calls == [30, "cancelled"]


def test_trusted_resume_does_not_open_saved_payloads_or_contact_server(
    tmp_path, monkeypatch, caplog
):
    data, cache = tmp_path / "data", tmp_path / "cache"
    cache.mkdir()
    Dataset.from_list([{"input_ids": [1, 2]}]).save_to_disk(data)
    (cache / "hs_0.safetensors").write_bytes(b"already saved")

    def forbidden(*args, **kwargs):
        pytest.fail("Trusted saved files must not be inspected or regenerated")

    monkeypatch.setattr(extraction, "check_hidden_state_file_header", forbidden)
    monkeypatch.setattr(extraction.openai, "AsyncOpenAI", forbidden)
    caplog.set_level("INFO", logger=extraction.__name__)
    asyncio.run(
        extraction._generate_and_save_hidden_states(
            model="target",
            endpoint="http://unused",
            preprocessed_data=str(data),
            output=str(cache),
            max_samples=None,
            concurrency=1,
            validate_outputs=True,
            request_timeout=600,
            max_retries=0,
            fail_on_error=True,
            max_consecutive_errors=None,
            world_size=1,
            rank=0,
            trust_existing_outputs=True,
        )
    )
    assert "1/1 reusable files; no requests needed" in caplog.text
