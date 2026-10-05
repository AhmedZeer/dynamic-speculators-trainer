"""
Offline Hidden States Generation Pipeline

This module generates hidden states and saves them to disk for offline training.

Usage::

    speculators generate-offline-data \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --preprocessed-data sharegpt \
        --output ./training_data \
        --max-samples 5000
"""

import asyncio
import faulthandler
import logging
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import openai
import typer
from datasets import load_from_disk
from tqdm import tqdm

from speculators.data_generation.offline import (
    check_hidden_state_file_header,
    get_existing_hidden_state_indices,
    get_indices_to_process,
    publish_hidden_states,
)
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    generate_hidden_states_async,
    wait_for_lock_async,
)
from speculators.train.data import build_client_item

logger = logging.getLogger(__name__)


class _StartupWatchdog:
    """Obtain thread stacks even when Python logging or its event loop is blocked."""

    def __init__(self):
        self.armed = False

    def __enter__(self):
        try:
            faulthandler.dump_traceback_later(30, file=sys.stderr, repeat=False)
            self.armed = True
        except (OSError, RuntimeError, ValueError):
            logger.warning(
                "Startup stack diagnostics unavailable on this stderr stream"
            )
        return self

    def finish(self):
        if self.armed:
            faulthandler.cancel_dump_traceback_later()
            self.armed = False

    def __exit__(self, *args):
        self.finish()


def _configure_extraction_logger():
    """Use a local plain console handler, independent of Rich/root filters."""
    if not any(
        handler.get_name() == "speculators-extraction" for handler in logger.handlers
    ):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name("speculators-extraction")
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s %(message)s", datefmt="%H:%M:%S"
            )
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


async def _run_logged_stage(label, function, *args, interval):
    """Keep startup visible while filesystem work runs outside the event loop."""
    started = time.perf_counter()
    logger.info("%s: starting", label)
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=interval)
            if done:
                result = task.result()
                logger.info(
                    "%s: completed in %.1fs", label, time.perf_counter() - started
                )
                return result
            logger.info(
                "%s: still running (%.1fs elapsed)",
                label,
                time.perf_counter() - started,
            )
    except BaseException:
        logger.info("%s: stopped after %.1fs", label, time.perf_counter() - started)
        # A filesystem thread cannot be cancelled. Consume its eventual exception.
        task.add_done_callback(
            lambda finished: finished.exception() if not finished.cancelled() else None
        )
        raise


class _ProgressLogger:
    """Periodic console summaries using in-memory counters only."""

    def __init__(self, stats: dict[str, Any], interval: float):
        self.stats = stats
        self.interval = interval
        self.stop = asyncio.Event()

    def report(self, phase: str):
        stats = self.stats
        completed = stats["reused"] + stats["ok"]
        elapsed = time.perf_counter() - stats["start_time"]
        logger.info(
            "Hidden states %s: %d/%d complete | reused=%d saved=%d "
            "failed=%d remaining=%d | %.2f files/s",
            phase,
            completed,
            stats["total"],
            stats["reused"],
            stats["ok"],
            stats["errors"],
            stats["total"] - completed,
            stats["ok"] / elapsed if elapsed > 0 else 0,
        )

    async def heartbeat(self):
        while not self.stop.is_set():
            try:
                await asyncio.wait_for(self.stop.wait(), timeout=self.interval)
            except TimeoutError:
                self.report("running")

    async def __aenter__(self):
        self.report("starting")
        self.task = asyncio.create_task(self.heartbeat())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.stop.set()
        await self.task
        self.report("stopped" if exc_type else "finished")


class _FailureTracker:
    """Tracks consecutive sample failures across async workers.

    When the number of consecutive failures (with no successes in between)
    reaches ``threshold``, the tracker signals that the run should abort.
    Because asyncio is single-threaded, no locking is needed.
    """

    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consecutive = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self) -> bool:
        """Record a failure. Returns True when the threshold is reached."""
        self._consecutive += 1
        return self._consecutive >= self.threshold


async def _worker(  # noqa: C901
    client,
    model: str,
    queue: "asyncio.Queue[dict[str, Any]]",
    pbar: tqdm,
    vllm_semaphore: asyncio.Semaphore,
    write_semaphore: asyncio.Semaphore,
    hidden_states_output_dir: Path,
    validate_outputs: bool,
    request_timeout: float | None,
    max_retries: int,
    fail_on_error: bool,
    skipped_indices: list[int],
    cancel_event: asyncio.Event,
    failure_tracker: _FailureTracker | None,
    stats: dict[str, Any],
):
    """Worker that pulls items from queue and sends them to the vLLM endpoint."""
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        idx = item["idx"]

        if cancel_event.is_set():
            queue.task_done()
            continue

        target_hidden_states_path = hidden_states_output_dir / f"hs_{idx}.safetensors"

        stage = "request"
        published = False
        try:
            async with vllm_semaphore:
                if cancel_event.is_set():
                    continue
                t_vllm = time.perf_counter()
                hidden_states_path = await generate_hidden_states_async(
                    client,
                    model,
                    item,
                    timeout=request_timeout,
                    max_retries=max_retries,
                )
                vllm_s = time.perf_counter() - t_vllm
            stage = "server-file lock"
            lock_path = hidden_states_path + ".lock"
            if Path(lock_path).exists():  # noqa: ASYNC240
                await wait_for_lock_async(lock_path, timeout=request_timeout)

            stage = "validation/publication"
            async with write_semaphore:
                t_write = time.perf_counter()
                save = asyncio.create_task(
                    asyncio.to_thread(
                        publish_hidden_states,
                        Path(hidden_states_path),
                        target_hidden_states_path,
                        item["input_ids"],
                        validate_outputs,
                    )
                )
                try:
                    await asyncio.shield(save)
                except asyncio.CancelledError:
                    # A copy thread cannot be cancelled; let its publication finish.
                    await save
                    published = True
                    write_s = time.perf_counter() - t_write
                    raise
                published = True
                write_s = time.perf_counter() - t_write
        except Exception as e:
            skipped_indices.append(idx)
            stats["errors"] += 1
            if fail_on_error:
                cancel_event.set()
                logger.exception("Fatal: sample %d failed during %s: %s", idx, stage, e)
                raise RuntimeError(
                    f"Extraction failed for row {idx} during {stage}: {e}. "
                    "Completed files are retained; rerun to resume missing rows."
                ) from e
            logger.warning("Skipping sample %d during %s: %s", idx, stage, e)
            if failure_tracker is not None and failure_tracker.record_failure():
                cancel_event.set()
                raise RuntimeError(
                    f"Aborting: {failure_tracker.threshold} consecutive samples "
                    "errored out. The vLLM server may be unreachable."
                ) from e
        else:
            logger.debug(
                "Sample %d: vLLM %.0f ms, write %.0f ms",
                idx,
                vllm_s * 1000,
                write_s * 1000,
            )
            if failure_tracker is not None:
                failure_tracker.record_success()
        finally:
            if published:
                stats["ok"] += 1
                stats["total_vllm_s"] += vllm_s
                stats["total_write_s"] += write_s
                pbar.update(1)
            elapsed = time.perf_counter() - stats["start_time"]
            postfix = {"ok": stats["ok"], "err": stats["errors"]}
            if elapsed > 0 and stats["ok"] > 0:
                postfix["rps"] = f"{stats['ok'] / elapsed:.1f}"
                postfix["vllm"] = f"{stats['total_vllm_s'] / stats['ok'] * 1000:.0f}ms"
                postfix["write"] = (
                    f"{stats['total_write_s'] / stats['ok'] * 1000:.0f}ms"
                )
            pbar.set_postfix(postfix, refresh=False)
            queue.task_done()


async def _feed_queue(to_process, dataset, queue, cancel_event):
    """Feed dataset items into the worker queue, respecting cancellation."""
    for i in to_process:
        if cancel_event.is_set():
            break

        dataset_item = dataset[i]
        client_item = build_client_item(dataset_item) | {"idx": i}

        while not cancel_event.is_set():
            try:
                queue.put_nowait(client_item)
                break
            except asyncio.QueueFull:
                await asyncio.sleep(0.1)


async def _shutdown_workers(workers, queue, cancel_event, *, propagate_errors=True):
    """Stop scheduling after failure, finish in-flight saves, and report errors."""
    logger.info("Waiting for in-flight requests and file saves to complete...")
    if cancel_event.is_set():
        while not queue.empty():
            queue.get_nowait()
            queue.task_done()
    for worker in workers:
        if not worker.done():
            await queue.put(None)
    results = await asyncio.gather(*workers, return_exceptions=True)
    if propagate_errors:
        for result in results:
            if isinstance(result, Exception):
                raise result


def _verified_existing_indices(directory, dataset, validate):
    existing = get_existing_hidden_state_indices(directory)
    if not validate:
        return existing
    verified = []
    for index in existing:
        if index < 0 or index >= len(dataset):
            continue
        try:
            tokens = dataset[index]["input_ids"]
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            check_hidden_state_file_header(
                directory / f"hs_{index}.safetensors", tokens
            )
        except Exception as exc:  # noqa: BLE001 -- invalid legacy files are regenerated
            logger.warning(
                "Cached row %d is incomplete or misaligned; regenerating: %s",
                index,
                exc,
            )
        else:
            verified.append(index)
    return verified


def _prepare_hidden_state_cache(directory, dataset, validate):
    directory.mkdir(parents=True, exist_ok=True)
    return _verified_existing_indices(directory, dataset, validate)


def _finish_startup(callback):
    if callback is not None:
        callback()


async def _generate_and_save_hidden_states(
    model: str | None,
    endpoint: str,
    preprocessed_data: str,
    output: str | None,
    max_samples: int | None,
    concurrency: int,
    validate_outputs: bool,
    request_timeout: float,
    max_retries: int,
    fail_on_error: bool,
    max_consecutive_errors: int | None,
    world_size: int,
    rank: int,
    write_concurrency: int = 2,
    progress_log_interval: float = 10.0,
    on_startup_complete: Callable[[], None] | None = None,
    trust_existing_outputs: bool = False,
):
    dataset = await _run_logged_stage(
        f"Loading preprocessed dataset from {preprocessed_data}",
        load_from_disk,
        preprocessed_data,
        interval=progress_log_interval,
    )
    logger.info("Dataset loaded: %d rows", len(dataset))

    if output is None:
        hidden_states_dir = Path(preprocessed_data) / "hidden_states"
    else:
        hidden_states_dir = Path(output)

    existing_file_indices = await _run_logged_stage(
        f"Checking existing hidden-state files for resume in {hidden_states_dir}",
        _prepare_hidden_state_cache,
        hidden_states_dir,
        dataset,
        validate_outputs and not trust_existing_outputs,
        interval=progress_log_interval,
    )
    num_samples = len(dataset)

    to_process = get_indices_to_process(
        num_samples,
        max_samples,
        existing_file_indices,
        world_size,
        rank,
    )
    target = max(
        0, min(max_samples, num_samples) if max_samples is not None else num_samples
    )
    shard_total = target // world_size + int(rank < target % world_size)
    reused = shard_total - len(to_process)
    if not to_process:
        _finish_startup(on_startup_complete)
        logger.info(
            "Hidden states complete: %d/%d reusable files; no requests needed",
            reused,
            shard_total,
        )
        return

    logger.info(
        "Cache: %d reusable files; %d rows pending; request concurrency=%d, writes=%d",
        reused,
        len(to_process),
        concurrency,
        write_concurrency,
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=concurrency * 4)
    vllm_semaphore = asyncio.Semaphore(concurrency)
    write_semaphore = asyncio.Semaphore(write_concurrency)

    skipped_indices: list[int] = []
    cancel_event = asyncio.Event()
    stats: dict[str, Any] = {
        "ok": 0,
        "reused": reused,
        "total": shard_total,
        "errors": 0,
        "total_vllm_s": 0.0,
        "total_write_s": 0.0,
        "start_time": time.perf_counter(),
    }

    max_consec = max_consecutive_errors
    if max_consec is None:
        max_consec = concurrency
    failure_tracker = _FailureTracker(max_consec) if not fail_on_error else None

    async with (
        _ProgressLogger(stats, progress_log_interval),
        openai.AsyncOpenAI(base_url=endpoint, api_key="EMPTY", max_retries=0) as client,
    ):
        list_models = await client.models.list()
        if not list_models.data:
            raise RuntimeError(
                "No models found on the vLLM server. "
                "Make sure the server is fully loaded."
            )
        model_id = list_models.data[0].id
        if model and model != model_id:
            raise ValueError(
                f"An explicit model name was passed ({model}) which doesn't match"
                f" found model_id {model_id}."
                "Please make sure --endpoint is set to the correct vllm instance."
            )

        _finish_startup(on_startup_complete)
        logger.info("Server ready: model=%s, endpoint=%s", model_id, endpoint)

        with tqdm(
            total=shard_total, initial=reused, desc="Hidden states", unit="files"
        ) as pbar:
            workers = [
                asyncio.create_task(
                    _worker(
                        client,
                        model_id,
                        queue,
                        pbar,
                        vllm_semaphore,
                        write_semaphore,
                        hidden_states_dir,
                        validate_outputs,
                        request_timeout,
                        max_retries,
                        fail_on_error,
                        skipped_indices,
                        cancel_event,
                        failure_tracker,
                        stats,
                    )
                )
                for _ in range(concurrency * 2)
            ]

            try:
                await _feed_queue(to_process, dataset, queue, cancel_event)
            except BaseException:
                cancel_event.set()
                await _shutdown_workers(
                    workers, queue, cancel_event, propagate_errors=False
                )
                raise
            else:
                await _shutdown_workers(workers, queue, cancel_event)

    elapsed = time.perf_counter() - stats["start_time"]
    if stats["ok"] > 0:
        logger.info(
            "Timing: %.1fs elapsed, %.1f samples/s, "
            "avg vLLM request %.0f ms, avg file write %.0f ms",
            elapsed,
            stats["ok"] / elapsed if elapsed > 0 else 0,
            stats["total_vllm_s"] / stats["ok"] * 1000,
            stats["total_write_s"] / stats["ok"] * 1000,
        )

    num_saved = stats["ok"]
    logger.info(f"Saved {num_saved} new data points to {hidden_states_dir}")
    if skipped_indices:
        logger.warning(
            f"Skipped {len(skipped_indices)} samples due to errors: {skipped_indices}"
        )


def generate_offline_data(
    model: Annotated[
        str | None,
        typer.Option(
            help=(
                "HuggingFace model ID or local path for target model "
                "(default auto select). For verification purposes only."
            ),
        ),
    ] = None,
    endpoint: Annotated[
        str,
        typer.Option(
            help=(
                "The address of the vLLM instance to use for hidden states "
                "generation. The instance must be configured for hidden states "
                "extraction."
            ),
        ),
    ] = "http://localhost:8000/v1",
    preprocessed_data: Annotated[
        str,
        typer.Option(
            help="Path to preprocessed dataset (produced by prepare-data)",
        ),
    ] = "./output",
    max_samples: Annotated[
        int | None,
        typer.Option(help="Maximum number of samples to process"),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            help=(
                "Directory to save generated hidden states files "
                "(default: {preprocessed-data}/hidden_states)"
            ),
        ),
    ] = None,
    concurrency: Annotated[
        int,
        typer.Option(
            help=(
                "Number of active vLLM requests at a time. "
                "Note: number of async workers set to 2*concurrency"
            ),
        ),
    ] = 32,
    write_concurrency: Annotated[
        int,
        typer.Option(
            min=1,
            help="Concurrent validation/file publications (independent of requests)",
        ),
    ] = 2,
    progress_log_interval: Annotated[
        float,
        typer.Option(min=0.1, help="Seconds between console progress summaries"),
    ] = 10.0,
    trust_existing_outputs: Annotated[
        bool,
        typer.Option(help="Reuse saved files by row ID without opening their payloads"),
    ] = False,
    validate_outputs: Annotated[
        bool,
        typer.Option(
            "--validate-outputs",
            help=(
                "Load generated safetensor files and check output token ids "
                "match prompt tokens and hidden states seq_len matches num tokens"
            ),
        ),
    ] = False,
    request_timeout: Annotated[
        float,
        typer.Option(
            help="Timeout in seconds for each individual vLLM request",
        ),
    ] = DEFAULT_REQUEST_TIMEOUT,
    max_retries: Annotated[
        int,
        typer.Option(
            help="Maximum number of retry attempts per request on failure",
        ),
    ] = DEFAULT_MAX_RETRIES,
    fail_on_error: Annotated[
        bool,
        typer.Option(
            "--fail-on-error",
            help=(
                "Abort when a request fails after all retries. "
                "By default, failed samples are skipped."
            ),
        ),
    ] = False,
    max_consecutive_errors: Annotated[
        int | None,
        typer.Option(
            help=(
                "Abort after this many consecutive sample failures (each sample "
                "already retried --max-retries times). Prevents silently churning "
                "through the entire dataset when the server is down. "
                "Ignored when --fail-on-error is set. "
                "(default: value of --concurrency)"
            ),
        ),
    ] = None,
    world_size: Annotated[
        int,
        typer.Option(
            help=(
                "World size for multi-node data generation offline. "
                "This is the number of nodes (not GPUs)."
            ),
        ),
    ] = 1,
    rank: Annotated[
        int,
        typer.Option(
            help=(
                "Rank for multi-node data generation offline. "
                "This is the node index, not a GPU index. "
                "Must be in range [0, world_size)."
            ),
        ),
    ] = 0,
) -> None:
    """Generate hidden states offline from a vLLM server.

    Connects to a running vLLM instance, sends preprocessed samples, and saves
    the extracted hidden states to disk for offline training.
    """
    if concurrency < 1 or write_concurrency < 1:
        raise typer.BadParameter("--concurrency and --write-concurrency must be >= 1")
    if rank < 0 or rank >= world_size:
        raise typer.BadParameter("--rank must be in range [0, world_size)")
    _configure_extraction_logger()

    try:
        with _StartupWatchdog() as watchdog:
            logger.info(
                "EAGLE Offline Data Generation; startup stacks after 30s if pending"
            )
            logger.info(
                "Starting extraction event loop; data=%s; endpoint=%s",
                preprocessed_data,
                endpoint,
            )
            asyncio.run(
                _generate_and_save_hidden_states(
                    model=model,
                    endpoint=endpoint,
                    preprocessed_data=preprocessed_data,
                    output=output,
                    max_samples=max_samples,
                    concurrency=concurrency,
                    write_concurrency=write_concurrency,
                    progress_log_interval=progress_log_interval,
                    on_startup_complete=watchdog.finish,
                    validate_outputs=validate_outputs,
                    trust_existing_outputs=trust_existing_outputs,
                    request_timeout=request_timeout,
                    max_retries=max_retries,
                    fail_on_error=fail_on_error,
                    max_consecutive_errors=max_consecutive_errors,
                    world_size=world_size,
                    rank=rank,
                )
            )
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception:
        logger.exception("Data generation failed")
        sys.exit(1)

    logger.info("Data generation complete!")
