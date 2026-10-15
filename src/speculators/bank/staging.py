"""Parallel, pinned activation downloads without rewriting tensor payloads."""

import json
import random
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path

import httpx
import yaml
from huggingface_hub import hf_hub_download

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.hub import hub_progress
from speculators.bank.progress import report, stage_progress


def _download_with_retries(ref, local_dir, on_retry=None):
    """Retry transient Hub failures while preserving resumable local downloads."""
    attempts = 6
    for attempt in range(1, attempts + 1):
        try:
            return Path(
                hf_hub_download(
                    repo_id=ref["repo"],
                    repo_type="dataset",
                    revision=ref["revision"],
                    filename=ref["path"],
                    local_dir=local_dir,
                )
            )
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            if isinstance(
                exc, httpx.HTTPStatusError
            ) and exc.response.status_code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == attempts:
                raise
            if on_retry is not None:
                on_retry()
            delay = min(2**attempt, 30) + random.uniform(0, 1)
            report(
                f"Activation download failed ({type(exc).__name__}): {ref['path']}; "
                f"retry {attempt}/{attempts - 1} in {delay:.1f}s"
            )
            time.sleep(delay)

    raise RuntimeError("Activation download retry loop exhausted")


def staging_plan(cfg, local_dir, subsets, include_validation=True):
    """Resolve immutable inputs and remaining bytes without downloading."""
    manifest = json.loads((cfg.output_root / "manifest.json").read_text())
    index = json.loads(
        (Path(manifest["hidden_states_path"]) / "index.json").read_text()
    )
    available = {s["id"]: s for s in manifest["subsets"]}
    selected = list(available) if subsets == ["all"] else subsets
    if not selected or set(selected) - available.keys():
        raise ValueError(f"Unknown/empty subset selection: {selected}")
    references = {}
    for name in selected:
        subset = available[name]
        indices = subset["train_indices"] + (
            subset["validation_indices"] if include_validation else []
        )
        for number in indices:
            row = index["rows"][str(number)]
            for kind in ("shard", "summary"):
                if kind in row:
                    ref = row[kind]
                    key = ref["path"]
                    if key in references and references[key] != ref:
                        raise ValueError(f"Conflicting activation reference: {key}")
                    references[key] = ref
    local_dir = local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for ref in references.values():
        path = (local_dir / ref["path"]).resolve()
        if not path.is_relative_to(local_dir):
            raise ValueError("Activation path escapes download folder")
        if not path.is_file() or path.stat().st_size != ref["bytes"]:
            missing_bytes += ref["bytes"]
    return selected, references, missing_bytes


def can_stage(cfg, local_dir, subsets, include_validation=True):
    _, _, missing = staging_plan(cfg, local_dir, subsets, include_validation)
    reserve = int(cfg.hub.cache_reserve_gib * 2**30) if cfg.hub else 0
    return shutil.disk_usage(local_dir).free >= missing + reserve


def stage_hidden(cfg, local_dir, subsets, workers=8, include_validation=True):  # noqa: C901 -- download/verify pipeline
    local_dir = local_dir.resolve()
    selected, references, missing_bytes = staging_plan(
        cfg, local_dir, subsets, include_validation
    )
    manifest = json.loads((cfg.output_root / "manifest.json").read_text())
    available = {s["id"]: s for s in manifest["subsets"]}
    reserve = int(cfg.hub.cache_reserve_gib * 2**30) if cfg.hub else 0
    if shutil.disk_usage(local_dir).free < missing_bytes + reserve:
        raise OSError(
            f"Staging needs {missing_bytes / 2**30:.1f} GiB plus "
            f"{reserve / 2**30:g} GiB reserve; select fewer subsets"
        )
    report(
        f"Staging {len(references)} files, {missing_bytes / 2**30:.1f} GiB "
        f"to download; workers={workers}; subsets={selected}"
    )

    def download(ref):
        start = time.perf_counter()
        path = local_dir / ref["path"]
        if not path.is_file() or path.stat().st_size != ref["bytes"]:
            path = _download_with_retries(ref, local_dir)
        return ref, path, time.perf_counter() - start

    def verify(result):
        ref, path, download_seconds = result
        start = time.perf_counter()
        stat = path.stat()
        identity = {
            "reference": ref,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "inode": stat.st_ino,
        }
        receipt = local_dir / ".verified" / (digest(ref["path"]) + ".json")
        try:
            reused = json.loads(receipt.read_text()) == identity
        except (OSError, ValueError):
            reused = False
        if not reused and (
            stat.st_size != ref["bytes"] or file_digest(path) != ref["sha256"]
        ):
            raise ValueError(f"Activation checksum mismatch: {path}")
        if not reused:
            after = path.stat()
            if (after.st_mtime_ns, after.st_ctime_ns, after.st_size) != (
                stat.st_mtime_ns,
                stat.st_ctime_ns,
                stat.st_size,
            ):
                raise ValueError(f"Activation changed during verification: {path}")
            write_json(receipt, identity)
        return download_seconds, time.perf_counter() - start, reused

    start = time.perf_counter()
    download_seconds = verification_seconds = reused = 0
    with (
        hub_progress(cfg) if cfg.hub else nullcontext(),
        stage_progress("Parallel activation staging"),
        ThreadPoolExecutor(max_workers=workers) as pool,
        ThreadPoolExecutor(max_workers=min(workers, 2)) as verification_pool,
    ):
        downloads = {pool.submit(download, ref) for ref in references.values()}
        pending = set(downloads)
        number = 0
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                if future in downloads:
                    downloads.remove(future)
                    pending.add(verification_pool.submit(verify, future.result()))
                    continue
                download_time, verify_time, cached = future.result()
                download_seconds += download_time
                verification_seconds += verify_time
                reused += cached
                number += 1
                if number % 100 == 0 or number == len(references):
                    report(f"Staged and verified {number}/{len(references)} files")
    elapsed = time.perf_counter() - start
    report(
        f"Staging timing: wall={elapsed:.1f}s; download worker-seconds="
        f"{download_seconds:.1f}; "
        f"verification worker-seconds={verification_seconds:.1f}; "
        f"reused checksums={reused}/{len(references)}; "
        f"effective throughput={missing_bytes / 2**20 / max(elapsed, 0.001):.1f} MiB/s"
    )
    training = cfg.model_copy(deep=True)
    training.execution.activation_local_dir = local_dir
    training.execution.training_subsets = [
        name for name in selected if available[name].get("role", "bank") == "bank"
    ]
    path = cfg.output_root / "raw_training.yaml"
    path.write_text(yaml.safe_dump(training.model_dump(mode="json"), sort_keys=False))
    return path
