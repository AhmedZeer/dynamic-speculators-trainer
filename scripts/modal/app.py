"""Modal batch runner. Importing this file never submits work.

Usage: modal run scripts/modal/app.py --command prepare
See examples/modal/README.md before launching paid compute.
"""

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal
import yaml

# Remote-only imports keep the local control plane free of GPU dependencies.
# ruff: noqa: PLC0415

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path("/artifacts")
MODEL_ROOT = Path("/models")
app = modal.App("speculators-prepare")
models = modal.Volume.from_name("speculators-models", create_if_missing=True)
artifacts = modal.Volume.from_name("speculators-data", create_if_missing=True)
# The published CUDA development image provides nvcc for FlashInfer JIT. The
# current source and connectors are mounted ahead of installed packages.
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "g++")
    .pip_install(
        "vllm==0.31.0",
        "datasets==5.0.1",
        "transformers==5.16.1",
        "pydantic==2.13.5",
        "pydantic-settings==2.15.0",
        "PyYAML==6.0.3",
        "openai==3.26.0",
        "ninja==1.13.2",
        "psutil==7.2.2",
        "loguru==0.7.3",
        "rich==15.0.0",
    )
    .run_commands("python -m pip check")
    .env(
        {
            "PYTHONPATH": "/repo/src:/repo/hs_connectors/src",
            "PYTHONUNBUFFERED": "1",
            "PYTHONNOUSERSITE": "1",
            "HF_HOME": "/models/hf",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_CACHE_ROOT": "/models/vllm",
            "XDG_CACHE_HOME": "/tmp/cache",  # noqa: S108 -- isolated container cache
            "OMP_NUM_THREADS": "1",
            "CUDA_HOME": "/usr/local/cuda",
        }
    )
    .add_local_dir(REPO / "src", "/repo/src", ignore=["**/__pycache__/**", "**/*.pyc"])
    .add_local_dir(
        REPO / "hs_connectors/src",
        "/repo/hs_connectors/src",
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
    .add_local_dir(
        REPO / "scripts", "/repo/scripts", ignore=["**/__pycache__/**", "**/*.pyc"]
    )
)
mounts = {"/models": models, "/artifacts": artifacts}


@app.function(
    image=image, volumes=mounts, cpu=4, memory=65536, timeout=86400, max_containers=1
)
def initialize(values, provenance):
    from speculators.data_generation.pipeline import PrepareConfig
    from speculators.data_generation.pipeline import initialize as select

    cfg = PrepareConfig.model_validate(values)
    artifacts.reload()
    models.reload()
    result = select(cfg, ARTIFACT_ROOT / cfg.run_id, MODEL_ROOT / "targets", provenance)
    models.commit()
    artifacts.commit()
    return result


@app.function(
    image=image,
    volumes=mounts,
    gpu="L40S",
    cpu=4,
    memory=65536,
    timeout=30000,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=1,
    retries=0,
)
def prepare(values, domains, subsets, provenance):  # noqa: C901 -- serial stages and recovery

    from speculators.bank.artifacts import write_json
    from speculators.data_generation.pipeline import (
        PrepareConfig,
        bank_status,
        prepare_subset,
    )

    cfg = PrepareConfig.model_validate(values)
    root = ARTIFACT_ROOT / cfg.run_id
    artifacts.reload()
    models.reload()
    selection = json.loads((root / "selection.json").read_text())
    if selection["identity"] != cfg.identity():
        raise ValueError("Configuration differs from selected data; use a new run_id")
    ledger_path = root / "gpu_runtime.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else {"seconds": 0, "attempts": 0}
    )
    previous = ledger["seconds"]
    limit = cfg.gpu_hours_limit * 3600
    if previous >= limit:
        raise RuntimeError(
            "Cumulative GPU-time guard reached. Inspect status and billing "
            "before increasing gpu_hours_limit."
        )
    started = time.monotonic()
    deadline = started + limit - previous - 20  # leave time to shut down and commit
    ledger["attempts"] += 1
    attempt = root / "provenance" / f"attempt-{ledger['attempts']}"
    write_json(attempt / "source.json", provenance)
    write_json(attempt / "config.json", cfg.model_dump(mode="json"))
    import sys

    (attempt / "requirements.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )
    lock = threading.Lock()
    done = threading.Event()
    errors = []
    peak_gpu_mib = 0
    peak_rss = 0

    def commit():
        # Reserve the next minute, so a killed attempt cannot lose unbounded
        # runtime accounting. Failed attempts retain this conservative reserve.
        with lock:
            ledger["seconds"] = min(limit, previous + time.monotonic() - started + 60)
            write_json(ledger_path, ledger)
            models.commit()
            artifacts.commit()

    def heartbeat():
        nonlocal peak_gpu_mib, peak_rss
        import psutil

        while not done.wait(30):
            try:
                commit()
                current = psutil.Process(os.getpid())
                peak_rss = max(
                    peak_rss,
                    sum(
                        p.memory_info().rss
                        for p in [current, *current.children(recursive=True)]
                        if p.is_running()
                    ),
                )
                memory = subprocess.check_output(  # noqa: S603 -- fixed diagnostic command
                    [
                        shutil.which("nvidia-smi") or "/usr/bin/nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                peak_gpu_mib = max(peak_gpu_mib, int(memory.strip().splitlines()[0]))
            except Exception as error:  # noqa: BLE001 -- best-effort diagnostics
                errors.append(str(error))
                # Periodic diagnostics cannot invalidate completed artifacts.
                print(f"Progress heartbeat failed: {error}", flush=True)

    commit()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    results = []
    try:
        for entry in selection["banks"]:
            if domains and entry["domain"] not in domains:
                continue
            if subsets and entry["subset"] not in subsets:
                continue
            if bank_status(entry["root"])["complete"]:
                results.append(bank_status(entry["root"]))
                continue
            print(f"Preparing {entry['domain']} subset {entry['subset']}", flush=True)
            results.append(
                prepare_subset(
                    cfg,
                    entry["root"],
                    selection["model_path"],
                    "/repo",
                    deadline,
                    commit=commit,
                )
            )
        return results
    finally:
        done.set()
        thread.join()
        with lock:
            ledger["seconds"] = previous + time.monotonic() - started
            write_json(ledger_path, ledger)
            write_json(
                root / "last_attempt.json",
                {
                    "elapsed_seconds": time.monotonic() - started,
                    "cumulative_gpu_seconds": ledger["seconds"],
                    "results": results,
                    "peak_gpu_mib": peak_gpu_mib,
                    "peak_process_rss_bytes": peak_rss,
                    "heartbeat_errors": errors,
                },
            )
            models.commit()
            artifacts.commit()


@app.function(
    image=image, volumes={"/artifacts": artifacts}, cpu=1, memory=4096, timeout=300
)
def status(run_id):
    from speculators.data_generation.pipeline import bank_status

    artifacts.reload()
    root = ARTIFACT_ROOT / run_id
    selection = root / "selection.json"
    if not selection.exists():
        return {"run_id": run_id, "selected": False}
    value = json.loads(selection.read_text())
    ledger = root / "gpu_runtime.json"
    return {
        "run_id": run_id,
        "selected": True,
        "gpu_runtime": json.loads(ledger.read_text()) if ledger.exists() else {},
        "banks": [bank_status(e["root"]) for e in value["banks"]],
    }


def source_provenance():
    def git(*args):
        return subprocess.check_output(  # noqa: S603 -- internally constructed git argv
            [shutil.which("git") or "/usr/bin/git", "-C", str(REPO), *args], text=True
        )

    # Include new source files too; `git diff` alone omits untracked additions.
    untracked = {}
    for path in git("ls-files", "--others", "--exclude-standard").splitlines():
        if path.startswith(("src/", "scripts/modal/", "examples/modal/")):
            untracked[path] = (REPO / path).read_text()
    return {
        "git_sha": git("rev-parse", "HEAD").strip(),
        "patch": git(
            "diff", "HEAD", "--", "src", "scripts", "hs_connectors", "pyproject.toml"
        ),
        "untracked_sources": untracked,
    }


@app.local_entrypoint()
def main(  # noqa: C901, PLR0917 -- Modal CLI parameters
    command: str = "status",
    config: str = "examples/modal/prepare.yaml",
    domains: str = "",
    subsets: str = "",
    destination: str = "output/modal",
    smoke: bool = False,
):
    values = yaml.safe_load(Path(config).read_text())
    run_id = values.get("run_id", "nemotron-v1")
    if not run_id or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        for c in run_id
    ):
        raise ValueError("Invalid run_id")
    selected_domains = [d for d in domains.split(",") if d]
    selected_subsets = [int(s) for s in subsets.split(",") if s]
    if set(selected_domains) - set(
        values.get("domains", ["chat", "math", "code", "stem"])
    ):
        raise ValueError("Requested domain is not in the configured selection")
    if set(selected_subsets) - set(values.get("subset_ids", list(range(6)))):
        raise ValueError("Requested subset is not in the configured selection")
    if command == "status":
        print(json.dumps(status.remote(run_id), indent=2))
    elif command == "prepare":
        if smoke:
            # A separate tiny experiment; never truncate production manifests.
            values = json.loads(json.dumps(values))
            values["run_id"] = run_id + "-smoke"
            values["subset_ids"] = [0]
            values.setdefault("bank", {}).setdefault("partition", {}).update(
                train_examples=2, validation_examples=1
            )
            values["bank"].setdefault("conditioning", {})["max_examples"] = 2
            values["bank"].setdefault("responses", {}).update(max_tokens=32)
            values["batch_size"] = 2
            selected_subsets = [0]
        initialize.remote(values, source_provenance())
        report = status.remote(values["run_id"])
        pending = [
            entry
            for entry in report["banks"]
            if not entry["complete"]
            and (not selected_domains or entry["domain"] in selected_domains)
            and (
                not selected_subsets
                or int(entry["subset"].rsplit("-", 1)[1]) in selected_subsets
            )
        ]
        if not pending:
            print(json.dumps(report, indent=2))
            return
        if (
            report["gpu_runtime"].get("seconds", 0)
            >= values.get("gpu_hours_limit", 8) * 3600
        ):
            raise RuntimeError(
                "Cumulative GPU-time guard reached; no GPU was requested."
            )
        print(
            json.dumps(
                prepare.remote(
                    values, selected_domains, selected_subsets, source_provenance()
                ),
                indent=2,
            )
        )
    elif command == "download":
        # Stream directly from the Volume without a persistent activation tar copy.
        output = Path(destination) / run_id
        for entry in artifacts.iterdir(run_id, recursive=True):
            if entry.type == modal.volume.FileEntryType.FILE:
                relative = Path(entry.path.lstrip("/")).relative_to(run_id)
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                pending = target.with_suffix(target.suffix + ".pending")
                with pending.open("wb") as handle:
                    for chunk in artifacts.read_file(entry.path):
                        handle.write(chunk)
                pending.replace(target)
        print(
            f"Downloaded to {output}; manifest paths still refer "
            "to the stable Modal mounts."
        )
    else:
        raise ValueError("command must be prepare, status, or download")
