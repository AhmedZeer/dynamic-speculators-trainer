"""Platform-independent, resumable preparation of a multi-domain EAGLE3 corpus.

Each subset is an ordinary bank directory. ``selection.json`` links those banks
and maps their local Arrow/hidden-state indices to immutable global row indices.
There is deliberately no training or Modal dependency in this module.
"""

# Dependencies for individual stages are imported only when needed.
# ruff: noqa: PLC0415

import argparse
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from http import HTTPStatus
from pathlib import Path

import yaml
from pydantic import Field, model_validator

from speculators.bank.artifacts import digest, file_digest, read_jsonl, write_json
from speculators.bank.config import BankConfig, Settings
from speculators.bank.workflow import cache_spec, prompt_messages
from speculators.provenance import atomic_write

DOMAINS = ("chat", "math", "code", "stem")


class PrepareConfig(Settings):
    run_id: str = "nemotron-v1"
    domains: list[str] = Field(default_factory=lambda: list(DOMAINS))
    subset_ids: list[int] = Field(default_factory=lambda: list(range(6)))
    batch_size: int = Field(default=100, ge=1)
    gpu_hours_limit: float = Field(default=8, gt=0)
    bank: BankConfig = Field(default_factory=BankConfig)

    @model_validator(mode="after")
    def valid(self):
        if not self.run_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in self.run_id
        ):
            raise ValueError("run_id must contain only letters, digits, '-' and '_'")
        if (
            not self.domains
            or len(set(self.domains)) != len(self.domains)
            or set(self.domains) - set(DOMAINS)
        ):
            raise ValueError(
                "domains must be a nonempty, unique selection of supported domains"
            )
        if (
            not self.subset_ids
            or min(self.subset_ids) < 0
            or len(set(self.subset_ids)) != len(self.subset_ids)
        ):
            raise ValueError("subset_ids must be nonempty, unique and nonnegative")
        return self

    @classmethod
    def load(cls, path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def identity(self):
        # Operational controls can change without altering selected examples.
        values = self.bank.model_dump(mode="json")
        for key in ("output_root", "execution", "training", "conditioning", "lora"):
            values.pop(key)
        values["partition"]["subset_ids"] = sorted(self.subset_ids)
        return digest(
            {"bank": values, "domains": [d for d in DOMAINS if d in self.domains]}
        )


def selected_rows(rows, cfg, excluded, scratch):
    """Disk-backed deduplication; only IDs and selected prompts occupy RAM."""
    with sqlite3.connect(scratch) as db:
        db.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY, payload TEXT)")
        db.execute(
            "CREATE TABLE aliases (id TEXT, source TEXT, PRIMARY KEY(id, source))"
        )
        for index, row in enumerate(rows):
            if (
                cfg.dataset.sample_limit is not None
                and index >= cfg.dataset.sample_limit
            ):
                break
            messages = prompt_messages(row)
            identity = digest(messages)
            if identity in excluded:
                continue
            source = str(row.get("uuid") or row.get("id") or digest(row))
            db.execute(
                "INSERT OR IGNORE INTO prompts VALUES (?, ?)",
                (identity, json.dumps(messages)),
            )
            db.execute(
                "INSERT OR IGNORE INTO aliases VALUES (?, ?)", (identity, source)
            )
            if index % 10000 == 0:
                db.commit()
        ids = [row[0] for row in db.execute("SELECT id FROM prompts ORDER BY id")]
        random.Random(cfg.partition.seed).shuffle(ids)
        size = cfg.partition.train_examples + cfg.partition.validation_examples
        if max(cfg.partition.subset_ids) >= len(ids) // size:
            raise ValueError(
                f"Not enough distinct {cfg.dataset.category} prompts "
                "for requested subsets"
            )
        result = []
        for subset in cfg.partition.subset_ids:
            for identity in ids[subset * size : (subset + 1) * size]:
                messages = json.loads(
                    db.execute(
                        "SELECT payload FROM prompts WHERE id=?", (identity,)
                    ).fetchone()[0]
                )
                aliases = [
                    r[0]
                    for r in db.execute(
                        "SELECT source FROM aliases WHERE id=? ORDER BY source",
                        (identity,),
                    )
                ]
                result.append(
                    {
                        "id": identity,
                        "messages": messages,
                        "source_ids": aliases,
                        "subset_id": f"{cfg.dataset.category}-{subset:05d}",
                    }
                )
        return result, {
            "unique_prompts": len(ids),
            "complete_subsets": len(ids) // size,
        }


def initialize(cfg, root, model_root, provenance, *, loader=None, revisions=None):
    """Select once on CPU; publish per-subset bank manifests with pinned revisions."""
    root, model_root = Path(root), Path(model_root)
    root.mkdir(parents=True, exist_ok=True)
    selection = root / "selection.json"
    if selection.exists():
        value = json.loads(selection.read_text())
        if value["identity"] != cfg.identity():
            raise ValueError("Selection differs: use a new run_id")
        return value
    if revisions is None:
        from huggingface_hub import HfApi, snapshot_download

        api = HfApi()
        revisions = {
            "target_sha": api.model_info(
                cfg.bank.models.target, revision=cfg.bank.models.target_revision
            ).sha,
            "drafter_sha": api.model_info(
                cfg.bank.models.drafter, revision=cfg.bank.models.drafter_revision
            ).sha,
            "dataset_sha": api.dataset_info(
                cfg.bank.dataset.source, revision=cfg.bank.dataset.revision
            ).sha,
        }
        snapshot_download(
            cfg.bank.models.target,
            revision=revisions["target_sha"],
            local_dir=model_root / revisions["target_sha"],
        )
    if loader is None:
        from datasets import load_dataset

        def loader(domain):
            return load_dataset(
                cfg.bank.dataset.source,
                name=cfg.bank.dataset.configuration,
                split=domain,
                revision=revisions["dataset_sha"],
                streaming=True,
            )

    excluded, entries = set(), []
    size = cfg.bank.partition.train_examples + cfg.bank.partition.validation_examples
    for domain in DOMAINS:
        if domain not in cfg.domains:
            continue
        bank = cfg.bank.model_copy(deep=True)
        bank.dataset.category = domain
        bank.dataset.split = domain
        bank.partition.subset_ids = sorted(cfg.subset_ids)
        with tempfile.TemporaryDirectory(prefix="speculators-selection-") as local:
            rows, stats = selected_rows(
                loader(domain), bank, excluded, Path(local) / "selection.db"
            )
        excluded.update(row["id"] for row in rows)
        for offset, subset in enumerate(bank.partition.subset_ids):
            subset_id = f"{domain}-{subset:05d}"
            directory = root / "banks" / domain / subset_id
            directory.mkdir(parents=True, exist_ok=True)
            subset_cfg = bank.model_copy(deep=True)
            subset_cfg.output_root = directory
            subset_cfg.partition.subset_ids = [subset]
            group = rows[offset * size : (offset + 1) * size]
            for index, row in enumerate(group):
                row["global_index"] = (
                    DOMAINS.index(domain) * (max(cfg.subset_ids) + 1) * size
                    + subset * size
                    + index
                )
            prompts = directory / "prompts.jsonl"
            atomic_write(prompts, "".join(json.dumps(row) + "\n" for row in group))
            # Preserve the ordinary bank schema; subset membership is explicit.
            train = subset_cfg.partition.train_examples
            spec = cache_spec(subset_cfg, revisions)
            manifest = {
                "version": 1,
                "category": domain,
                "cache_spec": spec,
                "cache_fingerprint": digest(spec),
                "revisions": revisions,
                "root": str(directory),
                "data_path": str(directory / "data"),
                "hidden_states_path": str(directory / "hidden_states"),
                "subsets": [
                    {
                        "id": subset_id,
                        "train_ids": [r["id"] for r in group[:train]],
                        "validation_ids": [r["id"] for r in group[train:]],
                        "train_indices": list(range(train)),
                        "validation_indices": list(range(train, size)),
                    }
                ],
                "partition_stats": stats,
                "prompts_sha256": file_digest(prompts),
                "prepared": False,
                "global_indices": [r["global_index"] for r in group],
            }
            write_json(directory / "manifest.json", manifest)
            atomic_write(
                directory / "bank.yaml",
                yaml.safe_dump(subset_cfg.model_dump(mode="json")),
            )
            entries.append(
                {
                    "domain": domain,
                    "subset": subset,
                    "root": str(directory),
                    "examples": size,
                }
            )
    value = {
        "version": 1,
        "identity": cfg.identity(),
        "revisions": revisions,
        "model_path": str(model_root / revisions["target_sha"]),
        "banks": sorted(
            entries, key=lambda e: (e["subset"], DOMAINS.index(e["domain"]))
        ),
        "provenance": provenance,
    }
    write_json(selection, value)
    write_json(root / "source.json", provenance)
    atomic_write(root / "speculators.patch", provenance.get("patch", ""))
    return value


def repair_responses(path, expected):
    """Recover a torn final append; reject corruption or duplicate identities."""
    path = Path(path)
    if not path.exists():
        return []
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    valid, seen = [], set()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break
            raise ValueError(f"Corrupt response record {index} in {path}") from None
        primary = row["primary_id"]
        if primary not in expected or primary in seen:
            raise ValueError(f"Unexpected or duplicate response identity: {primary}")
        seen.add(primary)
        valid.append(row)
    atomic_write(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in valid))
    return valid


def server_command(repo, bank, model_path, stage, local):
    command = [
        sys.executable,
        str(Path(repo) / "scripts/launch_vllm.py"),
        "responses" if stage == "responses" else "train",
        str(model_path),
        "--provenance-dir",
        str(bank.output_root / "provenance" / stage),
        "--no-hash-checkpoints",
    ]
    if stage == "hidden":
        command += [
            "--target-layer-ids",
            *map(str, bank.hidden_states.layer_ids),
            "--hidden-states-path",
            str(local),
        ]
    command += [
        "--",
        "--served-model-name",
        bank.models.target,
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(
            bank.responses.context_length
            if stage == "responses"
            else bank.hidden_states.sequence_length + 1
        ),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.85",
        "--max-num-seqs",
        "32",
        "--port",
        "8000",
    ]
    return command


def stop_process(process):
    """Stop the complete server/worker group, including orphaned descendants."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


@contextmanager
def server(command, log, deadline):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "OMP_NUM_THREADS": "1",
    }
    with log.open("a") as stream:
        process = subprocess.Popen(  # noqa: S603 -- internally constructed argv
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited: inspect {log}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "GPU runtime limit reached during server startup"
                    )
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:8000/health", timeout=2
                    ) as response:
                        if response.status == HTTPStatus.OK:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            yield
        finally:
            stop_process(process)


def run_child(command, deadline, commit, interval=30):
    process = subprocess.Popen(command, start_new_session=True)  # noqa: S603 -- internal argv
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("GPU runtime limit reached; rerun to resume")
            try:
                code = process.wait(timeout=min(interval, remaining))
            except subprocess.TimeoutExpired:
                commit()
                continue
            if code:
                raise subprocess.CalledProcessError(code, command)
            return
    finally:
        stop_process(process)
        commit()


def response_lines(path):
    with path.open(encoding="utf-8") as stream:
        yield from stream


def bank_status(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    prompts = read_jsonl(directory / "prompts.jsonl")
    response_ids = set()
    responses = directory / "responses.jsonl"
    if responses.exists():
        for line in response_lines(responses):
            try:
                response_ids.add(json.loads(line)["primary_id"])
            except json.JSONDecodeError:
                pass
    hidden = sum(
        (directory / "hidden_states" / f"hs_{i}.safetensors").is_file()
        for i in range(len(prompts))
    )
    return {
        "domain": manifest["category"],
        "subset": manifest["subsets"][0]["id"],
        "expected": len(prompts),
        "responses": len(response_ids & {p["id"] for p in prompts}),
        "hidden_states": hidden,
        "complete": (directory / "complete.json").exists()
        and hidden == len(prompts)
        and response_ids == {p["id"] for p in prompts}
        and manifest.get("prepared", False)
        and Path(manifest["data_path"]).exists(),
    }


@contextmanager
def stage_metrics(root, stage, commit):
    """Persist stage throughput, including partial progress after failures."""
    path = root / "metrics.json"
    metrics = json.loads(path.read_text()) if path.exists() else {}
    before = bank_status(root)
    started = time.monotonic()
    try:
        yield
    finally:
        after = bank_status(root)
        key = "responses" if stage == "responses" else "hidden_states"
        elapsed = time.monotonic() - started
        saved = after[key] - before[key]
        item = metrics.setdefault(stage, {"seconds": 0, "rows_saved": 0, "attempts": 0})
        item["seconds"] += elapsed
        item["rows_saved"] += saved
        item["attempts"] += 1
        item["rows_per_second"] = item["rows_saved"] / max(item["seconds"], 1e-9)
        if stage == "responses":
            responses = root / "responses.jsonl"
            tokens = 0
            if responses.exists():
                for line in response_lines(responses):
                    try:
                        tokens += sum(json.loads(line)["loss_mask"])
                    except json.JSONDecodeError:
                        continue
            item["generated_tokens"] = tokens
            item["tokens_per_second"] = tokens / max(item["seconds"], 1e-9)
        write_json(path, metrics)
        commit()


def prepare_subset(
    config, directory, model_path, repo, deadline, *, commit=lambda: None
):
    """Bounded response batches followed by Arrow preparation and extraction."""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    bank = BankConfig.load(Path(directory) / "bank.yaml")
    bank.execution = config.bank.execution.model_copy(deep=True)
    bank.execution.response_staging_dir = (
        None  # append to Volume; repair torn tail on resume
    )
    bank.execution.prompt_chunk_size = config.batch_size
    bank.execution.response_sync_interval = config.batch_size
    root = bank.output_root
    prompts = read_jsonl(root / "prompts.jsonl")
    expected = {p["id"] for p in prompts}
    recovered = repair_responses(root / "responses.jsonl", expected)
    command_base = [sys.executable, "-m", "speculators.data_generation.pipeline"]
    with tempfile.TemporaryDirectory(prefix="speculators-prepare-") as tmp:
        local = Path(tmp)
        # Only execution settings change; the original cache identity stays intact.
        effective = local / "bank.yaml"
        effective.write_text(yaml.safe_dump(bank.model_dump(mode="json")))
        if len(recovered) != len(prompts):
            missing = [
                p
                for p in prompts
                if p["id"] not in {r["primary_id"] for r in recovered}
            ]
            with (
                stage_metrics(root, "responses", commit),
                server(
                    server_command(repo, bank, model_path, "responses", local),
                    root / "provenance/responses/server.log",
                    deadline,
                ),
            ):
                for start in range(0, len(missing), config.batch_size):
                    batch = local / "prompts.jsonl"
                    batch.write_text(
                        "".join(
                            json.dumps(p) + "\n"
                            for p in missing[start : start + config.batch_size]
                        )
                    )
                    run_child(
                        command_base
                        + [
                            "responses",
                            "--config",
                            str(effective),
                            "--prompts",
                            str(batch),
                        ],
                        deadline,
                        commit,
                    )
                    repair_responses(root / "responses.jsonl", expected)
                    commit()
        run_child(command_base + ["data", "--config", str(effective)], deadline, commit)
        manifest = json.loads((root / "manifest.json").read_text())
        status = bank_status(root)
        if status["hidden_states"] != status["expected"]:
            with (
                stage_metrics(root, "hidden", commit),
                server(
                    server_command(repo, bank, model_path, "hidden", local),
                    root / "provenance/hidden/server.log",
                    deadline,
                ),
            ):
                run_child(
                    command_base + ["hidden", "--config", str(effective)],
                    deadline,
                    commit,
                )
        # Header validation checks every token sequence and expected layer shape.
        from datasets import load_from_disk
        from safetensors import safe_open

        from speculators.data_generation.offline import (
            check_hidden_state_file_header,
        )

        width = json.loads((Path(model_path) / "config.json").read_text())[
            "hidden_size"
        ]
        data = load_from_disk(manifest["data_path"]).with_format(None)
        for index, row in enumerate(data):
            path = root / "hidden_states" / f"hs_{index}.safetensors"
            check_hidden_state_file_header(path, row["input_ids"])
            with safe_open(path, framework="pt", device="cpu") as tensors:
                hidden_slice = tensors.get_slice("hidden_states")
                shape = hidden_slice.get_shape()
                if hidden_slice.get_dtype() != "BF16":
                    raise ValueError("Expected bf16 hidden states")
                if shape != [
                    len(row["input_ids"]),
                    len(bank.hidden_states.layer_ids),
                    width,
                ]:
                    raise ValueError(f"Unexpected hidden-state shape: {shape}")
        write_json(
            root / "complete.json",
            {
                "examples": len(data),
                "tokens": manifest["tokens"],
                "hidden_state_bytes": sum(
                    p.stat().st_size
                    for p in (root / "hidden_states").glob("hs_*.safetensors")
                ),
            },
        )
        commit()
    return bank_status(root)


def main():
    parser = argparse.ArgumentParser(
        description="Internal subprocess stages; no Modal dependency"
    )
    parser.add_argument("stage", choices=["responses", "data", "hidden"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--prompts", type=Path)
    args = parser.parse_args()
    bank = BankConfig.load(args.config)
    manifest = json.loads((bank.output_root / "manifest.json").read_text())
    if args.stage == "responses":
        from speculators.bank.workflow import regenerate

        regenerate(bank, manifest, prompts_path=args.prompts)
    else:
        from speculators.bank.workflow import extract, prepare_arrow

        (extract if args.stage == "hidden" else prepare_arrow)(bank, manifest)


if __name__ == "__main__":
    main()
