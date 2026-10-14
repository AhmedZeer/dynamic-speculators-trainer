"""Parallel, pinned activation downloads without rewriting tensor payloads."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

from speculators.bank.artifacts import file_digest
from speculators.bank.hub import hub_progress
from speculators.bank.progress import report, stage_progress


def stage_hidden(cfg, local_dir, subsets, workers=8):  # noqa: C901 -- select, budget and verify immutable inputs
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
        for number in subset["train_indices"] + subset["validation_indices"]:
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
        path = local_dir / ref["path"]
        if not path.is_file() or path.stat().st_size != ref["bytes"]:
            path = Path(
                hf_hub_download(
                    repo_id=ref["repo"],
                    repo_type="dataset",
                    revision=ref["revision"],
                    filename=ref["path"],
                    local_dir=local_dir,
                )
            )
        if path.stat().st_size != ref["bytes"] or file_digest(path) != ref["sha256"]:
            raise ValueError(f"Activation checksum mismatch: {path}")

    with (
        hub_progress(cfg) if cfg.hub else nullcontext(),
        stage_progress("Parallel activation staging"),
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        for number, _ in enumerate(pool.map(download, references.values()), 1):
            if number % 100 == 0 or number == len(references):
                report(f"Staged and verified {number}/{len(references)} files")
    training = cfg.model_copy(deep=True)
    training.execution.activation_local_dir = local_dir
    training.execution.training_subsets = [
        name for name in selected if available[name].get("role", "bank") == "bank"
    ]
    path = cfg.output_root / "raw_training.yaml"
    path.write_text(yaml.safe_dump(training.model_dump(mode="json"), sort_keys=False))
    return path
