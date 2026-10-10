"""Stable bank identities and atomic metadata writes."""

import hashlib
import json
import shutil
from pathlib import Path

from speculators.provenance import atomic_write

PROVENANCE_FILES = (
    "train_command.txt",
    "speculators.patch",
    "run.yaml",
    "resolved_train.json",
    "bank_context.json",
)


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_provenance(source: Path, destination: Path) -> None:
    for name in PROVENANCE_FILES:
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)


def load_subset(manifest_path: Path, subset_id: str):
    manifest = json.loads(manifest_path.read_text())
    subset = next((s for s in manifest["subsets"] if s["id"] == subset_id), None)
    if subset is None:
        raise ValueError(f"Unknown bank subset: {subset_id}")
    train, val = subset["train_indices"], subset["validation_indices"]
    if not train or not val or set(train) & set(val):
        raise ValueError("Bank training and validation memberships must be disjoint")
    if len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Bank memberships contain duplicate prepared rows")
    return manifest, subset


def read_jsonl(path: Path) -> list[dict]:
    """Read newline-delimited records without splitting Unicode text separators."""
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
