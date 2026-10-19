"""Local token trajectories; no bank files, Hub calls, or archive extraction."""

import json
import shlex
from dataclasses import dataclass

import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets

from speculators.bank.artifacts import file_digest, write_json
from speculators.generator.campaign_config import DOMAINS
from speculators.generator.campaign_inputs import campaign_banks

ORIGINAL_COLUMNS = {
    "input_ids",
    "loss_mask",
    "seq_len",
    "primary_id",
    "source_ids",
    "category",
    "subset_id",
    "prompt_length",
    "metadata",
}


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Local dataset path escapes download folder: {relative}")
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing local dataset file: {path}; download it before training"
        )
    return path


@dataclass
class LocalInputs:
    banks: dict
    manifests: dict
    rows: Dataset
    subsets: dict
    identity: dict
    ram_cache: object = None


def trajectory_record_path(cfg, bank):
    """Explain how to repair a parquet-only download without fetching in training."""
    relative = f"releases/{bank.hub.release}/{bank.dataset.category}/data.json"
    try:
        return local_path(cfg.trajectories_path, relative)
    except FileNotFoundError as exc:
        command = shlex.join(
            [
                "hf",
                "download",
                bank.hub.trajectories_repo,
                *[
                    f"releases/{bank.hub.release}/{domain}/data.json"
                    for domain in DOMAINS
                ],
                "--revision",
                bank.hub.trajectories_revision,
                "--local-dir",
                str(cfg.trajectories_path.resolve()),
                "--repo-type",
                "dataset",
            ]
        )
        raise FileNotFoundError(
            f"Missing trajectory metadata: {relative}. The subset Parquet files "
            "alone are insufficient: data.json records their order, membership, "
            "and pinned model revisions. Download only the four metadata files "
            f"(existing Parquet files are kept), then rerun training:\n{command}"
        ) from exc


def read_trajectories(cfg, banks):  # noqa: C901 -- local membership validation
    manifests, subsets, tables, sources = {}, {}, [], []
    offset = 0
    for domain in DOMAINS:
        bank = banks[domain]
        record_path = trajectory_record_path(cfg, bank)
        record = json.loads(record_path.read_text())
        manifest = record["manifest"]
        spec = manifest["cache_spec"]
        if spec["hidden_states"] != bank.hidden_states.model_dump(mode="json"):
            raise ValueError(f"Hidden-state recipe differs from local data: {domain}")
        if any(
            spec["models"][role] != getattr(bank.models, role)
            for role in ("target", "drafter")
        ):
            raise ValueError(f"Model IDs differ from local trajectories: {domain}")
        manifests[domain] = manifest
        domain_tables = []
        for ref in record["artifacts"]:
            path = local_path(cfg.trajectories_path, ref["path"])
            if path.stat().st_size != ref["bytes"]:
                raise ValueError(f"Incomplete local trajectory file: {path}")
            table = pq.read_table(path)
            domain_tables.append(
                Dataset(
                    table.select(
                        [c for c in table.column_names if c in ORIGINAL_COLUMNS]
                    )
                )
            )
        rows = concatenate_datasets(domain_tables).with_format(None)
        for subset in manifest["subsets"]:
            name = subset["id"]
            if name in subsets:
                raise ValueError(f"Duplicate local subset: {name}")
            for split, count in (
                ("train", cfg.expected_train_examples),
                ("validation", cfg.expected_validation_examples),
            ):
                indices = subset[f"{split}_indices"]
                if len(indices) != count or len(set(indices)) != len(indices):
                    raise ValueError(f"Unexpected {split} membership: {name}")
                if min(indices) < 0 or max(indices) >= len(rows):
                    raise ValueError(f"Out-of-range {split} indices: {name}")
                expected_ids = subset.get(f"{split}_ids")
                if (
                    expected_ids is not None
                    and list(rows.select(indices)["primary_id"]) != expected_ids
                ):
                    raise ValueError(
                        f"Local trajectory identity/order differs: {name}/{split}"
                    )
            subsets[name] = {
                **subset,
                **{
                    key: [i + offset for i in subset[key]]
                    for key in ("train_indices", "validation_indices")
                },
            }
        tables.append(rows)
        sources.append(
            {
                "category": domain,
                "prepared_fingerprint": manifest["prepared_fingerprint"],
                "metadata_sha256": file_digest(record_path),
            }
        )
        offset += len(rows)
        bank.models.target_revision = manifest["revisions"]["target_sha"]
        bank.models.drafter_revision = manifest["revisions"]["drafter_sha"]
        bank.output_root.mkdir(parents=True, exist_ok=True)
        # Small control metadata only: no reconstruction of PEFT/activation trees.
        write_json(bank.output_root / "manifest.json", manifest)
        write_json(cfg.input_root / f"bank_{domain}.json", bank.model_dump(mode="json"))
    first = manifests[DOMAINS[0]]
    if any(m["revisions"] != first["revisions"] for m in manifests.values()):
        raise ValueError("Local categories use different model/dataset revisions")
    for domain in DOMAINS:
        if any(f"{domain}-{i:05d}" not in subsets for i in range(6)):
            raise ValueError(f"Local trajectories lack required subsets for {domain}")
    members = [
        i
        for subset in subsets.values()
        for key in ("train_indices", "validation_indices")
        for i in subset[key]
    ]
    if len(set(members)) != len(members):
        raise ValueError("Local train/validation subsets overlap")
    return manifests, subsets, concatenate_datasets(tables), sources


def load_local_inputs(cfg):
    banks = campaign_banks(cfg)
    manifests, subsets, rows, sources = read_trajectories(cfg, banks)
    primary = manifests[DOMAINS[0]]
    identity = {
        "version": 2,
        "prepared_fingerprint": primary["prepared_fingerprint"],
        "cache_spec": primary["cache_spec"],
        "revisions": primary["revisions"],
        "sources": sources,
    }
    # Record trajectory identity without consulting saved adapter artifacts.
    write_json(
        cfg.input_root / "local_inputs.json",
        {
            "identity": identity,
            "subsets": {
                s: {
                    "train": len(v["train_indices"]),
                    "validation": len(v["validation_indices"]),
                }
                for s, v in subsets.items()
            },
        },
    )
    return LocalInputs(banks, manifests, rows, subsets, identity)
