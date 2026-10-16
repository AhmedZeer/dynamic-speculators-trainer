"""Local packed bank and token trajectories; no Hub calls or archive extraction."""

import json
from dataclasses import dataclass

import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets

from speculators.bank.artifacts import file_digest, write_json
from speculators.generator.campaign_config import DOMAINS, SNAPSHOTS_PER_SUBSET
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
    targets: dict
    identity: dict


def read_trajectories(cfg, banks):  # noqa: C901 -- local membership validation
    manifests, subsets, tables, sources = {}, {}, [], []
    offset = 0
    for domain in DOMAINS:
        bank = banks[domain]
        record_path = local_path(
            cfg.trajectories_path, f"releases/{bank.hub.release}/{domain}/data.json"
        )
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


def read_targets(cfg, manifests, banks):  # noqa: C901 -- packed format validation
    root = cfg.data_path / "releases" / cfg.bank_release
    catalog_path = local_path(root, "catalog.json")
    catalog = json.loads(catalog_path.read_text())
    if (
        catalog.get("format") != "speculators-packed-lora-bank"
        or catalog.get("version") != 1
    ):
        raise ValueError("Unsupported packed bank format")
    if catalog["release"] != cfg.bank_release:
        raise ValueError("Packed catalog release differs from configuration")
    targets = {}
    for run in catalog["runs"]:
        manifest_path = local_path(root, f"{run['path']}/manifest.json")
        if file_digest(manifest_path) != run["manifest_sha256"]:
            raise ValueError(f"Packed manifest identity differs: {manifest_path}")
        packed = json.loads(manifest_path.read_text())
        identity = packed["source_identity"]
        domain = identity["subset"]["id"].rsplit("-", 1)[0]
        if domain not in banks:
            raise ValueError(f"Unknown packed bank category: {domain}")
        original = manifests[domain]
        for key in ("prepared_fingerprint", "cache_fingerprint", "revisions"):
            if identity[key] != original[key]:
                raise ValueError(f"Packed bank and trajectories differ: {domain}/{key}")
        subset = identity["subset"]["id"]
        source_subset = next(s for s in original["subsets"] if s["id"] == subset)
        if identity["subset"] != source_subset or identity["lora"] != banks[
            domain
        ].lora.model_dump(mode="json"):
            raise ValueError(f"Packed membership or LoRA recipe differs: {subset}")
        entries = targets.setdefault(subset, [])
        first = next((e for e in packed["epochs"] if e["collection_epoch"] == 1), None)
        if first is None or len(first["snapshots"]) != SNAPSHOTS_PER_SUBSET:
            raise ValueError(f"Expected ten first-epoch snapshots: {subset}")
        weights = local_path(manifest_path.parent, first["file"])
        if weights.stat().st_size != first["bytes"]:
            raise ValueError(f"Incomplete packed weights: {weights}")
        for snapshot in first["snapshots"]:
            entries.append(
                {
                    "seed": identity["seed"],
                    "subset": subset,
                    "collection_step": snapshot["entry"]["collection_step"],
                    "entry": snapshot["entry"],
                    "path": {
                        "file": str(weights),
                        "snapshot_id": snapshot["snapshot_id"],
                        "tensor_keys": snapshot["tensor_keys"],
                    },
                }
            )
    for domain in DOMAINS:
        for i in range(4):
            if not targets.get(f"{domain}-{i:05d}"):
                raise ValueError(f"Missing packed bank targets: {domain}-{i:05d}")
    return targets, file_digest(catalog_path)


def load_local_inputs(cfg):
    banks = campaign_banks(cfg)
    manifests, subsets, rows, sources = read_trajectories(cfg, banks)
    targets, catalog_sha = read_targets(cfg, manifests, banks)
    primary = manifests[DOMAINS[0]]
    identity = {
        "version": 2,
        "prepared_fingerprint": primary["prepared_fingerprint"],
        "cache_spec": primary["cache_spec"],
        "revisions": primary["revisions"],
        "sources": sources,
        "packed_catalog_sha256": catalog_sha,
    }
    # Record data identity once; epoch weights remain in the user's download.
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
    return LocalInputs(banks, manifests, rows, subsets, targets, identity)
