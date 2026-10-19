"""Freeze existing validation rows for HTTP correctness checks and GuideLLM."""

import json
import random
import shutil
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.campaign import jobs
from speculators.generator.campaign_config import DOMAINS
from speculators.generator.campaign_inputs import campaign_banks
from speculators.generator.local_campaign import local_path, trajectory_record_path


def nemotron_domain(name):
    """Accept short domain selectors and explicit Nemotron group names."""
    for prefix in ("nemo-", "nem-"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name not in DOMAINS:
        raise ValueError(f"Unknown Nemotron category: {name}")
    return name


def sample_domain(rows, max_samples, seed):
    """Interleave seeded subset samples so small limits cover multiple subsets."""
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        buckets.setdefault(row["subset"], []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    ordered = []
    for index in range(max(map(len, buckets.values()), default=0)):
        for name in sorted(buckets):
            if index < len(buckets[name]):
                ordered.append(buckets[name][index])
    return ordered[:max_samples] if max_samples else ordered


def export_validation(  # noqa: C901 -- validate frozen dataset membership
    cfg,
    output,
    *,
    tokenizer_path=None,
    max_tokens=256,
    subsets=None,
    categories=None,
    max_samples=0,
    seed=42,
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new validation export directory: {output}")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    if max_samples < 0:
        raise ValueError("max_samples must be nonnegative; zero selects all rows")
    domains = (
        {nemotron_domain(name) for name in categories} if categories else set(DOMAINS)
    )
    selected = sorted(
        set(
            subsets or [name for job in jobs(cfg) for name in job["validation_subsets"]]
        )
    )
    selected = [name for name in selected if name.rsplit("-", 1)[0] in domains]
    if not selected:
        raise ValueError("No validation subsets selected")
    recipes = campaign_banks(cfg)
    tokenizer_path = tokenizer_path or cfg.online.target_model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    records, sources, model_refs, seen = [], [], None, set()
    for domain in DOMAINS:
        names = [name for name in selected if name.rsplit("-", 1)[0] == domain]
        if not names:
            continue
        metadata_path = trajectory_record_path(cfg, recipes[domain])
        record = json.loads(metadata_path.read_text())
        manifest = record["manifest"]
        current_refs = {
            role: {
                "id": manifest["cache_spec"]["models"][role],
                "revision": manifest["revisions"][f"{role}_sha"],
            }
            for role in ("target", "drafter")
        }
        if model_refs is not None and model_refs != current_refs:
            raise ValueError("Validation categories use different pinned models")
        model_refs = current_refs
        sources.append(
            {"category": domain, "metadata_sha256": file_digest(metadata_path)}
        )
        subset_map = {item["id"]: item for item in manifest["subsets"]}
        artifacts = {ref["path"]: ref for ref in record["artifacts"]}
        for name in names:
            subset = subset_map[name]
            if set(subset["validation_indices"]) & set(subset["train_indices"]):
                raise ValueError(f"Train/validation membership overlaps: {name}")
            relative = (
                f"releases/{recipes[domain].hub.release}/{domain}/"
                f"{name}/validation.parquet"
            )
            path = local_path(cfg.trajectories_path, relative)
            if path.stat().st_size != artifacts[relative]["bytes"]:
                raise ValueError(f"Incomplete validation file: {path}")
            rows = pq.read_table(path).to_pylist()
            if len(rows) != len(subset["validation_indices"]):
                raise ValueError(f"Validation membership size differs: {name}")
            if (
                subset.get("validation_ids") is not None
                and [row["primary_id"] for row in rows] != subset["validation_ids"]
            ):
                raise ValueError(f"Validation IDs/order differ: {name}")
            sources.append({"subset": name, "parquet_sha256": file_digest(path)})
            for row in rows:
                key = f"{name}/{row['primary_id']}"
                if (
                    key in seen
                    or row.get("subset_id", name) != name
                    or row.get("category", domain) != domain
                ):
                    raise ValueError(f"Duplicate or mismatched validation row: {key}")
                seen.add(key)
                boundary = row["prompt_length"]
                if not 0 < boundary < len(row["input_ids"]):
                    raise ValueError(f"Invalid prompt boundary: {key}")
                tokens = row["input_ids"][:boundary]
                text = tokenizer.decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if tokenizer.encode(text, add_special_tokens=False) != tokens:
                    raise ValueError(
                        f"Prompt text does not reproduce saved token IDs: {key}"
                    )
                cap = min(
                    max_tokens, recipes[domain].hidden_states.sequence_length - boundary
                )
                if cap < 1:
                    raise ValueError(f"Prompt leaves no generation space: {key}")
                records.append(
                    {
                        "row_id": key,
                        "subset": name,
                        "category": domain,
                        "dataset": "nemotron",
                        "prompt": text,
                        "prompt_token_ids": tokens,
                        "prompt_tokens_count": boundary,
                        "output_tokens_count": cap,
                    }
                )
    if {row["subset"] for row in records} != set(selected):
        raise ValueError("Unknown or empty validation subsets requested")
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"Previous partial export exists: {pending}")
    pending.mkdir(parents=True)
    try:
        groups = {}
        for domain in DOMAINS:
            rows = [row for row in records if row["category"] == domain]
            if not rows:
                continue
            rows = sample_domain(rows, max_samples, seed)
            group = f"nemo-{domain}"
            path = pending / f"{group}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            groups[group] = {
                "file": path.name,
                "sha256": file_digest(path),
                "rows": len(rows),
            }
        identity = {
            "version": 1,
            "models": model_refs,
            "subsets": selected,
            "sources": sources,
            "max_tokens": max_tokens,
            "dataset": "nemotron",
            "selection": {
                "categories": [d for d in DOMAINS if d in domains],
                "max_samples_per_category": max_samples,
                "seed": seed,
                "method": "seeded_subset_round_robin",
            },
            "groups": groups,
        }
        write_json(
            pending / "manifest.json", {**identity, "identity": digest(identity)}
        )
        pending.rename(output)
    except BaseException:
        shutil.rmtree(pending)
        raise
    return {
        "directory": str(output),
        "examples": sum(g["rows"] for g in groups.values()),
        "groups": groups,
    }


def load_validation(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    identity = {key: value for key, value in manifest.items() if key != "identity"}
    if manifest["version"] != 1 or digest(identity) != manifest["identity"]:
        raise ValueError("Validation manifest identity differs")
    groups = {}
    for name, info in manifest["groups"].items():
        data_path = local_path(path.parent, info["file"])
        if file_digest(data_path) != info["sha256"]:
            raise ValueError(f"Validation file checksum differs: {data_path}")
        rows = [json.loads(line) for line in data_path.read_text().splitlines() if line]
        if len(rows) != info["rows"]:
            raise ValueError(f"Validation row count differs: {name}")
        groups[name] = rows
    return manifest, groups
