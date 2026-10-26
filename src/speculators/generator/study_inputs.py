"""Freeze disjoint data membership and training-only conditioning donors once."""

import json
import random

from safetensors.torch import save_file
from transformers import AutoTokenizer

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.campaign_config import DOMAINS
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.online_activations import OnlineActivations
from speculators.generator.study_config import campaign_settings


def membership(study, inputs):
    groups = {}
    for category in DOMAINS:
        train = inputs.subsets[f"{category}-{study.training_subset:05d}"][
            "train_indices"
        ]
        held = list(
            inputs.subsets[f"{category}-{study.validation_subset:05d}"][
                "validation_indices"
            ]
        )
        if (
            len(train) < study.train_per_category
            or len(held) < study.tune_per_category + study.test_per_category
        ):
            raise ValueError(
                f"Insufficient examples for the requested split: {category}"
            )
        random.Random(study.seed).shuffle(held)
        train = list(train[: study.train_per_category])
        tune = held[: study.tune_per_category]
        test = held[
            study.tune_per_category : study.tune_per_category + study.test_per_category
        ]
        if set(train) & set(tune + test) or set(tune) & set(test):
            raise ValueError("Training/tuning/test memberships overlap")
        groups[category] = {"train": train, "tune": tune, "test": test}
    return {
        "version": 1,
        "data_identity": inputs.identity,
        "seed": study.seed,
        "groups": groups,
    }


def prepare_study(study, *, dry_run=False):  # noqa: C901 -- frozen split export and resumable donor preparation
    if dry_run:
        return {
            "configuration": study.model_dump(mode="json"),
            "output": str(study.inputs),
        }
    cfg = campaign_settings(study)
    inputs = load_local_inputs(cfg)
    value = membership(study, inputs)
    value["identity"] = digest(value)
    path = study.inputs / "membership.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("Study membership changed; choose a new workspace")
    write_json(path, value)
    tokenizer = AutoTokenizer.from_pretrained(cfg.online.target_model_path)
    models = {
        role: {
            "id": getattr(inputs.banks["math"].models, role),
            "revision": inputs.identity["revisions"][role + "_sha"],
        }
        for role in ("target", "drafter")
    }
    queries = {}
    for split in ("tune", "test"):
        output = study.inputs / split
        groups = {}
        for category, members in value["groups"].items():
            rows = []
            for index in members[split]:
                row = inputs.rows[index]
                tokens = row["input_ids"][: row["prompt_length"]]
                text = tokenizer.decode(
                    tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if tokenizer.encode(text, add_special_tokens=False) != tokens:
                    raise ValueError(
                        "Exported prompt does not round-trip through "
                        "the pinned tokenizer"
                    )
                key = digest(tokens)
                if key in queries and queries[key] != category:
                    raise ValueError(
                        "An identical benchmark prompt occurs in multiple categories; "
                        "conditioning donors would be ambiguous"
                    )
                queries[key] = category
                rows.append(
                    {
                        "row_id": row["primary_id"],
                        "subset": row.get(
                            "subset_id", f"{category}-{study.validation_subset:05d}"
                        ),
                        "category": category,
                        "dataset": "nemotron",
                        "prompt": text,
                        "prompt_token_ids": tokens,
                        "prompt_tokens_count": len(tokens),
                        "output_tokens_count": min(
                            study.max_output_tokens, 8192 - len(tokens)
                        ),
                    }
                )
            output.mkdir(parents=True, exist_ok=True)
            name = "nemo-" + category
            data = output / (name + ".jsonl")
            content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            if data.exists() and data.read_text() != content:
                raise ValueError("Frozen prompt export changed")
            data.write_text(content)
            groups[name] = {
                "file": data.name,
                "sha256": file_digest(data),
                "rows": len(rows),
            }
        manifest = {
            "version": 1,
            "dataset": "nemotron",
            "models": models,
            "max_tokens": study.max_output_tokens,
            "split": split,
            "membership_identity": value["identity"],
            "groups": groups,
        }
        write_json(output / "manifest.json", {**manifest, "identity": digest(manifest)})
    donors = study.inputs / "conditioning.safetensors"
    metadata_path = donors.with_suffix(".json")
    if donors.exists():
        saved = json.loads(metadata_path.read_text())
        if saved["membership_identity"] != value["identity"] or saved[
            "sha256"
        ] != file_digest(donors):
            raise ValueError("Conditioning donor bank identity/checksum changed")
        return {"membership": str(path), "donors": str(donors), "reused": True}
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    cache.check_server()
    tensors, groups = {}, {}
    try:
        # A fixed training-only donor pool is sufficient for causal conditioning
        # controls. Selection never searches benchmark performance.
        for category, members in value["groups"].items():
            selected = members["train"][:64]
            cache.prefetch([{"index": index, "summary": True} for index in selected])
            groups[category] = []
            for index in selected:
                key = "row_" + str(index)
                vector = cache.get_summary(index)
                tensors[key] = vector.float().contiguous()
                groups[category].append(
                    {
                        "key": key,
                        "length": inputs.rows[index]["prompt_length"],
                        "prompt_sha256": digest(
                            inputs.rows[index]["input_ids"][
                                : inputs.rows[index]["prompt_length"]
                            ]
                        ),
                    }
                )
        temporary = donors.with_suffix(".pending")
        save_file(tensors, temporary)
        temporary.replace(donors)
        write_json(
            metadata_path,
            {
                "membership_identity": value["identity"],
                "groups": groups,
                "queries": queries,
                "sha256": file_digest(donors),
                "seed": study.seed,
            },
        )
    finally:
        cache.close()
    return {"membership": str(path), "donors": str(donors), "reused": False}
