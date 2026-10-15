"""Metadata-only physical working-set estimates for the planned jobs."""

import json
from pathlib import Path

from speculators.bank.artifacts import digest
from speculators.generator.campaign_config import DOMAINS


def working_sets(cfg, planned):
    subsets = {}
    for domain in DOMAINS:
        root = cfg.input_root / domain
        manifest = json.loads((root / "manifest.json").read_text())
        index = json.loads(
            (Path(manifest["hidden_states_path"]) / "index.json").read_text()
        )
        for subset in manifest["subsets"]:
            refs = {}
            for split in ("train", "validation"):
                references = {}
                for number in subset[f"{split}_indices"]:
                    ref = index["rows"][str(number)]["shard"]
                    references[digest(ref)] = ref
                refs[split] = references
            subsets[subset["id"]] = refs
    result = {}
    for job in planned:
        refs = {}
        for subset in job["subsets"]:
            refs.update(subsets[subset]["train"])
        train_bytes = sum(r["bytes"] for r in refs.values())
        for subset in job["validation_subsets"]:
            refs.update(subsets[subset]["validation"])
        physical = sum(r["bytes"] for r in refs.values())
        result[job["label"]] = {
            "training_full_activation_bytes": train_bytes,
            "training_and_validation_full_activation_bytes": physical,
            "files": len(refs),
            "fits_activation_cache": physical <= cfg.downloads.capacity_gib * 2**30,
            "note": (
                "bank reconstruction retains only prompt summaries "
                "after reading source files"
            ),
        }
    return result
