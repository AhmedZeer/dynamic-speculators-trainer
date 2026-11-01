"""Deterministic category/subset-balanced views of existing held-out trajectories."""

import copy
import random
from dataclasses import replace

from speculators.generator.campaign_config import DOMAINS


def validation_view(cfg, inputs):  # noqa: C901 -- bounded stratified sampling across held-out subsets
    """Keep training intact; equally allocate each category's validation quota."""
    quota = cfg.validation_examples_per_category
    if quota is None:
        return inputs
    subsets = copy.deepcopy(inputs.subsets)
    for domain_index, domain in enumerate(DOMAINS):
        names = sorted(name for name in subsets if name.startswith(domain + "-"))
        available = sum(len(subsets[name]["validation_indices"]) for name in names)
        if available < quota or not names:
            raise ValueError(
                f"Only {available} validation examples available for {domain}; "
                f"need {quota}"
            )
        rng = random.Random(cfg.seed + domain_index * 1009)
        allocation = dict.fromkeys(names, 0)
        order = names.copy()
        rng.shuffle(order)
        remaining = quota
        while remaining:
            for name in order:
                if allocation[name] < len(subsets[name]["validation_indices"]):
                    allocation[name] += 1
                    remaining -= 1
                    if not remaining:
                        break
        for name in names:
            subset = subsets[name]
            positions = sorted(
                rng.sample(range(len(subset["validation_indices"])), allocation[name])
            )
            for key in ("validation_indices", "validation_ids"):
                if key in subset:
                    subset[key] = [subset[key][position] for position in positions]
    identity = {
        **inputs.identity,
        "validation_selection": {
            "policy": "balanced-category-subsets-v1",
            "seed": cfg.seed,
            "examples_per_category": quota,
            "indices": {
                name: subset["validation_indices"] for name, subset in subsets.items()
            },
        },
    }
    return replace(inputs, subsets=subsets, identity=identity, ram_cache=None)
