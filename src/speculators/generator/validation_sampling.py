"""Deterministic category/subset-balanced views of saved trajectory splits."""

import copy
import random
from dataclasses import replace

from speculators.generator.campaign_config import DOMAINS


def validation_view(cfg, inputs):
    """Keep training intact; equally allocate each category's validation quota."""
    return split_view(cfg, inputs, "validation", cfg.validation_examples_per_category)


def training_view(cfg, inputs):
    """Restrict pilot training to a balanced subset of the original training split."""
    return split_view(cfg, inputs, "train", cfg.training_examples_per_category)


def split_view(cfg, inputs, split, quota):  # noqa: C901 -- bounded stratified sampling
    if quota is None:
        return inputs
    subsets = copy.deepcopy(inputs.subsets)
    for domain_index, domain in enumerate(DOMAINS):
        names = sorted(name for name in subsets if name.startswith(domain + "-"))
        available = sum(len(subsets[name][f"{split}_indices"]) for name in names)
        if available < quota or not names:
            raise ValueError(
                f"Only {available} {split} examples available for {domain}; "
                f"need {quota}"
            )
        rng = random.Random(cfg.seed + domain_index * 1009)
        allocation = dict.fromkeys(names, 0)
        order = names.copy()
        rng.shuffle(order)
        remaining = quota
        while remaining:
            for name in order:
                if allocation[name] < len(subsets[name][f"{split}_indices"]):
                    allocation[name] += 1
                    remaining -= 1
                    if not remaining:
                        break
        for name in names:
            subset = subsets[name]
            positions = sorted(
                rng.sample(range(len(subset[f"{split}_indices"])), allocation[name])
            )
            for key in (f"{split}_indices", f"{split}_ids"):
                if key in subset:
                    subset[key] = [subset[key][position] for position in positions]
    identity = {
        **inputs.identity,
        f"{'training' if split == 'train' else 'validation'}_selection": {
            "policy": "balanced-category-subsets-v1",
            "seed": cfg.seed,
            "examples_per_category": quota,
            "indices": {
                name: subset[f"{split}_indices"] for name, subset in subsets.items()
            },
        },
    }
    return replace(inputs, subsets=subsets, identity=identity, ram_cache=None)
