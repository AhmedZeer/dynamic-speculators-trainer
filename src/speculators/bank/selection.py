"""Bounded uniform sampling of source rows, retaining repeated prompts."""

import random

from speculators.bank.artifacts import digest
from speculators.bank.progress import report

SELECTION_POLICY = "source-rows-v2"


def sample_rows(rows, cfg, *, skip_unsupported=False):
    from speculators.bank.workflow import prompt_messages  # noqa: PLC0415

    size = cfg.partition.train_examples + cfg.partition.validation_examples
    needed = (max(cfg.partition.subset_ids) + 1) * size
    rng = random.Random(cfg.partition.seed)
    sampled, valid, unsupported = [], 0, 0
    for index, row in enumerate(rows):
        if cfg.dataset.sample_limit is not None and index >= cfg.dataset.sample_limit:
            break
        try:
            messages = prompt_messages(row)
        except ValueError:
            if not skip_unsupported:
                raise
            unsupported += 1
            continue
        valid += 1
        # Reservoir sampling gives every eligible row equal probability without
        # retaining the entire source, hashing every prompt, or an SQL registry.
        position = valid - 1 if valid <= needed else rng.randrange(valid)
        if position < needed:
            source_id = str(row.get("uuid") or row.get("id") or f"row-{index}")
            value = {
                "messages": messages,
                "source_ids": [source_id],
                "source_aliases": [
                    {"id": source_id, "license": str(row.get("license", ""))}
                ],
                "source_row": index,
                "source_split": cfg.dataset.split,
            }
            if valid <= needed:
                sampled.append(value)
            else:
                sampled[position] = value
        if (index + 1) % 10000 == 0:
            report(
                f"Selecting {cfg.dataset.split}: scanned={index + 1}, "
                f"eligible={valid}, retained={len(sampled)}, unsupported={unsupported}"
            )
    if len(sampled) < needed:
        raise ValueError(
            f"Only {valid // size} complete subsets available for selection"
        )
    rng.shuffle(sampled)
    for value in sampled:
        # Occurrences have distinct IDs even for repeated text or source UUIDs.
        value["id"] = digest(
            {
                "source": cfg.dataset.source,
                "split": cfg.dataset.split,
                "row": value["source_row"],
                "source_id": value["source_ids"][0],
            }
        )
        value["prompt_hash"] = digest(value["messages"])
    return sampled, {
        "source_examples": valid,
        "unsupported_rows": unsupported,
        "complete_subsets": valid // size,
        "incomplete_group_examples": valid % size,
    }
