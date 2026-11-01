from types import SimpleNamespace

import pytest

from speculators.generator.campaign_config import DOMAINS
from speculators.generator.epoch_training import EpochSchedule
from speculators.generator.local_campaign import LocalInputs
from speculators.generator.validation_sampling import validation_view


def inputs():
    subsets = {}
    for domain in DOMAINS:
        for number in range(25):
            offset = len(subsets) * 1100
            subsets[f"{domain}-{number:05d}"] = {
                "train_indices": list(range(offset, offset + 1000)),
                "validation_indices": list(range(offset + 1000, offset + 1100)),
                "validation_ids": [
                    f"id-{i}" for i in range(offset + 1000, offset + 1100)
                ],
            }
    return LocalInputs({}, {}, None, subsets, {"sources": []})


def test_1k_validation_is_balanced_deterministic_and_training_is_unchanged():
    original = inputs()
    cfg = SimpleNamespace(
        validation_examples_per_category=250,
        seed=42,
        context=SimpleNamespace(min_examples=1, max_examples=8),
        optimization=SimpleNamespace(episodes_per_update=4),
    )
    selected = validation_view(cfg, original)
    repeated = validation_view(cfg, original)
    assert selected.subsets == repeated.subsets
    training, held_out = set(), set()
    for domain in DOMAINS:
        members = [
            s for name, s in selected.subsets.items() if name.startswith(domain + "-")
        ]
        assert sum(len(s["validation_indices"]) for s in members) == 250
        assert all(len(s["validation_indices"]) == 10 for s in members)
    for name, subset in selected.subsets.items():
        source = original.subsets[name]
        assert len(source["validation_indices"]) == 100
        assert subset["train_indices"] == source["train_indices"]
        assert set(subset["validation_indices"]) <= set(source["validation_indices"])
        assert subset["validation_ids"] == [
            f"id-{i}" for i in subset["validation_indices"]
        ]
        training.update(subset["train_indices"])
        held_out.update(subset["validation_indices"])
    assert len(training) == 100000
    assert len(held_out) == 1000
    assert not training.intersection(held_out)
    before = EpochSchedule(
        cfg,
        SimpleNamespace(indices=lambda n: original.subsets[n]["train_indices"]),
        sorted(original.subsets),
        1,
    )
    after = EpochSchedule(
        cfg,
        SimpleNamespace(indices=lambda n: selected.subsets[n]["train_indices"]),
        sorted(selected.subsets),
        1,
    )
    assert before.updates == after.updates
    cfg.seed = 43
    assert validation_view(cfg, original).identity != selected.identity


def test_validation_cap_is_opt_in_and_rejects_insufficient_holdout():
    original = inputs()
    cfg = SimpleNamespace(validation_examples_per_category=None, seed=42)
    assert validation_view(cfg, original) is original
    cfg.validation_examples_per_category = 2501
    with pytest.raises(ValueError, match="Only 2500"):
        validation_view(cfg, original)


def test_pilot_has_exactly_1k_training_rows_and_unchanged_1k_holdout():
    from speculators.generator.validation_sampling import training_view  # noqa: PLC0415

    source = inputs()
    cfg = SimpleNamespace(
        training_examples_per_category=250,
        validation_examples_per_category=250,
        seed=42,
    )
    reduced = validation_view(cfg, training_view(cfg, source))
    train, validation = set(), set()
    for domain in DOMAINS:
        names = [n for n in reduced.subsets if n.startswith(domain + "-")]
        assert sum(len(reduced.subsets[n]["train_indices"]) for n in names) == 250
        assert sum(len(reduced.subsets[n]["validation_indices"]) for n in names) == 250
    for name, subset in reduced.subsets.items():
        assert len(subset["train_indices"]) == 10
        assert len(source.subsets[name]["train_indices"]) == 1000
        assert set(subset["train_indices"]) <= set(
            source.subsets[name]["train_indices"]
        )
        train.update(subset["train_indices"])
        validation.update(subset["validation_indices"])
    assert len(train) == 1000
    assert len(validation) == 1000
    assert not train.intersection(validation)
    assert reduced.identity == validation_view(cfg, training_view(cfg, source)).identity
