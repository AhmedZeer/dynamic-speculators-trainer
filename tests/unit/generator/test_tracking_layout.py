"""Quality/performance separation, compact validation details, and project routing."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.tracking import ExperimentTracker, replay_campaign_history
from speculators.generator.tracking_layout import sectioned_metrics


def test_section_routing_and_units():
    values = sectioned_metrics(
        {
            "optimizer_step": 5,
            "validation/seconds": 4,
            "validation/examples": 100,
            "validation/cumulative_seconds": 12,
            "train/reconstruction_l1": 0.3,
            "train/lr": 0.01,
            "train/host_rss_bytes": 2**30,
            "train/update_seconds": 2,
            "train/examples": 8,
            "train/response_tokens": 100,
            "cache/cache_bytes": 2**31,
            "cache/hit_rate": 0.75,
            "train/data_wait_fraction": 0.2,
            "validation/subset_math-00000/seconds": 3,
        }
    )
    assert values["validation_performance/pass_seconds"] == 4
    assert values["validation_performance/examples_per_second"] == 25
    assert values["training_performance/examples_per_second"] == 4
    assert values["training_performance/supervised_tokens_per_second"] == 50
    assert values["resources/host_rss_gib"] == 1
    assert values["cache_performance/cache_gib"] == 2
    assert values["cache_performance/hit_percent"] == 75
    assert values["training_performance/data_wait_percent"] == 20
    assert values["training_losses/reconstruction_l1"] == 0.3
    assert values["optimization/lr"] == 0.01
    assert not any(k.startswith(("validation/", "train/", "cache/")) for k in values)


def test_sectioned_tracker_creates_one_subset_table(tmp_path):
    cfg = CampaignConfig.load(
        Path("examples/eagle3-lora/generator_campaign_online.yaml")
    )
    tracker = ExperimentTracker(cfg, {"kind": "pretraining"}, tmp_path, {})
    logs = []
    tracker.run = SimpleNamespace(log=logs.append, summary={})
    tracker.module = SimpleNamespace(Table=lambda **kwargs: kwargs)
    context = {"full_acc_0": 0.6, "loss_0": 0.4, "cond_acc_0": 0.7}
    metrics = {
        "score": 0.6,
        "validation_examples": 100,
        "contexts": {"4": context},
        "categories": {"math": 0.6},
        "subsets": {
            f"math-{i:05d}": {"contexts": {"4": context}, "elapsed_seconds": 2}
            for i in range(6)
        },
    }
    tracker.validation(metrics, 5, training_seconds=12)
    values = logs[0]
    assert values["validation_scores/overall_agreement"] == 0.6
    assert values["validation_losses/context_4_loss_step_0"] == 0.4
    assert values["diagnostics/context_4_cond_acc_0"] == 0.7
    assert len(values["validation_details/subsets"]["data"]) == 24
    assert not any("subset_math" in key for key in values)
    assert values["training_performance/cumulative_update_seconds"] == 12


def test_sectioned_layout_with_real_offline_wandb(tmp_path):
    pytest.importorskip("wandb")
    cfg = CampaignConfig.load(
        Path("examples/eagle3-lora/generator_campaign_online.yaml")
    )
    cfg.wandb.mode = "offline"
    cfg.staging_dir = tmp_path / "staging"
    job = {"kind": "pretraining", "condition": "last", "label": "sections-smoke"}
    tracker = ExperimentTracker(cfg, job, tmp_path / "run", {})
    metrics = {
        "score": 0.5,
        "validation_examples": 4,
        "contexts": {"4": {"full_acc_0": 0.5, "loss_0": 0.7}},
        "subsets": {
            "math-00000": {
                "elapsed_seconds": 1,
                "contexts": {"4": {"full_acc_0": 0.5, "loss_0": 0.7}},
            }
        },
    }
    with tracker:
        tracker.training(
            1,
            {"train/update_seconds": 2, "train/drafter_loss": 0.8, "train/examples": 4},
        )
        tracker.validation(metrics, 1, training_seconds=2)
        tracker.log(
            {"optimizer_step": 1, "validation/seconds": 1, "validation/examples": 4}
        )
        tracker.result(
            {
                "score": 0.5,
                "trainable_parameters": 10,
                "elapsed_seconds": 3,
                "training_seconds": 2,
                "history": [{"score": 0.5}],
            }
        )
        assert tracker.run.summary["run_summary/training_seconds"] == 2
    assert (tmp_path / "run/wandb.json").is_file()


def test_project_migration_replays_baseline_and_partial_training_in_order():
    calls = []
    tracker = SimpleNamespace(
        training=lambda step, values, **kwargs: calls.append((step, "train")),
        validation=lambda row, step, **kwargs: calls.append((step, "validation")),
        log=lambda values: None,
    )
    replay_campaign_history(
        tracker, [{"update": 0}, {"update": 2}], [{"update": i} for i in (1, 2, 3)]
    )
    assert calls == [
        (0, "validation"),
        (1, "train"),
        (2, "train"),
        (2, "validation"),
        (3, "train"),
    ]
