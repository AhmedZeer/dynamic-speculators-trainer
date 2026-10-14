"""Stage one subset on scratch, collect its LoRAs, then publish completed runs."""

import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from filelock import FileLock

from speculators.bank import workflow
from speculators.bank.hub import prepare
from speculators.bank.progress import report
from speculators.bank.publication import BankPublisher
from speculators.bank.staging import can_stage, stage_hidden


def selected_subsets(manifest, requested=None):
    subsets = [s for s in manifest["subsets"] if s.get("role", "bank") == "bank"]
    if requested is not None:
        if not requested or set(requested) - {s["id"] for s in subsets}:
            raise ValueError(f"Invalid bank subset selection: {requested}")
        subsets = [s for s in subsets if s["id"] in requested]
    return subsets


def seeds_for(cfg, subset):
    return (
        [subset["training_seed"]]
        if subset.get("training_seed") is not None
        else cfg.training.seeds
    )


def train_cloud(cfg, subsets=None, workers=8, keep_activations=False):  # noqa: C901 -- bounded two-subset pipeline
    if cfg.hub is None or cfg.hub.trajectories_revision == "main":
        raise ValueError(
            "Cloud training requires pinned Hub inputs; use bank_*_cloud.yaml"
        )
    if cfg.execution.n_workers != 1:
        raise ValueError("train-cloud currently requires execution.n_workers: 1")
    consumer = cfg.model_copy(deep=True)
    consumer.hub.mode = "consumer"
    consumer.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(consumer.output_root / ".cloud-training.lock"), timeout=0):
        manifest = prepare(consumer, "hidden")
        publisher = BankPublisher(consumer)
        selected = selected_subsets(manifest, subsets or cfg.execution.training_subsets)
        pending_runs = []
        for subset in selected:
            seeds = seeds_for(consumer, subset)
            pending = [
                seed
                for seed in seeds
                if publisher.completed(manifest, subset, seed) is None
            ]
            if not pending:
                report(
                    f"Skipping published subset {subset['id']}; no activation download"
                )
                continue
            pending_runs.append((subset, pending))
        with ThreadPoolExecutor(max_workers=1) as pool:
            prefetched = None
            for position, (subset, pending) in enumerate(pending_runs):
                activation_dir = _activation_dir(consumer, subset)
                if prefetched is not None:
                    prefetched.result()
                    prefetched = None
                else:
                    stage_hidden(consumer, activation_dir, [subset["id"]], workers)
                if consumer.execution.prefetch_next_subset and position + 1 < len(
                    pending_runs
                ):
                    next_subset = pending_runs[position + 1][0]
                    next_dir = _activation_dir(consumer, next_subset)
                    if can_stage(consumer, next_dir, [next_subset["id"]]):
                        report(
                            f"Prefetching next subset {next_subset['id']} "
                            "during training"
                        )
                        prefetched = pool.submit(
                            stage_hidden,
                            consumer,
                            next_dir,
                            [next_subset["id"]],
                            workers,
                        )
                    else:
                        report(
                            "Next-subset prefetch deferred: insufficient scratch space"
                        )
                for seed in pending:
                    training = consumer.model_copy(deep=True)
                    training.execution.activation_local_dir = activation_dir
                    training.execution.training_subsets = [subset["id"]]
                    training.training.seeds = [seed]
                    workflow.train(training)
                    publisher.publish_run(manifest, subset, seed)
                if not keep_activations:
                    shutil.rmtree(activation_dir)
                    report(f"Freed staged activations for {subset['id']}")
        return {"repo": consumer.hub.banks_repo, "release": consumer.hub.bank_release}


def _activation_dir(consumer, subset):
    # Dedicated input folders survive failures and same-machine restarts.
    scratch_root = consumer.output_root.resolve() / ".cloud-activations"
    activation_dir = scratch_root / subset["id"]
    if (
        "/" in subset["id"]
        or subset["id"] in (".", "..")
        or not activation_dir.resolve().is_relative_to(scratch_root)
    ):
        raise ValueError("Invalid subset scratch path")
    return activation_dir


def publish_banks(cfg, subsets=None):
    manifest = json.loads((cfg.output_root / "manifest.json").read_text())
    publisher = BankPublisher(cfg)
    for subset in selected_subsets(manifest, subsets or cfg.execution.training_subsets):
        for seed in seeds_for(cfg, subset):
            publisher.publish_run(manifest, subset, seed)


def restore_banks(cfg, revision, subsets=None):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Bank restore requires a complete Hub commit SHA")
    manifest = json.loads((cfg.output_root / "manifest.json").read_text())
    publisher = BankPublisher(cfg, create=False)
    restored = 0
    for subset in selected_subsets(manifest, subsets or cfg.execution.training_subsets):
        for seed in seeds_for(cfg, subset):
            if publisher.restore_run(manifest, subset, seed, revision) is not None:
                restored += 1
    return restored
