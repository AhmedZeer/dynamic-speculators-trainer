"""Per-experiment W&B runs with durable IDs and small reproducibility artifacts."""

import importlib
import json

from speculators.bank.artifacts import digest, write_json
from speculators.bank.progress import report
from speculators.generator.tracking_layout import (
    SECTIONS,
    sectioned_metrics,
    subset_table,
    validation_values,
)

METADATA_FILES = (
    "resolved_experiment.json",
    "tracking_config.json",
    "train_command.txt",
    "speculators.patch",
    "metrics.json",
    "result.json",
)


def wandb_module():
    try:
        return importlib.import_module("wandb")
    except ImportError as exc:
        raise ImportError(
            "W&B tracking requires pip install 'speculators[generator]' "
            "or pip install wandb. Set wandb.enabled: false to disable tracking."
        ) from exc


def run_name(cfg, job):
    if "label" in job:
        return cfg.wandb.name_prefix + job["label"] + f"-gseed{cfg.seed}"
    stage = job["kind"]
    subset = cfg.bank_subset if stage.startswith("heatmap") else cfg.conditioning_subset
    parts = [stage, subset]
    if stage == "pretraining":
        parts = [stage, "+".join(job["subsets"])]
    if "arm" in job:
        parts.append(job["arm"])
    parts.append(job["condition"])
    if "stride" in job:
        parts.extend(
            [f"stride{job['stride']}", "bankseeds" + "+".join(map(str, job["seeds"]))]
        )
    parts.extend(
        [
            f"gseed{cfg.seed}",
            f"ctx{cfg.context.min_examples}-{cfg.context.max_examples}",
        ]
    )
    return cfg.wandb.name_prefix + "-".join(parts)


class ExperimentTracker:
    def __init__(self, cfg, job, output, identity):
        self.cfg, self.job, self.output, self.identity = cfg, job, output, identity
        self.run = self.module = None
        self.new_destination = False
        stage = "heatmap" if job["kind"].startswith("heatmap") else job["kind"]
        self.project = (
            cfg.wandb.pretraining_project
            if stage == "pretraining"
            else cfg.wandb.projects[stage]
        )
        self.run_id = digest(
            {
                "identity": identity,
                "project": self.project,
                "entity": cfg.wandb.entity,
            }
        )[:16]
        self.destination = {
            "id": self.run_id,
            "project": self.project,
            "entity": cfg.wandb.entity,
            "mode": cfg.wandb.mode,
        }

    def __enter__(self):
        if not self.cfg.wandb.enabled:
            return self
        self.module = wandb_module()
        self.output.mkdir(parents=True, exist_ok=True)
        previous_path = self.output / "wandb.json"
        previous = (
            json.loads(previous_path.read_text()) if previous_path.is_file() else {}
        )
        self.new_destination = any(
            previous.get(key) != value for key, value in self.destination.items()
        )
        local = self.cfg.staging_dir / "wandb"
        local.mkdir(parents=True, exist_ok=True)
        settings = self.cfg.model_dump(mode="json")
        settings.pop("wandb")
        job_metadata = {
            key: value for key, value in self.job.items() if key != "snapshots"
        }
        if "snapshots" in self.job:
            job_metadata["checkpoint_count"] = len(self.job["snapshots"])
        kwargs = {
            **self.destination,
            "name": run_name(self.cfg, self.job),
            "dir": str(local),
            "group": self.cfg.wandb.group or self.cfg.output_root.name,
            "job_type": self.job["kind"],
            "tags": [self.job["kind"], self.job["condition"], f"gseed-{self.cfg.seed}"],
            "config": {
                "experiment": settings,
                "job": job_metadata,
                "training_identity": digest(self.identity),
            },
        }
        if self.cfg.wandb.mode == "online":
            kwargs["resume"] = "allow"
        self.run = self.module.init(**kwargs)
        self.run.define_metric("optimizer_step")
        if self.cfg.wandb.layout == "sections":
            for section in SECTIONS:
                self.run.define_metric(section + "/*", step_metric="optimizer_step")
            self.run.define_metric(
                "diagnostics/*", step_metric="optimizer_step", hidden=True
            )
            self.run.define_metric(
                "validation_scores/overall_agreement_by_training_seconds",
                step_metric="training_performance/cumulative_update_seconds",
            )
        else:
            self.run.define_metric("train/*", step_metric="optimizer_step")
            self.run.define_metric("cache/*", step_metric="optimizer_step")
            self.run.define_metric("validation/*", step_metric="optimizer_step")
        write_json(
            self.output / "tracking_config.json", self.cfg.wandb.model_dump(mode="json")
        )
        self.record("running")
        report(
            f"W&B project={self.project}, run={run_name(self.cfg, self.job)}, "
            f"id={self.run_id}"
        )
        if self.run.url:
            report(f"W&B URL: {self.run.url}")
        return self

    def record(self, status):
        write_json(
            self.output / "wandb.json",
            {
                **self.destination,
                "name": run_name(self.cfg, self.job),
                "status": status,
                "url": self.run.url,
            },
        )

    def __exit__(self, exc_type, exc, traceback):
        if self.run is not None:
            try:
                self.run.finish(exit_code=1 if exc_type else 0)
                self.record("failed" if exc_type else "finished")
            except Exception as finish_error:
                if exc_type is None:
                    raise
                report(f"W&B finalization failed: {finish_error}")
        return False

    def log(self, values):
        if self.run is not None:
            if self.cfg.wandb.layout == "sections":
                values = sectioned_metrics(values)
            self.run.log(values)

    def training(self, step, values, final=False):
        if step == 1 or step % self.cfg.wandb.log_interval == 0 or final:
            self.log({"optimizer_step": step, **values})

    def validation(self, metrics, step, epoch=None, training_seconds=None):
        if self.cfg.wandb.layout == "sections":
            if self.run is None:
                return
            values = validation_values(metrics, step)
            if training_seconds is not None:
                values.update(
                    {
                        "training_performance/cumulative_update_seconds": (
                            training_seconds
                        ),
                        "validation_scores/overall_agreement_by_training_seconds": (
                            metrics["score"]
                        ),
                    }
                )
            rows = subset_table(metrics, step)
            if rows:
                values["validation_details/subsets"] = self.module.Table(
                    columns=["update", "subset", "context_size", "metric", "value"],
                    data=rows,
                )
            self.log(values)
            return
        values = {
            "optimizer_step": step,
            "validation/score_mean_full_acc_0": metrics["score"],
            "validation/examples": metrics["validation_examples"],
        }
        if epoch is not None:
            values["validation/epoch"] = epoch
        for size, context in metrics["contexts"].items():
            values.update(
                {
                    f"validation/context_{size}/{key}": value
                    for key, value in context.items()
                }
            )
        if "subsets" in metrics:
            for subset, result in metrics["subsets"].items():
                values[f"validation/subset_{subset}/score"] = result["score"]
                for size, context in result["contexts"].items():
                    values.update(
                        {
                            f"validation/subset_{subset}/context_{size}/{key}": value
                            for key, value in context.items()
                        }
                    )
        self.log(values)

    def result(self, result):
        if self.run is None:
            return
        if self.cfg.wandb.layout == "sections":
            self.run.summary.update(
                {
                    "run_summary/final_agreement": result["score"],
                    "run_summary/best_agreement": max(
                        r["score"] for r in result["history"]
                    ),
                    **{
                        f"run_summary/{key}": result[key]
                        for key in (
                            "elapsed_seconds",
                            "setup_seconds",
                            "preparation_seconds",
                            "compiler_cache_dir",
                            "training_seconds",
                            "validation_seconds",
                            "data_wait_seconds",
                            "trainable_parameters",
                            "score_change",
                        )
                        if key in result
                    },
                }
            )
            if not self.cfg.wandb.upload_artifacts:
                return
        self.run.summary.update(
            {
                "final_score": result["score"],
                "trainable_parameters": result["trainable_parameters"],
                "elapsed_seconds": result["elapsed_seconds"],
                "best_score": max(row["score"] for row in result["history"]),
            }
        )
        if "score_change" in result:
            self.run.summary.update(
                {
                    "score_change": result["score_change"],
                    "baseline_score": result["history"][0]["score"],
                }
            )
        for key in (
            "training_seconds",
            "validation_seconds",
            "data_wait_seconds",
            "setup_seconds",
        ):
            if key in result:
                self.run.summary[key] = result[key]
        if "snapshots" in self.job:
            self.run.summary["checkpoint_count"] = len(self.job["snapshots"])
        if self.cfg.wandb.upload_artifacts:
            self.artifact(
                "run-metadata",
                [self.output / name for name in METADATA_FILES]
                + list((self.output / "provenance").glob("attempt-*/*")),
            )

    def artifact(self, label, paths):
        if self.run is None or not self.cfg.wandb.upload_artifacts:
            return
        artifact = self.module.Artifact(f"{label}-{self.run_id}", type="evaluation")
        for path in paths:
            if path.is_file():
                artifact.add_file(str(path), name=str(path.relative_to(self.output)))
        self.run.log_artifact(artifact)

    def already_finished(self):
        path = self.output / "wandb.json"
        if not path.exists():
            return False
        metadata = json.loads(path.read_text())
        return metadata.get("status") == "finished" and all(
            metadata.get(key) == value for key, value in self.destination.items()
        )


def epoch_updates(cfg, corpus, subset, completed_epochs):
    from speculators.generator.data import groups  # noqa: PLC0415

    return sum(
        len(
            list(
                groups(
                    corpus.indices(subset),
                    cfg.seed + epoch,
                    cfg.context.min_examples,
                    cfg.context.max_examples,
                )
            )
        )
        for epoch in range(completed_epochs)
    )


def log_completed(cfg, corpus, job, output, identity, result):
    if not cfg.wandb.enabled:
        return
    tracker = ExperimentTracker(cfg, job, output, identity)
    if tracker.already_finished():
        return
    with tracker:
        timing_path = output / "timing.json"
        timing = iter(
            json.loads(timing_path.read_text())
            if cfg.wandb.layout == "sections" and timing_path.is_file()
            else []
        )
        if cfg.wandb.layout == "sections" and all(
            "update" in r for r in result["history"]
        ):
            replay_campaign_history(tracker, result["history"], list(timing))
            tracker.result(result)
            return
        pending = next(timing, None)
        for metrics in result["history"]:
            epoch = metrics.get("epoch")
            step = metrics.get("optimizer_step")
            if step is None:
                step = (
                    metrics.get("update", 0)
                    if epoch is None
                    else epoch_updates(cfg, corpus, cfg.conditioning_subset, epoch)
                )
            while pending is not None and pending["update"] <= step:
                tracker.training(
                    pending["update"],
                    {k: v for k, v in pending.items() if k != "update"},
                    final=pending["update"] == result["history"][-1].get("update"),
                )
                pending = next(timing, None)
            tracker.validation(
                metrics, step, epoch, training_seconds=metrics.get("training_seconds")
            )
            if cfg.wandb.layout == "sections" and "validation_seconds" in metrics:
                tracker.log(
                    {
                        "optimizer_step": step,
                        "validation/seconds": metrics["validation_seconds"],
                        "validation/examples": metrics["validation_examples"],
                    }
                )
        tracker.result(result)


def replay_campaign_history(tracker, history, timing):
    """Replay recorded measurements in step order when moving to a new project."""
    events = [(r["update"], 0, r) for r in timing]
    events.extend((r["update"], 1, r) for r in history)
    for step, validation, row in sorted(events, key=lambda event: event[:2]):
        if not validation:
            tracker.training(
                step,
                {k: v for k, v in row.items() if k != "update"},
                final=step == timing[-1]["update"],
            )
            continue
        tracker.validation(row, step, training_seconds=row.get("training_seconds"))
        if "validation_seconds" in row:
            tracker.log(
                {
                    "optimizer_step": step,
                    "validation/seconds": row["validation_seconds"],
                    "validation/examples": row["validation_examples"],
                }
            )


def log_heatmap_summary(cfg, results, root):
    if not cfg.wandb.enabled:
        return
    job = {"kind": "heatmap-summary", "condition": results[0]["job"]["condition"]}
    # Local plotting/export is complete before the W&B summary is published.
    identity = {"version": 1, "results": results, "output": str(root.resolve())}
    tracker = ExperimentTracker(cfg, job, root, identity)
    if tracker.already_finished():
        return
    with tracker:
        columns = ["stride", "bank_seeds", "checkpoint_count", "score"] + [
            f"context_{size}_full_acc_0" for size in cfg.context.evaluation_sizes
        ]
        rows = [
            [
                r["job"]["stride"],
                "+".join(map(str, r["job"]["seeds"])),
                len(r["job"]["snapshots"]),
                r["score"],
                *(
                    r["history"][-1]["contexts"][str(size)]["full_acc_0"]
                    for size in cfg.context.evaluation_sizes
                ),
            ]
            for r in results
        ]
        tracker.log(
            {
                "heatmap/cells": tracker.module.Table(columns=columns, data=rows),
                **{
                    f"heatmap/{path.stem}": tracker.module.Image(str(path))
                    for path in sorted(root.glob("*.png"))
                },
            }
        )
        tracker.run.summary["best_score"] = max(r["score"] for r in results)
        files = [
            p for p in root.iterdir() if p.suffix in (".png", ".pdf", ".csv", ".json")
        ]
        files.extend(
            p for p in root.glob("stride-*/seeds-*/*") if p.name in METADATA_FILES
        )
        files.extend(root.glob("stride-*/seeds-*/provenance/attempt-*/*"))
        tracker.artifact("heatmap-results", files)
