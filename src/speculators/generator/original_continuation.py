"""Original saved-response training lifecycle, with optional causal feedback."""

import json
import time
from types import SimpleNamespace

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.progress import report
from speculators.generator import study as common
from speculators.generator.epoch_training import EpochSchedule
from speculators.generator.saved_response_feedback import update, validation
from speculators.generator.tracking import ExperimentTracker


def original_schedule(study, runtime, members, subsets):
    """Original epoch/cohort sampler, restricted to the explicit 4K membership."""
    rows = sum(len(value["train"]) for value in members.values())
    epochs = (
        study.steps
        * study.examples_per_update
        * runtime.cfg.context.max_examples
        // rows
        + 1
    )
    restricted = SimpleNamespace(
        indices=lambda name: members[name.rsplit("-", 1)[0]]["train"]
    )
    return EpochSchedule(runtime.cfg, restricted, subsets, epochs)


def prepare_selected_summaries(runtime, members):
    """Use the original summary cache for selected train/tune rows only."""
    store = runtime.corpus.summary_store
    store.subsets = dict(store.subsets)
    names = []
    for category, splits in sorted(members.items()):
        for role in ("train", "tune"):
            indices = splits[role]
            name = f"continuation-{category}-{role}-{digest(indices)[:12]}"
            store.subsets[name] = {"train_indices": indices, "validation_indices": []}
            names.append(name)
    return runtime.corpus.prepare_summaries(names)


def run_original_loop(  # noqa: C901, PLR0917 -- durable original-pipeline lifecycle
    study,
    runtime,
    cfg,
    job,
    split,
    identity,
    generator,
    optimizer,
    progress,
    history,
    started,
):
    output = study.output
    schedule = original_schedule(study, runtime, split["groups"], job["subsets"])
    report(
        "Original saved-response pipeline: cached activations; teacher-forced TTT; "
        "unchanged LK; no continuation generation or speculative rollouts"
    )
    with ExperimentTracker(runtime.cfg, job, output, identity) as tracker:
        if tracker.run is not None:
            tracker.run.summary.update(
                {
                    "run_summary/training_pipeline": "original_saved_response",
                    "run_summary/source_optimizer_step": study.parent.optimizer_step,
                    "run_summary/parent_adam_restored": study.inherit_parent_optimizer,
                    "run_summary/training_examples": job["training_examples"],
                    "run_summary/episodes_per_update": study.examples_per_update,
                }
            )
            for section in (
                "training_losses",
                "training_performance",
                "validation_scores",
                "validation_losses",
                "validation_performance",
                "feedback_diagnostics",
            ):
                tracker.run.define_metric(section + "/*", step_metric="optimizer_step")
        report(f"{study.name}: preparing original reusable prompt summaries")
        prepare_selected_summaries(runtime, split["groups"])

        def save():
            snapshot = common.save_snapshot(
                study,
                runtime,
                cfg,
                job,
                identity,
                generator,
                optimizer,
                progress,
                history,
            )
            common.upload_snapshot(study, cfg, snapshot, progress)

        def validate():
            tick = time.perf_counter()
            scores = validation(runtime, study, split["groups"], generator)
            history.append({"optimizer_step": progress, **scores})
            write_json(output / "validation.json", history)
            tracker.log(
                {
                    "optimizer_step": progress,
                    "validation_scores/overall_agreement": scores["score"],
                    "validation_performance/seconds": time.perf_counter() - tick,
                    **{
                        "validation_scores/" + name + "_agreement": value
                        for name, value in scores["categories"].items()
                    },
                    **{
                        "validation_losses/" + name: value
                        for name, value in scores["losses"].items()
                    },
                }
            )
            report(
                f"{study.name}: original validation update={progress}; "
                f"agreement={scores['score']:.5f}"
            )

        if progress and (
            progress % study.checkpoint_interval == 0 or progress == study.steps
        ):
            save()
        if not history or history[-1]["optimizer_step"] != progress:
            validate()
        while progress < study.steps:
            tick = time.perf_counter()
            episodes = schedule.update(progress)
            runtime.corpus.set_training_window(
                [
                    episode["indices"]
                    for step in range(progress, min(study.steps, progress + 3))
                    for episode in schedule.update(step)
                ],
                full=True,
            )
            values, observations = update(
                runtime,
                study,
                generator,
                optimizer,
                episodes,
                study.parent.optimizer_step + progress,
            )
            progress += 1
            metrics = {
                "optimizer_step": progress,
                "training_losses/drafter_loss": values["train/drafter_loss"],
                "training_performance/update_seconds": time.perf_counter() - tick,
                "training_performance/response_tokens": values["train/response_tokens"],
                "training_performance/examples": values["train/examples"],
                "cache_performance/cache_gib": runtime.corpus.cache.ram.bytes / 2**30,
                **{
                    "training_performance/" + key.removeprefix("train/"): value
                    for key, value in values.items()
                    if key in ("train/data_wait_seconds", "train/gpu_compute_seconds")
                },
            }
            if study.feedback:
                metrics["feedback_diagnostics/projector_weight_norm"] = float(
                    generator.feedback_encoder.weight.detach().norm()
                )
                write_json(
                    output / "feedback_observations" / f"update-{progress:08d}.json",
                    observations,
                )
            tracker.log(metrics)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            if progress == 1 or progress % 10 == 0:
                report(
                    f"{study.name}: update={progress}/{study.steps}; "
                    f"wall={metrics['training_performance/update_seconds']:.2f}s; "
                    f"loss={values['train/drafter_loss']:.5f}; "
                    f"response_tokens={values['train/response_tokens']}; "
                    "pipeline=original"
                )
            if progress % study.validation_interval == 0 or progress == study.steps:
                validate()
            if (
                progress % study.checkpoint_interval == 0
                or progress % study.hub_interval == 0
                or progress == study.steps
            ):
                save()
        result = {
            "status": "complete",
            "job": job,
            "completed_updates": progress,
            "source_optimizer_step": study.parent.optimizer_step,
            "lineage_optimizer_step": study.parent.optimizer_step + progress,
            "score": history[-1]["score"],
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(
                p.numel() for p in generator.parameters() if p.requires_grad
            ),
            "checkpoint_sha256": file_digest(output / "checkpoint.pt"),
            "training_pipeline": "original_saved_response",
        }
        write_json(output / "result.json", result)
        common.publish_study_result(tracker, result)
        tracker.artifact(
            "saved-response-feedback-observations",
            list((output / "feedback_observations").glob("*.json")),
        )
    return result
