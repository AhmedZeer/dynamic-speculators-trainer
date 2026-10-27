"""Two independent online-feedback continuations from a pinned serving bundle."""

import json
import shutil
import time
from pathlib import Path

import torch
from filelock import FileLock

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.progress import report
from speculators.benchmark.config import Model
from speculators.benchmark.prepare import resolve_model
from speculators.generator import study as common
from speculators.generator.config import Architecture
from speculators.generator.engine import clip_gradients, run_identity, set_learning_rate
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.model import LoRAGenerator
from speculators.generator.online_feedback import OnlineFeedbackTrainer
from speculators.generator.serving.bundle import archive_source, bundle_state
from speculators.generator.study_config import campaign_settings
from speculators.generator.study_inputs import membership
from speculators.generator.tracking import ExperimentTracker
from speculators.train.utils import save_train_command


def parent_source(study):
    return resolve_model(
        Model(
            name="parent",
            type="generator",
            **study.parent.model_dump(exclude={"optimizer_step"}),
        ),
        Path.cwd(),
        study.workspace / "sources",
    )


def initialize_generator(runtime, study, metadata, state):
    if metadata.get("feedback_dim") or metadata["condition"] != "last":
        raise ValueError("Parent must be a prompt-only generator without feedback")
    if (
        metadata["rank"] != runtime.bank.lora.rank
        or metadata["alpha"] != runtime.bank.lora.alpha
        or metadata["shapes"] != {k: list(v) for k, v in runtime.shapes.items()}
        or metadata["input_dim"] != runtime.drafter.fc.weight.shape[1]
        or metadata["layer_ids"] != runtime.bank.hidden_states.layer_ids[:-1]
        or metadata["max_examples"] != runtime.cfg.context.max_examples
    ):
        raise ValueError(
            "Parent adapter/generator dimensions differ from training recipe"
        )
    source = runtime.corpus.manifest
    for role in ("target", "drafter"):
        expected = {
            "id": source["cache_spec"]["models"][role],
            "revision": source["revisions"][role + "_sha"],
        }
        if metadata["models"][role] != expected:
            raise ValueError(
                f"Parent {role} revision differs from the local training inputs"
            )
    model = LoRAGenerator(
        metadata["input_dim"],
        runtime.shapes,
        metadata["rank"],
        Architecture.model_validate(metadata["architecture"]),
        metadata["max_examples"],
        feedback_dim=3 if study.feedback else 0,
    ).to(runtime.device)
    mismatch = model.load_state_dict(state, strict=False)
    expected_missing = ["feedback_encoder.weight"] if study.feedback else []
    if sorted(mismatch.missing_keys) != expected_missing or mismatch.unexpected_keys:
        raise ValueError(
            "Parent generator state does not match the continuation architecture"
        )
    model.requires_grad_(study.trainable == "generator")
    if study.trainable == "projector":
        model.feedback_encoder.requires_grad_(True)
    return model.eval()


def trace_metrics(traces):
    rounds = [row for trace in traces for row in trace["rounds"]]
    accepted = sum(row["accepted"] for row in rounds)
    proposed = sum(row["proposed_count"] for row in rounds)
    return {
        "accepted": accepted,
        "proposed": proposed,
        "rounds": len(rounds),
        "draft_acceptance_rate": accepted / max(proposed, 1),
        "accepted_prefix_length": accepted / max(len(rounds), 1),
        "generated_tokens": sum(len(trace["response"]) for trace in traces),
    }


def evaluate_online(trainer, members):
    runtime = trainer.runtime
    common.synchronize(runtime)
    started = time.perf_counter()
    categories, traces, losses, positions = {}, [], 0.0, 0
    for category, split in sorted(members.items()):
        category_traces = []
        for number, index in enumerate(split["tune"]):
            trainer.prefetch(split["tune"][number : number + 8])
            metrics, trace = trainer.evaluate_episode(index)
            losses += metrics["loss_sum"]
            positions += metrics["positions"]
            category_traces.append(trace)
        values = trace_metrics(category_traces)
        categories[category] = values["draft_acceptance_rate"]
        traces.extend(category_traces)
        report(
            f"Online feedback validation {category}: "
            f"acceptance={categories[category]:.5f}"
        )
    values = trace_metrics(traces)
    common.synchronize(runtime)
    return {
        "score": sum(categories.values()) / len(categories),
        "categories": categories,
        "loss": losses / max(positions, 1),
        "positions": positions,
        "seconds": time.perf_counter() - started,
        "protocol": "current-policy greedy verification; prompt conditioning; "
        "observed causal feedback",
        **values,
    }, traces


def publish_result(tracker, result):
    common.publish_study_result(tracker, result)
    tracker.artifact(
        "online-feedback-traces",
        list((tracker.output / "online_traces").glob("*.json"))
        + list((tracker.output / "validation_traces").glob("*.json")),
    )


def train_feedback(study, *, resume=False, dry_run=False):
    if dry_run:
        return {
            "configuration": study.model_dump(mode="json"),
            "training_examples": 4 * study.train_per_category,
            "additional_updates": study.steps,
            "source_optimizer_step": study.parent.optimizer_step,
            "final_lineage_step": study.parent.optimizer_step + study.steps,
        }
    study.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(study.output / ".training.lock"), timeout=0):
        cfg = campaign_settings(study)
        inputs = load_local_inputs(cfg)
        split = membership(study, inputs)
        split["identity"] = digest(split)
        membership_path = study.inputs / "membership.json"
        with FileLock(str(study.workspace / ".membership.lock"), timeout=0):
            if (
                membership_path.exists()
                and json.loads(membership_path.read_text()) != split
            ):
                raise ValueError(
                    "Feedback study membership changed; choose a new workspace"
                )
            write_json(membership_path, split)
        extraction = study.workspace / "provenance/vllm-extractor"
        if not (extraction / "vllm_command.txt").is_file():
            raise FileNotFoundError(
                "Start the extractor with --provenance-dir " + str(extraction)
            )
        parent = parent_source(study)
        metadata, state = bundle_state(parent["path"])
        runtime, cfg, job = common.build_runtime(
            study,
            inputs,
            architecture=Architecture.model_validate(metadata["architecture"]),
        )
        job.update(
            training_mode="online_verified_continuation",
            trainable=study.trainable,
            source_optimizer_step=study.parent.optimizer_step,
            additional_updates=study.steps,
            refresh="verification_round",
            conditioning_policy="fixed_original_prompt",
            feedback_dim=3 if study.feedback else 0,
            rollout_tokens=study.rollout_tokens,
        )
        try:
            return run_training(
                study, runtime, cfg, job, split, parent, metadata, state, resume=resume
            )
        finally:
            runtime.corpus.cache.close()


def run_training(study, runtime, cfg, job, split, parent, metadata, state, *, resume):  # noqa: PLR0917, C901 -- durable online training lifecycle
    output, started = study.output, time.perf_counter()
    identity = run_identity(runtime.cfg, runtime.bank, runtime.corpus, job)
    identity.update(
        study=study.model_dump(mode="json"),
        membership_identity=split["identity"],
        parent={k: v for k, v in parent.items() if k != "path"},
    )
    identity_path = output / "resolved_experiment.json"
    if identity_path.exists():
        if not resume:
            raise FileExistsError("Feedback experiment exists; pass --resume")
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError(
                "Feedback experiment configuration changed; choose a new name/workspace"
            )
        if (output / "result.json").is_file():
            result = json.loads((output / "result.json").read_text())
            tracker = ExperimentTracker(runtime.cfg, job, output, identity)
            if not tracker.already_finished():
                with tracker:
                    publish_result(tracker, result)
            return result
    write_json(identity_path, identity)
    write_json(output / "configuration.json", runtime.cfg.model_dump(mode="json"))
    write_json(output / "study.json", study.model_dump(mode="json"))
    write_json(output / "membership.json", split)
    provenance = output / "provenance" / f"attempt-{time.time_ns()}"
    save_train_command(str(provenance))
    for name in ("train_command.txt", "speculators.patch"):
        if not (output / name).exists():
            shutil.copy2(provenance / name, output / name)
    shutil.copytree(
        study.workspace / "provenance/vllm-extractor", provenance / "vllm-extractor"
    )
    archive_source(Path(parent["path"]), output / "provenance/parent")
    torch.manual_seed(study.seed)
    generator = initialize_generator(runtime, study, metadata, state)
    params = [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=study.lr, weight_decay=study.weight_decay)
    progress, history = 0, []
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != digest(identity):
            raise ValueError("Checkpoint identity differs")
        generator.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        progress, history = saved["progress"], saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if runtime.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    # Keep the live generator and exported parent provenance.
    del state
    trainer = OnlineFeedbackTrainer(runtime, study, generator)
    report(
        "Online feedback: SDPA streaming; fixed prompt summary; "
        "LK excludes invalidated suffix contexts"
    )
    report(
        f"{study.name}: source_step={study.parent.optimizer_step}; "
        f"additional_updates={study.steps}; "
        f"training_examples={job['training_examples']}; "
        f"trainable_parameters={sum(p.numel() for p in params):,}"
    )
    with ExperimentTracker(runtime.cfg, job, output, identity) as tracker:
        if tracker.run is not None:
            tracker.run.summary.update(
                {
                    "run_summary/trainable_parameters": sum(p.numel() for p in params),
                    "run_summary/source_optimizer_step": study.parent.optimizer_step,
                }
            )
            for section in ("training_scores", "feedback_diagnostics"):
                tracker.run.define_metric(section + "/*", step_metric="optimizer_step")

        def validate():
            scores, traces = evaluate_online(trainer, split["groups"])
            history.append({"optimizer_step": progress, **scores})
            write_json(output / "validation.json", history)
            write_json(
                output / "validation_traces" / f"step-{progress:08d}.json", traces
            )
            tracker.log(
                {
                    "optimizer_step": progress,
                    "validation_scores/online_macro_draft_acceptance": scores["score"],
                    "validation_scores/accepted_prefix_length": scores[
                        "accepted_prefix_length"
                    ],
                    "validation_losses/verified_prefix_lk": scores["loss"],
                    "validation_performance/seconds": scores["seconds"],
                    **{
                        "validation_scores/" + k + "_draft_acceptance": v
                        for k, v in scores["categories"].items()
                    },
                }
            )

        if progress and (
            progress % study.checkpoint_interval == 0 or progress == study.steps
        ):
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
        if not history or history[-1]["optimizer_step"] != progress:
            validate()
        while progress < study.steps:
            beginning = time.perf_counter()
            episodes = common.update_members(study, split["groups"], progress)
            teacher_before = (
                trainer.teacher_requests,
                trainer.teacher_seconds,
                trainer.teacher_tokens,
            )
            trainer.prefetch([e["indices"][0] for e in episodes])
            optimizer.zero_grad(set_to_none=True)
            set_learning_rate(optimizer, runtime.cfg, progress + 1)
            loss, positions, traces = 0.0, 0, []
            for episode, item in enumerate(episodes):
                torch.manual_seed(study.seed + progress * 100003 + episode * 1009)
                value, count, trace = trainer.train_episode(
                    item["indices"][0], len(episodes)
                )
                trace["policy_optimizer_step"] = progress
                traces.append(trace)
                loss += value / len(episodes)
                positions += count
            if not positions:
                raise ValueError(
                    "All requests ended before a supervised draft decision"
                )
            gradients = clip_gradients(optimizer, study.max_grad_norm)
            # All rollouts in this batch used the same pre-update policy.
            optimizer.step()
            common.synchronize(runtime)
            progress += 1
            scores = trace_metrics(traces)
            metrics = {
                "optimizer_step": progress,
                "training_losses/verified_prefix_lk": loss,
                "training_scores/draft_acceptance": scores["draft_acceptance_rate"],
                "training_scores/accepted_prefix_length": scores[
                    "accepted_prefix_length"
                ],
                "training_performance/supervised_positions": positions,
                "training_performance/generated_tokens": scores["generated_tokens"],
                "training_performance/teacher_requests": trainer.teacher_requests
                - teacher_before[0],
                "training_performance/teacher_seconds": trainer.teacher_seconds
                - teacher_before[1],
                "training_performance/teacher_input_tokens": trainer.teacher_tokens
                - teacher_before[2],
                "training_performance/update_seconds": time.perf_counter() - beginning,
                "training_performance/example_presentations": len(episodes),
                "cache_performance/prompt_cache_gib": runtime.corpus.cache.ram.bytes
                / 2**30,
                "optimization/lr": optimizer.param_groups[0]["lr"],
                **{
                    "optimization/" + key.removeprefix("train/"): value
                    for key, value in gradients.items()
                },
            }
            if study.feedback:
                metrics["feedback_diagnostics/projector_weight_norm"] = float(
                    generator.feedback_encoder.weight.detach().norm()
                )
            tracker.log(metrics)
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics) + "\n")
            write_json(output / "online_traces" / f"update-{progress:08d}.json", traces)
            if progress == 1 or progress % 10 == 0:
                report(
                    f"{study.name}: update={progress}/{study.steps}; "
                    f"wall={metrics['training_performance/update_seconds']:.2f}s; "
                    f"acceptance={scores['draft_acceptance_rate']:.4f}; "
                    f"supervised_positions={positions}; "
                    f"prompt_cache={runtime.corpus.cache.ram.bytes / 2**30:.2f} GiB"
                )
            if progress % study.validation_interval == 0 or progress == study.steps:
                validate()
            if (
                progress % study.checkpoint_interval == 0
                or progress % study.hub_interval == 0
                or progress == study.steps
            ):
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
        result = {
            "status": "complete",
            "job": job,
            "completed_updates": progress,
            "source_optimizer_step": study.parent.optimizer_step,
            "lineage_optimizer_step": study.parent.optimizer_step + progress,
            "score": history[-1]["score"],
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(p.numel() for p in params),
            "checkpoint_sha256": file_digest(checkpoint),
            "budget": "same prompts and additional updates; "
            "verifier and supervised-token costs measured separately",
        }
        write_json(output / "result.json", result)
        publish_result(tracker, result)
        return result
