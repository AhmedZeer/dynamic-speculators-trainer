"""Continue the original saved-response generator pipeline from a pinned bundle."""

import json
import shutil
import time
from pathlib import Path

import torch
from filelock import FileLock

from speculators.bank.artifacts import digest, write_json
from speculators.bank.progress import report, stage_progress
from speculators.benchmark.config import Model
from speculators.benchmark.prepare import resolve_model
from speculators.generator import study as common
from speculators.generator.config import Architecture
from speculators.generator.engine import run_identity
from speculators.generator.feedback_execution import original_execution_settings
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.model import LoRAGenerator
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
    # AdapterLinear placeholders also stay frozen; functional substitution uses
    # the generated factors, whose gradients belong to the generator optimizer.
    runtime.drafter.requires_grad_(False)
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
    model.requires_grad_(study.trainable in {"generator", "generator_and_projector"})
    if study.trainable == "projector":
        model.feedback_encoder.requires_grad_(True)
    return model.eval()


def train_feedback(
    study, *, resume=False, dry_run=False, extraction_concurrency=0, ram_cache_gib=None
):
    if dry_run:
        return {
            "configuration": study.model_dump(
                mode="json",
                exclude={"rollout_tokens", "replay_rounds_per_episode", "initial_step"},
            ),
            "training_examples": 4 * study.train_per_category,
            "additional_updates": study.steps,
            "execution": {
                "extraction_concurrency": extraction_concurrency or "configured",
                "ram_cache_gib": ram_cache_gib or "auto",
            },
            "training_pipeline": "original_saved_response",
            "parent_optimizer": "restore"
            if study.inherit_parent_optimizer
            else "fresh",
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
        runtime.cfg.wandb.group = study.wandb.group
        execution = original_execution_settings(
            runtime, concurrency=extraction_concurrency, ram_cache_gib=ram_cache_gib
        )
        job.update(
            training_mode="original_saved_response",
            training_schedule="original_epoch_cohorts",
            trainable=study.trainable,
            source_optimizer_step=study.parent.optimizer_step,
            additional_updates=study.steps,
            feedback_source="current_model_saved_prefix_ttt_agreement"
            if study.feedback
            else "none",
            conditioning_policy="fixed_original_prompt",
            feedback_dim=3 if study.feedback else 0,
        )
        try:
            return run_training(
                study,
                runtime,
                cfg,
                job,
                split,
                parent,
                metadata,
                state,
                resume=resume,
                execution=execution,
            )
        finally:
            runtime.corpus.cache.close()


def run_training(  # noqa: PLR0917, C901 -- durable saved-response continuation lifecycle
    study, runtime, cfg, job, split, parent, metadata, state, *, resume, execution
):
    output, started = study.output, time.perf_counter()
    identity = run_identity(runtime.cfg, runtime.bank, runtime.corpus, job)
    identity.update(
        study=study.model_dump(
            mode="json",
            exclude={"rollout_tokens", "replay_rounds_per_episode", "initial_step"},
        ),
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
                    common.publish_study_result(tracker, result)
            return result
    write_json(identity_path, identity)
    write_json(output / "configuration.json", runtime.cfg.model_dump(mode="json"))
    write_json(
        output / "study.json",
        study.model_dump(
            mode="json",
            exclude={"rollout_tokens", "replay_rounds_per_episode", "initial_step"},
        ),
    )
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
    elif study.inherit_parent_optimizer:
        from speculators.generator.continuation_state import restore_parent_optimizer  # noqa: PLC0415

        with stage_progress(
            f"Restoring parent Adam/RNG at step {study.parent.optimizer_step}"
        ):
            receipt = restore_parent_optimizer(study, metadata, generator, optimizer)
        write_json(output / "parent_training_state.json", receipt)
    # Keep the live generator and exported parent provenance.
    del state
    if not checkpoint.exists() and (output / "validation.json").exists():
        previous = json.loads((output / "validation.json").read_text())
        history = [row for row in previous if row["optimizer_step"] == 0][:1]
        if history:
            report("Reusing matching original-pipeline step-0 validation")
    from speculators.generator.original_continuation import run_original_loop  # noqa: PLC0415

    write_json(output / "execution.json", execution)
    write_json(provenance / "execution.json", execution)
    return run_original_loop(
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
    )
