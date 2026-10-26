"""Individually runnable, resumable 4K generator and static-LoRA studies."""

import json
import os
import random
import shutil
import time

import torch
from filelock import FileLock

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.progress import report
from speculators.generator.campaign import job_config
from speculators.generator.campaign_train import synchronize, train_update
from speculators.generator.data import atomic_torch_save
from speculators.generator.engine import (
    Runtime,
    clip_gradients,
    run_identity,
    set_learning_rate,
)
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.model import LoRAGenerator, factor_parameters
from speculators.generator.online_activations import (
    ActivationRAM,
    OnlineActivations,
    OnlineCorpus,
)
from speculators.generator.serving.bundle import PROVENANCE, export_run
from speculators.generator.study_config import campaign_settings
from speculators.generator.study_inputs import membership
from speculators.generator.tracking import ExperimentTracker
from speculators.losses import resolve_loss_config
from speculators.train.utils import normalize_counted_metrics, save_train_command


def update_members(study, members, step):
    """Balanced request-level gradient accumulation; no shared conditioning sets."""
    categories = sorted(members)
    values = []
    for episode in range(study.examples_per_update):
        category = categories[
            (step * study.examples_per_update + episode) % len(categories)
        ]
        pool = members[category]["train"]
        # Epoch shuffles cover each category's entire pool before repeating.
        occurrence = (step * study.examples_per_update + episode) // len(categories)
        epoch, offset = divmod(occurrence, len(pool))
        order = list(pool)
        random.Random(study.seed + epoch * 100003 + categories.index(category)).shuffle(
            order
        )
        values.append(
            {
                "subset": f"{category}-{study.training_subset:05d}",
                "indices": [order[offset]],
            }
        )
    return values


def build_runtime(study, inputs):
    cfg = campaign_settings(study)
    subsets = [f"{d}-{study.training_subset:05d}" for d in inputs.banks]
    held = [f"{d}-{study.validation_subset:05d}" for d in inputs.banks]
    job = {
        "kind": "pretraining",
        "stage": "pretraining",
        "label": study.name,
        "condition": "last",
        "arm": "fresh_lora" if study.arm == "lora" else "direct",
        "objective": "direct",
        "subsets": subsets,
        "validation_subsets": held,
        "updates": study.steps,
        "training_examples": study.train_per_category * len(inputs.banks),
        "feedback_dim": 3 if study.feedback else 0,
        "conditioning_policy": study.conditioning,
        "training_mode": "replay" if study.replay else "offline",
    }
    local = job_config(cfg, job)
    ram = ActivationRAM(cfg.online.ram_cache_gib)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"], ram=ram)
    cache.check_server()
    if cfg.online.compiler_cache_dir is not None:
        compiler = cfg.online.compiler_cache_dir.resolve()
        compiler.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compiler)
        os.environ["TRITON_CACHE_DIR"] = str(compiler / "triton")
    corpus = OnlineCorpus(local, inputs.banks["math"], cache, inputs)
    runtime = Runtime(
        local,
        inputs.banks["math"],
        corpus_factory=lambda _c, _b: corpus,
        model_paths={
            "target": cfg.online.target_model_path,
            "drafter": cfg.online.drafter_model_path,
        },
    )
    runtime.call_kwargs["loss_config"] = resolve_loss_config(
        study.loss, study.loss_implementation
    )
    return runtime, cfg, job


def initialize_parent(study, model, optimizer):
    if study.initial_checkpoint is None:
        return 0
    parent = torch.load(study.initial_checkpoint, map_location="cpu", weights_only=True)
    if parent["progress"] != study.initial_step:
        raise ValueError("Parent checkpoint step differs from initial_step")
    parent_identity = json.loads(
        (study.initial_checkpoint.parent / "resolved_experiment.json").read_text()
    )
    source_study = parent_identity.get("study", {})
    if (
        source_study.get("loss") != "lk_hybrid"
        or source_study.get("replay")
        or source_study.get("arm") != "generator"
    ):
        raise ValueError("Replay branches require an ordinary LK generator parent")
    for key in (
        "seed",
        "training_subset",
        "train_per_category",
        "examples_per_update",
        "lr",
        "weight_decay",
    ):
        if source_study.get(key) != getattr(study, key):
            raise ValueError(f"Parent training setup differs: {key}")
    incompatible = model.load_state_dict(parent["model"], strict=False)
    expected = {"feedback_encoder.weight"} if study.feedback else set()
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError("Parent generator architecture differs")
    # Map optimizer slots by parameter names, including zero-initialized new
    # feedback parameters, rather than guessing positions after insertion.
    names = list(model.state_dict())
    source_names = parent["parameter_names"]
    old = parent["optimizer"]
    new = optimizer.state_dict()
    mapping = dict(zip(source_names, old["param_groups"][0]["params"], strict=True))
    for name, identifier in zip(names, new["param_groups"][0]["params"], strict=True):
        if name in mapping and mapping[name] in old["state"]:
            new["state"][identifier] = old["state"][mapping[name]]
    optimizer.load_state_dict(new)
    torch.set_rng_state(parent["torch_rng"])
    if torch.cuda.is_available() and parent["cuda_rng"]:
        torch.cuda.set_rng_state_all(parent["cuda_rng"])
    return parent["progress"]


@torch.no_grad()
def validation(runtime, study, members, generator):
    runtime.drafter.eval()
    if generator is not None:
        generator.eval()
    categories, losses = {}, {}
    for category, splits in members.items():
        accumulated = {}
        indices = splits["tune"]
        for number, index in enumerate(indices):
            upcoming = indices[number : number + 8]
            runtime.corpus.cache.prefetch(
                runtime.corpus.references(upcoming, full=True)
            )
            kwargs = (
                {"feedback": torch.zeros(3, device=runtime.device)}
                if study.feedback
                else {}
            )
            factors = (
                generator(
                    *runtime.corpus.context([index], "last", runtime.device), **kwargs
                )
                if generator is not None
                else None
            )
            for batch, _ in runtime.corpus.microbatches(
                [index], runtime.device, runtime.dtype
            ):
                _, _, metrics = runtime.forward(batch, factors)
                for name, value in metrics.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(value)
        normalized = normalize_counted_metrics(accumulated)
        categories[category] = normalized["full_acc_0"]
        for key, value in normalized.items():
            if key.startswith("loss_"):
                losses[key] = losses.get(key, 0.0) + value / len(members)
    # This fixed prompt-state diagnostic is shared even by prefix/feedback arms;
    # it is not reported as an on-policy refresh validation measurement.
    return {
        "score": sum(categories.values()) / len(categories),
        "categories": categories,
        "losses": losses,
        "protocol": "singleton prompt-conditioned TTT; zero feedback; tuning split",
    }


def save_snapshot(  # noqa: PLR0917 -- immutable training snapshot and portable export
    study, runtime, cfg, job, identity, generator, optimizer, progress, history
):
    output = study.output
    state = (
        generator.state_dict()
        if generator is not None
        else factor_parameters(runtime.paths, runtime.adapter_state())
    )
    names = (
        list(generator.state_dict())
        if generator is not None
        else [name for name, p in runtime.drafter.named_parameters() if p.requires_grad]
    )
    payload = {
        "identity": digest(identity),
        "model": {k: v.detach().cpu() for k, v in state.items()},
        "optimizer": optimizer.state_dict(),
        "parameter_names": names,
        "progress": progress,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all()
        if runtime.device.type == "cuda"
        else [],
    }
    atomic_torch_save(output / "checkpoint.pt", payload)
    snapshot = output / "checkpoints" / f"step-{progress:08d}"
    if not (snapshot / "bundle/bundle.json").is_file():
        pending = snapshot.with_name(snapshot.name + ".pending")
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        try:
            (pending / "checkpoint.pt").hardlink_to(output / "checkpoint.pt")
        except OSError:
            shutil.copy2(output / "checkpoint.pt", pending / "checkpoint.pt")
        for name in PROVENANCE:
            if name == "result.json":
                continue
            source = output / name
            if source.is_file():
                shutil.copy2(source, pending / name)
            elif source.is_dir():
                shutil.copytree(source, pending / name)
        write_json(
            pending / "result.json",
            {"status": "training_snapshot", "completed_updates": progress, "job": job},
        )
        export_run(
            None,
            pending,
            pending / "bundle",
            drafter_model_path=cfg.online.drafter_model_path,
        )
        if snapshot.exists():
            shutil.rmtree(snapshot)
        pending.replace(snapshot)
    return snapshot


def upload_snapshot(study, cfg, snapshot, progress):
    if not study.repo_id or (progress % study.hub_interval and progress != study.steps):
        return
    receipt = snapshot / "hub.json"
    if receipt.exists():
        return
    from speculators.generator.publish import push_run  # noqa: PLC0415

    result = push_run(
        snapshot,
        study.repo_id,
        drafter_model_path=cfg.online.drafter_model_path,
        provenance_dir=study.workspace / "provenance/vllm-extractor",
        include_training_checkpoint=True,
        path_in_repo=snapshot.name,
    )
    write_json(receipt, result)


def train_study(study, *, resume=False, dry_run=False):
    if dry_run:
        return {
            "configuration": study.model_dump(mode="json"),
            "output": str(study.output),
            "new_updates": study.steps
            - (study.initial_step if study.initial_checkpoint else 0),
        }
    study.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(study.output / ".training.lock"), timeout=0):
        return _train_study(study, resume=resume)


def publish_study_result(tracker, result):
    if tracker.run is None:
        return
    tracker.run.summary.update(
        {
            "run_summary/final_agreement": result["score"],
            "run_summary/completed_updates": result["completed_updates"],
            "run_summary/elapsed_seconds": result["elapsed_seconds"],
        }
    )
    tracker.artifact(
        "study-evidence",
        [
            tracker.output / name
            for name in (
                "study.json",
                "resolved_experiment.json",
                "membership.json",
                "result.json",
                "validation.json",
                "metrics.jsonl",
                "train_command.txt",
                "speculators.patch",
            )
        ]
        + list((tracker.output / "provenance").rglob("*"))
        + list((tracker.output / "replay").glob("*.json")),
    )


def _train_study(study, *, resume):
    if not (study.workspace / "provenance/vllm-extractor/vllm_command.txt").is_file():
        raise FileNotFoundError(
            "Start the extractor with the study's --provenance-dir first"
        )
    cfg = campaign_settings(study)
    inputs = load_local_inputs(cfg)
    split = membership(study, inputs)
    split["identity"] = digest(split)
    stored = json.loads((study.inputs / "membership.json").read_text())
    if split != stored:
        raise ValueError("Run prepare-experiments for this exact data membership first")
    runtime, cfg, job = build_runtime(study, inputs)
    try:
        return _train_runtime(study, runtime, cfg, job, split, resume=resume)
    finally:
        runtime.corpus.cache.close()


def _train_runtime(study, runtime, cfg, job, split, *, resume):  # noqa: C901 -- recovery, validation, replay and checkpoint lifecycle
    output, started = study.output, time.perf_counter()
    runtime_cfg = runtime.cfg
    identity = run_identity(runtime_cfg, runtime.bank, runtime.corpus, job)
    identity["study"] = study.model_dump(mode="json")
    identity["membership_identity"] = split["identity"]
    if study.initial_checkpoint is not None:
        identity["parent_checkpoint_sha256"] = file_digest(study.initial_checkpoint)
        parent_identity = json.loads(
            (study.initial_checkpoint.parent / "resolved_experiment.json").read_text()
        )
        if (
            parent_identity.get("membership_identity") != split["identity"]
            or parent_identity["lora"] != identity["lora"]
        ):
            raise ValueError("Parent data or LoRA recipe differs")
    resolved = output / "resolved_experiment.json"
    if resolved.exists():
        if not resume:
            raise FileExistsError("Experiment exists; pass --resume")
        if json.loads(resolved.read_text()) != identity:
            raise ValueError(
                "Experiment configuration changed; choose a different name/workspace"
            )
        if (output / "result.json").exists():
            result = json.loads((output / "result.json").read_text())
            tracker = ExperimentTracker(runtime_cfg, job, output, identity)
            if not tracker.already_finished():
                with tracker:
                    publish_study_result(tracker, result)
            return result
    write_json(resolved, identity)
    write_json(output / "configuration.json", runtime_cfg.model_dump(mode="json"))
    write_json(output / "study.json", study.model_dump(mode="json"))
    write_json(output / "membership.json", split)
    provenance = output / "provenance" / f"attempt-{time.time_ns()}"
    save_train_command(str(provenance))
    for name in ("train_command.txt", "speculators.patch"):
        if not (output / name).exists():
            shutil.copy2(provenance / name, output / name)
    extractor = study.workspace / "provenance/vllm-extractor"
    if not extractor.is_dir():
        raise FileNotFoundError(
            "Start the extractor with --provenance-dir "
            "<workspace>/provenance/vllm-extractor"
        )
    shutil.copytree(extractor, provenance / "vllm-extractor", dirs_exist_ok=True)
    torch.manual_seed(study.seed)
    generator = None
    if study.arm == "lora":
        runtime.initialize_direct()
        params = [p for p in runtime.drafter.parameters() if p.requires_grad]
    else:
        generator = LoRAGenerator(
            runtime.drafter.fc.weight.shape[1],
            runtime.shapes,
            runtime.bank.lora.rank,
            runtime_cfg.architecture,
            runtime_cfg.context.max_examples,
            feedback_dim=3 if study.feedback else 0,
        ).to(runtime.device)
        params = list(generator.parameters())
    optimizer = torch.optim.AdamW(params, lr=study.lr, weight_decay=study.weight_decay)
    progress, history = 0, []
    checkpoint = output / "checkpoint.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != digest(identity):
            raise ValueError("Checkpoint identity differs")
        (generator or runtime.drafter).load_state_dict(
            saved["model"], strict=generator is not None
        )
        optimizer.load_state_dict(saved["optimizer"])
        progress, history = saved["progress"], saved["history"]
        torch.set_rng_state(saved["torch_rng"])
        if runtime.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    elif generator is not None:
        progress = initialize_parent(study, generator, optimizer)
    replay = None
    if study.replay:
        from speculators.generator.study_replay import ReplayTrainer  # noqa: PLC0415

        replay = ReplayTrainer(runtime, study, generator)
    metrics_path = output / "metrics.jsonl"
    with ExperimentTracker(runtime_cfg, job, output, identity) as tracker:
        if generator is not None:
            report(f"{study.name}: preparing reusable prompt summaries")
            prepared = runtime.corpus.prepare_summaries(job["subsets"])
            tracker.log(
                {
                    "optimizer_step": progress,
                    "preparation_performance/seconds": prepared["seconds"],
                }
            )
        if tracker.run is not None:
            for section in (
                "replay_performance",
                "replay_scores",
                "training_losses",
                "validation_scores",
                "training_performance",
                "validation_performance",
            ):
                tracker.run.define_metric(section + "/*", step_metric="optimizer_step")
        if progress:
            snapshot = output / "checkpoints" / f"step-{progress:08d}"
            if progress % study.checkpoint_interval == 0 or progress == study.steps:
                if not (snapshot / "bundle/bundle.json").is_file():
                    snapshot = save_snapshot(
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
                upload_snapshot(study, cfg, snapshot, progress)

        def validate():
            beginning = time.perf_counter()
            scores = validation(runtime, study, split["groups"], generator)
            row = {"optimizer_step": progress, **scores}
            history.append(row)
            tracker.log(
                {
                    "optimizer_step": progress,
                    "validation_scores/prompt_probe_agreement": scores["score"],
                    **{
                        "validation_losses/prompt_probe_" + key: value
                        for key, value in scores["losses"].items()
                    },
                    **{
                        "validation_scores/" + k + "_prompt_probe_agreement": v
                        for k, v in scores["categories"].items()
                    },
                    "validation_performance/seconds": time.perf_counter() - beginning,
                }
            )
            write_json(output / "validation.json", history)
            report(
                f"{study.name}: validation update={progress}; "
                f"agreement={scores['score']:.5f}"
            )

        if not history or history[-1]["optimizer_step"] != progress:
            validate()
        while progress < study.steps:
            beginning = time.perf_counter()
            episodes = update_members(study, split["groups"], progress)
            # Bounded prefetch keeps ordinary training on the established RAM cache.
            future = [
                i
                for step in range(progress, min(study.steps, progress + 3))
                for e in update_members(study, split["groups"], step)
                for i in e["indices"]
            ]
            if not replay:
                runtime.corpus.cache.prefetch(
                    runtime.corpus.references(future, full=True)
                )
                values = train_update(
                    runtime,
                    generator,
                    optimizer,
                    episodes,
                    progress,
                    "direct",
                    conditioning_examples=1,
                )
                values = {
                    "training_losses/drafter_loss": values["train/drafter_loss"],
                    "training_performance/response_tokens": values[
                        "train/response_tokens"
                    ],
                    "training_performance/data_wait_seconds": values[
                        "train/data_wait_seconds"
                    ],
                    "training_performance/gpu_compute_seconds": values[
                        "train/gpu_compute_seconds"
                    ],
                }
            else:
                generator.train()
                set_learning_rate(optimizer, runtime_cfg, progress + 1)
                optimizer.zero_grad(set_to_none=True)
                replay.trace_records.clear()
                teacher_start = (
                    replay.teacher_requests,
                    replay.teacher_seconds,
                    replay.teacher_tokens,
                )
                values = {
                    "loss": 0.0,
                    "tokens": 0,
                    "accepted": 0,
                    "proposed": 0,
                    "rounds": 0,
                }
                for number, episode in enumerate(episodes):
                    torch.manual_seed(study.seed + progress * 100003 + number * 1009)
                    result = replay.episode(
                        episode["indices"][0], progress + 1, number, len(episodes)
                    )
                    for key, value in result.items():
                        values[key] += value
                clip_gradients(optimizer, runtime_cfg.optimization.max_grad_norm)
                optimizer.step()
                write_json(
                    output / "replay" / f"update-{progress + 1:08d}.json",
                    replay.trace_records,
                )
                values = {
                    "training_losses/drafter_loss": values["loss"] / len(episodes),
                    "training_performance/replayed_positions": values["tokens"],
                    "replay_scores/draft_acceptance_rate": values["accepted"]
                    / max(1, values["proposed"]),
                    "replay_performance/verification_rounds": values["rounds"],
                    "replay_performance/teacher_requests": replay.teacher_requests
                    - teacher_start[0],
                    "replay_performance/teacher_seconds": replay.teacher_seconds
                    - teacher_start[1],
                    "replay_performance/teacher_input_tokens": replay.teacher_tokens
                    - teacher_start[2],
                }
            synchronize(runtime)
            progress += 1
            cache = runtime.corpus.cache.snapshot()
            values["cache_performance/cache_gib"] = cache["cache_bytes"] / 2**30
            values["training_performance/example_presentations"] = len(episodes)
            values.update(
                optimizer_step=progress,
                **{
                    "training_performance/update_seconds": time.perf_counter()
                    - beginning,
                    "training_performance/session_seconds": time.perf_counter()
                    - started,
                },
            )
            tracker.log(values)
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(values) + "\n")
            if progress == 1 or progress % 10 == 0:
                report(
                    f"{study.name}: update={progress}/{study.steps}; "
                    f"wall={values['training_performance/update_seconds']:.2f}s"
                )
            if progress % study.validation_interval == 0 or progress == study.steps:
                validate()
            if (
                progress % study.checkpoint_interval == 0
                or progress % study.hub_interval == 0
                or progress == study.steps
            ):
                snapshot = save_snapshot(
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
                upload_snapshot(study, cfg, snapshot, progress)
        result = {
            "status": "complete",
            "job": job,
            "completed_updates": progress,
            "score": history[-1]["score"],
            "history": history,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint_sha256": file_digest(checkpoint),
            "initial_step": study.initial_step if study.initial_checkpoint else 0,
            "budget": "fixed optimizer updates; replay token/teacher costs "
            "are measured separately",
        }
        write_json(output / "result.json", result)
        publish_study_result(tracker, result)
        return result
