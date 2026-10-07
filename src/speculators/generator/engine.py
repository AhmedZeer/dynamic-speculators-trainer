"""Frozen-drafter evaluation, bank reconstruction, and matched adaptation."""

import json
import random
import shutil
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch.func import functional_call

from speculators.bank.artifacts import digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.progress import report, stage_progress
from speculators.generator.config import ExperimentConfig
from speculators.generator.data import Corpus, FactorCache, atomic_torch_save, groups
from speculators.generator.model import (
    LoRAGenerator,
    backward_factors,
    detached_factors,
    factor_loss,
    factor_parameters,
    install_adapters,
)
from speculators.generator.tracking import ExperimentTracker, epoch_updates
from speculators.losses import resolve_loss_config
from speculators.model import SpeculatorModel
from speculators.train.utils import normalize_counted_metrics, save_train_command


def clip_gradients(optimizer, max_norm):
    """Clip accumulated optimizer gradients and report their original L2 norm."""
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, max_norm, error_if_nonfinite=True
    ).item()
    count = sum(p.numel() for p in parameters if p.grad is not None)
    return {
        "train/grad_norm": norm,
        "train/gradient_clipped": float(norm > max_norm),
        "train/clip_scale": min(1.0, max_norm / (norm + 1e-6)),
        "train/grad_rms": norm / max(count, 1) ** 0.5,
    }


def set_learning_rate(optimizer, cfg, step):
    warmup = cfg.optimization.warmup_updates
    scale = min(step / warmup, 1.0) if warmup else 1.0
    for group in optimizer.param_groups:
        group["lr"] = cfg.optimization.lr * scale


class Runtime:
    def __init__(self, cfg, bank):
        self.cfg, self.bank = cfg, bank
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; use device: cpu for a synthetic smoke test"
            )
        self.dtype = getattr(torch, cfg.dtype)
        self.corpus = Corpus(cfg, bank)
        paths = {}
        for role in ("target", "drafter"):
            with stage_progress(f"Resolving pinned {role} snapshot"):
                paths[role] = snapshot_download(
                    getattr(bank.models, role),
                    revision=self.corpus.manifest["revisions"][f"{role}_sha"],
                )
        with stage_progress("Loading frozen EAGLE3 and verifier head/norm"):
            # Convert external drafter format before setting its attention backend.
            from speculators.convert.entrypoints import (  # noqa: PLC0415
                maybe_convert_external_checkpoint,
            )

            source = maybe_convert_external_checkpoint(
                paths["drafter"], verifier=paths["target"]
            )
            config = SpeculatorModel.config_class.from_pretrained(source)
            config.speculators_config.verifier.name_or_path = paths["target"]
            config.transformer_layer_config._attn_implementation = (  # noqa: SLF001
                bank.execution.draft_attn_impl
            )
            self.drafter = SpeculatorModel.from_pretrained(
                source, config=config, dtype=self.dtype
            )
            if list(self.drafter.target_layer_ids) != bank.hidden_states.layer_ids[:-1]:
                raise ValueError("Drafter and saved target layers differ")
            self.drafter.to(self.device).requires_grad_(False).eval()
        self.corpus.prepare(self.drafter)
        self.paths = install_adapters(
            self.drafter, bank.lora.rank, bank.lora.alpha, bank.lora.dropout
        )
        self.shapes = {
            name: (
                self.drafter.get_submodule(path).base.in_features,
                self.drafter.get_submodule(path).base.out_features,
            )
            for name, path in self.paths.items()
        }
        self.cache = FactorCache(cfg.factor_cache_mib, self.shapes, bank.lora.rank)
        self.call_kwargs = {
            "ttt_steps": bank.training.ttt_steps,
            "ttt_step_loss_decay": bank.training.ttt_step_loss_decay,
            "loss_config": resolve_loss_config(
                cfg.optimization.loss_fn or bank.training.loss_fn
            ),
        }
        # Keep functional parameter substitution outside the model graph while
        # allowing standalone attention kernels to compile and retain fusion.
        forward = type(self.drafter).forward
        forward = getattr(forward, "_torchdynamo_orig_callable", forward)
        self.drafter.forward = torch.compiler.disable(
            forward.__get__(self.drafter), recursive=False
        )
        report(
            f"Generator drafter backend={bank.execution.draft_attn_impl}; "
            "model forward uncompiled, CUDA FlexAttention kernel compiled separately"
        )

    def new_generator(self, condition):
        torch.manual_seed(self.cfg.seed)
        width = self.corpus.summaries[condition].shape[-1]
        generator = LoRAGenerator(
            width,
            self.shapes,
            self.bank.lora.rank,
            self.cfg.architecture,
            self.cfg.context.max_examples,
        ).to(self.device)
        report(
            f"Generator parameters={sum(p.numel() for p in generator.parameters()):,}; "
            f"condition={condition}"
        )
        return generator

    def adapter_state(self):
        return {
            name: tuple(
                getattr(self.drafter.get_submodule(path), suffix)
                for suffix in ("a", "b")
            )
            for name, path in self.paths.items()
        }

    def initialize_direct(self, source=None):
        torch.manual_seed(self.cfg.seed)
        targets = self.cache.get(source) if source else None
        self.drafter.requires_grad_(False)
        for name, path in self.paths.items():
            adapter = self.drafter.get_submodule(path)
            with torch.no_grad():
                if targets:
                    adapter.a.copy_(targets[name][0])
                    adapter.b.copy_(targets[name][1])
                else:
                    torch.nn.init.kaiming_uniform_(adapter.a, a=5**0.5)
                    adapter.b.zero_()
            adapter.a.requires_grad_(True)
            adapter.b.requires_grad_(True)

    def forward(self, batch, factors=None):
        with torch.autocast(
            self.device.type, dtype=self.dtype, enabled=self.dtype != torch.float32
        ):
            kwargs = {**batch, **self.call_kwargs}
            if factors is None:
                return self.drafter(**kwargs)
            return functional_call(
                self.drafter, factor_parameters(self.paths, factors), (), kwargs
            )

    @torch.no_grad()
    def evaluate(self, subset, condition, generator=None):
        self.drafter.eval()
        if generator is not None:
            generator.eval()
        indices = self.corpus.indices(subset, validation=True)
        results = {}
        for size in self.cfg.context.evaluation_sizes:
            accumulated = {}
            cohorts = list(groups(indices, self.cfg.seed + 9000, size, size))
            with stage_progress(
                f"Validation subset={subset}, context={size}, examples={len(indices)}"
            ):
                for number, cohort in enumerate(cohorts, 1):
                    factors = (
                        generator(*self.corpus.context(cohort, condition, self.device))
                        if generator is not None
                        else None
                    )
                    for batch, _ in self.corpus.microbatches(
                        cohort, self.device, self.dtype
                    ):
                        _, _, metrics = self.forward(batch, factors)
                        for key, value in metrics.items():
                            accumulated[key] = accumulated.get(key, 0.0) + float(value)
                    if (
                        number == 1
                        or number % self.cfg.optimization.log_interval == 0
                        or number == len(cohorts)
                    ):
                        report(
                            f"Validation context={size}: group {number}/{len(cohorts)}"
                        )
            normalized = normalize_counted_metrics(accumulated)
            if "full_acc_0" not in normalized:
                raise ValueError("Drafter did not report full_acc_0")
            results[str(size)] = normalized
        return {
            "contexts": results,
            "score": sum(v["full_acc_0"] for v in results.values()) / len(results),
            "validation_examples": len(indices),
            "tail_policy": "include all examples; short final group uses actual size",
        }

    def train_epoch(
        self,
        subset,
        condition,
        generator,
        optimizer,
        epoch,
        tracker=None,
        optimizer_step=0,
    ):
        self.drafter.train()
        if generator is not None:
            self.drafter.requires_grad_(False)
            generator.train()
        cohorts = list(
            groups(
                self.corpus.indices(subset),
                self.cfg.seed + epoch,
                self.cfg.context.min_examples,
                self.cfg.context.max_examples,
            )
        )
        started = time.monotonic()
        for update, cohort in enumerate(cohorts, 1):
            update_started = time.monotonic()
            set_learning_rate(optimizer, self.cfg, optimizer_step + update)
            torch.manual_seed(self.cfg.seed + 100000 * (epoch + 1) + update)
            optimizer.zero_grad(set_to_none=True)
            factors = (
                generator(*self.corpus.context(cohort, condition, self.device))
                if generator is not None
                else None
            )
            # Compiled attention backward may donate its saved buffers. Give
            # each drafter microbatch an independent graph ending at adapter
            # leaves, then backpropagate their total derivative through G once.
            leaves = detached_factors(factors) if factors is not None else None
            token_total = sum(sum(self.corpus.rows[i]["loss_mask"][1:]) for i in cohort)
            if not token_total:
                raise ValueError("Training cohort has no response tokens")
            total_loss = 0.0
            data_seconds = 0.0
            for batch, _ in self.corpus.microbatches(
                cohort, self.device, self.dtype, training=True
            ):
                data_seconds += self.corpus.last_batch_data_seconds
                _, loss, _ = self.forward(batch, leaves)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite drafter training loss")
                weight = batch["loss_mask"].sum().item() / token_total
                (loss * weight).backward()
                total_loss += loss.detach().item() * weight
            if factors is not None:
                backward_factors(factors, leaves)
            gradient_metrics = clip_gradients(
                optimizer, self.cfg.optimization.max_grad_norm
            )
            optimizer.step()
            if tracker is not None:
                tracker.training(
                    optimizer_step + update,
                    {
                        "train/drafter_loss": total_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                        "train/context_examples": len(cohort),
                        "train/response_tokens": token_total,
                        "train/data_wait_seconds": data_seconds,
                        "train/update_seconds": time.monotonic() - update_started,
                        "train/activation_cache_hits": self.corpus.activations.hits,
                        "train/activation_cache_misses": self.corpus.activations.misses,
                        "train/activation_cache_bypasses": (
                            self.corpus.activations.bypasses
                        ),
                        **gradient_metrics,
                    },
                    final=update == len(cohorts),
                )
            if (
                update == 1
                or update % self.cfg.optimization.log_interval == 0
                or update == len(cohorts)
            ):
                report(
                    f"Epoch {epoch + 1}: update={update}/{len(cohorts)}, "
                    f"loss={total_loss:.6f}, examples={len(cohort)}, "
                    f"data wait/preparation={data_seconds:.2f}s, "
                    f"grad norm={gradient_metrics['train/grad_norm']:.3f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.3g}, "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
        return len(cohorts)


def run_identity(cfg, bank, corpus, job):
    settings = cfg.model_dump(mode="json")
    for key in (
        "n_workers",
        "staging_dir",
        "factor_cache_mib",
        "wandb",
        "preprocessing_workers",
        "activation_cache_gib",
        "activation_cache_reserve_gib",
        "activation_prefetch_workers",
        "condition",
        "pretrained_checkpoint",
    ):
        settings.pop(key)
    if not cfg.optimization.warmup_updates:
        settings["optimization"].pop("warmup_updates")
    if cfg.optimization.loss_fn is None:
        settings["optimization"].pop("loss_fn")
    return {
        "version": 1,
        "configuration": settings,
        "bank_data": corpus.identity,
        "lora": bank.lora.model_dump(mode="json"),
        "drafter_training": {
            **{
                key: getattr(bank.training, key)
                for key in ("noise_std", "ttt_steps", "ttt_step_loss_decay")
            },
            "loss_fn": cfg.optimization.loss_fn or bank.training.loss_fn,
        },
        "attention": bank.execution.draft_attn_impl,
        "job": job,
    }


def begin_run(path, identity):
    path.mkdir(parents=True, exist_ok=True)
    previous = path / "resolved_experiment.json"
    if previous.exists() and json.loads(previous.read_text()) != identity:
        raise ValueError(
            f"Experiment configuration changed at {path}; choose a new output_root"
        )
    write_json(previous, identity)
    if (path / "result.json").exists():
        report(f"Skipping completed matching experiment: {path}")
        return False
    if (path / "train_command.txt").exists():
        archive = path / "provenance" / f"attempt-{time.time_ns()}"
        archive.mkdir(parents=True)
        for name in ("train_command.txt", "speculators.patch"):
            if (path / name).exists():
                shutil.copy2(path / name, archive / name)
    # A restart may use a bug fix: record the code/command actually resumed,
    # retaining earlier attempts so the full run can still be reproduced.
    save_train_command(str(path))
    return True


def run_job(runtime, job, output):
    identity = run_identity(runtime.cfg, runtime.bank, runtime.corpus, job)
    with ExperimentTracker(runtime.cfg, job, output, identity) as tracker:
        return _run_job(runtime, job, output, tracker)


def _run_job(runtime, job, output, tracker):  # noqa: C901 -- recovery and two training schedules
    cfg, bank = runtime.cfg, runtime.bank
    condition = job["condition"]
    generator = (
        None
        if job["kind"] == "adaptation"
        and job["arm"] in ("fresh_lora", "transferred_lora")
        else runtime.new_generator(condition)
    )
    if generator is None:
        runtime.initialize_direct(job.get("transfer_path"))
        parameters = [p for p in runtime.drafter.parameters() if p.requires_grad]
    else:
        runtime.drafter.requires_grad_(False)
        parameters = list(generator.parameters())
        if job.get("pretrained_path"):
            pretrained = torch.load(
                job["pretrained_path"], map_location="cpu", weights_only=True
            )
            generator.load_state_dict(pretrained["model"])
            del pretrained
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.optimization.lr, weight_decay=cfg.optimization.weight_decay
    )
    identity = run_identity(cfg, bank, runtime.corpus, job)
    checkpoint = output / "checkpoint.pt"
    local_checkpoint = (
        runtime.corpus.local / "runs" / digest(identity) / "checkpoint.pt"
    )
    history, progress = [], 0
    optimizer_step = 0
    rng = random.Random(cfg.seed)

    def model_state():
        state = (
            generator.state_dict()
            if generator is not None
            else factor_parameters(runtime.paths, runtime.adapter_state())
        )
        return {key: value.detach().cpu().clone() for key, value in state.items()}

    def save(progress_value):
        with stage_progress(
            f"Saving recovery checkpoint at {output}, progress={progress_value}"
        ):
            payload = {
                "model": model_state(),
                "optimizer": optimizer.state_dict(),
                "progress": progress_value,
                "optimizer_step": optimizer_step,
                "history": history,
                "rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
            }
            atomic_torch_save(local_checkpoint, payload)
            pending = checkpoint.with_suffix(".pt.pending")
            # Streaming copy avoids an extra multi-GB serialization buffer.
            shutil.copyfile(local_checkpoint, pending)
            pending.replace(checkpoint)
            local_checkpoint.unlink()

    if checkpoint.exists():
        with stage_progress(f"Restoring recovery checkpoint {checkpoint}"):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if generator is not None:
                generator.load_state_dict(state["model"])
            else:
                runtime.drafter.load_state_dict(state["model"], strict=False)
            optimizer.load_state_dict(state["optimizer"])
            progress, history = state["progress"], state["history"]
            optimizer_step = state.get("optimizer_step")
            if optimizer_step is None:
                optimizer_step = (
                    progress
                    if job["kind"] == "heatmap"
                    else epoch_updates(
                        cfg, runtime.corpus, cfg.conditioning_subset, progress
                    )
                )
            rng.setstate(state["rng"])
            torch.set_rng_state(state["torch_rng"])
            if state["cuda_rng"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            del state
    started = time.monotonic()
    subset = cfg.bank_subset if job["kind"] == "heatmap" else cfg.conditioning_subset
    if job["kind"] == "heatmap":
        selected = job["snapshots"]
        by_seed = {
            seed: [s for s in selected if s["seed"] == seed] for seed in job["seeds"]
        }
        indices = runtime.corpus.indices(subset)
        generator.train()
        for update in range(progress, cfg.optimization.pretraining_updates):
            set_learning_rate(optimizer, cfg, update + 1)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            context_sizes = []
            for _ in range(cfg.optimization.episodes_per_update):
                size = rng.randint(cfg.context.min_examples, cfg.context.max_examples)
                context_sizes.append(size)
                cohort = rng.sample(indices, size)
                target = rng.choice(by_seed[rng.choice(job["seeds"])])
                generated = generator(
                    *runtime.corpus.context(cohort, condition, runtime.device)
                )
                loss = factor_loss(generated, runtime.cache.get(target["path"]))
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite bank reconstruction loss")
                (loss / cfg.optimization.episodes_per_update).backward()
                losses.append(loss.detach().item())
            gradient_metrics = clip_gradients(optimizer, cfg.optimization.max_grad_norm)
            optimizer.step()
            progress = update + 1
            optimizer_step = progress
            tracker.training(
                optimizer_step,
                {
                    "train/reconstruction_l1": sum(losses) / len(losses),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/context_examples_mean": sum(context_sizes)
                    / len(context_sizes),
                    **gradient_metrics,
                },
                final=progress == cfg.optimization.pretraining_updates,
            )
            if progress == 1 or progress % cfg.optimization.log_interval == 0:
                report(
                    f"Bank reconstruction {output.name}: "
                    f"update={progress}/{cfg.optimization.pretraining_updates}, "
                    f"L1={sum(losses) / len(losses):.6f}"
                )
            if progress % cfg.optimization.checkpoint_interval == 0:
                save(progress)
        if progress % cfg.optimization.checkpoint_interval or not checkpoint.exists():
            save(progress)
        metrics = runtime.evaluate(subset, condition, generator)
        history.append({"update": progress, **metrics})
        tracker.validation(metrics, optimizer_step)
    else:
        epochs = (
            cfg.optimization.conditioning_epochs
            if job["kind"] == "conditioning"
            else cfg.optimization.adaptation_epochs
        )
        if not history:
            history.append(
                {"epoch": 0, **runtime.evaluate(subset, condition, generator)}
            )
            tracker.validation(history[-1], optimizer_step, epoch=0)
            save(0)
        for epoch in range(progress, epochs):
            with stage_progress(
                f"Training {job['kind']} {output.name}, epoch={epoch + 1}/{epochs}"
            ):
                optimizer_step += runtime.train_epoch(
                    subset,
                    condition,
                    generator,
                    optimizer,
                    epoch,
                    tracker=tracker,
                    optimizer_step=optimizer_step,
                )
            history.append(
                {
                    "epoch": epoch + 1,
                    "optimizer_step": optimizer_step,
                    **runtime.evaluate(subset, condition, generator),
                }
            )
            tracker.validation(history[-1], optimizer_step, epoch=epoch + 1)
            save(epoch + 1)
    result = {
        "job": job,
        "history": history,
        "score": history[-1]["score"],
        "elapsed_seconds": time.monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "checkpoint_path": str(checkpoint.resolve()),
    }
    if job["kind"] == "adaptation":
        result["score_change"] = result["score"] - history[0]["score"]
    write_json(output / "metrics.json", history)
    write_json(output / "result.json", result)
    tracker.result(result)
    report(f"Experiment complete: {output}; score={result['score']:.6f}")
    return result


def worker(job_path):
    payload = json.loads(job_path.read_text())
    cfg = ExperimentConfig.model_validate(payload["configuration"])
    bank = BankConfig.load(cfg.bank_config)
    corpus = Corpus(cfg, bank)
    job, output = payload["job"], Path(payload["output"])
    if begin_run(output, run_identity(cfg, bank, corpus, job)):
        run_job(Runtime(cfg, bank), job, output)
