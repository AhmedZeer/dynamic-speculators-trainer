"""Epoch-based LoRA collection using the existing EAGLE3 training loop."""

import json
import random
import shutil
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from speculators.bank.artifacts import copy_provenance, file_digest, write_json
from speculators.bank.progress import report, stage_progress
from speculators.train.checkpointer import SingleGPUCheckpointer
from speculators.train.graceful_shutdown import with_graceful_shutdown
from speculators.train.lora import is_lora_model, save_lora_checkpoint
from speculators.train.trainer import Trainer


@dataclass(frozen=True)
class BankSchedule:
    warmup_epochs: int = 1
    collect_epochs: int = 4
    collect_lr: float = 1e-6
    save_interval: int = 10

    def __post_init__(self):
        if min(self.warmup_epochs, self.collect_epochs, self.save_interval) < 1:
            raise ValueError("Bank epoch counts and save interval must be positive")
        if self.collect_lr <= 0:
            raise ValueError("Collection LR must be positive")

    def phase(self, epoch: int) -> str:
        if epoch < self.warmup_epochs:
            return "warmup"
        if epoch < self.warmup_epochs + self.collect_epochs:
            return "collect"
        return "done"


def capture_rng() -> dict:
    np_state = np.random.get_state()  # noqa: NPY002 -- capture existing global RNG
    return {
        "python": random.getstate(),
        "numpy": [np_state[0], np_state[1].tolist(), *np_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state(  # noqa: NPY002 -- restore existing global RNG
        (name, np.array(keys, dtype=np.uint32), position, gaussian, cached)
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def raw_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


class BankCheckpointer(SingleGPUCheckpointer):
    """Only fully committed recovery directories participate in discovery."""

    def _get_previous_epoch(self) -> int:
        if not self.path.exists():
            return -1
        return max(
            (
                int(p.name)
                for p in self.path.iterdir()
                if p.is_dir()
                and p.name.isdigit()
                and (p / "training_state.json").is_file()
                and (p / "rng.pt").is_file()
                and (p / "optimizer_state_dict.pt").is_file()
                and (p / "adapter/adapter_model.safetensors").is_file()
            ),
            default=-1,
        )

    def load_optimizer_state_dict(
        self,
        model,
        optimizer,
        float_dtype=None,  # noqa: ARG002
    ):
        device = next(raw_model(model).parameters()).device
        payload = torch.load(
            self.optimizer_path(self.previous_epoch),
            map_location=device,
            weights_only=True,
        )
        states = payload if isinstance(payload, list) else [payload]
        opts = optimizer if isinstance(optimizer, list) else [optimizer]
        for opt, state in zip(opts, states, strict=True):
            opt.load_state_dict(state)

    def save_state(self, model, optimizers, state: dict) -> None:
        rng = capture_rng()
        rngs = [rng]
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            rngs = [None] * dist.get_world_size()
            dist.all_gather_object(rngs, rng)
        error = None
        if not distributed or dist.get_rank() == 0:
            pending = self.path / f".pending-{uuid.uuid4().hex}"
            try:
                pending.mkdir(parents=True)
                if not save_lora_checkpoint(
                    raw_model(model), pending, float_dtype=torch.float32
                ):
                    raise ValueError("Bank recovery requires a LoRA model")
                payload = [opt.state_dict() for opt in optimizers]
                torch.save(
                    payload[0] if len(payload) == 1 else payload,
                    pending / "optimizer_state_dict.pt",
                )
                torch.save(rngs, pending / "rng.pt")
                copy_provenance(self.path.parent, pending)
                write_json(pending / "training_state.json", state)
                destination = self.path / str(self._get_previous_epoch() + 1)
                pending.rename(destination)
                # Keep the latest and its predecessor; snapshots live elsewhere.
                committed = sorted(
                    (p for p in self.path.iterdir() if p.name.isdigit()),
                    key=lambda p: int(p.name),
                )
                for obsolete in committed[:-2]:
                    shutil.rmtree(obsolete)
            except Exception as exc:  # noqa: BLE001 -- broadcast failure to all ranks
                error = str(exc)
            finally:
                if pending.exists():
                    shutil.rmtree(pending)
        if distributed:
            result = [error]
            dist.broadcast_object_list(result, src=0)
            error = result[0]
        if error:
            raise RuntimeError(f"Bank recovery save failed: {error}")
        self.previous_epoch = self._get_previous_epoch()
        self.prev_path = self.path / str(self.previous_epoch)


class BankTrainer(Trainer):
    """One-way warmup/collect with immutable snapshots and durable resume state."""

    def __init__(self, model, config, train_loader, val_loader, *, schedule, context):
        self.schedule = schedule
        self.context = context
        self.collection_step = 0
        self._last_epoch = 0
        self._last_local_step = 0
        self.completed = False
        if config.num_epochs != schedule.warmup_epochs + schedule.collect_epochs:
            raise ValueError("Bank epoch budget must equal warmup + collection")
        if config.max_steps is not None or config.scheduler_type != "none":
            raise ValueError("Step limits and LR schedulers conflict with bank epochs")
        if config.save_best or config.fsdp_shard or val_loader is None:
            raise ValueError(
                "Bank runs require validation, no FSDP, and no best-only saves"
            )
        if not is_lora_model(raw_model(model)):
            raise ValueError("Bank runs require a PEFT LoRA model")
        if config.optimizer != "adamw" or schedule.collect_lr >= config.lr:
            raise ValueError("Use AdamW and a collection LR below the warmup LR")
        super().__init__(model, config, train_loader, val_loader)
        if self.resume_from_checkpoint and self.checkpointer.previous_epoch != -1:
            state = self._load_training_state()
            rngs = torch.load(
                self.checkpointer.prev_path / "rng.pt",
                weights_only=True,
                map_location="cpu",
            )
            if len(rngs) != (dist.get_world_size() if self.is_distributed else 1):
                raise ValueError("Resume requires the original distributed world size")
            restore_rng(rngs[self.rank])
            # Recovery is committed before the snapshot; repair an interrupted publish.
            if (
                self.collection_step
                and self.collection_step % schedule.save_interval == 0
            ):
                self._publish_snapshot(state["epoch"])
        else:
            if self.checkpointer.previous_epoch != -1:
                raise ValueError(
                    "A bank run exists; resume it or use a new output root"
                )
            self._save_recovery(0, 0, epoch_complete=False)

    def _label(self, message):
        return (
            f"subset={self.context.get('subset_id', 'bank')} "
            f"seed={self.context.get('seed', '?')} | {message}"
        )

    def _report(self, message):
        if self.rank == 0:
            report(self._label(message))

    def _stage(self, message):
        return stage_progress(self._label(message)) if self.rank == 0 else nullcontext()

    def on_train_epoch_start(self, epoch, num_steps, skip_steps):
        self._epoch_steps = num_steps
        self._epoch_started = time.perf_counter()
        self._epoch_resume_step = skip_steps
        self._report(
            f"Epoch {epoch + 1}: {num_steps} optimizer batches; "
            f"resumed after {skip_steps}; remaining={num_steps - skip_steps}"
        )

    def create_checkpointer(self):
        return BankCheckpointer(Path(self.config.save_path) / "recovery")

    def _load_training_state(self):
        if self.checkpointer.prev_path is None:
            return {}
        return json.loads(
            (self.checkpointer.prev_path / "training_state.json").read_text()
        )

    def setup_trainer(self):
        state = self._load_training_state() if self.resume_from_checkpoint else {}
        if state and state.get("context") != self.context:
            raise ValueError("Saved bank experiment identity differs from this run")
        if state and state.get("schedule") != self.schedule.__dict__:
            raise ValueError("Saved bank schedule differs from this run")
        self.current_epoch = state.get("epoch", 0) + bool(state.get("epoch_complete"))
        self._resume_local_step = (
            0 if state.get("epoch_complete") else state.get("local_step", 0)
        )
        self._report(
            f"{'Resuming' if state else 'Starting new run'}: "
            f"phase={self.schedule.phase(self.current_epoch)}, "
            f"epoch={min(self.current_epoch + 1, self.config.num_epochs)}/"
            f"{self.config.num_epochs}, "
            f"restored step={self._resume_local_step}, "
            f"global_step={state.get('global_step', 0)}, "
            f"collection_step={state.get('collection_step', 0)}"
        )
        self.global_step = state.get("global_step", 0)
        self._resume_global_step = self.global_step
        self.collection_step = state.get("collection_step", 0)
        self.best_val_loss = state.get("best_val_loss")
        if self.best_val_loss is None:
            self.best_val_loss = float("inf")
        self._last_epoch = state.get("epoch", 0)
        self._last_local_step = state.get("local_step", 0)

    def _save_recovery(self, epoch, local_step, *, epoch_complete):
        with self._stage(
            f"Saving recovery: epoch={epoch + 1}, step={local_step}, "
            f"epoch_complete={epoch_complete}"
        ):
            self.checkpointer.save_state(
                self.model,
                self.optimizers,
                {
                    "epoch": epoch,
                    "local_step": local_step,
                    "epoch_complete": epoch_complete,
                    "global_step": self.global_step,
                    "collection_step": self.collection_step,
                    "phase": self.schedule.phase(epoch + int(epoch_complete)),
                    "best_val_loss": self.best_val_loss
                    if np.isfinite(self.best_val_loss)
                    else None,
                    "schedule": self.schedule.__dict__,
                    "context": self.context,
                },
            )

    def _publish_snapshot(self, epoch):
        self._report(
            f"Publishing LoRA snapshot: collection step={self.collection_step}"
        )
        error = None
        if self.rank == 0:
            try:
                source = self.checkpointer.prev_path
                root = Path(self.config.save_path) / "snapshots"
                root.mkdir(parents=True, exist_ok=True)
                destination = root / f"step-{self.collection_step:08d}"
                expected_hash = file_digest(
                    source / "adapter/adapter_model.safetensors"
                )
                if destination.exists():
                    if (
                        file_digest(destination / "adapter/adapter_model.safetensors")
                        != expected_hash
                    ):
                        raise ValueError(
                            "Existing snapshot differs from recovery state"
                        )
                else:
                    self._write_snapshot(
                        source, root, destination, expected_hash, epoch
                    )
            except Exception as exc:  # noqa: BLE001 -- broadcast failure to all ranks
                error = str(exc)
        if self.is_distributed:
            result = [error]
            dist.broadcast_object_list(result, src=0)
            error = result[0]
        if error:
            raise RuntimeError(f"Bank snapshot save failed: {error}")
        destination = (
            Path(self.config.save_path)
            / "snapshots"
            / f"step-{self.collection_step:08d}"
        )
        self._report(f"LoRA snapshot ready: {destination}")

    def _write_snapshot(self, source, root, destination, expected_hash, epoch):
        pending = root / f".pending-{uuid.uuid4().hex}"
        try:
            pending.mkdir()
            shutil.copytree(source / "adapter", pending / "adapter")
            copy_provenance(Path(self.config.save_path), pending)
            write_json(
                pending / "entry.json",
                {
                    **self.context,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "collection_step": self.collection_step,
                    "adapter_sha256": expected_hash,
                    "adapter_config_sha256": file_digest(
                        pending / "adapter/adapter_config.json"
                    ),
                },
            )
            pending.rename(destination)
        finally:
            if pending.exists():
                shutil.rmtree(pending)

    def after_optimizer_step(self, epoch, local_step):
        self._last_epoch, self._last_local_step = epoch, local_step
        phase = self.schedule.phase(epoch)
        if phase == "collect":
            self.collection_step += 1
        if (
            local_step == 1
            or local_step % self.config.log_freq == 0
            or local_step == self._epoch_steps
        ):
            elapsed = time.perf_counter() - self._epoch_started
            steps = local_step - self._epoch_resume_step
            rate = steps / elapsed if elapsed > 0 else 0
            eta = (self._epoch_steps - local_step) / rate if rate > 0 else 0
            lr = self.optimizers[0].param_groups[0]["lr"]
            next_save = (
                self.schedule.save_interval
                - self.collection_step % self.schedule.save_interval
            )
            self._report(
                f"{phase}: epoch={epoch + 1}/{self.config.num_epochs}, "
                f"step={local_step}/{self._epoch_steps}, "
                f"global_step={self.global_step}, "
                f"collection_step={self.collection_step}, lr={lr:g}, "
                f"{rate:.2f} steps/s, epoch ETA={eta:.0f}s"
                + (
                    f", next snapshot in {next_save} collection steps"
                    if phase == "collect"
                    else ""
                )
            )
        if (
            phase == "collect"
            and self.collection_step % self.schedule.save_interval == 0
        ):
            self._save_recovery(epoch, local_step, epoch_complete=False)
            self._publish_snapshot(epoch)

    def maybe_save_checkpoint(self, epoch, local_step=0):  # noqa: ARG002
        # Signals may interrupt an optimizer update. Resume from the last committed
        # boundary instead of persisting a partially updated model or stale counters.
        self._report("Interrupted; restart from the last durable recovery checkpoint")

    @with_graceful_shutdown()
    def run_training(self):
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            lr = (
                self.config.lr
                if self.schedule.phase(epoch) == "warmup"
                else self.schedule.collect_lr
            )
            for opt in self.optimizers:
                for group in opt.param_groups:
                    group["lr"] = lr
            with self._stage(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"{self.schedule.phase(epoch)}, lr={lr:g}"
            ):
                self.train_epoch(epoch)
            self._save_recovery(epoch, self._last_local_step, epoch_complete=False)
            with self._stage(f"Epoch {epoch + 1}: evaluating held-out examples"):
                metrics = self.val_epoch(epoch)
            self._report(
                f"Epoch {epoch + 1} validation metrics: "
                f"{json.dumps(metrics, sort_keys=True)}"
            )
            if metrics and "loss_epoch" in metrics:
                self.best_val_loss = min(self.best_val_loss, metrics["loss_epoch"])
            if self.rank == 0:
                write_json(
                    Path(self.config.save_path) / "epochs" / f"{epoch}.json",
                    {
                        "epoch": epoch,
                        "phase": self.schedule.phase(epoch),
                        "global_step": self.global_step,
                        "collection_step": self.collection_step,
                        "lr": lr,
                        "metrics": metrics,
                    },
                )
            transfer = (
                getattr(self.train_loader.dataset, "transfer", None)
                if hasattr(self.train_loader, "dataset")
                else None
            )
            if hasattr(transfer, "stats"):
                self._report(
                    f"Raw data cache: {json.dumps(transfer.stats(), sort_keys=True)}"
                )
            self._save_recovery(epoch, self._last_local_step, epoch_complete=True)
            self._last_local_step = 0
        self.completed = True
        self._report(
            f"Run complete: optimizer updates={self.global_step}, "
            f"collected LoRAs={self.collection_step // self.schedule.save_interval}, "
            f"output={self.config.save_path}"
        )
