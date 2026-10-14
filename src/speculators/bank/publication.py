"""Publish completed LoRA runs atomically and restore their portable artifacts."""

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.hub import Publisher, fetch, hub_progress
from speculators.bank.inspect import inspect_adapters
from speculators.bank.progress import report


def run_identity(cfg, manifest, subset, seed):
    return {
        "version": 1,
        "prepared_fingerprint": manifest["prepared_fingerprint"],
        "cache_fingerprint": manifest["cache_fingerprint"],
        "revisions": manifest["revisions"],
        "models": {"target": cfg.models.target, "drafter": cfg.models.drafter},
        "source": {
            key: getattr(cfg.hub, key)
            for key in (
                "trajectories_repo",
                "trajectories_revision",
                "activations_repo",
                "release",
                "selection",
            )
        },
        "subset": subset,
        "seed": seed,
        "lora": cfg.lora.model_dump(mode="json"),
        "training": cfg.training.model_dump(mode="json", exclude={"seeds", "log_freq"}),
        "attention": cfg.execution.draft_attn_impl,
        "processes": cfg.execution.processes,
    }


class BankPublisher(Publisher):
    """Reuse immutable batching, with a separate bank repo and local journal."""

    def __init__(self, cfg, api=None, *, create=True):
        if cfg.hub is None:
            raise ValueError("LoRA publication requires Hub configuration")
        if cfg.hub.trajectories_revision == "main":
            raise ValueError("LoRA publication requires a pinned trajectories_revision")
        self.cfg = cfg
        self.root = cfg.output_root.resolve()
        self.journal_path = self.root / "bank_publication.json"
        self.journal = (
            json.loads(self.journal_path.read_text())
            if self.journal_path.exists()
            else {}
        )
        self.api = api or HfApi()
        self.repo = cfg.hub.banks_repo
        if create:
            self.api.create_repo(
                self.repo, repo_type="dataset", private=False, exist_ok=True
            )
        if self.api.repo_info(self.repo, repo_type="dataset").private:
            raise ValueError("LoRA banks must use a public dataset repository")

    def prefix(self, subset, seed):
        return (
            f"releases/{self.cfg.hub.bank_release}/{self.cfg.dataset.category}/"
            f"{subset['id']}/seed-{seed}"
        )

    def completed(self, manifest, subset, seed, revision=None):
        parent = revision or self.api.repo_info(self.repo, repo_type="dataset").sha
        path = f"{self.prefix(subset, seed)}/completion.json"
        if path not in self.api.list_repo_files(
            self.repo, repo_type="dataset", revision=parent
        ):
            return None
        record = json.loads(
            Path(
                hf_hub_download(self.repo, path, repo_type="dataset", revision=parent)
            ).read_text()
        )
        if record["identity"] != run_identity(self.cfg, manifest, subset, seed):
            raise ValueError(
                f"Published run settings differ for {subset['id']}/seed-{seed}; "
                "choose a new hub.bank_release"
            )
        return record, parent

    def validate_run(self, manifest, subset, seed):
        run = self.root / "runs" / subset["id"] / f"seed-{seed}"
        diagnostics = inspect_adapters(run, self.cfg.lora)
        if not diagnostics["candidate_count"] or not diagnostics["recovery_state"]:
            raise ValueError(f"No completed LoRA collection at {run}")
        if diagnostics["recovery_state"]["phase"] != "done":
            raise ValueError(
                f"Run is unfinished; publication requires completion: {run}"
            )
        states = sorted(
            (p for p in (run / "recovery").iterdir() if p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        checkpoint = states[-1]
        required_checkpoint = (
            "training_state.json",
            "optimizer_state_dict.pt",
            "rng.pt",
            "adapter/adapter_model.safetensors",
            "adapter/adapter_config.json",
        )
        if any(not (checkpoint / name).is_file() for name in required_checkpoint):
            raise ValueError(f"Incomplete final recovery checkpoint: {checkpoint}")
        required = (
            "train_command.txt",
            "speculators.patch",
            "bank_context.json",
            "resolved_train.json",
        )
        if any(not (run / name).is_file() for name in required):
            raise ValueError(f"Missing training provenance at {run}")
        context = json.loads((run / "bank_context.json").read_text())
        if any(
            context.get(key) != value
            for key, value in {
                "prepared_fingerprint": manifest["prepared_fingerprint"],
                "cache_fingerprint": manifest["cache_fingerprint"],
                "subset_id": subset["id"],
                "seed": seed,
            }.items()
        ):
            raise ValueError(f"Run provenance does not match selected data: {run}")
        actual = json.loads((run / "resolved_train.json").read_text())
        expected = {
            "lr": self.cfg.training.warmup_lr,
            "weight_decay": self.cfg.training.weight_decay,
            "total_seq_len": self.cfg.training.token_budget,
            "noise_std": self.cfg.training.noise_std,
            "ttt_steps": self.cfg.training.ttt_steps,
            "ttt_step_loss_decay": self.cfg.training.ttt_step_loss_decay,
            "loss_fn": self.cfg.training.loss_fn,
            "gradient_checkpointing": self.cfg.training.gradient_checkpointing,
            "bank_warmup_epochs": self.cfg.training.warmup_epochs,
            "bank_collect_epochs": self.cfg.training.collect_epochs,
            "bank_collect_lr": self.cfg.training.collect_lr,
            "bank_loras_per_epoch": self.cfg.training.loras_per_epoch,
        }
        if any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Publication config differs from trained settings: {run}")
        return run, diagnostics

    def publish_run(self, manifest, subset, seed):
        previous = self.completed(manifest, subset, seed)
        if previous is not None:
            report(f"Already published {subset['id']}/seed-{seed}")
            return previous
        run, diagnostics = self.validate_run(manifest, subset, seed)
        files = {}
        for directory in (run, self.root / "provenance"):
            for path in sorted(directory.rglob("*")):
                relative = path.relative_to(self.root)
                if path.is_file() and not any(
                    part.startswith(".") or part.endswith(".lock")
                    for part in relative.parts
                ):
                    files[relative.as_posix()] = path
        identity = run_identity(self.cfg, manifest, subset, seed)
        record = {
            "identity": identity,
            "candidate_count": diagnostics["candidate_count"],
            "files": {
                name: {"sha256": file_digest(path), "bytes": path.stat().st_size}
                for name, path in files.items()
            },
        }
        metadata = self.root / ".bank-exports" / (digest(identity) + ".json")
        write_json(metadata, record)
        prefix = self.prefix(subset, seed)
        with hub_progress(self.cfg):
            references = self.put_many(
                self.repo,
                [
                    *[(f"{prefix}/{name}", path) for name, path in files.items()],
                    (f"{prefix}/completion.json", metadata),
                ],
            )
        report(
            f"Published {subset['id']}/seed-{seed}: {record['candidate_count']} LoRAs"
        )
        report(f"Bank repository commit: {references[-1]['revision']}")
        return record, references[-1]["revision"]

    def restore_run(self, manifest, subset, seed, revision=None):
        completed = self.completed(manifest, subset, seed, revision)
        if completed is None:
            return None
        record, commit = completed
        prefix = self.prefix(subset, seed)
        with hub_progress(self.cfg):
            for name, metadata in record["files"].items():
                destination = (self.root / name).resolve()
                if not destination.is_relative_to(self.root):
                    raise ValueError("Published bank path escapes output root")
                fetch(
                    {
                        "repo": self.repo,
                        "path": f"{prefix}/{name}",
                        "revision": commit,
                        **metadata,
                    },
                    destination,
                )
        report(f"Restored {subset['id']}/seed-{seed} from {commit}")
        return record
