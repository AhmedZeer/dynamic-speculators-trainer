"""Exact data passes and synchronous Hub backups for full-data generator training."""

import bisect
import json
import random
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from filelock import FileLock
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.progress import report
from speculators.generator.publish import extraction_provenance, push_run
from speculators.generator.serving.bundle import PROVENANCE


class EpochSchedule:
    """Shuffle every training row once per epoch, retaining single-subset episodes."""

    def __init__(self, cfg, corpus, subsets, epochs):
        self.updates, self.epoch_ends = [], []
        for epoch in range(epochs):
            rng = random.Random(cfg.seed + epoch * 100003)
            episodes = []
            for subset in sorted(subsets):
                indices = list(corpus.indices(subset))
                rng.shuffle(indices)
                offset = 0
                while offset < len(indices):
                    size = rng.randint(
                        cfg.context.min_examples, cfg.context.max_examples
                    )
                    cohort = indices[offset : offset + size]
                    episodes.append({"subset": subset, "indices": cohort})
                    offset += len(cohort)
            rng.shuffle(episodes)
            batch = cfg.optimization.episodes_per_update
            self.updates.extend(
                episodes[start : start + batch]
                for start in range(0, len(episodes), batch)
            )
            self.epoch_ends.append(len(self.updates))

    def update(self, step):
        return self.updates[step]

    def completed_epochs(self, progress):
        return bisect.bisect_right(self.epoch_ends, progress)


def backup_epoch(output, job, identity, epoch, progress):
    """Commit a resumable checkpoint and inference bundle before continuing."""
    output = Path(output)
    checkpoint = output / "checkpoint.pt"
    sha = file_digest(checkpoint)
    ledger_path = output / "hub_checkpoints.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    key = f"epoch-{epoch:04d}"
    if ledger.get(key, {}).get("checkpoint_sha256") == sha:
        return
    report(f"{job['label']}: uploading {key} to {job['hub_repo']}")
    with tempfile.TemporaryDirectory(prefix="epoch-backup-", dir=output) as temporary:
        folder = Path(temporary) / "run"
        folder.mkdir()
        for name in PROVENANCE:
            source = output / name
            if source.is_file() and name != "result.json":
                shutil.copy2(source, folder / name)
            elif source.is_dir():
                shutil.copytree(source, folder / name)
        # The training loop is stopped throughout this upload, so the hard link
        # stays immutable and avoids another multi-GB checkpoint copy here.
        try:
            (folder / "checkpoint.pt").hardlink_to(checkpoint)
        except OSError:
            # Some mounted filesystems (including Drive) do not support links.
            shutil.copy2(checkpoint, folder / "checkpoint.pt")
        write_json(folder / "resolved_experiment.json", identity)
        write_json(
            folder / "result.json",
            {
                "status": "epoch_snapshot",
                "job": job,
                "completed_epochs": epoch,
                "completed_updates": progress,
                "checkpoint_sha256": sha,
            },
        )
        published = push_run(
            folder,
            job["hub_repo"],
            drafter_model_path=job["drafter_model_path"],
            provenance_dir=job["vllm_provenance_dir"],
            include_training_checkpoint=True,
            path_in_repo=key,
        )
    ledger[key] = {**published, "checkpoint_sha256": sha, "completed_updates": progress}
    write_json(ledger_path, ledger)
    report(f"{job['label']}: {key} backed up at commit {published['commit_sha']}")


def train_full_data(
    cfg,
    output_root,
    repo_id,
    provenance_dir,
    *,
    epochs=10,
    save_every_epochs=2,
    private=False,
    dry_run=False,
):
    from speculators.generator.campaign import run_one  # noqa: PLC0415
    from speculators.generator.local_campaign import load_local_inputs  # noqa: PLC0415

    if cfg.activation_source != "vllm":
        raise ValueError("Full-data training requires the online activation config")
    if epochs < 1 or save_every_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    validate_repo_id(repo_id)
    inputs = load_local_inputs(cfg)
    subsets = sorted(inputs.subsets)
    corpus = SimpleNamespace(indices=lambda name: inputs.subsets[name]["train_indices"])
    schedule = EpochSchedule(cfg, corpus, subsets, epochs)
    job = {
        "kind": "pretraining",
        "stage": "pretraining",
        "condition": "last",
        "label": f"all-data-{epochs}epochs",
        "arm": "direct",
        "objective": "direct",
        "subsets": subsets,
        "validation_subsets": [f"{domain}-00000" for domain in inputs.banks],
        "training_examples": sum(len(corpus.indices(name)) for name in subsets),
        "updates": len(schedule.updates),
        "epochs": epochs,
        "epoch_ends": schedule.epoch_ends,
        "save_every_epochs": save_every_epochs,
        "epoch_schedule_version": 1,
        "hub_repo": repo_id,
        "vllm_provenance_dir": str(Path(provenance_dir).resolve()),
        "drafter_model_path": str(cfg.online.drafter_model_path.resolve()),
    }
    cfg = cfg.model_copy(deep=True)
    cfg.output_root = Path(output_root)
    cfg.optimization.pretraining_updates = job["updates"]
    if dry_run:
        return {"job": job, "configuration": cfg.model_dump(mode="json")}
    if extraction_provenance(cfg.output_root, provenance_dir) is None:
        raise ValueError("Missing extraction provenance")
    api = HfApi()
    api.whoami()
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_root / ".campaign.lock"), timeout=0):
        return run_one(cfg, job, local_inputs=inputs)
