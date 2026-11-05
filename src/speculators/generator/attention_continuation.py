"""Restore the pinned TTT3 step-2000 training state without replacing local progress."""

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCE_REPO = "azeeer/eagle3-generator-attention-100k-2k-v1"
SOURCE_REVISION = "cf09002ae6fbca1fe0e9139ab8a0eee3b149a124"
SOURCE_FOLDER = "step-00002000"
TRAJECTORY_REVISION = "93750ccc4ba1701c2ec6b75ca5e4c1a5d867daa6"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def restore(root, plan):
    """Install checkpoint last; completed local restores always take precedence."""
    output = Path(root) / "pretraining" / plan["job"]["label"]
    if (output / "checkpoint.pt").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "provenance" / "hf-continuation-source"
    archive.mkdir(parents=True, exist_ok=True)
    for name in (
        "resolved_experiment.json", "result.json", "configuration.json",
        "train_command.txt", "speculators.patch", "checkpoint_sha256.txt",
        "provenance/vllm/vllm_command.txt", "provenance/vllm/vllm.patch",
        "provenance/vllm/checkpoint_sha256.txt", "provenance/vllm/run.json",
    ):
        source = hf_hub_download(
            SOURCE_REPO, f"{SOURCE_FOLDER}/{name}", revision=SOURCE_REVISION
        )
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    identity = json.loads((archive / "resolved_experiment.json").read_text())
    result = json.loads((archive / "result.json").read_text())
    pin = json.loads((archive / "provenance/vllm/run.json").read_text())
    if (
        identity["drafter_training"]["ttt_steps"] != 3
        or result["completed_updates"] != 2000
        or plan["job"]["updates"] != 10000
        or plan["job"]["epoch_ends"] != identity["job"]["epoch_ends"]
        or pin["trajectories_revision"] != TRAJECTORY_REVISION
    ):
        raise ValueError("Continuation source or epoch schedule differs")
    source = Path(hf_hub_download(
        SOURCE_REPO, f"{SOURCE_FOLDER}/checkpoint.pt", revision=SOURCE_REVISION
    ))
    checksum = sha256(source)
    if checksum != result["checkpoint_sha256"]:
        raise ValueError("Downloaded continuation checkpoint checksum differs")
    shutil.copy2(archive / "resolved_experiment.json", output / "resolved_experiment.json")
    shutil.copy2(archive / "configuration.json", output / "configuration.json")
    receipt = {
        "repo_id": SOURCE_REPO,
        "revision": SOURCE_REVISION,
        "subfolder": SOURCE_FOLDER,
        "checkpoint_sha256": checksum,
        "identity_sha256": sha256(archive / "resolved_experiment.json"),
        "completed_updates": 2000,
    }
    (output / "hf_continuation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    # The source snapshot is already durable; do not republish step 2000.
    if plan["job"]["hub_repo"] == SOURCE_REPO:
        (output / "hub_checkpoints.json").write_text(json.dumps({
            SOURCE_FOLDER: {
                "checkpoint_sha256": checksum,
                "completed_updates": 2000,
                "commit_sha": SOURCE_REVISION,
            }
        }, indent=2) + "\n")
    pending = output / "checkpoint.pt.pending"
    shutil.copy2(source, pending)
    pending.replace(output / "checkpoint.pt")
    print("Restored TTT3 step 2000; continuing through total step 10000.", flush=True)
