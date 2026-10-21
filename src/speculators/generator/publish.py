"""Publish portable generator/ordinary-adapter bundles and run provenance."""

import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id

from speculators.bank.artifacts import file_digest, write_json
from speculators.generator.serving.bundle import export_run

EXTRACTION_FILES = (
    "vllm_command.txt",
    "vllm.patch",
    "checkpoint_sha256.txt",
    "drafter_checkpoint_sha256.txt",
    "eval_command.txt",
)


def extraction_provenance(run_dir, explicit=None):
    if explicit is not None:
        candidates = [Path(explicit)]
    else:
        candidates = [
            candidate
            for parent in (run_dir, *run_dir.parents)
            for candidate in (parent / "provenance" / "vllm", parent)
        ]
    for candidate in candidates:
        if (candidate / "vllm_command.txt").is_file():
            for name in ("vllm.patch", "checkpoint_sha256.txt"):
                if not (candidate / name).is_file():
                    raise FileNotFoundError(
                        f"Missing vLLM provenance: {candidate / name}"
                    )
            return candidate
    return None


def push_run(
    run_dir,
    repo_id,
    *,
    cfg=None,
    drafter_model_path=None,
    provenance_dir=None,
    private=False,
    include_training_checkpoint=False,
    path_in_repo=None,
    dry_run=False,
):
    """Build and validate locally before creating or updating a model repo."""
    validate_repo_id(repo_id)
    run_dir = Path(run_dir).resolve()
    result = json.loads((run_dir / "result.json").read_text())
    checkpoint = run_dir / "checkpoint.pt"
    if file_digest(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Completed checkpoint checksum differs from result.json")
    identity = json.loads((run_dir / "resolved_experiment.json").read_text())
    provenance = extraction_provenance(run_dir, provenance_dir)
    if provenance is None and (
        provenance_dir is not None
        or identity.get("campaign", {}).get("activation_source") == "vllm"
    ):
        raise FileNotFoundError(
            "Missing vLLM provenance. Pass --provenance-dir pointing to the "
            "directory written by launch_vllm.py --provenance-dir."
        )
    with tempfile.TemporaryDirectory(prefix="speculators-publish-") as temporary:
        folder = Path(temporary) / "bundle"
        metadata = export_run(
            cfg, run_dir, folder, drafter_model_path=drafter_model_path
        )
        if metadata["source_checkpoint_sha256"] != result["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during export; retry after training")
        if provenance is not None:
            destination = folder / "provenance" / "vllm"
            destination.mkdir(parents=True, exist_ok=True)
            for name in EXTRACTION_FILES:
                if (provenance / name).is_file():
                    shutil.copy2(provenance / name, destination / name)
        (folder / "checkpoint_sha256.txt").write_text(
            f"{metadata['source_checkpoint_sha256']}  checkpoint.pt\n"
        )
        if include_training_checkpoint:
            shutil.copy2(checkpoint, folder / "checkpoint.pt")
        label = identity["job"].get("label", run_dir.name)
        (folder / "README.md").write_text(
            "---\ntags:\n- speculators\n- eagle3\n---\n\n"
            f"# {label}\n\n"
            f"Portable Speculators `{metadata['kind']}` inference bundle.\n\n"
            "Load with `speculators.generator.serving.bundle.bundle_state`. "
            "Use the downloaded folder as a bundle in the generator benchmark "
            "configuration. This bundle requires the pinned target and drafter "
            "models recorded in `bundle.json`.\n\n"
            "`resolved_experiment.json` records architecture, training data identity, "
            "and optimization settings. Training commands, patches, metrics, and "
            "available extraction provenance are included.\n\n"
            f"Source checkpoint SHA256: `{metadata['source_checkpoint_sha256']}`.\n"
        )
        files = {
            str(path.relative_to(folder)): {
                "bytes": path.stat().st_size,
                "sha256": file_digest(path),
            }
            for path in sorted(folder.rglob("*"))
            if path.is_file()
        }
        write_json(
            folder / "upload_manifest.json",
            {"repo_id": repo_id, "path_in_repo": path_in_repo, "files": files},
        )
        report = {
            "repo_id": repo_id,
            "kind": metadata["kind"],
            "dry_run": dry_run,
            "files": [*files, "upload_manifest.json"],
            "bytes": sum(item["bytes"] for item in files.values())
            + (folder / "upload_manifest.json").stat().st_size,
        }
        if path_in_repo:
            report["path_in_repo"] = path_in_repo
        if not dry_run:
            api = HfApi()
            api.create_repo(
                repo_id=repo_id, repo_type="model", private=private, exist_ok=True
            )
            upload_options = {"path_in_repo": path_in_repo} if path_in_repo else {}
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=folder,
                commit_message=f"Publish {label} inference bundle and provenance",
                **upload_options,
            )
            report.update(commit_sha=commit.oid, commit_url=commit.commit_url)
        return report
