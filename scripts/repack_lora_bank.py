#!/usr/bin/env python3
"""Pack a downloaded LoRA bank and publish it to a separate dataset repository.

Example:
    python scripts/repack_lora_bank.py --local-dir /path/to/download

Requires a Hugging Face login with write access. No GPU or speculators import
is needed. Original provenance and recovery artifacts are retained in archives.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from safetensors.torch import load_file, save_file
from tqdm.auto import tqdm
from tqdm.utils import CallbackIOWrapper

SOURCE_REPO = "azeeer/eagle3-qwen3-8b-lora-bank-r32-v1"
DEST_REPO = "azeeer/eagle3-qwen3-8b-lora-bank-r32-packed-v1"


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def checked_path(root, relative, metadata):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Source path escapes downloaded run: {relative}")
    if not path.is_file() or path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"Missing or incomplete downloaded file: {path}")
    return path


def pack_epochs(record, files, destination, label):  # noqa: C901 -- group and pack snapshots by epoch
    epochs = defaultdict(list)
    adapter_weights = set()
    shared_config = None
    for name, entry_path in sorted(files.items()):
        if "/snapshots/" not in name or not name.endswith("/entry.json"):
            continue
        entry = json.loads(entry_path.read_text())
        snapshot_dir = Path(name).parent
        weights_name = (snapshot_dir / "adapter/adapter_model.safetensors").as_posix()
        config_name = (snapshot_dir / "adapter/adapter_config.json").as_posix()
        config = json.loads(files[config_name].read_text())
        if shared_config is None:
            shared_config = config
        elif config != shared_config:
            raise ValueError(f"Adapter configurations differ within {label}")
        epoch = entry["epoch"] - record["identity"]["training"]["warmup_epochs"] + 1
        if epoch < 1:
            raise ValueError(f"Unexpected warmup snapshot: {entry_path}")
        epochs[epoch].append((snapshot_dir.name, weights_name, entry))
        adapter_weights.add(weights_name)
    count = sum(map(len, epochs.values()))
    if not count or count != record["candidate_count"]:
        raise ValueError(f"Snapshot count mismatch in {label}")

    packed = []
    with tqdm(
        total=count, desc=f"Snapshots {label}", unit="snapshot", leave=False
    ) as bar:
        for epoch, snapshots in sorted(epochs.items()):
            tensors, entries = {}, []
            for snapshot_id, weights_name, entry in snapshots:
                mapping = {}
                for original_key, tensor in load_file(
                    files[weights_name], device="cpu"
                ).items():
                    key = f"{snapshot_id}/{original_key}"
                    if key in tensors:
                        raise ValueError(f"Duplicate tensor key: {key}")
                    tensors[key] = tensor.contiguous().clone()
                    mapping[original_key] = key
                entries.append(
                    {"snapshot_id": snapshot_id, "entry": entry, "tensor_keys": mapping}
                )
                bar.update()
            filename = f"epoch-{epoch:02d}.safetensors"
            path = destination / filename
            save_file(
                tensors,
                path,
                metadata={"format": "speculators-packed-lora-epoch", "version": "1"},
            )
            packed.append(
                {
                    "collection_epoch": epoch,
                    "file": filename,
                    "bytes": path.stat().st_size,
                    "sha256": checksum(path),
                    "snapshots": entries,
                }
            )
            del tensors
    return packed, adapter_weights, shared_config, count


def archive_files(path, members, description):
    """Archive file bytes without compression; recovery tensors compress poorly."""
    total = sum(local.stat().st_size for _, local in members)
    with (
        tqdm(
            total=total, desc=description, unit="B", unit_scale=True, leave=False
        ) as bar,
        tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive,
    ):
        for name, local in members:
            # Stream through tqdm so large recovery files show byte progress too.
            info = archive.gettarinfo(str(local), arcname=name)
            if not info.isfile():
                raise ValueError(f"Expected a regular downloaded file: {local}")
            with local.open("rb") as handle:
                archive.addfile(info, CallbackIOWrapper(bar.update, handle, "read"))
    return {"bytes": path.stat().st_size, "sha256": checksum(path)}


def pack_run(completion, source, destination):
    root = completion.parent
    record = json.loads(completion.read_text())
    files = {
        name: checked_path(root, name, metadata)
        for name, metadata in record["files"].items()
    }
    relative = root.relative_to(source)
    packed_run = destination / relative
    packed_run.mkdir(parents=True)
    epochs, weights, config, count = pack_epochs(
        record, files, packed_run, str(relative)
    )
    manifest = {
        "format": "speculators-packed-lora-run",
        "version": 1,
        "source_identity": record["identity"],
        "source_completion_sha256": checksum(completion),
        "adapter_config": config,
        "snapshot_count": count,
        "epochs": epochs,
        "archives": {},
    }
    for archive_name, recovery in (("provenance.tar", False), ("recovery.tar", True)):
        members = [] if recovery else [("completion.json", completion)]
        members.extend(
            (name, local)
            for name, local in sorted(files.items())
            if name not in weights and ("recovery" in Path(name).parts) == recovery
        )
        manifest["archives"][archive_name] = archive_files(
            packed_run / archive_name, members, f"{archive_name} {relative}"
        )
    path = packed_run / "manifest.json"
    write_json(path, manifest)
    return {
        "path": relative.as_posix(),
        "manifest_sha256": checksum(path),
        "snapshot_count": count,
    }


def pack_bank(local_dir, output_dir, source_repo=SOURCE_REPO, release="b0002"):
    if not release or any(
        c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in release
    ):
        raise ValueError("Release must be a safe lowercase path component")
    source = local_dir.resolve() / "releases" / release
    completions = sorted(source.glob("*/*/seed-*/completion.json"))
    if not completions:
        raise FileNotFoundError(f"No completed bank runs under {source}")
    destination = output_dir.resolve() / "releases" / release
    if output_dir.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the source release")
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    destination.mkdir(parents=True)
    catalog = {
        "format": "speculators-packed-lora-bank",
        "version": 1,
        "release": release,
        "source_release": release,
        "source_repo": source_repo,
        "runs": [],
    }
    try:
        for completion in tqdm(completions, desc="Packing bank runs", unit="run"):
            catalog["runs"].append(pack_run(completion, source, destination))
        provenance = destination / "repack-provenance"
        provenance.mkdir()
        shutil.copyfile(__file__, provenance / "repack_lora_bank.py")
        write_json(
            provenance / "command.json",
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "argv": sys.argv,
                "source_repo": source_repo,
                "source_release": release,
                "source_identity": (
                    "per-run completion.json SHA-256 values in manifests"
                ),
            },
        )
        write_json(destination / "catalog.json", catalog)
    except BaseException:
        # Only this invocation's newly created release folder is removed.
        shutil.rmtree(destination)
        raise
    return catalog


def publish_bank(
    output_dir,
    destination_repo,
    source_repo=SOURCE_REPO,
    release="b0002",
    private=False,
    api=None,
):
    if destination_repo.lower() == source_repo.lower():
        raise ValueError("Destination must be a different repository from the source")
    prefix = f"releases/{release}/"
    root = output_dir.resolve()
    files = sorted(p for p in (root / prefix).rglob("*") if p.is_file())
    catalog = root / prefix / "catalog.json"
    if not catalog.is_file():
        raise FileNotFoundError("Build the packed release before uploading")
    api = api or HfApi()
    api.create_repo(
        destination_repo, repo_type="dataset", private=private, exist_ok=True
    )
    parent = api.repo_info(destination_repo, repo_type="dataset").sha
    existing = (
        api.list_repo_files(destination_repo, repo_type="dataset", revision=parent)
        if parent
        else []
    )
    if any(path.startswith(prefix) for path in existing):
        raise FileExistsError(
            f"Destination already contains {prefix}; refusing to overwrite"
        )
    operations = [
        CommitOperationAdd(
            path_in_repo=p.relative_to(root).as_posix(), path_or_fileobj=str(p)
        )
        for p in tqdm(files, desc="Preparing upload", unit="file")
    ]
    tqdm.write(
        f"Uploading {len(files)} files "
        f"({sum(p.stat().st_size for p in files) / 2**20:.1f} MiB)."
    )
    # huggingface_hub supplies its own tqdm transfer bars. One atomic commit
    # publishes weights, manifests, provenance, recovery and the release catalog.
    commit = api.create_commit(
        repo_id=destination_repo,
        repo_type="dataset",
        parent_commit=parent,
        operations=operations,
        commit_message=f"Publish packed LoRA bank {release}",
    )
    tqdm.write(f"Published: {commit.commit_url}\nPinned revision: {commit.oid}")
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-dir", type=Path, default=Path("."), help="Root created by hf download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local packed output; defaults to a new temporary directory",
    )
    parser.add_argument(
        "--source-repo",
        default=SOURCE_REPO,
        help="Source identity recorded in provenance",
    )
    parser.add_argument(
        "--dest-repo", default=DEST_REPO, help="Separate destination dataset repository"
    )
    parser.add_argument(
        "--release", default="b0002", help="Same source and destination release name"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a new destination repository as private",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Build locally without contacting the Hub",
    )
    args = parser.parse_args()
    if args.dest_repo.lower() == args.source_repo.lower():
        parser.error("--dest-repo must differ from --source-repo")
    output = args.output_dir or Path(tempfile.mkdtemp(prefix="packed-lora-bank-"))
    catalog = pack_bank(args.local_dir, output, args.source_repo, args.release)
    tqdm.write(f"Packed {len(catalog['runs'])} runs at {output.resolve()}")
    if not args.no_push:
        publish_bank(
            output, args.dest_repo, args.source_repo, args.release, args.private
        )


if __name__ == "__main__":
    main()
