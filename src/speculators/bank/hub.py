"""Immutable public preparation artifacts and relocatable, pinned consumers."""

# Stage-specific dependencies are intentionally lazy.
# ruff: noqa: PLC0415

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml
from filelock import FileLock
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)

from speculators.bank.artifacts import digest, file_digest, read_jsonl, write_json
from speculators.bank.progress import report, stage_progress

DOMAINS = ("chat", "math", "code", "stem")
STATE_NDIM = 3


@contextmanager
def hub_progress(cfg):
    """Suppress nested transfer bars during preparation and restore callers' state."""
    disabled = are_progress_bars_disabled()
    if not cfg.hub.progress_bars and not disabled:
        disable_progress_bars()
    try:
        yield
    finally:
        if not cfg.hub.progress_bars and not disabled:
            enable_progress_bars()


def fetch(reference, destination):
    """Restore one verified object without retaining a second Hub-cache copy."""
    destination = Path(destination)
    if destination.exists() and file_digest(destination) == reference["sha256"]:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
        source = hf_hub_download(
            reference["repo"],
            reference["path"],
            repo_type="dataset",
            revision=reference["revision"],
            local_dir=scratch,
        )
        if file_digest(Path(source)) != reference["sha256"]:
            raise ValueError(f"Downloaded object checksum differs: {reference['path']}")
        Path(source).replace(destination)
    return destination


class Publisher:
    """One coordinator, immutable paths, and a durable successful-upload journal."""

    def __init__(self, cfg, api=None):
        self.cfg = cfg
        self.root = cfg.output_root.resolve()
        self.journal_path = self.root / "publication.json"
        self.journal = (
            json.loads(self.journal_path.read_text())
            if self.journal_path.exists()
            else {}
        )
        self.api = api or HfApi()
        for repo in (cfg.hub.trajectories_repo, cfg.hub.activations_repo):
            self.api.create_repo(
                repo, repo_type="dataset", private=False, exist_ok=True
            )
            if self.api.repo_info(repo, repo_type="dataset").private:
                raise ValueError(f"Expected public dataset repository: {repo}")

    def put(self, repo, path, source):
        return self.put_many(repo, [(path, source)])[0]

    def put_many(self, repo, files):  # noqa: C901 -- immutable batch verification and commit
        """Publish related immutable objects and checksums in one atomic commit."""
        if not files:
            return []
        parent = self.api.repo_info(repo, repo_type="dataset").sha
        references, operations = [], []
        checksums = {}
        for path, filename in files:
            source = Path(filename)
            checksum = file_digest(source)
            if path in checksums:
                if checksums[path] != checksum:
                    raise ValueError(f"Conflicting batch object: {path}")
                continue
            checksums[path] = checksum
            previous = self.journal.get(f"{repo}:{path}")
            if previous and previous["sha256"] != checksum:
                raise ValueError(f"Immutable published object changed: {path}")
            if not previous:
                try:
                    sidecar = hf_hub_download(
                        repo,
                        path + ".sha256.json",
                        repo_type="dataset",
                        revision=parent,
                    )
                except EntryNotFoundError:
                    sidecar = None
                if sidecar:
                    if json.loads(Path(sidecar).read_text())["sha256"] != checksum:
                        raise ValueError(f"Immutable Hub object differs: {path}")
                else:
                    operations.extend(
                        [
                            CommitOperationAdd(
                                path_in_repo=path, path_or_fileobj=source
                            ),
                            CommitOperationAdd(
                                path_in_repo=path + ".sha256.json",
                                path_or_fileobj=json.dumps(
                                    {"sha256": checksum}
                                ).encode(),
                            ),
                        ]
                    )
            references.append(
                {
                    "repo": repo,
                    "path": path,
                    "revision": parent,
                    "sha256": checksum,
                    "bytes": source.stat().st_size,
                }
            )
        if operations:
            size = sum(r["bytes"] for r in references)
            if size >= 2**20:
                report(
                    f"Publishing {len(operations) // 2} objects to {repo}: "
                    f"{size / 2**20:.1f} MiB in one commit"
                )
            with stage_progress(
                "Hub batch upload",
                interval=self.cfg.hub.upload_log_interval,
                quiet=True,
            ):
                commit = self.api.create_commit(
                    repo,
                    repo_type="dataset",
                    parent_commit=parent,
                    commit_message=f"Publish {len(operations) // 2} immutable objects",
                    operations=operations,
                )
            for reference in references:
                reference["revision"] = commit.oid
        for reference in references:
            self.journal[f"{repo}:{reference['path']}"] = reference
        write_json(self.journal_path, self.journal)
        return references

    def metadata(self, repo, path, value):
        local = self.root / ".publication" / (digest(path) + ".json")
        write_json(local, value)
        return self.put(repo, path, local)


def profile(cfg):
    return f"releases/{cfg.hub.release}/{cfg.dataset.category}"


def portable(manifest):
    return {
        **manifest,
        "root": ".",
        "data_path": "data",
        "hidden_states_path": "hidden_states",
    }


def provenance(publisher, stage):
    from speculators.train.utils import save_train_command

    root = publisher.root / "provenance" / "hub" / stage
    # Keep each attempt; never replace earlier invocation provenance.
    attempt = root / f"attempt-{len(list(root.glob('attempt-*'))):05d}"
    save_train_command(str(attempt))
    files = []
    for file in sorted((publisher.root / "provenance").rglob("*")):
        if file.is_file() and file.suffix != ".log":
            # Content-addressed paths permit additional server/resume attempts.
            path = f"provenance/{file_digest(file)}/{file.name}"
            files.append((path, file))
    return publisher.put_many(publisher.cfg.hub.trajectories_repo, files)


def stage_record(publisher, stage, manifest, artifacts):
    path = f"{profile(publisher.cfg)}/{stage}.json"
    parent = publisher.api.repo_info(
        publisher.cfg.hub.trajectories_repo, repo_type="dataset"
    ).sha
    try:
        existing = hf_hub_download(
            publisher.cfg.hub.trajectories_repo,
            path,
            repo_type="dataset",
            revision=parent,
        )
    except EntryNotFoundError:
        existing = None
    if existing:
        previous = json.loads(Path(existing).read_text())

        def identity(items):
            return [(a["repo"], a["path"], a["sha256"]) for a in items]

        if previous["manifest"]["cache_fingerprint"] != manifest[
            "cache_fingerprint"
        ] or identity(previous["artifacts"]) != identity(artifacts):
            raise ValueError(f"Published {stage} differs; choose a new release")
        return previous
    return publisher.metadata(
        publisher.cfg.hub.trajectories_repo,
        path,
        {
            "version": 1,
            "stage": stage,
            "manifest": portable(manifest),
            "artifacts": artifacts,
            "provenance": provenance(publisher, stage),
        },
    )


def select(cfg, publisher):
    """Freeze all configured domains once; per-domain preparation uses that registry."""
    from datasets import load_dataset

    from speculators.bank.selection import SELECTION_POLICY
    from speculators.bank.workflow import cache_spec, partition_rows

    catalog_path = f"selections/{cfg.hub.selection}/catalog.json"
    cfg.hub.selection_root.mkdir(parents=True, exist_ok=True)
    local = cfg.hub.selection_root / "catalog.json"
    settings = {
        "models": cfg.models.model_dump(),
        "source": cfg.dataset.source,
        "source_revision": cfg.dataset.revision,
        "configuration": cfg.dataset.configuration,
        "sample_limit": cfg.dataset.sample_limit,
        "partition": cfg.partition.model_dump(),
        "domains": cfg.hub.domains,
        "reserved": cfg.hub.adaptation_subset_ids,
    }
    identity = digest({"selection_policy": SELECTION_POLICY, **settings})
    # A sampler change only affects new selections. Older frozen catalogs have
    # the same configuration identity without a policy field; reuse their rows.
    legacy_identity = digest(settings)
    with FileLock(str(local) + ".lock"):
        if not local.exists():
            try:
                remote = hf_hub_download(
                    cfg.hub.trajectories_repo, catalog_path, repo_type="dataset"
                )
            except EntryNotFoundError:
                remote = None
            if remote:
                shutil.copyfile(remote, local)
        if local.exists():
            catalog = json.loads(local.read_text())
            if catalog["identity"] not in (identity, legacy_identity):
                raise ValueError(
                    "Frozen selection changed; use a new selection ID "
                    "and selection_root"
                )
        else:
            api = publisher.api
            revisions = {
                "target_sha": api.model_info(
                    cfg.models.target, revision=cfg.models.target_revision
                ).sha,
                "drafter_sha": api.model_info(
                    cfg.models.drafter, revision=cfg.models.drafter_revision
                ).sha,
                "dataset_sha": api.dataset_info(
                    cfg.dataset.source, revision=cfg.dataset.revision
                ).sha,
            }
            catalog = {
                "version": 1,
                "identity": identity,
                "revisions": revisions,
                "domains": {},
            }
            for domain in DOMAINS:
                if domain not in cfg.hub.domains:
                    continue
                domain_cfg = cfg.model_copy(deep=True)
                domain_cfg.dataset.category = domain
                domain_cfg.dataset.split = domain
                report(
                    f"Selecting source rows for {domain}; duplicate filtering disabled"
                )
                rows = load_dataset(
                    cfg.dataset.source,
                    name=cfg.dataset.configuration,
                    split=domain,
                    revision=revisions["dataset_sha"],
                    streaming=True,
                )
                selected, _, stats = partition_rows(
                    rows, domain_cfg, skip_unsupported=True
                )
                for value in selected:
                    subset = int(value["subset_id"].rsplit("-", 1)[1])
                    value["role"] = (
                        "adaptation"
                        if subset in cfg.hub.adaptation_subset_ids
                        else "bank"
                    )
                catalog["domains"][domain] = {"prompts": selected, **stats}
            write_json(local, catalog)
        publisher.put(cfg.hub.trajectories_repo, catalog_path, local)
    prompts = catalog["domains"][cfg.dataset.category]["prompts"]
    root = cfg.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    prompt_file = root / "prompts.jsonl"
    prompt_file.write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in prompts)
    )
    subsets = []
    size = cfg.partition.train_examples + cfg.partition.validation_examples
    for offset, subset in enumerate(cfg.partition.subset_ids):
        group = prompts[offset * size : (offset + 1) * size]
        train = cfg.partition.train_examples
        subsets.append(
            {
                "id": f"{cfg.dataset.category}-{subset:05d}",
                "role": group[0]["role"],
                "training_seed": 42 + 5 * DOMAINS.index(cfg.dataset.category) + subset
                if group[0]["role"] == "bank"
                else None,
                "train_ids": [p["id"] for p in group[:train]],
                "validation_ids": [p["id"] for p in group[train:]],
                "train_indices": list(range(offset * size, offset * size + train)),
                "validation_indices": list(
                    range(offset * size + train, (offset + 1) * size)
                ),
            }
        )
    spec = cache_spec(cfg, catalog["revisions"])
    manifest = {
        "version": 1,
        "category": cfg.dataset.category,
        "cache_spec": spec,
        "cache_fingerprint": digest(spec),
        "revisions": catalog["revisions"],
        "root": str(root),
        "data_path": str(root / "data"),
        "hidden_states_path": str(root / "hidden_states"),
        "subsets": subsets,
        "prompts_sha256": file_digest(prompt_file),
        "prepared": False,
        "partition_stats": {"selected_prompts": len(prompts)},
    }
    write_json(root / "manifest.json", manifest)
    artifact = publisher.put(
        cfg.hub.trajectories_repo, f"{profile(cfg)}/prompts.jsonl", prompt_file
    )
    stage_record(publisher, "select", manifest, [artifact])
    return manifest


def publish_data(cfg, publisher, manifest):
    from datasets import load_from_disk

    data = load_from_disk(manifest["data_path"]).with_format(None)
    prompts = {p["id"]: p for p in read_jsonl(publisher.root / "prompts.jsonl")}
    files = []
    for subset in manifest["subsets"]:
        for split in ("train", "validation"):
            path = f"{profile(cfg)}/{subset['id']}/{split}.parquet"
            local = publisher.root / ".publication" / subset["id"] / f"{split}.parquet"
            local.parent.mkdir(parents=True, exist_ok=True)
            from datasets import Dataset

            rows = [
                {
                    **data[i],
                    "role": subset["role"],
                    "membership": split,
                    "source_aliases": prompts[data[i]["primary_id"]].get(
                        "source_aliases", []
                    ),
                    "source_repo": cfg.dataset.source,
                    "source_revision": manifest["revisions"]["dataset_sha"],
                }
                for i in subset[f"{split}_indices"]
            ]
            Dataset.from_list(rows).to_parquet(local)
            files.append(publisher.put(cfg.hub.trajectories_repo, path, local))
    stage_record(publisher, "data", manifest, files)
    # Explicit configs prevent stage manifests from becoming dataset-viewer rows.
    parent = publisher.api.repo_info(cfg.hub.trajectories_repo, repo_type="dataset").sha
    paths = publisher.api.list_repo_files(
        cfg.hub.trajectories_repo, repo_type="dataset", revision=parent
    )
    configs = []
    for domain in DOMAINS:
        splits = [
            {
                "split": split,
                "path": sorted(
                    p
                    for p in paths
                    if p.startswith(f"releases/{cfg.hub.release}/{domain}/")
                    and p.endswith(f"/{split}.parquet")
                ),
            }
            for split in ("train", "validation")
        ]
        if all(s["path"] for s in splits):
            configs.append({"config_name": domain, "data_files": splits})
    card = "---\n" + yaml.safe_dump({"configs": configs}, sort_keys=False) + "---\n"
    card += "# Frozen EAGLE3 target trajectories\n\n"
    card += (
        "Prompt memberships, target token trajectories, "
        "and references to pinned activation shards.\n\n"
    )
    card += f"Source: https://huggingface.co/datasets/{cfg.dataset.source}\n\n"
    card += (
        "Retained source UUIDs and licenses are in source_aliases. "
        "Preserve attribution and respect each source license.\n\n"
    )
    card += (
        "The train split contains both bank and reserved adaptation subsets. "
        "Filter by subset_id and role before training.\n"
    )
    try:
        previous = hf_hub_download(
            cfg.hub.trajectories_repo, "README.md", repo_type="dataset", revision=parent
        )
    except EntryNotFoundError:
        previous = None
    if previous is None or Path(previous).read_text() != card:
        publisher.api.create_commit(
            cfg.hub.trajectories_repo,
            repo_type="dataset",
            parent_commit=parent,
            commit_message="Update trajectory configurations and attribution",
            operations=[
                CommitOperationAdd(
                    path_in_repo="README.md", path_or_fileobj=card.encode()
                )
            ],
        )


def responses(cfg, publisher, manifest):
    from speculators.bank.workflow import regenerate
    from speculators.data_generation.pipeline import repair_responses

    prompts = read_jsonl(publisher.root / "prompts.jsonl")
    expected = {p["id"] for p in prompts}
    response_file = publisher.root / "responses.jsonl"
    saved = {r["primary_id"]: r for r in repair_responses(response_file, expected)}
    parent = publisher.api.repo_info(cfg.hub.trajectories_repo, repo_type="dataset").sha
    for path in publisher.api.list_repo_files(
        cfg.hub.trajectories_repo, repo_type="dataset", revision=parent
    ):
        if path.startswith(f"{profile(cfg)}/responses/") and path.endswith(".jsonl"):
            remote = hf_hub_download(
                cfg.hub.trajectories_repo, path, repo_type="dataset", revision=parent
            )
            for row in read_jsonl(Path(remote)):
                primary = row["primary_id"]
                if primary not in expected or (
                    primary in saved and saved[primary] != row
                ):
                    raise ValueError(
                        "Published response chunk differs from local responses"
                    )
                saved[primary] = row
    response_file.write_text(
        "".join(json.dumps(saved[p["id"]]) + "\n" for p in prompts if p["id"] in saved)
    )
    files = []
    for start in range(0, len(prompts), cfg.hub.response_chunk_size):
        batch = prompts[start : start + cfg.hub.response_chunk_size]
        temporary = publisher.root / ".publication" / "request-prompts.jsonl"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text("".join(json.dumps(p) + "\n" for p in batch))
        if any(p["id"] not in saved for p in batch):
            regenerate(cfg, manifest, prompts_path=temporary)
        available = {
            r["primary_id"]: r for r in read_jsonl(publisher.root / "responses.jsonl")
        }
        missing = [p["id"] for p in batch if p["id"] not in available]
        if missing:
            raise ValueError(
                f"Response chunk incomplete ({len(missing)} missing); rerun to resume"
            )
        chunk = temporary.with_name(f"responses-{start:08d}.jsonl")
        chunk.write_text("".join(json.dumps(available[p["id"]]) + "\n" for p in batch))
        files.append(
            publisher.put(
                cfg.hub.trajectories_repo,
                f"{profile(cfg)}/responses/{chunk.name}",
                chunk,
            )
        )
    stage_record(publisher, "responses", manifest, files)


def hidden(cfg, publisher, manifest):  # noqa: C901 -- bounded extraction and durable publication
    """Extract bounded batches, seal/upload shards, then release local payloads."""
    import torch
    from datasets import load_from_disk
    from safetensors.torch import load_file, save_file

    from speculators.cli.generate_offline_data import generate_offline_data

    root = Path(manifest["hidden_states_path"])
    root.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.json"
    index = (
        json.loads(index_path.read_text())
        if index_path.exists()
        else {
            "version": 1,
            "rows": {},
            "cache_gib": cfg.hub.cache_gib,
            "reserve_gib": cfg.hub.cache_reserve_gib,
            "prepared_fingerprint": manifest["prepared_fingerprint"],
        }
    )
    if index["prepared_fingerprint"] != manifest["prepared_fingerprint"]:
        raise ValueError("Activation index belongs to another trajectory dataset")
    data = load_from_disk(manifest["data_path"]).with_format(None)
    # Recover sealed shards even when the original machine/local index was lost.
    parent = publisher.api.repo_info(cfg.hub.activations_repo, repo_type="dataset").sha
    remote_paths = publisher.api.list_repo_files(
        cfg.hub.activations_repo, repo_type="dataset", revision=parent
    )
    final_index = f"{profile(cfg)}/index.json"
    if final_index in remote_paths:
        published = hf_hub_download(
            cfg.hub.activations_repo, final_index, repo_type="dataset", revision=parent
        )
        final = json.loads(Path(published).read_text())
        if final["prepared_fingerprint"] != manifest["prepared_fingerprint"]:
            raise ValueError("Published index belongs to different trajectories")
        index = final
        # A final index already contains every sealed record. Avoid hundreds of
        # metadata downloads when resuming a fully published category.
        remote_paths = []
    for path in remote_paths:
        if path.startswith(profile(cfg) + "/") and path.endswith("-complete.json"):
            saved = hf_hub_download(
                cfg.hub.activations_repo, path, repo_type="dataset", revision=parent
            )
            complete = json.loads(Path(saved).read_text())
            if complete["prepared_fingerprint"] != manifest["prepared_fingerprint"]:
                raise ValueError("Published shard belongs to different trajectories")
            for row in complete["rows"].values():
                for kind in ("shard", "summary"):
                    if row[kind]["revision"] == "__commit__":
                        row[kind]["revision"] = parent
            for key, row in complete["rows"].items():
                index["rows"].setdefault(key, row)
    write_json(index_path, index)
    raw = root / ".raw-states"
    raw.mkdir(exist_ok=True)
    packaged = root / ".packaged"
    packaged.mkdir(exist_ok=True)
    packages = []
    local_rows = set()
    for marker in sorted(packaged.glob("*/local.json")):
        record = json.loads(marker.read_text())
        if record["prepared_fingerprint"] != manifest["prepared_fingerprint"]:
            raise ValueError("Local packaged shard belongs to different trajectories")
        packages.append((marker.parent, record))
        local_rows.update(record["rows"])

    # Recover extraction files from the previous per-batch layout without asking
    # vLLM to compute them again. Token IDs guard against mismatched trajectories.
    row_ids = {data[i]["primary_id"]: i for i in range(len(data))}
    for old in sorted((root / ".pending-extraction").glob("*/data")):
        previous = load_from_disk(old).with_format(None)
        for offset, row in enumerate(previous):
            i = row_ids.get(row["primary_id"])
            source = old.parent / "states" / f"hs_{offset}.safetensors"
            if i is None or not source.exists():
                continue
            if row["input_ids"] != data[i]["input_ids"]:
                raise ValueError("Pending extraction tokens differ from trajectories")
            target = raw / f"hs_{i}.safetensors"
            if not target.exists():
                source.replace(target)

    skipped = set(index["rows"]) | local_rows
    with stage_progress("Continuous hidden-state extraction", interval=30):
        generate_offline_data(
            model=cfg.models.target,
            endpoint=cfg.execution.extraction_endpoint,
            preprocessed_data=manifest["data_path"],
            output=str(raw),
            max_samples=None,
            concurrency=cfg.execution.hidden_state_concurrency,
            write_concurrency=cfg.execution.hidden_state_write_concurrency,
            progress_log_interval=cfg.execution.hidden_state_log_interval,
            validate_outputs=False,
            trust_existing_outputs=True,
            skip_indices=sorted(int(i) for i in skipped),
            request_timeout=cfg.execution.request_timeout,
            max_retries=cfg.execution.max_retries,
            fail_on_error=True,
            max_consecutive_errors=None,
            world_size=1,
            rank=0,
        )

    def seal(tensors, summaries, entries, location):
        identity = digest([data[i]["primary_id"] for i in entries])
        path = f"{location}/{identity}"
        directory = packaged / identity
        directory.mkdir(exist_ok=True)
        shard, summary = (
            directory / "shard.safetensors",
            directory / "conditioning.safetensors",
        )
        save_file(tensors, shard)
        save_file(summaries, summary)

        def reference(remote_path, local):
            return {
                "repo": cfg.hub.activations_repo,
                "path": remote_path,
                "revision": "__commit__",
                "sha256": file_digest(local),
                "bytes": local.stat().st_size,
            }

        shard_ref = reference(path + ".safetensors", shard)
        summary_ref = reference(path + "-conditioning.safetensors", summary)
        rows = {
            str(i): {
                "primary_id": data[i]["primary_id"],
                "shard": shard_ref.copy(),
                "summary": summary_ref.copy(),
            }
            for i in entries
        }
        public_record = {
            "version": 2,
            "prepared_fingerprint": manifest["prepared_fingerprint"],
            "rows": rows,
        }
        write_json(directory / "complete.json", public_record)
        record = {
            **public_record,
            "objects": [
                [path + ".safetensors", shard.name],
                [path + "-conditioning.safetensors", summary.name],
                [path + "-complete.json", "complete.json"],
            ],
        }
        # The atomic marker makes completed local packages reusable on restart.
        write_json(directory / "local.json", record)
        packages.append((directory, record))
        local_rows.update(rows)
        for i in entries:
            (raw / f"hs_{i}.safetensors").unlink(missing_ok=True)

    with stage_progress("Packaging extracted hidden states", interval=30):
        for subset in manifest["subsets"]:
            for split in ("train", "validation"):
                location = f"{profile(cfg)}/{subset['id']}/{split}"
                tensors, summaries, entries, size = {}, {}, [], 0
                for i in subset[f"{split}_indices"]:
                    if str(i) in skipped or str(i) in local_rows:
                        continue
                    row = data[i]
                    payload = load_file(raw / f"hs_{i}.safetensors")
                    states = payload["hidden_states"]
                    if (
                        states.ndim != STATE_NDIM
                        or states.shape[0] != row["seq_len"]
                        or states.shape[1] != len(cfg.hidden_states.layer_ids)
                        or states.dtype != getattr(torch, cfg.hidden_states.dtype)
                        or payload["token_ids"].tolist() != row["input_ids"]
                    ):
                        raise ValueError(
                            f"Extraction shape/dtype/tokens differ for row {i}"
                        )
                    byte_count = sum(
                        t.numel() * t.element_size() for t in payload.values()
                    )
                    if entries and size + byte_count > cfg.hub.shard_bytes:
                        seal(tensors, summaries, entries, location)
                        tensors, summaries, entries, size = {}, {}, [], 0
                    primary = row["primary_id"]
                    tensors.update(
                        {f"{primary}/{key}": value for key, value in payload.items()}
                    )
                    summaries[primary] = (
                        states[row["prompt_length"] - 1, :-1].flatten().float().clone()
                    )
                    entries.append(i)
                    size += byte_count
                if entries:
                    seal(tensors, summaries, entries, location)

    # All GPU work and packaging finish before the first activation upload.
    # Bound each commit by payload/row count, but publish every group only here.
    pending_files, pending_rows, pending_dirs = [], {}, []
    size = 0

    def publish():
        nonlocal size
        if not pending_files:
            return
        references = publisher.put_many(cfg.hub.activations_repo, pending_files)
        by_path = {r["path"]: r for r in references}
        for row in pending_rows.values():
            for kind in ("shard", "summary"):
                row[kind] = by_path[row[kind]["path"]]
        index["rows"].update(pending_rows)
        write_json(index_path, index)
        report(f"Durable hidden states: {len(index['rows'])}/{len(data)}")
        for directory in pending_dirs:
            shutil.rmtree(directory)
        for i in pending_rows:
            (raw / f"hs_{i}.safetensors").unlink(missing_ok=True)
        pending_files.clear()
        pending_rows.clear()
        pending_dirs.clear()
        size = 0

    for directory, record in packages:
        if all(i in index["rows"] for i in record["rows"]):
            shutil.rmtree(directory)
            continue
        pending_files.extend(
            (path, directory / filename) for path, filename in record["objects"]
        )
        pending_rows.update(record["rows"])
        pending_dirs.append(directory)
        size += sum(
            (directory / filename).stat().st_size for _, filename in record["objects"]
        )
        if (
            len(pending_rows) >= cfg.hub.upload_batch_examples
            or size >= cfg.hub.upload_batch_bytes
        ):
            publish()
    publish()
    artifact = publisher.put(
        cfg.hub.activations_repo, f"{profile(cfg)}/index.json", index_path
    )
    stage_record(publisher, "hidden", manifest, [artifact])


def restore(cfg, stage):  # noqa: C901 -- restore/verify four dependent stages
    """Hydrate metadata/trajectories; full activations remain lazy shard references."""
    from datasets import Dataset, concatenate_datasets, load_from_disk

    root = cfg.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def record(name):
        path = hf_hub_download(
            cfg.hub.trajectories_repo,
            f"{profile(cfg)}/{name}.json",
            repo_type="dataset",
            revision=cfg.hub.trajectories_revision,
        )
        return json.loads(Path(path).read_text())

    selection = record("select")
    manifest = selection["manifest"]
    existing_manifest = root / "manifest.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text())
        if existing["cache_fingerprint"] != manifest["cache_fingerprint"]:
            raise ValueError(
                "Restore selection differs from local data; use a fresh output_root"
            )
    manifest.update(
        root=str(root),
        data_path=str(root / "data"),
        hidden_states_path=str(root / "hidden_states"),
    )
    for reference in selection["artifacts"]:
        fetch(reference, root / "prompts.jsonl")
    if stage in ("responses", "data", "hidden", "all"):
        response_record = record("responses")
        chunks = []
        for reference in response_record["artifacts"]:
            chunks.append(
                fetch(reference, root / ".restore" / Path(reference["path"]).name)
            )
        (root / "responses.jsonl").write_bytes(b"".join(p.read_bytes() for p in chunks))
    if stage in ("data", "hidden", "all"):
        prepared = record("data")
        datasets = []
        original_columns = {
            "input_ids",
            "loss_mask",
            "seq_len",
            "primary_id",
            "source_ids",
            "category",
            "subset_id",
            "prompt_length",
            "metadata",
        }
        for r in prepared["artifacts"]:
            table = Dataset.from_parquet(
                str(fetch(r, root / ".restore" / (digest(r["path"]) + ".parquet")))
            )
            datasets.append(
                table.remove_columns(
                    [
                        name
                        for name in table.column_names
                        if name not in original_columns
                    ]
                )
            )
        rows = concatenate_datasets(datasets)
        if digest(rows.to_list()) != prepared["manifest"]["prepared_fingerprint"]:
            raise ValueError(
                "Restored trajectory order or contents differ from publication"
            )
        # Train then validation for each subset preserves local order.
        if not (root / "data").exists():
            rows.with_format("torch").save_to_disk(root / "data")
        elif (
            digest(load_from_disk(root / "data").with_format(None).to_list())
            != prepared["manifest"]["prepared_fingerprint"]
        ):
            raise ValueError(
                "Existing local trajectories differ; use a fresh output_root"
            )
        manifest = {
            **prepared["manifest"],
            "root": str(root),
            "data_path": str(root / "data"),
            "hidden_states_path": str(root / "hidden_states"),
        }
    if stage in ("hidden", "all"):
        activation = record("hidden")["artifacts"][0]
        fetch(activation, root / "hidden_states" / "index.json")
    write_json(root / "manifest.json", manifest)
    return manifest


def prepare(cfg, stage, *, resume_from_hf=False):
    with hub_progress(cfg):
        return _prepare(cfg, stage, resume_from_hf=resume_from_hf)


def _prepare(cfg, stage, *, resume_from_hf=False):  # noqa: C901 -- ordered stage restoration and publication
    from speculators.bank.workflow import prepare_arrow, select_data

    root = cfg.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".hub-prepare.lock"), timeout=0):
        if cfg.hub.mode == "consumer":
            return restore(cfg, stage)
        if resume_from_hf:
            if stage not in ("responses", "data", "hidden"):
                raise ValueError(
                    "--resume-from-hf requires responses, data, or hidden stage"
                )
            consumer = cfg.model_copy(deep=True)
            if consumer.hub.trajectories_revision == "main":
                consumer.hub.trajectories_revision = (
                    HfApi()
                    .repo_info(consumer.hub.trajectories_repo, repo_type="dataset")
                    .sha
                )
            restore(
                consumer,
                {"responses": "select", "data": "responses", "hidden": "data"}[stage],
            )
            report(f"Restored published inputs at {consumer.hub.trajectories_revision}")
        publisher = Publisher(cfg)
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            if stage not in ("select", "all"):
                restore(
                    cfg,
                    {"responses": "select", "data": "responses", "hidden": "data"}[
                        stage
                    ],
                )
            else:
                select(cfg, publisher)
        manifest = select_data(cfg)
        if stage not in ("hidden", "all"):
            provenance(publisher, stage)
        for current in ["responses", "data", "hidden"] if stage == "all" else [stage]:
            if current == "select":
                # Retry publication after an interrupted select stage.
                path = root / "prompts.jsonl"
                stage_record(
                    publisher,
                    "select",
                    manifest,
                    [
                        publisher.put(
                            cfg.hub.trajectories_repo,
                            f"{profile(cfg)}/prompts.jsonl",
                            path,
                        )
                    ],
                )
            elif current == "responses":
                responses(cfg, publisher, manifest)
            elif current == "data":
                prepare_arrow(cfg, manifest)
                publish_data(cfg, publisher, manifest)
            elif current == "hidden":
                if not manifest.get("prepared"):
                    raise ValueError(
                        "Run prepare --stage data before hidden extraction"
                    )
                hidden(cfg, publisher, manifest)
        write_json(root / "experiment.json", cfg.model_dump(mode="json"))
        revision = publisher.api.repo_info(
            cfg.hub.trajectories_repo, repo_type="dataset"
        ).sha
        report(f"Hub trajectories revision: {revision}")
        return manifest
