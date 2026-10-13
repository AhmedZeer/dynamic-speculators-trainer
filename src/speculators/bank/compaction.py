"""Build a separate training release from already published activation data."""

import json
from pathlib import Path

import yaml
from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import save_file

from speculators.bank.artifacts import write_json
from speculators.bank.hub import Publisher, hidden, prepare, profile, stage_record
from speculators.bank.hub_cache import HubActivationStore
from speculators.bank.progress import report, stage_progress


def compact(cfg, release, output_root):  # noqa: C901 -- ordered source restore and release publication
    if cfg.hub is None:
        raise ValueError("Compaction requires a Hub-enabled config")
    if release == cfg.hub.release:
        raise ValueError("Training release must differ from the source release")
    output_root = Path(output_root).resolve()
    if output_root == cfg.output_root.resolve():
        raise ValueError("Compaction requires a separate output root")
    output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output_root / ".compaction.lock"), timeout=0):
        source = cfg.model_copy(deep=True)
        source.output_root = output_root / ".source"
        source.hub.mode = "consumer"
        if source.hub.trajectories_revision == "main":
            source.hub.trajectories_revision = (
                HfApi().repo_info(source.hub.trajectories_repo, repo_type="dataset").sha
            )
        manifest = prepare(source, "hidden")
        store = HubActivationStore(Path(manifest["hidden_states_path"]))
        target = cfg.model_copy(deep=True)
        target.output_root = output_root
        target.hub.release = release
        target.hub.mode = "producer"
        target.hub.activation_format = "sharded"
        target.hub.extraction_scope = "subset"
        # Validate externally provided release identifiers before publishing.
        target = type(cfg).model_validate(target.model_dump())
        manifest = {
            **manifest,
            "root": str(output_root),
            "hidden_states_path": str(output_root / "hidden_states"),
        }
        existing = output_root / "manifest.json"
        if existing.exists() and json.loads(existing.read_text()).get(
            "prepared_fingerprint"
        ) != manifest["prepared_fingerprint"]:
            raise ValueError("Compaction output belongs to different trajectories")
        write_json(output_root / "manifest.json", manifest)
        publisher = Publisher(target)
        # Reference original immutable trajectories instead of copying payloads.
        for stage in ("select", "responses", "data"):
            record = hf_hub_download(
                source.hub.trajectories_repo,
                f"{profile(source)}/{stage}.json",
                repo_type="dataset",
                revision=source.hub.trajectories_revision,
            )
            saved = json.loads(Path(record).read_text())
            stage_record(publisher, stage, saved["manifest"], saved["artifacts"])
        raw = output_root / "hidden_states" / ".raw-states"
        raw.mkdir(parents=True, exist_ok=True)
        for subset in manifest["subsets"]:
            local_index = output_root / "hidden_states" / "index.json"
            completed = (
                json.loads(local_index.read_text())["rows"]
                if local_index.exists()
                else {}
            )
            with stage_progress(
                f"Restoring activations for {subset['id']}", interval=30
            ):
                for i in subset["train_indices"] + subset["validation_indices"]:
                    destination = raw / f"hs_{i}.safetensors"
                    if str(i) in completed or destination.exists():
                        continue
                    payload = store.read(i)
                    if payload is None:
                        raise ValueError(f"Source activations missing for row {i}")
                    pending = destination.with_suffix(".pending")
                    save_file(payload, pending)
                    pending.replace(destination)
            # Existing packager builds summaries and shards; it never calls vLLM.
            hidden(target, publisher, manifest, publish_only=True, subsets=[subset])
        revision = publisher.api.repo_info(
            target.hub.trajectories_repo, repo_type="dataset"
        ).sha
        target.hub.mode = "consumer"
        target.hub.trajectories_revision = revision
        config_path = output_root / "bank_training.yaml"
        config_path.write_text(
            yaml.safe_dump(target.model_dump(mode="json"), sort_keys=False)
        )
        report(
            f"Training release ready: {release}; config={config_path}; "
            f"revision={revision}"
        )
        return config_path
