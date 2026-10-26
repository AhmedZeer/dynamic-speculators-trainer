"""Resolve immutable model revisions and cache only the required workload files."""

import json
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.campaign import jobs
from speculators.generator.campaign_inputs import campaign_banks
from speculators.generator.serving.bundle import PROVENANCE, read_bundle
from speculators.generator.serving.external_validation import export_external_validation
from speculators.generator.serving.validation import export_validation, load_validation


def local_source(value, base):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _local_drafter(source, revision, output):  # noqa: C901 -- validate and stage files
    if revision is None and re.fullmatch(r"[0-9a-f]{40}", source.name):
        revision = source.name
    cache_metadata = source / ".cache/huggingface/download/config.json.metadata"
    if revision is None and cache_metadata.exists():
        revision = cache_metadata.read_text().splitlines()[0]
    if revision is None or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            "Local drafters require a full revision SHA or Hub download metadata"
        )
    destination = output / revision
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Local drafter staging must be outside its source folder")
    hashes = {}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or any(
            part in (".cache", ".git") for part in relative.parts
        ):
            continue
        hashes[str(relative)] = file_digest(path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_digest(target) != hashes[str(relative)]:
                raise ValueError(
                    "Staged local drafter changed; choose a new output root"
                )
        else:
            try:
                target.hardlink_to(path.resolve())
            except OSError:
                shutil.copy2(path, target)
    if "config.json" not in hashes:
        raise FileNotFoundError("Local drafter has no config.json")
    return destination, revision, hashes


def resolve_model(spec, base, output, pinned=None):  # noqa: C901 -- pin local/remote model and serving policy
    source = local_source(spec.model, base)
    local = source.is_dir()
    if not local and (spec.model.startswith((".", "/", "~"))):
        raise FileNotFoundError(f"Local model directory does not exist: {source}")
    revision = pinned or spec.revision
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        revision = HfApi().model_info(spec.model, revision=revision or "main").sha
    result = {
        "name": spec.name,
        "type": spec.type,
        "model": spec.model,
        "revision": revision,
        "subfolder": spec.subfolder,
    }
    if spec.adapter_policy == "per_request":
        result.update(adapter_policy="per_request", conditioning_examples=1)
    if spec.vanilla_vllm:
        result["vanilla_vllm"] = True
    if spec.batching_policy is not None:
        result["batching_policy"] = spec.batching_policy
    if spec.adapter_experiment is not None:
        policy = spec.adapter_experiment.model_dump(mode="json")
        if policy["donor_bank"]:
            bank = Path(policy["donor_bank"]).resolve()
            policy["donor_bank"] = str(bank)
            result["donor_bank_sha256"] = file_digest(bank)
            result["donor_metadata_sha256"] = file_digest(bank.with_suffix(".json"))
        result["adapter_experiment"] = policy
    if spec.type == "drafter":
        if spec.subfolder:
            raise ValueError(
                "Drafter subfolders are unsupported; supply a local model folder"
            )
        if local:
            path, revision, hashes = _local_drafter(
                source, revision, output / spec.name
            )
            result.update(path=str(path), revision=revision, local_files_sha256=hashes)
        else:
            result["path"] = spec.model
        return result
    prefix = (spec.subfolder.rstrip("/") + "/") if spec.subfolder else ""
    if not local:
        source = Path(
            snapshot_download(
                spec.model,
                revision=revision,
                allow_patterns=[
                    prefix + "bundle.json",
                    prefix + "model.safetensors",
                    *[prefix + name for name in PROVENANCE if name != "provenance"],
                    prefix + "provenance/**",
                ],
                ignore_patterns=["checkpoint.pt", "*/checkpoint.pt"],
            )
        )
    folder = source / (spec.subfolder or "")
    metadata = read_bundle(folder)
    expected = "static_lora" if spec.type == "lora" else "generator"
    if metadata["kind"] != expected:
        raise ValueError(
            f"{spec.name}: declared {spec.type}, bundle is {metadata['kind']}"
        )
    result.update(
        path=str(folder),
        weights_sha256=metadata["weights_sha256"],
        bundle_sha256=file_digest(folder / "bundle.json"),
        models=metadata["models"],
    )
    return result


def ensure_inputs(cfg, *, nemotron):
    """Download metadata and selected validation files without training payloads."""
    banks = campaign_banks(cfg)
    names = {name for job in jobs(cfg) for name in job["validation_subsets"]}
    for domain, bank in banks.items():
        if not nemotron and domain != "math":
            continue
        root = cfg.trajectories_path
        relative = f"releases/{bank.hub.release}/{domain}/data.json"

        def fetch(name, root=root, bank=bank):
            destination = root / name
            if not destination.exists():
                cached = hf_hub_download(
                    bank.hub.trajectories_repo,
                    name,
                    repo_type="dataset",
                    revision=bank.hub.trajectories_revision,
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + ".suite-partial")
                shutil.copy2(cached, partial)
                partial.replace(destination)
            return destination

        metadata = fetch(relative)
        if nemotron:
            manifest = json.loads(metadata.read_text())["manifest"]
            for subset in manifest["subsets"]:
                if subset["id"] in names:
                    fetch(
                        f"releases/{bank.hub.release}/{domain}/{subset['id']}/validation.parquet"
                    )


def prepare_workload(benchmark, cfg, output, settings, tokenizer):  # noqa: C901 -- explicit manifests and legacy workload exports
    request = {
        "benchmark": benchmark.model_dump(mode="json"),
        "max_samples": settings.max_samples,
        "max_tokens": settings.max_tokens,
        "seed": settings.seed,
    }
    request_path = output.with_name(output.name + ".request.json")
    if request_path.exists() and json.loads(request_path.read_text()) != request:
        raise ValueError("Frozen workload settings changed; choose a new output root")
    write_json(request_path, request)
    manifest_path = output / "manifest.json"
    if benchmark.validation_manifest is not None and not manifest_path.exists():
        source = Path(benchmark.validation_manifest).resolve()
        source_manifest, source_groups = load_validation(source)
        selected = benchmark.categories
        names = (
            list(source_groups)
            if selected == ["all"]
            else [
                name if name in source_groups else "nemo-" + name for name in selected
            ]
        )
        if any(name not in source_groups for name in names):
            raise ValueError(
                "Selected category is absent from the explicit validation manifest"
            )
        if any(len(source_groups[name]) > settings.max_samples for name in names):
            raise ValueError(
                "Explicit workload exceeds max_samples; "
                "prepare the intended split first"
            )
        if source_manifest["max_tokens"] != settings.max_tokens:
            raise ValueError(
                "Explicit workload token cap differs from benchmark settings"
            )
        partial = output.with_name(output.name + ".pending")
        if partial.exists():
            shutil.rmtree(partial)
        partial.mkdir(parents=True)
        source_manifest["groups"] = {
            name: source_manifest["groups"][name] for name in names
        }
        for name in names:
            item = source_manifest["groups"][name]
            shutil.copy2(source.parent / item["file"], partial / item["file"])
        source_manifest.pop("identity")
        source_manifest["identity"] = digest(source_manifest)
        write_json(partial / "manifest.json", source_manifest)
        partial.replace(output)
    if not manifest_path.exists():
        partial = output.with_name(output.name + ".pending")
        if partial.exists():
            shutil.rmtree(partial)
        ensure_inputs(cfg, nemotron=benchmark.dataset == "nemotron")
        exporter = (
            export_validation
            if benchmark.dataset == "nemotron"
            else export_external_validation
        )
        options = (
            {}
            if benchmark.dataset == "nemotron"
            else {"prompt_format": benchmark.prompt_format}
        )
        exporter(
            cfg,
            output,
            tokenizer_path=tokenizer,
            max_samples=settings.max_samples,
            max_tokens=settings.max_tokens,
            seed=settings.seed,
            categories=None
            if benchmark.categories == ["all"]
            else benchmark.categories,
            **options,
        )
    manifest, groups = load_validation(manifest_path)
    if "drafter" in manifest["models"]:
        # A shared workload constrains target/tokenizer, while each suite entry
        # defines its own drafter. Preserve the original export for provenance.
        write_json(output / "source_manifest.json", manifest)
        manifest["training_reference_drafter"] = manifest["models"].pop("drafter")
        manifest["drafter_policy"] = "model-defined"
        manifest.pop("identity")
        manifest["identity"] = digest(manifest)
        write_json(manifest_path, manifest)
    return {
        "path": str(manifest_path),
        "identity": manifest["identity"],
        "models": manifest["models"],
        "groups": {
            name: {"rows": len(rows), "sha256": manifest["groups"][name]["sha256"]}
            for name, rows in groups.items()
        },
    }
