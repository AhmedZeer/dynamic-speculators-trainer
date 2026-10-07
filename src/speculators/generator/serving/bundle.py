"""Portable, optimizer-free exports of generator and ordinary LoRA runs."""

import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.generator.config import Architecture
from speculators.generator.model import LoRAGenerator

BUNDLE_VERSION = 1
VLLM_VERSION = "0.31.0"
VLLM_COMMIT = "db9527a46873454610df6dbedf79a36d6bf1a7f6"
PROJECTIONS = ("o_proj", "v_proj")
PROVENANCE = (
    "resolved_experiment.json",
    "train_command.txt",
    "speculators.patch",
    "result.json",
    "metrics.json",
    "provenance",
)


def read_bundle(path):
    path = Path(path)
    value = json.loads((path / "bundle.json").read_text())
    if value["version"] != BUNDLE_VERSION or value["kind"] not in (
        "generator",
        "static_lora",
    ):
        raise ValueError("Unsupported inference bundle")
    if value["kind"] == "generator" and value["condition"] != "last":
        raise ValueError("The serial runtime supports only last conditioning")
    if set(value["shapes"]) != set(PROJECTIONS):
        raise ValueError("Expected o_proj and v_proj in the inference bundle")
    if file_digest(path / "model.safetensors") != value["weights_sha256"]:
        raise ValueError("Inference bundle weights do not match their checksum")
    return value


def load_generator(metadata, state):
    generator = LoRAGenerator(
        metadata["input_dim"],
        metadata["shapes"],
        metadata["rank"],
        Architecture.model_validate(metadata["architecture"]),
        metadata["max_examples"],
    ).float()
    generator.load_state_dict(state, strict=True)
    return generator.eval().requires_grad_(False)


def export_run(cfg, run_dir, output):  # noqa: C901 -- validate and atomically export a run
    run_dir, output = Path(run_dir), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new inference bundle directory: {output}")
    identity = json.loads((run_dir / "resolved_experiment.json").read_text())
    if not (run_dir / "result.json").is_file():
        raise ValueError("Export requires a completed experiment run")
    for name in ("train_command.txt", "speculators.patch"):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"Missing training provenance: {run_dir / name}")
    bank = BankConfig.load(cfg.bank_config)
    manifest = json.loads((bank.output_root / "manifest.json").read_text())
    if (
        identity["bank_data"]["prepared_fingerprint"]
        != manifest["prepared_fingerprint"]
    ):
        raise ValueError("Run and bank data identities differ")
    if identity["bank_data"]["revisions"] != manifest["revisions"]:
        raise ValueError("Run and bank model revisions differ")
    if identity["lora"] != bank.lora.model_dump(mode="json"):
        raise ValueError("Run and bank LoRA configurations differ")
    model_ids = manifest["cache_spec"]["models"]
    models = {
        role: {"id": model_ids[role], "revision": manifest["revisions"][f"{role}_sha"]}
        for role in ("target", "drafter")
    }
    source_config = json.loads(
        Path(
            hf_hub_download(
                models["drafter"]["id"],
                "config.json",
                revision=models["drafter"]["revision"],
            )
        ).read_text()
    )
    if source_config.get("quantization_config"):
        raise ValueError("The merge prototype requires an unquantized drafter")
    job = identity["job"]
    kind = (
        "static_lora"
        if job.get("arm") in ("fresh_lora", "transferred_lora")
        else "generator"
    )
    if kind == "generator" and job["condition"] != "last":
        raise ValueError("The serial runtime supports only last conditioning")
    checkpoint = run_dir / "checkpoint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model"]
    rank = identity["lora"]["rank"]
    shapes = {}
    if kind == "generator":
        for name in PROJECTIONS:
            shapes[name] = [
                state[f"heads.{name}.A.bias"].numel() // rank,
                state[f"heads.{name}.B.bias"].numel() // rank,
            ]
    else:
        factors = {}
        for name in PROJECTIONS:
            for suffix, label in (("a", "A"), ("b", "B")):
                found = [
                    tensor
                    for key, tensor in state.items()
                    if key.endswith(f".{name}.{suffix}")
                ]
                if len(found) != 1:
                    raise ValueError(f"Expected exactly one {name}.{suffix} factor")
                factors[f"{name}.{label}"] = found[0]
            shapes[name] = [
                factors[f"{name}.A"].shape[1],
                factors[f"{name}.B"].shape[0],
            ]
        state = factors
    metadata = {
        "version": BUNDLE_VERSION,
        "kind": kind,
        "models": models,
        "condition": job["condition"] if kind == "generator" else None,
        "layer_ids": manifest["cache_spec"]["hidden_states"]["layer_ids"][:-1],
        "rank": rank,
        "alpha": identity["lora"]["alpha"],
        "shapes": shapes,
        "source_identity": digest(identity),
        "source_job": job,
        "source_checkpoint_sha256": file_digest(checkpoint),
        "draft_config": source_config,
    }
    if kind == "generator":
        metadata.update(
            input_dim=state["encoder.0.weight"].shape[1] - 1,
            architecture=identity["configuration"]["architecture"],
            max_examples=identity["configuration"]["context"]["max_examples"],
        )
        load_generator(metadata, state)
    else:
        for name in PROJECTIONS:
            inputs, outputs = shapes[name]
            if state[f"{name}.A"].shape != (rank, inputs) or state[
                f"{name}.B"
            ].shape != (outputs, rank):
                raise ValueError(f"Invalid {name} factor shape")
    state = {key: tensor.detach().float().contiguous() for key, tensor in state.items()}
    if not all(torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Non-finite exported weights")
    pending = output.with_name(output.name + ".pending")
    if pending.exists():
        raise FileExistsError(f"Previous partial export exists: {pending}")
    pending.mkdir(parents=True)
    try:
        save_file(state, pending / "model.safetensors")
        metadata["weights_sha256"] = file_digest(pending / "model.safetensors")
        write_json(pending / "bundle.json", metadata)
        for name in PROVENANCE:
            source = run_dir / name
            if source.is_dir():
                shutil.copytree(source, pending / name)
            elif source.is_file():
                shutil.copy2(source, pending / name)
        pending.rename(output)
    except BaseException:
        shutil.rmtree(pending)
        raise
    return metadata


def bundle_state(path):
    metadata = read_bundle(path)
    return metadata, load_file(str(Path(path) / "model.safetensors"), device="cpu")
