"""Factorization-invariant diagnostics without materializing dense updates."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from speculators.bank.artifacts import file_digest

MATRIX_NDIM = 2


def factors(state: dict, rank: int) -> dict:
    result = {}
    for name, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite adapter tensor: {name}")
        if name.endswith("lora_A.weight"):
            prefix = name.removesuffix("lora_A.weight")
            partner = prefix + "lora_B.weight"
            if partner not in state:
                raise ValueError(f"Missing B factor for {name}")
            a, b = value.double(), state[partner].double()
            if (
                a.ndim != MATRIX_NDIM
                or b.ndim != MATRIX_NDIM
                or a.shape[0] != rank
                or b.shape[1] != rank
            ):
                raise ValueError(f"Incompatible factor shapes for {name}")
            result[prefix] = (a, b)
    if not result or len(state) != 2 * len(result):
        raise ValueError("Expected paired LoRA A/B tensors only")
    return result


def update_norm_squared(pair):
    a, b = pair
    return ((b.T @ b) * (a @ a.T)).sum()


def update_distance(left: dict, right: dict, scale: float) -> float:
    if left.keys() != right.keys():
        raise ValueError("Adapter modules differ")
    distance = 0.0
    for name, (a, b) in left.items():
        c, d = right[name]
        cross = ((b.T @ d) * (a @ c.T)).sum()
        distance += float(
            (
                update_norm_squared((a, b)) + update_norm_squared((c, d)) - 2 * cross
            ).clamp_min(0)
        )
    return distance**0.5 * scale


def inspect_adapters(run: Path, lora):
    entries, norms, distances, hashes = [], [], [], set()
    previous = None
    duplicates = 0
    for path in sorted((run / "snapshots").glob("step-*")):
        entry = json.loads((path / "entry.json").read_text())
        adapter = path / "adapter/adapter_model.safetensors"
        config_path = path / "adapter/adapter_config.json"
        config = json.loads(config_path.read_text())
        if config["r"] != lora.rank or config["lora_alpha"] != lora.alpha:
            raise ValueError(f"Rank/alpha mismatch: {path}")
        if set(config["target_modules"]) != set(lora.target_modules):
            raise ValueError(f"Target modules mismatch: {path}")
        if (
            file_digest(adapter) != entry["adapter_sha256"]
            or file_digest(config_path) != entry["adapter_config_sha256"]
        ):
            raise ValueError(f"Adapter checksum mismatch: {path}")
        state = load_file(str(adapter))
        pairs = factors(state, lora.rank)
        expected = {"o_proj": (4096, 4096), "v_proj": (8192, 1024)}
        modules = []
        for name, (a, b) in pairs.items():
            module = name.rstrip(".").split(".")[-1]
            if module not in expected or (a.shape[1], b.shape[0]) != expected[module]:
                raise ValueError(f"Unexpected EAGLE3 module or shape: {name}")
            modules.append(module)
        if sorted(modules) != sorted(lora.target_modules):
            raise ValueError(f"Expected exactly one o_proj and one v_proj: {path}")
        tensor_hash = hashlib.sha256()
        for name, value in sorted(state.items()):
            tensor_hash.update(name.encode())
            tensor_hash.update(value.float().numpy().tobytes())
        identity = tensor_hash.hexdigest()
        duplicates += identity in hashes
        hashes.add(identity)
        scale = lora.alpha / lora.rank
        norm = (
            sum(float(update_norm_squared(pair)) for pair in pairs.values()) ** 0.5
            * scale
        )
        norms.append(norm)
        if previous is not None:
            distances.append(update_distance(previous, pairs, scale))
        previous = pairs
        metrics_path = run / "epochs" / f"{entry['epoch']}.json"
        entries.append(
            {
                **entry,
                "path": str(path),
                "effective_update_norm": norm,
                "factor_sha256": identity,
                "module_shapes": {
                    name: [list(a.shape), list(b.shape)]
                    for name, (a, b) in pairs.items()
                },
                "epoch_validation": json.loads(metrics_path.read_text())
                if metrics_path.exists()
                else None,
            }
        )
    recovery = run / "recovery"
    states = sorted(
        (
            p
            for p in recovery.glob("*")
            if p.name.isdigit() and (p / "training_state.json").exists()
        ),
        key=lambda p: int(p.name),
    )
    latest = (
        json.loads((states[-1] / "training_state.json").read_text()) if states else None
    )
    return {
        "entries": entries,
        "candidate_count": len(entries),
        "exact_duplicates": duplicates,
        "effective_update_norm_range": [min(norms), max(norms)] if norms else None,
        "mean_adjacent_update_distance": sum(distances) / len(distances)
        if distances
        else None,
        "recovery_state": {
            k: latest[k] for k in ("phase", "global_step", "collection_step")
        }
        if latest
        else None,
    }
