"""Restore pinned parent Adam/RNG state by parameter name for continuation."""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from speculators.bank.artifacts import file_digest


def restore_parent_optimizer(study, metadata, generator, optimizer):  # noqa: C901 -- strict verified optimizer mapping
    folder = Path(study.parent.model)
    if folder.is_dir():
        path = folder / (study.parent.subfolder or "") / "checkpoint.pt"
    else:
        prefix = (
            (study.parent.subfolder.rstrip("/") + "/") if study.parent.subfolder else ""
        )
        path = Path(
            hf_hub_download(
                study.parent.model,
                prefix + "checkpoint.pt",
                revision=study.parent.revision,
            )
        )
    expected = metadata.get("source_checkpoint_sha256")
    if not expected or file_digest(path) != expected:
        raise ValueError(
            "Parent optimizer checkpoint differs from the pinned bundle checksum"
        )
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source["progress"] != study.parent.optimizer_step:
        raise ValueError(
            "Parent optimizer progress differs from the requested source step"
        )
    old = source["optimizer"]
    if len(old["param_groups"]) != 1:
        raise ValueError("Expected a single original generator optimizer group")
    # Older campaign checkpoints predate parameter_names. Their ordered model
    # state contains only parameters, in the original optimizer's module order.
    names = source.get("parameter_names", list(source["model"]))
    source_ids = old["param_groups"][0]["params"]
    if len(names) != len(source_ids) or len(set(names)) != len(names):
        raise ValueError("Cannot map parent optimizer state unambiguously")
    current_names = [
        name for name, p in generator.named_parameters() if p.requires_grad
    ]
    expected_names = set(current_names) - (
        {"feedback_encoder.weight"} if study.feedback else set()
    )
    if set(names) != expected_names:
        raise ValueError(
            "Parent optimizer parameter names differ from the original generator"
        )
    result = optimizer.state_dict()
    ids = dict(zip(names, source_ids, strict=True))
    for name, identifier in zip(
        current_names, result["param_groups"][0]["params"], strict=True
    ):
        if name in ids and ids[name] in old["state"]:
            result["state"][identifier] = old["state"][ids[name]]
    for key, value in old["param_groups"][0].items():
        if key != "params":
            result["param_groups"][0][key] = value
    if (
        result["param_groups"][0]["lr"] != study.lr
        or result["param_groups"][0]["weight_decay"] != study.weight_decay
    ):
        raise ValueError(
            "Continuation LR/weight decay differ from the parent optimizer"
        )
    optimizer.load_state_dict(result)
    torch.set_rng_state(source["torch_rng"])
    if next(generator.parameters()).device.type == "cuda" and source.get("cuda_rng"):
        torch.cuda.set_rng_state_all(source["cuda_rng"])
    return {
        "checkpoint_sha256": expected,
        "optimizer_step": source["progress"],
        "restored_parameters": len(names),
        "new_parameters": sorted(set(current_names) - set(names)),
    }
