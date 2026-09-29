"""Optional PEFT LoRA integration for pretrained speculator fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from speculators.model import SpeculatorModel

logger = logging.getLogger("speculators")

ADAPTER_DIRNAME = "adapter"


def _import_peft():
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from peft.utils.save_and_load import set_peft_model_state_dict
    except ImportError as exc:
        raise ImportError(
            "LoRA training requires PEFT. Install it with "
            "`pip install 'speculators[lora]'` (or `pip install -e '.[lora]'` "
            "from a source checkout)."
        ) from exc
    return LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict


def is_lora_model(model: torch.nn.Module) -> bool:
    """Return whether *model* was wrapped by :func:`apply_lora`."""
    return bool(getattr(model, "_speculators_lora_enabled", False))


def base_speculator_model(model: torch.nn.Module) -> SpeculatorModel:
    """Return the underlying SpeculatorModel for dense or PEFT-wrapped models."""
    if is_lora_model(model):
        model = model.get_base_model()  # type: ignore[attr-defined]
    if not isinstance(model, SpeculatorModel):
        raise TypeError(
            f"Expected a SpeculatorModel or PEFT-wrapped SpeculatorModel, got "
            f"{type(model).__name__}"
        )
    return model


def apply_lora(
    model: SpeculatorModel,
    *,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    save_merged: bool,
) -> torch.nn.Module:
    """Freeze *model* and attach trainable LoRA matrices to selected linears."""
    lora_config_cls, _, get_peft_model, _ = _import_peft()
    config = lora_config_cls(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    wrapped = get_peft_model(model, config)
    wrapped._speculators_lora_enabled = True  # noqa: SLF001
    wrapped._speculators_lora_save_merged = save_merged  # noqa: SLF001

    matched = sorted(
        name
        for name, module in wrapped.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A")
    )
    if not matched:
        raise ValueError(
            "None of the requested LoRA target modules matched the drafter: "
            + ", ".join(target_modules)
        )

    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    logger.info("LoRA target modules (%d): %s", len(matched), ", ".join(matched))
    logger.info(
        "LoRA trainable parameters: %d / %d (%.4f%%)",
        trainable,
        total,
        100.0 * trainable / total,
    )
    return wrapped


def load_adapter_checkpoint(model: torch.nn.Module, checkpoint_dir: Path) -> bool:
    """Load an adapter saved below *checkpoint_dir*; return whether one existed."""
    adapter_dir = checkpoint_dir / ADAPTER_DIRNAME
    adapter_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_path.exists():
        return False
    if not is_lora_model(model):
        raise ValueError(
            f"Checkpoint {checkpoint_dir} contains a LoRA adapter but the current "
            "run did not enable LoRA. Reuse the saved run configuration."
        )
    _, _, _, set_peft_model_state_dict = _import_peft()
    state_dict = load_file(
        str(adapter_path), device=str(next(model.parameters()).device)
    )
    result: Any = set_peft_model_state_dict(model, state_dict)
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise ValueError(
            f"Unexpected keys while loading LoRA adapter from {adapter_dir}: "
            f"{unexpected}"
        )
    logger.info("Loaded LoRA adapter from %s", adapter_dir)
    return True


def save_lora_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: Path,
    *,
    float_dtype: torch.dtype,
) -> bool:
    """Save adapter and optional merged model; return whether *model* uses LoRA."""
    if not is_lora_model(model):
        return False

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = checkpoint_dir / ADAPTER_DIRNAME
    model.save_pretrained(adapter_dir, safe_serialization=True)  # type: ignore[attr-defined]

    if getattr(model, "_speculators_lora_save_merged", True):
        model.merge_adapter()  # type: ignore[attr-defined]
        try:
            base_model = base_speculator_model(model)
            state_dict = {}
            for name, value in base_model.state_dict().items():
                # A merged PEFT Linear still contains its wrapper and adapter
                # tensors until unmerge/unload. Save the merged base_layer under
                # the original model key and omit adapter-only tensors so the
                # checkpoint reloads as a plain SpeculatorModel.
                if ".lora_" in name:
                    continue
                name = name.replace(".base_layer.", ".")
                state_dict[name] = (
                    value.to(float_dtype) if value.is_floating_point() else value
                )
            base_model.save_pretrained(checkpoint_dir, state_dict=state_dict)
        finally:
            model.unmerge_adapter()  # type: ignore[attr-defined]
    return True
