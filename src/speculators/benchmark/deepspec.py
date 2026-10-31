"""Metadata-only preparation of the pinned DeepSpec Qwen3-8B EAGLE3 checkpoint."""

import copy
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from speculators.bank.artifacts import file_digest, write_json
from speculators.provenance import atomic_write

REPOSITORY = "deepseek-ai/eagle3_qwen3_8b_ttt7"
REVISION = "f6485ba8d21e11942958617dbe7e71b467f38f38"
TARGET_LAYERS = [1, 9, 17, 25, 33]


def serving_config(original):
    """Use vLLM's Qwen3 loader and its pre-decoder auxiliary indexing convention."""
    expected = {
        "architectures": ["Qwen3Eagle3Model"],
        "model_type": "qwen3",
        "target_model_name_or_path": "Qwen/Qwen3-8B",
        "target_layer_ids": TARGET_LAYERS,
        "num_hidden_layers": 1,
        "hidden_size": 4096,
        "num_target_layers": 36,
        "ttt_length": 7,
    }
    if any(original.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected DeepSpec checkpoint configuration")
    result = copy.deepcopy(original)
    result.update(
        architectures=["Eagle3Qwen3ForCausalLM"],
        # DeepSpec reads HF hidden_states[layer_id + 1]; vLLM collects before
        # decoder layer i, so these offsets select the same raw decoder outputs.
        eagle_aux_hidden_state_layer_ids=[i + 1 for i in TARGET_LAYERS],
        num_aux_hidden_states=len(TARGET_LAYERS),
        norm_before_residual=False,
        norm_output=False,
    )
    return result


def prepare(output_root):
    """Keep source config and exact weights; stage a distinct serving config."""
    source = Path(
        snapshot_download(
            REPOSITORY,
            revision=REVISION,
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    original = source / "config.json"
    config = serving_config(json.loads(original.read_text()))
    weights = source / "model.safetensors"
    output = Path(output_root) / REVISION
    output.mkdir(parents=True, exist_ok=True)
    target = output / "model.safetensors"
    # Preserve cached tensor bytes; avoid allocating another multi-GB copy.
    if target.is_symlink():
        if target.resolve() != weights.resolve():
            raise ValueError("Prepared weights point to a different checkpoint")
    elif target.exists():
        raise FileExistsError("Prepared model weights must reference the pinned cache")
    else:
        target.symlink_to(weights.resolve())
    atomic_write(output / "original_config.json", original.read_text())
    write_json(output / "config.json", config)
    write_json(
        output / "preparation.json",
        {
            "source_repository": REPOSITORY,
            "source_revision": REVISION,
            "original_config_sha256": file_digest(original),
            "serving_config_sha256": file_digest(output / "config.json"),
            "weights_sha256": file_digest(weights),
            "weights_changed": False,
            "source_target_layer_ids": TARGET_LAYERS,
            "vllm_aux_hidden_state_layer_ids": config[
                "eagle_aux_hidden_state_layer_ids"
            ],
            "vllm_version": "0.31.0",
        },
    )
    return output


def native_training_config(original, verifier):
    """Retain Qwen3 layers/QK norms and full-vocabulary weights in native format."""
    from transformers import Qwen3Config  # noqa: PLC0415

    from speculators.config import SpeculatorsConfig, VerifierConfig  # noqa: PLC0415
    from speculators.models.eagle3 import Eagle3SpeculatorConfig  # noqa: PLC0415
    from speculators.proposals.greedy import GreedyTokenProposalConfig  # noqa: PLC0415

    serving = serving_config(original)
    return Eagle3SpeculatorConfig(
        transformer_layer_config=Qwen3Config(**original),
        draft_vocab_size=original["vocab_size"],
        eagle_aux_hidden_state_layer_ids=serving["eagle_aux_hidden_state_layer_ids"],
        norm_before_residual=False,
        norm_output=False,
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=str(verifier), architectures=["Qwen3ForCausalLM"]
            ),
        ),
    )


def prepare_training(serving_path, output_root, verifier):
    """Write native training metadata after checking every source tensor shape."""
    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415

    from speculators.models.eagle3 import Eagle3DraftModel  # noqa: PLC0415

    source = Path(serving_path)
    original = json.loads((source / "original_config.json").read_text())
    config = native_training_config(original, verifier)
    with torch.device("meta"):
        model = Eagle3DraftModel(config)
    expected = {key: list(value.shape) for key, value in model.state_dict().items()}
    weights = source / "model.safetensors"
    with safe_open(weights, framework="pt", device="cpu") as tensors:
        found = set(tensors.keys())
        for key in found:
            if (
                key not in expected
                or tensors.get_slice(key).get_shape() != expected[key]
            ):
                raise ValueError(f"Unexpected DeepSpec tensor: {key}")
    missing = set(expected) - found
    if missing != {"verifier_norm.weight", "verifier_lm_head.weight"}:
        raise ValueError(f"Missing DeepSpec model tensors: {sorted(missing)}")
    output = Path(output_root) / REVISION
    output.mkdir(parents=True, exist_ok=True)
    target = output / "model.safetensors"
    if target.is_symlink():
        if target.resolve() != weights.resolve():
            raise ValueError("Native training weights point to another checkpoint")
    elif target.exists():
        raise FileExistsError("Native training weights must reference the pinned cache")
    else:
        target.symlink_to(weights.resolve())
    atomic_write(output / "config.json", config.to_json_string())
    atomic_write(
        output / "original_config.json", (source / "original_config.json").read_text()
    )
    write_json(
        output / "preparation.json",
        {
            **json.loads((source / "preparation.json").read_text()),
            "native_training_config_sha256": file_digest(output / "config.json"),
            "training_verifier": str(verifier),
            "missing_tensors_loaded_from_verifier": sorted(missing),
            "weights_changed": False,
        },
    )
    return output
