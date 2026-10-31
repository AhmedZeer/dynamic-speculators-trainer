"""DeepSpec metadata translation preserves source and weight bytes."""

import json
from pathlib import Path

import pytest

from speculators.benchmark import deepspec
from speculators.benchmark.config import Suite


def source_config():
    return {
        "architectures": ["Qwen3Eagle3Model"],
        "model_type": "qwen3",
        "target_model_name_or_path": "Qwen/Qwen3-8B",
        "target_layer_ids": [1, 9, 17, 25, 33],
        "num_hidden_layers": 1,
        "num_target_layers": 36,
        "hidden_size": 4096,
        "ttt_length": 7,
    }


def test_metadata_maps_architecture_and_five_raw_target_outputs():
    original = source_config()
    result = deepspec.serving_config(original)
    assert original == source_config()
    assert result["architectures"] == ["Eagle3Qwen3ForCausalLM"]
    assert result["eagle_aux_hidden_state_layer_ids"] == [2, 10, 18, 26, 34]
    assert result["num_aux_hidden_states"] == 5
    assert result["norm_before_residual"] is False
    assert result["norm_output"] is False
    for key in original.keys() - {"architectures"}:
        assert result[key] == original[key]


def test_unexpected_checkpoint_is_rejected():
    original = source_config()
    original["target_layer_ids"] = [2, 18, 33]
    with pytest.raises(ValueError, match="Unexpected"):
        deepspec.serving_config(original)


def test_prepare_preserves_weights_and_original_metadata(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    original = json.dumps(source_config())
    (source / "config.json").write_text(original)
    payload = b"unchanged model tensor bytes"
    (source / "model.safetensors").write_bytes(payload)

    def download(repo, *, revision, allow_patterns):
        assert repo == deepspec.REPOSITORY
        assert revision == deepspec.REVISION
        assert allow_patterns == ["config.json", "model.safetensors"]
        return source

    monkeypatch.setattr(deepspec, "snapshot_download", download)
    root = tmp_path / "prepared"
    output = deepspec.prepare(root)
    assert (output / "original_config.json").read_text() == original
    assert (source / "config.json").read_text() == original
    assert (output / "model.safetensors").read_bytes() == payload
    record = json.loads((output / "preparation.json").read_text())
    assert record["weights_sha256"] == deepspec.file_digest(
        source / "model.safetensors"
    )
    assert record["weights_changed"] is False
    assert deepspec.prepare(root) == output
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"other checkpoint")
    (output / "model.safetensors").unlink()
    (output / "model.safetensors").symlink_to(unrelated)
    with pytest.raises(ValueError, match="different checkpoint"):
        deepspec.prepare(root)


def test_suite_retains_matched_settings_and_only_one_model():
    root = (
        Path(__file__).resolve().parents[3]
        / "examples/eagle3-lora/structure_benchmarks_v1"
    )
    candidate = Suite.load(root / "deepseek_ttt7_math_chat_c1.yaml")
    reference = Suite.load(root / "math_chat_c1.yaml")
    assert candidate.settings == reference.settings
    assert candidate.benchmarks == reference.benchmarks
    assert len(candidate.models) == 1
    model = candidate.models[0]
    assert model.type == "drafter"
    assert model.revision == deepspec.REVISION
