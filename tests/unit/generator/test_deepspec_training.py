"""DeepSpec generator settings and explicit saved-response/runtime separation."""

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from speculators.benchmark import deepspec
from speculators.generator.campaign_inputs import campaign_banks
from speculators.generator.local_campaign import (
    load_local_inputs,
    runtime_trajectory_manifests,
)
from speculators.generator.serving import server
from speculators.generator.structure_study import StructureStudy, training_job
from speculators.models.eagle3 import Eagle3DraftModel
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401

ROOT = Path("examples/eagle3-lora/deepseek_ttt7_generator_v1")


def test_config_keeps_original_4k_budget_and_cohort_schedule():
    cfg = StructureStudy.load(ROOT / "train.yaml")
    reference = StructureStudy.load(
        Path("examples/eagle3-lora/structure_ablations_v1/modules/attention.yaml")
    )
    job = training_job(cfg)
    assert job["training_examples"] == 4000
    assert job["updates"] == 1500
    assert job["subsets"] == job["validation_subsets"]
    assert cfg.optimization == reference.optimization
    assert cfg.context == reference.context
    assert cfg.architecture == reference.architecture
    assert cfg.target_modules == reference.target_modules
    assert cfg.trajectory_recipe.is_file()
    bank = campaign_banks(cfg)["math"]
    assert bank.models.drafter == deepspec.REPOSITORY
    assert bank.models.drafter_revision == deepspec.REVISION
    assert bank.hidden_states.layer_ids == [2, 10, 18, 26, 34, 36]
    assert bank.training.ttt_steps == 3
    assert cfg.hub_upload_interval == 500
    assert cfg.generator_batching
    assert cfg.generator_dtype == "bfloat16"


def test_effective_inputs_keep_original_membership_and_source_lineage(
    online_campaign,  # noqa: F811
):
    cfg, _, _ = online_campaign
    source = load_local_inputs(cfg)
    banks = campaign_banks(cfg)
    for domain, bank in banks.items():
        bank.models.target_revision = source.manifests[domain]["revisions"][
            "target_sha"
        ]
        bank.models.drafter = deepspec.REPOSITORY
        bank.models.drafter_revision = deepspec.REVISION
        bank.hidden_states.layer_ids = [2, 10, 18, 26, 34, 36]
    before = copy.deepcopy(source.manifests)
    effective = runtime_trajectory_manifests(banks, source.banks, source.manifests)
    assert source.manifests == before
    for domain, manifest in effective.items():
        assert manifest["subsets"] == before[domain]["subsets"]
        assert manifest["trajectory_source"]["revisions"] == before[domain]["revisions"]
        assert manifest["cache_spec"]["models"]["drafter"] == deepspec.REPOSITORY
        assert manifest["revisions"]["drafter_sha"] == deepspec.REVISION
        assert manifest["cache_spec"]["hidden_states"]["layer_ids"] == [
            2,
            10,
            18,
            26,
            34,
            36,
        ]
    banks["math"].models.target_revision = "f" * 40
    with pytest.raises(ValueError, match="same pinned target"):
        runtime_trajectory_manifests(banks, source.banks, source.manifests)


def test_real_loader_separates_source_tokens_and_runtime_layers(
    online_campaign,  # noqa: F811
    tmp_path,
):
    cfg, _, _ = online_campaign
    original = load_local_inputs(cfg)
    source = original.banks["math"].model_copy(deep=True)
    source.hub.mode = "consumer"
    source_path = tmp_path / "source-recipe.json"
    source_path.write_text(source.model_dump_json())
    runtime = source.model_copy(deep=True)
    runtime.models.drafter = deepspec.REPOSITORY
    runtime.models.drafter_revision = deepspec.REVISION
    runtime.hidden_states.layer_ids = [2, 10, 18, 26, 34, 36]
    runtime_path = tmp_path / "runtime-recipe.json"
    runtime_path.write_text(runtime.model_dump_json())
    cfg.training_recipe = runtime_path
    cfg.trajectory_recipe = source_path
    effective = load_local_inputs(cfg)
    assert effective.subsets == original.subsets
    assert effective.rows.to_dict() == original.rows.to_dict()
    assert (
        effective.identity["trajectory_source"]["revisions"]
        == original.identity["revisions"]
    )
    assert effective.identity["revisions"]["drafter_sha"] == deepspec.REVISION
    assert effective.banks["math"].hidden_states.layer_ids == [2, 10, 18, 26, 34, 36]


def test_training_preparation_preserves_all_qwen3_weights(tmp_path, monkeypatch):
    # Build the real native config from pinned-layout metadata, then reduce only
    # tensor dimensions for a CPU checkpoint round trip.
    original = {
        "architectures": ["Qwen3Eagle3Model"],
        "model_type": "qwen3",
        "target_model_name_or_path": "Qwen/Qwen3-8B",
        "target_layer_ids": [1, 9, 17, 25, 33],
        "num_hidden_layers": 1,
        "hidden_size": 4096,
        "num_target_layers": 36,
        "ttt_length": 7,
        "vocab_size": 32,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }
    config = deepspec.native_training_config(original, "unused")
    config.transformer_layer_config.hidden_size = 8
    config.transformer_layer_config._attn_implementation = "eager"
    source_model = Eagle3DraftModel(config)
    state = {
        key: torch.randn_like(value).contiguous()
        for key, value in source_model.state_dict().items()
        if key not in {"verifier_lm_head.weight", "verifier_norm.weight"}
    }
    serving = tmp_path / "serving"
    serving.mkdir()
    (serving / "original_config.json").write_text(json.dumps(original))
    (serving / "preparation.json").write_text(
        json.dumps({"source_revision": deepspec.REVISION})
    )
    save_file(state, serving / "model.safetensors")
    monkeypatch.setattr(deepspec, "native_training_config", lambda *_: config)
    output = deepspec.prepare_training(serving, tmp_path / "native", "unused")
    # Verifier tensors are separately loaded at runtime; source tensors are exact.
    from safetensors.torch import load_file  # noqa: PLC0415

    recovered = load_file(output / "model.safetensors")
    assert recovered.keys() == state.keys()
    for key, value in state.items():
        assert torch.equal(recovered[key], value)
    assert "layers.0.self_attn.q_norm.weight" in recovered
    assert "layers.0.self_attn.k_norm.weight" in recovered
    assert recovered["fc.weight"].shape == (8, 40)
    assert (
        json.loads((output / "config.json").read_text())["speculators_model_type"]
        == "eagle3"
    )
    assert deepspec.prepare_training(serving, tmp_path / "native", "unused") == output
    state.pop("layers.0.self_attn.q_norm.weight")
    save_file(state, serving / "model.safetensors")
    with pytest.raises(ValueError, match="Missing DeepSpec"):
        deepspec.prepare_training(serving, tmp_path / "bad", "unused")


def test_generator_serving_uses_same_metadata_translation_as_baseline(
    monkeypatch, tmp_path
):
    calls = []
    path = tmp_path / deepspec.REVISION
    monkeypatch.setattr(deepspec, "prepare", lambda root: calls.append(root) or path)
    assert server.pinned_model(deepspec.REPOSITORY, deepspec.REVISION) == str(path)
    assert len(calls) == 1
