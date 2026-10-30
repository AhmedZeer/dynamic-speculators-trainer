"""Packed attention/MLP adapters agree with explicit deltas and restore exactly."""

import copy
import json

import pytest
import torch
from torch import nn

from speculators.generator.serving.controller import (
    AdapterMerger,
    DraftAdapterController,
    validate_runtime,
)
from tests.unit.generator.test_serving import draft, valid_config, write_bundle

SHAPES = {
    "q_proj": (8, 4),
    "k_proj": (8, 2),
    "v_proj": (8, 2),
    "o_proj": (4, 4),
    "gate_proj": (4, 6),
    "up_proj": (4, 6),
    "down_proj": (6, 4),
}
SETS = {
    "attention": ("q_proj", "k_proj", "v_proj", "o_proj"),
    "all": tuple(SHAPES),
    "mlp": ("up_proj", "gate_proj", "down_proj"),
    "q_only": ("q_proj",),
    "gate_only": ("gate_proj",),
}


def model(dtype=torch.float32):
    result = draft(dtype)
    layer = result.model.layers[0]
    layer.self_attn.q_size = 4
    layer.self_attn.kv_size = 2
    layer.mlp = nn.Module()
    layer.mlp.gate_up_proj = nn.Linear(4, 12, bias=False, dtype=dtype)
    layer.mlp.down_proj = nn.Linear(6, 4, bias=False, dtype=dtype)
    return result


def unpack(result):
    layer = result.model.layers[0]
    qkv = layer.self_attn.qkv_proj.weight
    packed = layer.mlp.gate_up_proj.weight
    return {
        "q_proj": qkv[:4],
        "k_proj": qkv[4:6],
        "v_proj": qkv[6:],
        "o_proj": layer.self_attn.o_proj.weight,
        "gate_proj": packed[:6],
        "up_proj": packed[6:],
        "down_proj": layer.mlp.down_proj.weight,
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("targets", SETS.values(), ids=SETS)
def test_merged_weights_match_explicit_lora_and_preserve_unselected_blocks(
    dtype, targets
):
    result = model(dtype)
    original = {name: value.detach().clone() for name, value in unpack(result).items()}
    shapes = {name: SHAPES[name] for name in targets}
    merger = AdapterMerger(result, {"shapes": shapes, "rank": 2, "alpha": 4})
    factors = {
        name: (
            torch.randn(2, SHAPES[name][0]) * 0.02,
            torch.randn(SHAPES[name][1], 2) * 0.02,
        )
        for name in targets
    }
    merger.apply(factors)
    for name, weight in unpack(result).items():
        if name in factors:
            a, b = factors[name]
            torch.testing.assert_close(
                weight, (original[name].float() + 2 * b @ a).to(dtype), rtol=0, atol=0
            )
        else:
            assert torch.equal(weight, original[name])
    first = copy.deepcopy(result.state_dict())
    merger.apply(factors)
    assert all(
        torch.equal(first[key], value) for key, value in result.state_dict().items()
    )
    merger.restore()
    assert all(
        torch.equal(value, original[name]) for name, value in unpack(result).items()
    )


def test_bad_late_factor_restores_already_modified_projections():
    result = model()
    before = copy.deepcopy(result.state_dict())
    merger = AdapterMerger(result, {"shapes": SHAPES, "rank": 2, "alpha": 4})
    factors = {
        name: (torch.randn(2, inputs), torch.randn(outputs, 2))
        for name, (inputs, outputs) in SHAPES.items()
    }
    factors["down_proj"] = (torch.ones(1, 1), torch.ones(1, 1))
    with pytest.raises(ValueError, match="down_proj"):
        merger.apply(factors)
    assert all(
        torch.equal(before[key], value) for key, value in result.state_dict().items()
    )


@pytest.mark.parametrize("targets", [SETS["attention"], SETS["all"]])
@pytest.mark.parametrize("kind", ["generator", "static_lora"])
def test_projection_controllers_serve_generators_and_permanently_merged_static_loras(
    tmp_path, monkeypatch, targets, kind
):
    shapes = {name: SHAPES[name] for name in targets}
    metadata, _ = write_bundle(tmp_path / "bundle", kind, shapes=shapes)
    validate_runtime(valid_config(metadata), metadata)
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG",
        json.dumps({"static_adapter_policy": "merged_once"}),
    )
    result = model()
    before = copy.deepcopy(result.state_dict())
    controller = DraftAdapterController(tmp_path / "bundle", result)
    static = copy.deepcopy(result.state_dict())
    controller.prepare("request", 3, torch.arange(3), torch.randn(3, 12))
    record = controller.finish("request")
    assert record["generator_invocations"] == int(kind == "generator")
    expected = before if kind == "generator" else static
    assert all(
        torch.equal(expected[key], value) for key, value in result.state_dict().items()
    )
    if kind == "static_lora":
        assert any(not torch.equal(before[key], value) for key, value in static.items())
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG", json.dumps({"adapter_policy": "per_request"})
    )
    with pytest.raises(ValueError, match="Per-request adapters"):
        validate_runtime(valid_config(metadata), metadata)
