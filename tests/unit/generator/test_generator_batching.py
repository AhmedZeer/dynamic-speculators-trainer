"""Independent cohort outputs and gradients match the serial generator path."""

import copy

import pytest
import torch

from speculators.bank.artifacts import write_json
from speculators.generator.campaign_resume import resume_identity
from speculators.generator.model import (
    LoRAGenerator,
    backward_factors,
    detached_factors,
)
from tests.unit.generator.test_experiments import small_architecture


def generator(dtype=torch.float32):
    torch.manual_seed(8)
    model = LoRAGenerator(
        12, {"o_proj": (4, 4), "v_proj": (8, 2)}, 2, small_architecture()
    )
    with torch.no_grad():
        for head in model.heads.values():
            for projection in head.values():
                projection.weight.normal_(std=0.02)
                projection.bias.normal_(std=0.02)
    return model.to(dtype)


def contexts():
    return [(torch.randn(n, 12), torch.arange(1, n + 1)) for n in (1, 2, 5, 8)]


def test_batched_outputs_gradients_and_cohort_independence_match_serial():
    serial = generator()
    batched = copy.deepcopy(serial)
    inputs = contexts()
    expected = [serial(*context) for context in inputs]
    outputs = batched.forward_cohorts(inputs)
    leaves = detached_factors(outputs)
    for cohort, factors in enumerate(expected):
        weight = (cohort + 1) / 10
        sum(
            value.square().sum() * weight for pair in factors.values() for value in pair
        ).backward()
        sum(
            value[cohort].square().sum() * weight
            for pair in leaves.values()
            for value in pair
        ).backward()
        for name, pair in factors.items():
            for position, value in enumerate(pair):
                torch.testing.assert_close(outputs[name][position][cohort], value)
    backward_factors(outputs, leaves)
    for original, packed in zip(serial.parameters(), batched.parameters(), strict=True):
        assert original.grad is not None
        torch.testing.assert_close(packed.grad, original.grad, atol=1e-6, rtol=1e-4)
    altered = [(inputs[0][0] + 10, inputs[0][1]), *inputs[1:]]
    changed = batched.forward_cohorts(altered)
    for name, pair in outputs.items():
        for position, value in enumerate(pair):
            torch.testing.assert_close(changed[name][position][1:], value[1:])


def test_bf16_generator_and_optimizer_have_no_fp32_master_weights():
    model = generator(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    outputs = model.forward_cohorts(contexts())
    assert all(
        value.dtype == torch.bfloat16 for pair in outputs.values() for value in pair
    )
    sum(
        value.float().square().mean() for pair in outputs.values() for value in pair
    ).backward()
    assert all(p.grad.dtype == torch.bfloat16 for p in model.parameters())
    optimizer.step()
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())
    assert all(
        state["exp_avg"].dtype == torch.bfloat16 for state in optimizer.state.values()
    )
    single = model(*contexts()[0])
    assert all(
        value.dtype == torch.bfloat16 for pair in single.values() for value in pair
    )


@pytest.mark.parametrize("unsafe_change", [False, True])
def test_execution_change_retains_checkpoint_identity_and_records_settings(
    tmp_path, unsafe_change
):
    previous = {
        "configuration": {
            "output_root": "output",
            "optimization": {"pretraining_updates": 1500},
            "architecture": {"decoder_width": 512},
        },
        "job": {"updates": 1500},
    }
    current = copy.deepcopy(previous)
    current["configuration"].update(generator_dtype="bfloat16", generator_batching=True)
    if unsafe_change:
        current["configuration"]["architecture"]["decoder_width"] = 1024
    write_json(tmp_path / "resolved_experiment.json", previous)
    resolved = resume_identity(tmp_path, current)
    assert resolved == (current if unsafe_change else previous)
    receipts = list((tmp_path / "provenance").glob("resume-*/configuration.json"))
    assert len(receipts) == (0 if unsafe_change else 1)
