import math

import pytest
import torch

from speculators.generator.compact import (
    CompactLoRAGenerator,
    component_groups,
    optimizer_groups,
)
from speculators.generator.config import Architecture, Optimization
from speculators.generator.dynamics import (
    DynamicsLogger,
    effective_rank,
    factor_gram,
    gram_statistics,
)
from speculators.generator.model import LoRAGenerator, build_generator, cast_generator
from speculators.generator.online_activations import summary_vector

SHAPES = {"q_proj": (16, 12), "v_proj": (16, 4)}
RANK = 4
WIDTH = 9


def small(**kw):
    return Architecture(encoder_hidden=8, example_dim=6, condition_dim=5,
                        descriptor_dim=3, decoder_width=7, residual_blocks=1, **kw)


def context(count=3, width=WIDTH):
    return torch.randn(count, width), torch.tensor([5, 2, 9][:count])


def test_default_architecture_dump_is_unchanged():
    dumped = Architecture().model_dump(mode="json")
    assert set(dumped) == {"encoder_hidden", "example_dim", "condition_dim",
                           "descriptor_dim", "decoder_width", "residual_blocks"}
    assert "encoder_lr_scale" not in Optimization().model_dump(mode="json")
    assert Optimization(encoder_lr_scale=4).model_dump()["encoder_lr_scale"] == 4
    assert Architecture(kind="core").model_dump()["kind"] == "core"


def test_full_generator_state_unchanged_without_dropout():
    generator = build_generator(WIDTH, SHAPES, RANK, small())
    assert isinstance(generator, LoRAGenerator)
    assert not any("null" in k for k in generator.state_dict())


@pytest.mark.parametrize("kind", ["core", "gates"])
def test_compact_starts_at_static_lora(kind):
    generator = build_generator(WIDTH, SHAPES, RANK, small(kind=kind))
    assert isinstance(generator, CompactLoRAGenerator)
    factors = generator(*context())
    for name, (a, b) in factors.items():
        assert a.shape == (RANK, SHAPES[name][0])
        assert b.shape == (SHAPES[name][1], RANK)
        torch.testing.assert_close(a, generator.static_a[name])
        assert torch.count_nonzero(b) == 0


def test_dynamic_gate_starts_static_and_has_gradient():
    generator = build_generator(WIDTH, SHAPES, RANK, small(kind="core",
                                                           dynamic_gate=True))
    with torch.no_grad():
        for b in generator.static_b.values():
            b.normal_()
    factors = generator(*context())
    for name, (a, _) in factors.items():
        torch.testing.assert_close(a, generator.static_a[name])
    loss = sum((a.sum() + b.sum()) for a, b in factors.values())
    loss.backward()
    assert all(g.grad is not None and g.grad.abs() > 0
               for g in generator.gates.values())


def test_batched_cohorts_match_serial():
    generator = build_generator(WIDTH, SHAPES, RANK, small(kind="core"))
    with torch.no_grad():
        for head in generator.mod_heads.values():
            head.weight.normal_()
    contexts = [context(3), context(2)]
    batched = generator.forward_cohorts(contexts)
    for index, ctx in enumerate(contexts):
        serial = generator(*ctx)
        for name in SHAPES:
            torch.testing.assert_close(batched[name][0][index], serial[name][0])


def test_perceiver_reader_shapes_and_padding():
    tokens, hidden = 4, 5
    width = (tokens + 1) * 3 * hidden
    arch = small(kind="perceiver_core", summary_tokens=tokens, perceiver_width=8,
                 perceiver_heads=2, perceiver_queries=3, perceiver_blocks=1)
    generator = build_generator(width, SHAPES, RANK, arch)
    summaries = torch.randn(2, width)
    lengths = torch.tensor([1, 10])  # first prompt shorter than the window
    factors = generator(summaries, lengths)
    assert factors["q_proj"][0].shape == (RANK, 16)
    assert all(torch.isfinite(a).all() for a, _ in factors.values())


def test_summary_vector_window():
    states = torch.arange(6 * 4 * 2, dtype=torch.float32).view(6, 4, 2)
    last = summary_vector(states, 3)
    torch.testing.assert_close(last, states[2, :-1].flatten())
    window = summary_vector(states, 2, tokens=3).view(4, 3, 2)
    assert torch.count_nonzero(window[0]) == 0  # left padding
    torch.testing.assert_close(window[1:3], states[:2, :-1])
    torch.testing.assert_close(window[3], states[:2, :-1].mean(0))


def test_condition_dropout_and_force_null():
    generator = build_generator(WIDTH, SHAPES, RANK,
                                small(kind="core", condition_dropout=0.5))
    with torch.no_grad():
        for head in generator.mod_heads.values():
            head.weight.normal_()
        for b in generator.static_b.values():
            b.normal_()
    generator.force_null = True
    one = generator(*context(3))
    two = generator(*context(2))
    for name in SHAPES:
        torch.testing.assert_close(one[name][0], two[name][0])
    generator.force_null = False
    generator.eval()
    three = generator(*context(3))
    assert not torch.allclose(one["q_proj"][0], three["q_proj"][0])


def test_full_generator_condition_dropout():
    generator = build_generator(WIDTH, SHAPES, RANK, small(condition_dropout=0.3))
    assert "null_condition" in generator.state_dict()
    generator.force_null = True
    generator(*context())


def test_component_and_optimizer_groups_cover_parameters():
    for arch in (small(), small(kind="gates", dynamic_gate=True,
                                condition_dropout=0.1)):
        generator = build_generator(WIDTH, SHAPES, RANK, arch)
        groups = component_groups(generator)
        assert sum(len(v) for v in groups.values()) == len(
            list(generator.parameters())
        )
        single = optimizer_groups(generator, 1.0)
        assert len(single) == 1
        scaled = optimizer_groups(generator, 4.0)
        scales = {g["name"]: g["lr_scale"] for g in scaled}
        assert scales["encoder"] == 4.0
        assert scales.get("head.W", scales.get("mod_head")) == 1.0


def test_cast_keeps_static_float32():
    generator = cast_generator(build_generator(WIDTH, SHAPES, RANK,
                                               small(kind="core")), torch.bfloat16)
    assert generator.encoder[0].weight.dtype == torch.bfloat16
    assert generator.static_a["q_proj"].dtype == torch.float32
    assert generator(*context())["q_proj"][0].dtype == torch.float32


def test_gram_statistics_and_rank():
    a = torch.randn(RANK, 6)
    b = torch.randn(5, RANK)
    same = factor_gram([(a, b)] * 4)
    stats = gram_statistics(same)
    assert stats["rho"] == pytest.approx(0.0, abs=1e-9)
    direct = torch.trace((b @ a).T @ (b @ a)).double()
    assert same[0, 0] == pytest.approx(float(direct), rel=1e-6)
    varied = factor_gram([(torch.randn(RANK, 6), b) for _ in range(6)])
    assert gram_statistics(varied)["rho"] > 0
    assert 1 <= effective_rank(torch.randn(40, 5)) <= 5


def test_dynamics_lca_tracks_quadratic_loss_change():
    model = build_generator(WIDTH, SHAPES, RANK, small(kind="gates"))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    logger = DynamicsLogger(model, interval=1)
    data = context()

    def loss_fn():
        return sum((a.pow(2).sum() + b.pow(2).sum())
                   for a, b in model(*data).values())

    losses, lcas = [], []
    for update in range(3):
        optimizer.zero_grad()
        loss = loss_fn()
        losses.append(float(loss))
        loss.backward()
        metrics = logger.pre_clip(update)
        if "dynamics/lca/total" in metrics:
            lcas.append(metrics["dynamics/lca/total"])
        optimizer.step()
        ratio = logger.post_step(update)
        assert all(math.isfinite(v) for v in ratio.values())
    # LCA for update t is logged at t+1 and approximates L_{t+1} - L_t.
    for index, value in enumerate(lcas):
        assert value == pytest.approx(losses[index + 1] - losses[index], rel=1e-2)
