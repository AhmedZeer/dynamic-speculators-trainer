"""Kernel dispatch for model graphs and independent CUDA attention calls."""

from types import SimpleNamespace

import torch

from speculators.models import attention


def test_uncompiled_cuda_uses_standalone_fused_kernel(monkeypatch):
    class CUDAQuery:
        shape = (1, 2, 128, 4)
        is_cuda = True

        def contiguous(self):
            return self

    calls = []

    def fused(*args, **kwargs):
        calls.append("fused")
        return torch.zeros(1, 2, 128, 4)

    def native(*args, **kwargs):
        calls.append("native")
        return torch.zeros(1, 2, 128, 4)

    compiling = False
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(attention, "_standalone_flex_attention", lambda: fused)
    monkeypatch.setattr(attention, "flex_attention", native)
    query = CUDAQuery()
    key = torch.zeros(1, 1, 128, 4)
    output, _ = attention.flex_attention_forward(
        SimpleNamespace(), query, key, key, None
    )
    assert output.shape == (1, 128, 2, 4)
    assert calls == ["fused"]
    compiling = True
    attention.flex_attention_forward(SimpleNamespace(), query, key, key, None)
    assert calls == ["fused", "native"]


def test_cpu_attention_does_not_request_cuda_compilation(monkeypatch):
    def forbidden():
        raise AssertionError("CPU attention requested CUDA compilation")

    monkeypatch.setattr(attention, "_standalone_flex_attention", forbidden)
    monkeypatch.setattr(
        attention, "flex_attention", lambda query, *args, **kwargs: query
    )
    tensor = torch.zeros(1, 2, 8, 4)
    output, _ = attention.flex_attention_forward(None, tensor, tensor, tensor, None)
    assert output.shape == (1, 8, 2, 4)


def test_streaming_sdpa_matches_causal_gqa_outputs_and_gradients(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Streaming SDPA called FlexAttention")

    monkeypatch.setattr(attention, "_standalone_flex_attention", forbidden)
    monkeypatch.setattr(attention, "flex_attention", forbidden)
    query = torch.randn(1, 4, 3, 8, requires_grad=True)
    key = torch.randn(1, 1, 8, 8, requires_grad=True)
    value = torch.randn(1, 1, 8, 8, requires_grad=True)
    allowed = torch.arange(8)[None, :] <= torch.arange(3)[:, None] + 5
    mask = torch.zeros(1, 1, 3, 8).masked_fill(~allowed, float("-inf"))
    output, _ = attention.flex_attention_forward(
        None,
        query,
        key,
        value,
        mask,
        scaling=0.25,
        speculators_streaming_sdpa=True,
    )
    scores = query @ key.repeat_interleave(4, 1).transpose(-1, -2) * 0.25 + mask
    expected = (
        (scores.softmax(-1) @ value.repeat_interleave(4, 1))
        .transpose(1, 2)
        .contiguous()
    )
    torch.testing.assert_close(output, expected)
    actual_grads = torch.autograd.grad(
        output.square().sum(), (query, key, value), retain_graph=True
    )
    expected_grads = torch.autograd.grad(expected.square().sum(), (query, key, value))
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference)
