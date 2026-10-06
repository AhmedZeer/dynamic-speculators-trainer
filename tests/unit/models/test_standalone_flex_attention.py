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
