"""Numerical and work-reduction checks for current-policy training execution."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from torch.func import functional_call
from transformers import DynamicCache

from speculators.generator.model import factor_parameters
from speculators.generator.streaming import StreamingDraft
from speculators.generator.target_verifier import FrozenTarget
from tests.unit.generator.test_studies import streaming_runtime


@pytest.mark.parametrize("backend", ["eager", "simple_flex_attention"])
@pytest.mark.parametrize("norm_output", [False, True])
def test_last_only_prefill_matches_full_model_logits_and_adapter_gradients(
    backend, norm_output
):
    runtime, generator = streaming_runtime(backend=backend)
    runtime.drafter.config.norm_output = norm_output
    # Eager full-model reference supports CPU backward. The optimized streaming
    # path is checked against it with both eager and SDPA attention.
    reference, reference_generator = streaming_runtime(backend="eager")
    reference.drafter.config.norm_output = norm_output
    reference.drafter.load_state_dict(runtime.drafter.state_dict())
    reference_generator.load_state_dict(generator.state_dict())
    summary, length = torch.randn(1, 24), torch.tensor([7])
    states, ids = torch.randn(7, 24), torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    full_logits, projected_lengths = [], []
    hook = reference.drafter.lm_head.register_forward_hook(
        lambda m, args, out: full_logits.append(out)
    )
    functional_call(
        reference.drafter,
        factor_parameters(reference.paths, reference_generator(summary, length)),
        (),
        {
            "hidden_states": states[None],
            "input_ids": ids,
            "document_ids": torch.zeros_like(ids),
            "position_ids": torch.arange(7)[None],
            "ttt_steps": 1,
        },
    )
    hook.remove()
    expected = full_logits[-1][:, -1:].clone()
    assert full_logits[-1].shape[1] == 7
    expected.square().mean().backward()
    gradients = {
        name: p.grad.clone()
        for name, p in reference_generator.named_parameters()
        if p.grad is not None
    }
    hook = runtime.drafter.lm_head.register_forward_pre_hook(
        lambda m, args: projected_lengths.append(args[0].shape[1])
    )
    _, actual = StreamingDraft(runtime).propose(
        states,
        ids[0].tolist(),
        0,
        DynamicCache(),
        generator(summary, length),
        count=1,
    )
    hook.remove()
    assert projected_lengths == [1]
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    for name, parameter in generator.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name])


def test_forced_proposals_do_not_compute_greedy_ids(monkeypatch):
    runtime, generator = streaming_runtime(backend="simple_flex_attention")
    states = torch.randn(4, 24)
    factors = generator(torch.randn(1, 24), torch.tensor([4]))
    stream = StreamingDraft(runtime)
    forced = [4, 5, 6]
    _, expected = stream.propose(
        states, [1, 2, 3, 4], 0, DynamicCache(), factors, forced=forced
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Teacher-forced proposals must not call argmax")

    monkeypatch.setattr(torch.Tensor, "argmax", forbidden)
    actual_ids, actual = stream.propose(
        states, [1, 2, 3, 4], 0, DynamicCache(), factors, forced=forced
    )
    assert actual_ids == forced
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    assert generator.heads["v_proj"]["B"].weight.grad.abs().sum() > 0


def test_verifier_checks_suffix_finiteness_and_complete_token_alignment(tmp_path):
    runtime, _ = streaming_runtime()
    cache = SimpleNamespace(
        root=tmp_path,
        settings=SimpleNamespace(request_timeout=2, max_retries=0),
        client=None,
        bank=SimpleNamespace(
            models=SimpleNamespace(target="target"),
            hidden_states=SimpleNamespace(layer_ids=[2, 18, 33, -1]),
        ),
        cfg=SimpleNamespace(online=SimpleNamespace(target_model_path="unused")),
    )
    runtime.corpus = SimpleNamespace(cache=cache)
    tokens = list(range(7))
    states = torch.zeros(7, 4, 8)
    states[0] = float("nan")  # Previously consumed prefix, unused in this round.
    payload = {"token_ids": torch.tensor(tokens), "hidden_states": states}
    path = tmp_path / "states.safetensors"

    def extract(*args, **kwargs):
        save_file(payload, path)
        return str(path)

    target = FrozenTarget(
        runtime,
        loader=lambda *args: {"lm_head.weight": torch.zeros(16, 8)},
        extractor=extract,
    )
    with pytest.raises(ValueError, match="non-finite"):
        target.extract(tokens)
    assert not path.exists()
    actual = target.extract(tokens, validate_from=5)
    assert torch.isfinite(actual[5:]).all()
    assert target.teacher_validated_tokens == 2
    assert not path.exists()
    payload["token_ids"][0] = 15
    with pytest.raises(ValueError, match="token IDs"):
        target.extract(tokens, validate_from=5)
    payload["token_ids"][0] = tokens[0]
    payload["hidden_states"][6] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        target.extract(tokens, validate_from=5)
    assert not path.exists()
