"""Differentiable, singleton EAGLE3 proposals with explicit causal KV state."""

import torch
from torch import nn
from torch.func import functional_call
from transformers import DynamicCache

from speculators.generator.model import factor_parameters


def detached_cache(cache):
    result = DynamicCache()
    for index, layer in enumerate(cache.layers):
        if layer.keys is not None:
            result.update(
                layer.keys.detach().clone(), layer.values.detach().clone(), index
            )
    return result


class DraftStep(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states, tokens, positions, cache, target_states=True):
        model = self.model
        if target_states:
            if model.input_norm is not None:
                states = model.input_norm(states)
            if model.fc_norm is not None:
                states = torch.cat(
                    [
                        norm(chunk)
                        for norm, chunk in zip(
                            model.fc_norm,
                            states.chunk(len(model.fc_norm), dim=-1),
                            strict=True,
                        )
                    ],
                    dim=-1,
                )
            states = model.fc(states)
        with torch.no_grad():
            embeddings = model.embed_tokens(tokens)
        hidden = torch.cat([embeddings, states], dim=-1)
        past, query = cache.get_seq_length(), tokens.shape[1]
        if model._attn_impl == "simple_flex_attention":  # noqa: SLF001
            from torch.nn.attention.flex_attention import create_block_mask  # noqa: PLC0415

            mask = create_block_mask(
                lambda _b, _h, q, k: k <= q + past,
                B=None,
                H=None,
                Q_LEN=query,
                KV_LEN=past + query,
                device=str(tokens.device),
            )
        else:
            allowed = torch.arange(past + query, device=tokens.device)[None, :] <= (
                torch.arange(query, device=tokens.device)[:, None] + past
            )
            mask = torch.zeros(
                query, past + query, device=tokens.device, dtype=hidden.dtype
            )
            mask.masked_fill_(~allowed, float("-inf"))
            mask = mask[None, None]
        rope = model.rotary_emb(hidden, positions)
        for layer in model.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=cache,
                cache_position=torch.arange(past, past + query, device=tokens.device),
                position_embeddings=rope,
                use_cache=True,
            )
        if model.config.norm_output:
            hidden = model.norm(hidden)
            logits = model.lm_head(hidden)
        else:
            logits = model.lm_head(model.norm(hidden))
        return logits, hidden


class StreamingDraft:
    def __init__(self, runtime):
        self.runtime = runtime
        self.step = DraftStep(runtime.drafter)

    def forward(self, states, tokens, positions, cache, factors, *, target_states=True):
        kwargs = {"target_states": target_states}
        args = (states, tokens, positions, cache)
        with torch.autocast(
            self.runtime.device.type,
            dtype=self.runtime.dtype,
            enabled=self.runtime.dtype != torch.float32,
        ):
            return functional_call(
                self.step,
                {
                    "model." + key: value
                    for key, value in factor_parameters(
                        self.runtime.paths, factors
                    ).items()
                },
                args,
                kwargs,
            )

    def propose(  # noqa: PLR0917 -- explicit causal streaming state
        self,
        states,
        shifted_tokens,
        start_position,
        cache,
        factors,
        count=3,
        forced=None,
    ):
        """Return greedy target-vocabulary IDs and differentiable local logits.

        Replay can force previously collected proposals; teacher and student then
        see the same drafted prefixes. Cache updates are cropped by the caller.
        """
        device = self.runtime.device
        tokens = torch.tensor([shifted_tokens], device=device)
        positions = torch.arange(
            start_position, start_position + len(shifted_tokens), device=device
        )[None]
        logits, hidden = self.forward(
            states[None].to(device), tokens, positions, cache, factors
        )
        proposals, outputs = [], []
        for index in range(count):
            current = logits[:, -1:].float()
            outputs.append(current)
            draft_id = int(current.argmax(-1).item())
            target_id = (
                draft_id + int(self.runtime.drafter.d2t[draft_id].item())
                if self.runtime.drafter.use_draft_vocab
                else draft_id
            )
            token = forced[index] if forced is not None else target_id
            proposals.append(token)
            if index + 1 < count:
                logits, hidden = self.forward(
                    hidden[:, -1:],
                    torch.tensor([[token]], device=device),
                    positions[:, -1:] + index + 1,
                    cache,
                    factors,
                    target_states=False,
                )
        return proposals, torch.cat(outputs, dim=1)
