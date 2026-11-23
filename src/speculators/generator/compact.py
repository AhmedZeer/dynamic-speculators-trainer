"""Compact conditional LoRA generators (next-arms A1/A2/A4/A5).

All variants keep a *static* rank-r LoRA pair (A, B) per adapted projection as
ordinary trainable parameters and generate only a small per-request
modulation of it:

* ``core``  (A1):  dW(x) = B M(x) A,          M(x) = I + g * H(d(x)) in R^{r x r}
* ``gates`` (A2):  dW(x) = B diag(s(x)) A,    s(x) = 1 + g * H(d(x)) in R^r
* ``perceiver_core`` (A5): ``core`` behind a learned cross-attention reader over
  the last K prompt positions (plus the prompt mean) at the three target layers.
* ``dynamic_gate`` (A4): g is a learned scalar per module initialized at 0, so
  training starts exactly at the static LoRA (warm-startable from a static run).
  Without it g = 1 and H is zero-initialized instead.

Generated factors are returned in the original API shape, A_eff = M(x) A
[r, in] and B_eff = B [out, r], so the drafter injection, serving merge and
checkpointing paths are unchanged. Static factors, the modulation heads and
the gates are kept in float32 regardless of the generator dtype.
"""

import math

import torch
from torch import nn
from torch.nn import functional

from speculators.generator.model import ResidualMLP, drop_condition

SUMMARY_NDIM = 2
TARGET_LAYERS = 3


class PerceiverReader(nn.Module):
    """Learned queries cross-attend the last K prompt states of 3 target layers."""

    def __init__(self, hidden, tokens, cfg):
        super().__init__()
        width = cfg.perceiver_width
        self.hidden, self.tokens = hidden, tokens
        self.layer_proj = nn.ModuleList(
            nn.Linear(hidden, width) for _ in range(TARGET_LAYERS)
        )
        self.layer_gate = nn.Parameter(torch.zeros(TARGET_LAYERS))
        # Positions: K window slots (most recent last) + 1 prompt-mean slot.
        self.position = nn.Parameter(torch.zeros(tokens + 1, width))
        self.length = nn.Linear(1, width)
        self.queries = nn.Parameter(torch.randn(cfg.perceiver_queries, width) * 0.02)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "norm_q": nn.LayerNorm(width),
                    "norm_kv": nn.LayerNorm(width),
                    "attn": nn.MultiheadAttention(
                        width, cfg.perceiver_heads, batch_first=True
                    ),
                    "mlp": ResidualMLP(width),
                }
            )
            for _ in range(cfg.perceiver_blocks)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, cfg.example_dim),
            nn.LayerNorm(cfg.example_dim),
        )

    def forward(self, summaries, lengths):
        count = summaries.shape[0]
        states = summaries.view(count, self.tokens + 1, TARGET_LAYERS, self.hidden)
        weights = torch.softmax(self.layer_gate, 0)
        tokens = sum(
            weights[i] * self.layer_proj[i](states[:, :, i]) for i in range(TARGET_LAYERS)
        )
        tokens = tokens + self.position.to(tokens)
        tokens = tokens + self.length(lengths.float().log1p().to(tokens).unsqueeze(-1)
                                      ).unsqueeze(1)
        # Window slots before the prompt start are zero-padded; mask them.
        valid = torch.arange(self.tokens, device=lengths.device).flip(0).unsqueeze(0)
        pad = valid >= lengths.unsqueeze(1)  # [count, K]
        pad = torch.cat([pad, pad.new_zeros(count, 1)], dim=1)  # mean slot valid
        queries = self.queries.to(tokens).unsqueeze(0).expand(count, -1, -1)
        for block in self.blocks:
            keys = block["norm_kv"](tokens)
            attended, _ = block["attn"](
                block["norm_q"](queries), keys, keys, key_padding_mask=pad,
                need_weights=False,
            )
            queries = block["mlp"](queries + attended)
        return self.out(queries.mean(1))


class CompactLoRAGenerator(nn.Module):
    def __init__(self, input_dim, shapes, rank, architecture, max_examples=8, *,
                 feedback_dim=0):
        super().__init__()
        if feedback_dim:
            raise ValueError("Compact generators do not support feedback inputs")
        cfg = architecture
        if cfg.kind not in ("core", "gates", "perceiver_core"):
            raise ValueError(f"Unsupported compact generator kind: {cfg.kind}")
        self.kind, self.feedback_dim = cfg.kind, 0
        self.shapes, self.rank, self.max_examples = shapes, rank, max_examples
        self.condition_dropout = cfg.condition_dropout
        self.force_null = cfg.constant_condition
        if cfg.kind == "perceiver_core":
            tokens = cfg.summary_tokens
            if input_dim % ((tokens + 1) * TARGET_LAYERS):
                raise ValueError("Summary width does not match summary_tokens")
            self.encoder = PerceiverReader(
                input_dim // ((tokens + 1) * TARGET_LAYERS), tokens, cfg
            )
        else:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim + 1, cfg.encoder_hidden),
                nn.GELU(),
                nn.Linear(cfg.encoder_hidden, cfg.example_dim),
                nn.LayerNorm(cfg.example_dim),
            )
        self.condition = nn.Sequential(
            nn.Linear(2 * cfg.example_dim + 1, cfg.example_dim),
            nn.GELU(),
            nn.Linear(cfg.example_dim, cfg.condition_dim),
        )
        if cfg.condition_dropout or cfg.constant_condition:
            self.null_condition = nn.Parameter(torch.zeros(cfg.condition_dim))
        self.module_embedding = nn.Embedding(len(shapes), cfg.descriptor_dim)
        self.layer_embedding = nn.Embedding(1, cfg.descriptor_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.condition_dim + 2 * cfg.descriptor_dim, cfg.decoder_width),
            nn.GELU(),
            *(ResidualMLP(cfg.decoder_width) for _ in range(cfg.residual_blocks)),
        )
        width = rank if cfg.kind == "gates" else rank * rank
        self.mod_heads = nn.ModuleDict()
        self.static_a = nn.ParameterDict()
        self.static_b = nn.ParameterDict()
        self.dynamic_gate = cfg.dynamic_gate
        if cfg.dynamic_gate:
            self.gates = nn.ParameterDict()
        for name, (in_features, out_features) in shapes.items():
            head = nn.Linear(cfg.decoder_width, width)
            nn.init.zeros_(head.bias)
            if cfg.dynamic_gate:
                # g = 0 keeps the static LoRA; a small random head gives g a
                # nonzero gradient from the first update.
                nn.init.normal_(head.weight, std=1.0 / math.sqrt(cfg.decoder_width))
                self.gates[name] = nn.Parameter(torch.zeros(()))
            else:
                nn.init.zeros_(head.weight)
            self.mod_heads[name] = head
            a = torch.empty(rank, in_features)
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.static_a[name] = nn.Parameter(a)
            self.static_b[name] = nn.Parameter(torch.zeros(out_features, rank))

    # Keep static factors/heads/gates in float32 after a dtype cast.
    def fp32_static_(self):
        for module in (self.static_a, self.static_b, self.mod_heads):
            module.float()
        if self.dynamic_gate:
            self.gates.float()
        return self

    def encode_examples(self, summaries, lengths):
        count = summaries.shape[0]
        if summaries.ndim != SUMMARY_NDIM or not 1 <= count <= self.max_examples:
            raise ValueError(f"Expected 1–{self.max_examples} conditioning examples")
        if lengths.shape != (count,) or torch.any(lengths <= 0):
            raise ValueError("Each conditioning example needs a positive prompt length")
        if self.kind == "perceiver_core":
            dtype = self.encoder.queries.dtype
            return self.encoder(summaries.to(dtype), lengths)
        dtype = self.encoder[0].weight.dtype
        return self.encoder(
            torch.cat(
                [summaries.to(dtype), lengths.float().log1p().to(dtype).unsqueeze(-1)],
                dim=-1,
            )
        )

    @staticmethod
    def pool_examples(encoded):
        return torch.cat(
            [
                encoded.mean(0),
                encoded.std(0, correction=0),
                encoded.new_tensor([math.log1p(len(encoded))]),
            ]
        )

    def forward(self, summaries, lengths, feedback=None):
        if feedback is not None:
            raise ValueError("This generator was trained without feedback inputs")
        condition = self.condition(self.pool_examples(self.encode_examples(summaries,
                                                                           lengths)))
        return self.decode_factors(drop_condition(self, condition), None)

    def forward_cohorts(self, contexts, feedback=None):
        if feedback is not None:
            raise ValueError("This generator was trained without feedback inputs")
        if not contexts:
            raise ValueError("At least one conditioning cohort is required")
        pooled = torch.stack(
            [self.pool_examples(self.encode_examples(s, n)) for s, n in contexts]
        )
        return self.decode_factors(drop_condition(self, self.condition(pooled)), None)

    def modulation(self, condition):
        """Per-module modulation (M or s), float32, leading shape of condition."""
        leading = condition.shape[:-1]
        result = {}
        for index, name in enumerate(self.shapes):
            descriptor = torch.cat(
                [
                    condition,
                    self.module_embedding.weight[index].expand(*leading, -1),
                    self.layer_embedding.weight[0].expand(*leading, -1),
                ],
                dim=-1,
            )
            decoded = self.decoder(descriptor)
            head = self.mod_heads[name]
            delta = functional.linear(decoded.float(), head.weight.float(),
                                      head.bias.float())
            if self.dynamic_gate:
                delta = self.gates[name].float() * delta
            if self.kind == "gates":
                result[name] = 1.0 + delta
            else:
                eye = torch.eye(self.rank, device=delta.device, dtype=delta.dtype)
                result[name] = eye + delta.view(*leading, self.rank, self.rank)
        return result

    def decode_factors(self, condition, feedback=None):
        if feedback is not None:
            raise ValueError("This generator was trained without feedback inputs")
        leading = condition.shape[:-1]
        result = {}
        for name, mod in self.modulation(condition).items():
            a = self.static_a[name].float()
            b = self.static_b[name].float()
            if self.kind == "gates":
                a_eff = mod.unsqueeze(-1) * a  # [*, r, 1] * [r, in]
            else:
                a_eff = torch.matmul(mod, a)  # [*, r, r] @ [r, in]
            result[name] = (a_eff, b.expand(*leading, *b.shape))
        return result

    def load_static(self, factors):
        """Warm start: copy a static LoRA {name: (A [r,in], B [out,r])}."""
        with torch.no_grad():
            for name, (a, b) in factors.items():
                self.static_a[name].copy_(a)
                self.static_b[name].copy_(b)


def component_groups(generator):
    """Named parameter groups for per-group LR and training-dynamics metrics."""
    groups = {}

    def add(key, params):
        params = [p for p in params if p.requires_grad]
        if params:
            groups.setdefault(key, []).extend(params)

    add("encoder", generator.encoder.parameters())
    add("condition", generator.condition.parameters())
    if getattr(generator, "null_condition", None) is not None:
        add("condition", [generator.null_condition])
    add("decoder", [*generator.decoder.parameters(),
                    *generator.module_embedding.parameters(),
                    *generator.layer_embedding.parameters()])
    if getattr(generator, "feedback_dim", 0):
        add("condition", generator.feedback_encoder.parameters())
    if isinstance(generator, CompactLoRAGenerator):
        add("mod_head", generator.mod_heads.parameters())
        add("static_ab", [*generator.static_a.values(), *generator.static_b.values()])
        if generator.dynamic_gate:
            add("gate", generator.gates.values())
    else:
        add("head.W", [h.weight for m in generator.heads.values() for h in m.values()])
        add("head.bias", [h.bias for m in generator.heads.values() for h in m.values()])
    seen = sum(len(v) for v in groups.values())
    total = sum(1 for p in generator.parameters() if p.requires_grad)
    if seen != total:
        raise ValueError("Component groups do not cover every generator parameter")
    return groups


READING_GROUPS = ("encoder", "condition", "decoder")


def optimizer_groups(generator, encoder_lr_scale=1.0):
    """AdamW param groups; reading side scaled, heads/static at the base LR."""
    if encoder_lr_scale == 1.0:
        # One group keeps optimizer state compatible with pre-existing runs.
        return [{"params": list(generator.parameters()), "name": "all",
                 "lr_scale": 1.0}]
    groups = component_groups(generator)
    return [
        {"params": params, "name": name,
         "lr_scale": encoder_lr_scale if name in READING_GROUPS else 1.0}
        for name, params in groups.items()
    ]
