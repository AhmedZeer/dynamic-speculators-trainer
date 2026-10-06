"""Set-conditioned full-factor hypernetwork and differentiable LoRA injection."""

import math

import torch
from torch import nn
from torch.nn import functional

from speculators.generator.config import Architecture

SUMMARY_NDIM = 2


class ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.net(x)


class LoRAGenerator(nn.Module):
    """Permutation-invariant context encoder, one A/B pair per projection."""

    def __init__(self, input_dim, shapes, rank, architecture=None, max_examples=8):
        super().__init__()
        cfg = architecture or Architecture()
        self.shapes, self.rank, self.max_examples = shapes, rank, max_examples
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
        self.module_embedding = nn.Embedding(len(shapes), cfg.descriptor_dim)
        self.layer_embedding = nn.Embedding(1, cfg.descriptor_dim)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.condition_dim + 2 * cfg.descriptor_dim, cfg.decoder_width),
            nn.GELU(),
            *(ResidualMLP(cfg.decoder_width) for _ in range(cfg.residual_blocks)),
        )
        self.heads = nn.ModuleDict()
        for name, (in_features, out_features) in shapes.items():
            self.heads[name] = nn.ModuleDict(
                {
                    "A": nn.Linear(cfg.decoder_width, rank * in_features),
                    "B": nn.Linear(cfg.decoder_width, out_features * rank),
                }
            )
            for head in self.heads[name].values():
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            nn.init.kaiming_uniform_(
                self.heads[name]["A"].bias.view(rank, in_features), a=math.sqrt(5)
            )

    def forward(self, summaries, lengths):
        count = summaries.shape[0]
        if summaries.ndim != SUMMARY_NDIM or not 1 <= count <= self.max_examples:
            raise ValueError(f"Expected 1–{self.max_examples} conditioning examples")
        if lengths.shape != (count,) or torch.any(lengths <= 0):
            raise ValueError("Each conditioning example needs a positive prompt length")
        encoded = self.encoder(
            torch.cat([summaries, lengths.to(summaries).log1p().unsqueeze(-1)], dim=-1)
        )
        pooled = torch.cat(
            [
                encoded.mean(0),
                encoded.std(0, correction=0),
                encoded.new_tensor([math.log1p(count)]),
            ]
        )
        condition = self.condition(pooled)
        result = {}
        for index, (name, (in_features, out_features)) in enumerate(
            self.shapes.items()
        ):
            descriptor = torch.cat(
                [
                    condition,
                    self.module_embedding.weight[index],
                    self.layer_embedding.weight[0],
                ]
            )
            decoded = self.decoder(descriptor)
            result[name] = (
                self.heads[name]["A"](decoded).view(self.rank, in_features),
                self.heads[name]["B"](decoded).view(out_features, self.rank),
            )
        return result


class AdapterLinear(nn.Module):
    """Ordinary LoRA parameters can be substituted by torch.func.functional_call."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # Both generated and directly optimized factors use float32 optimizer
        # state, while autocast handles the frozen drafter's compute precision.
        self.a = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * functional.linear(
            functional.linear(self.dropout(x), self.a), self.b
        )


def install_adapters(model, rank, alpha, dropout):
    """Return stable module names; freeze everything except the two adapters."""
    model.requires_grad_(False)
    paths = {}
    for path, layer in list(model.named_modules()):
        name = path.rsplit(".", 1)[-1]
        if name not in ("o_proj", "v_proj"):
            continue
        if name in paths or not isinstance(layer, nn.Linear):
            raise ValueError("Expected exactly one linear o_proj and v_proj")
        parent, _, child = path.rpartition(".")
        setattr(
            model.get_submodule(parent),
            child,
            AdapterLinear(layer, rank, alpha, dropout),
        )
        paths[name] = path
    if set(paths) != {"o_proj", "v_proj"}:
        raise ValueError("Drafter must expose o_proj and v_proj")
    return dict(sorted(paths.items()))


def factor_parameters(paths, factors):
    return {
        f"{paths[name]}.{suffix}": tensor
        for name, pair in factors.items()
        for suffix, tensor in zip(("a", "b"), pair, strict=True)
    }


def detached_factors(factors):
    """Leaf adapters collect per-microbatch gradients without retaining graphs."""
    return {
        name: tuple(value.detach().requires_grad_(True) for value in pair)
        for name, pair in factors.items()
    }


def backward_factors(generated, leaves):
    """Apply accumulated adapter derivatives to the generator exactly once."""
    values, gradients = [], []
    for name, pair in generated.items():
        for value, leaf in zip(pair, leaves[name], strict=True):
            if leaf.grad is not None:
                values.append(value)
                gradients.append(leaf.grad)
    if not values:
        raise ValueError("Drafter produced no gradients for generated LoRA factors")
    torch.autograd.backward(values, gradients)


def factor_loss(generated, target):
    """Equal weight for each saved A/B factor; no gauge alignment or conversion."""
    return torch.stack(
        [
            functional.l1_loss(value.float(), reference.to(value).float())
            for name, pair in generated.items()
            for value, reference in zip(pair, target[name], strict=True)
        ]
    ).mean()
