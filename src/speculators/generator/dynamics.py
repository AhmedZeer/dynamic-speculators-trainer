"""Training-dynamics metrics for conditional LoRA generators (run plan §4).

Core (every ``interval`` updates, nearly free):

* ``dynamics/lca/<group>``: loss-change allocation, the first-order share of the
  loss change on update t attributed to a parameter group,
  0.5 * (g_t + g_{t+1}) . (theta_{t+1} - theta_t), with pre-clip minibatch
  gradients (midpoint rule; Lan et al. 2019). Negative = the group lowered the
  loss. Logged one update late (it needs g_{t+1}). ``lca_share`` normalizes by
  the summed absolute allocation.
* ``dynamics/update_ratio/<group>``: ||theta_{t+1} - theta_t|| / ||theta_t||.
* ``dynamics/grad_norm/<group>`` and ``grad_share``: pre-clip gradient norms;
  global clipping scales all groups equally, so the post-clip share is equal.

Probe (every validation): see :func:`probe_metrics`. All values are scalars
logged to W&B and appended to ``dynamics.jsonl`` in the run directory.
"""

import math
from contextlib import contextmanager

import torch

from speculators.generator.compact import component_groups


class DynamicsLogger:
    def __init__(self, generator, interval=25):
        self.groups = component_groups(generator)
        self.interval = interval
        self.pending = None  # (update, {group: delta}) awaiting g_{t+1}
        self.pending_grads = None
        self.before = None
        self.gradients = None

    def due(self, update):
        return (update + 1) % self.interval == 0

    @torch.no_grad()
    def pre_clip(self, update):
        """Call after backward and before clipping; returns metrics to log."""
        metrics = {}
        grads = {
            name: [p.grad if p.grad is not None else torch.zeros_like(p) for p in ps]
            for name, ps in self.groups.items()
        }
        if self.pending is not None:
            step, deltas = self.pending
            lca = {
                name: sum(
                    float((0.5 * (g0 + g1.float()) * d).sum())
                    for g0, g1, d in zip(
                        self.pending_grads[name], grads[name], deltas[name], strict=True
                    )
                )
                for name in self.groups
            }
            total = sum(abs(v) for v in lca.values()) or 1.0
            for name, value in lca.items():
                metrics[f"dynamics/lca/{name}"] = value
                metrics[f"dynamics/lca_share/{name}"] = abs(value) / total
            metrics["dynamics/lca/total"] = sum(lca.values())
            metrics["dynamics/lca_update"] = step + 1
            self.pending = self.pending_grads = None
        if self.due(update):
            norms = {
                name: math.sqrt(sum(float(g.float().pow(2).sum()) for g in gs))
                for name, gs in grads.items()
            }
            total = math.sqrt(sum(v * v for v in norms.values())) or 1.0
            for name, value in norms.items():
                metrics[f"dynamics/grad_norm/{name}"] = value
                metrics[f"dynamics/grad_share/{name}"] = (value / total) ** 2
            self.before = {
                name: [p.detach().float().clone() for p in ps]
                for name, ps in self.groups.items()
            }
            self.gradients = {
                name: [g.detach().float().clone() for g in gs]
                for name, gs in grads.items()
            }
        return metrics

    @torch.no_grad()
    def post_step(self, update):
        """Call after optimizer.step(); returns update-ratio metrics."""
        if self.before is None:
            return {}
        metrics, deltas = {}, {}
        for name, ps in self.groups.items():
            deltas[name] = [
                p.detach().float() - b for p, b in zip(ps, self.before[name],
                                                       strict=True)
            ]
            step = math.sqrt(sum(float(d.pow(2).sum()) for d in deltas[name]))
            weight = math.sqrt(sum(float(b.pow(2).sum()) for b in self.before[name]))
            metrics[f"dynamics/update_ratio/{name}"] = step / max(weight, 1e-12)
            metrics[f"dynamics/update_norm/{name}"] = step
        self.pending = (update, deltas)
        self.pending_grads = self.gradients
        self.before = self.gradients = None
        return metrics


@contextmanager
def forced_null(generator):
    previous = generator.force_null
    generator.force_null = True
    try:
        yield
    finally:
        generator.force_null = previous


def factor_gram(factors_list, scale=1.0):
    """Gram matrix <dW_i, dW_j>_F of dW = scale * B A without forming dW.

    <B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)). Inputs: list of (A [r,in], B [out,r]).
    """
    count = len(factors_list)
    gram = torch.zeros(count, count, dtype=torch.float64)
    for i in range(count):
        a_i, b_i = (x.double() for x in factors_list[i])
        for j in range(i, count):
            a_j, b_j = (x.double() for x in factors_list[j])
            value = torch.trace((b_i.T @ b_j) @ (a_j @ a_i.T)) * scale * scale
            gram[i, j] = gram[j, i] = value
    return gram


def gram_statistics(gram):
    """Dynamic-energy ratio rho and participation ratio of a dW Gram matrix."""
    count = gram.shape[0]
    mean_norm = gram.sum() / count**2  # ||mean dW||^2
    spread = torch.diagonal(gram).mean() - mean_norm  # E||dW - mean||^2
    centered = gram - gram.mean(0, keepdim=True) - gram.mean(1, keepdim=True) + (
        gram.mean()
    )
    eig = torch.linalg.eigvalsh(centered).clamp_min(0)
    participation = float(eig.sum() ** 2 / eig.pow(2).sum()) if eig.sum() > 0 else 0.0
    return {
        "rho": float(spread / mean_norm.clamp_min(1e-30)),
        "participation_ratio": participation,
        "mean_norm": float(mean_norm.sqrt()),
    }


def effective_rank(matrix):
    """RankMe: exp(entropy of normalized singular values) of centered rows."""
    values = torch.linalg.svdvals((matrix - matrix.mean(0)).double())
    p = values / values.sum().clamp_min(1e-30)
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def centroid_accuracy(features, labels):
    """Leave-one-out nearest-centroid domain accuracy (a training-free probe)."""
    features = torch.nn.functional.normalize(features.double(), dim=-1)
    labels = torch.as_tensor(labels)
    correct = 0
    classes = labels.unique()
    sums = {int(c): features[labels == c].sum(0) for c in classes}
    counts = {int(c): int((labels == c).sum()) for c in classes}
    for x, y in zip(features, labels, strict=True):
        best, best_score = None, -math.inf
        for c in classes:
            c = int(c)
            total, n = sums[c], counts[c]
            if c == int(y):
                total, n = total - x, n - 1
            if n == 0:
                continue
            score = float(torch.dot(x, total / n))
            if score > best_score:
                best, best_score = c, score
        correct += int(best == int(y))
    return correct / len(labels)


def within_between_ratio(features, labels):
    features, labels = features.double(), torch.as_tensor(labels)
    overall = features.mean(0)
    within = between = 0.0
    for c in labels.unique():
        group = features[labels == c]
        centre = group.mean(0)
        within += float((group - centre).pow(2).sum())
        between += float(len(group) * (centre - overall).pow(2).sum())
    return within / max(between, 1e-30)


@torch.no_grad()
def probe_metrics(runtime, generator, probe, *, scale):  # noqa: C901
    """Swap gaps, dW statistics and condition geometry on a fixed probe set.

    ``probe`` = list of (domain_label, cohort_indices) with >= 2 cohorts per
    domain. Conditioning policies per cohort: own, another cohort of the same
    domain (within), a cohort of the next domain (cross), and the learned/zero
    null condition. Loss is the drafter's training objective, token-weighted.
    """
    runtime.drafter.eval()
    generator.eval()
    by_domain = {}
    for position, (label, _) in enumerate(probe):
        by_domain.setdefault(label, []).append(position)
    domains = sorted(by_domain)
    contexts = [runtime.corpus.context(cohort, "last", runtime.device)
                for _, cohort in probe]

    def loss_for(cohort, factors):
        total = weight = 0.0
        for batch, _ in runtime.corpus.microbatches(cohort, runtime.device,
                                                    runtime.dtype):
            _, loss, _ = runtime.forward(batch, factors)
            tokens = float(batch["loss_mask"].sum())
            total += float(loss) * tokens
            weight += tokens
        return total / max(weight, 1.0)

    own = [generator(*context) for context in contexts]
    with forced_null(generator):
        null = generator(*contexts[0])
    gaps = {"within": [], "cross": [], "null": []}
    for position, (label, cohort) in enumerate(probe):
        same = by_domain[label]
        within = same[(same.index(position) + 1) % len(same)]
        other = by_domain[domains[(domains.index(label) + 1) % len(domains)]]
        cross = other[same.index(position) % len(other)]
        base = loss_for(cohort, own[position])
        gaps["within"].append(loss_for(cohort, own[within]) - base)
        gaps["cross"].append(loss_for(cohort, own[cross]) - base)
        gaps["null"].append(loss_for(cohort, null) - base)
    metrics = {f"probe/swap_gap/{k}": sum(v) / len(v) for k, v in gaps.items()}
    # dW geometry per module, over the probe cohorts.
    for name in own[0]:
        pairs = [(f[name][0].float().cpu(), f[name][1].float().cpu()) for f in own]
        stats = gram_statistics(factor_gram(pairs, scale))
        for key, value in stats.items():
            metrics[f"probe/{key}/{name}"] = value
    # Condition geometry (pre-dropout condition of each cohort).
    conditions = torch.stack(
        [
            generator.condition(
                generator.pool_examples(generator.encode_examples(*context))
            ).float().cpu()
            for context in contexts
        ]
    )
    labels = [domains.index(label) for label, _ in probe]
    metrics["probe/condition_effective_rank"] = effective_rank(conditions)
    metrics["probe/condition_effective_rank_fraction"] = (
        metrics["probe/condition_effective_rank"] / conditions.shape[1]
    )
    metrics["probe/domain_centroid_accuracy"] = centroid_accuracy(conditions, labels)
    metrics["probe/within_between_ratio"] = within_between_ratio(conditions, labels)
    return metrics


def probe_cohorts(runtime, subsets, per_domain=40, cohort_size=4, seed=9100):
    """Fixed probe: first ``per_domain`` validation rows per domain, in cohorts."""
    import random  # noqa: PLC0415

    by_domain = {}
    for subset in sorted(subsets):
        by_domain.setdefault(subset.rsplit("-", 1)[0], []).extend(
            runtime.corpus.indices(subset, validation=True)
        )
    probe = []
    for domain, indices in sorted(by_domain.items()):
        chosen = sorted(indices)[:per_domain]
        random.Random(seed).shuffle(chosen)
        probe.extend(
            (domain, chosen[i : i + cohort_size])
            for i in range(0, len(chosen) - cohort_size + 1, cohort_size)
        )
    return probe
