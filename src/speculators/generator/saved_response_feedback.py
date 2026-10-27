"""Feedback on saved responses using the original EAGLE3 forward and LK loss.

No autoregressive generation, speculative rollout, or target-extraction RPC occurs
here. Prefix observations use the current model and already-cached teacher states.
All original response positions remain supervised exactly once.
"""

import time

import torch

from speculators.generator.campaign_train import train_update
from speculators.generator.data import groups
from speculators.generator.engine import clip_gradients, set_learning_rate
from speculators.generator.feedback import AdapterExperiment, VerifierFeedback
from speculators.generator.model import backward_factors, detached_factors
from speculators.losses.utils import _LOSS_REDUCTION_EPS
from speculators.train.utils import normalize_counted_metrics


class RegionalLoss:
    """Preserve the original per-TTT full-response loss denominator."""

    def __init__(self, criterion, region_mask, full_mask):
        self.criterion, self.region, self.full = criterion, region_mask, full_mask

    def __call__(self, logits, targets):
        step = self.region.shape[1] - logits.shape[1]
        numerator = self.region[:, step:].sum(1) + _LOSS_REDUCTION_EPS
        denominator = self.full[:, step:].sum(1) + _LOSS_REDUCTION_EPS
        return self.criterion(logits, targets) * (numerator / denominator)[:, None]


def regions(batch, block_tokens, block_count, alignment):
    """Disjoint response masks; earlier parts cannot read future response tokens."""
    positions = batch["loss_mask"][0].nonzero().flatten().tolist()
    if not positions:
        raise ValueError("Saved response has no supervised tokens")
    offsets = list(
        range(0, min(len(positions), block_tokens * block_count), block_tokens)
    )
    offsets.append(min(len(positions), block_tokens * block_count))
    if offsets[-1] < len(positions):
        offsets.append(len(positions))
    for lower, upper in zip(offsets, offsets[1:], strict=False):
        if lower == upper:
            continue
        start, stop = positions[lower], positions[upper - 1] + 1
        last = upper == len(positions)
        padded = (
            batch["input_ids"].shape[1]
            if last
            else ((stop + alignment - 1) // alignment * alignment)
        )
        part = {key: value[:, :padded].clone() for key, value in batch.items()}
        if not last:
            for key, value in part.items():
                value[:, stop:] = -1 if key == "document_ids" else 0
        mask = part["loss_mask"]
        mask[:, :start] = False
        mask[:, stop:] = False
        yield part, start, stop


def add_metrics(total, values):
    for key, value in values.items():
        # Regional losses already use the full-response denominator. Sum their
        # contributions but count their common original microbatch only once.
        if "loss" in key and key.endswith("_total"):
            total[key] = value
        else:
            total[key] = total.get(key, 0) + value


def feedback_batch(runtime, study, generator, batch, context, *, training, weight=1):
    """Original packed batch with disjoint causal feedback regions."""
    signal = VerifierFeedback(AdapterExperiment(feedback=True))
    accumulated, trace, value, count = {}, [], 0.0, 0
    for part, start, stop in regions(
        batch,
        study.feedback_prefix_tokens,
        study.feedback_prefix_blocks,
        runtime.corpus.attention_block_size(),
    ):
        before = signal.tensor(runtime.device)
        factors = generator(*context, feedback=before)
        leaves = detached_factors(factors) if training else factors
        configured = {
            name: (RegionalLoss(fn, part["loss_mask"], batch["loss_mask"]), weight)
            for name, (fn, weight) in runtime.call_kwargs["loss_config"].items()
        }
        _, loss, metrics = runtime.forward(part, leaves, loss_config=configured)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite original-pipeline feedback loss")
        if training:
            (loss * weight).backward()
            backward_factors(factors, leaves)
        value += float(loss.detach())
        count += int(part["loss_mask"].sum())
        add_metrics(accumulated, metrics)
        # Existing EAGLE3 cumulative TTT agreement counts, not claims of
        # acceptance on a separately generated speculative continuation.
        accepted = sum(
            int(metrics[f"full_acc_{step}_sum"])
            for step in range(runtime.call_kwargs["ttt_steps"])
        )
        proposed = sum(
            int(metrics[f"full_acc_{step}_total"])
            for step in range(runtime.call_kwargs["ttt_steps"])
        )
        trace.append(
            {
                "start": start,
                "stop": stop,
                "feedback": before.cpu().tolist(),
                "supervised_positions": int(part["loss_mask"].sum()),
                "ttt_correct_slots": accepted,
                "ttt_total_slots": proposed,
            }
        )
        if proposed:
            signal.observe(accepted, proposed)
    return value, count, accumulated, trace


def feedback_sample(runtime, study, generator, index, *, training, divisor=1):
    context = runtime.corpus.context([index], "last", runtime.device)
    accumulated, traces, value, count = {}, [], 0.0, 0
    for batch, _ in runtime.corpus.microbatches(
        [index], runtime.device, runtime.dtype, training=training
    ):
        loss, size, metrics, trace = feedback_batch(
            runtime,
            study,
            generator,
            batch,
            context,
            training=training,
            weight=1 / divisor,
        )
        value += loss
        count += size
        traces.extend(trace)
        for key, number in metrics.items():
            accumulated[key] = accumulated.get(key, 0) + number
    return value, count, accumulated, traces


def update(runtime, study, generator, optimizer, episodes, step):  # noqa: PLR0917 -- original training-update API
    if not study.feedback:
        # The original training implementation, without changing data or loss.
        return train_update(runtime, generator, optimizer, episodes, step, "direct"), []
    runtime.drafter.train()
    generator.train()
    optimizer.zero_grad(set_to_none=True)
    set_learning_rate(optimizer, runtime.cfg, step + 1)
    started, loss, positions, traces = time.perf_counter(), 0.0, 0, []
    examples = 0
    for number, episode in enumerate(episodes):
        cohort = episode["indices"]
        runtime.corpus.prepare_episode(cohort, full=True)
        torch.manual_seed(study.seed + step * 100003 + number * 1009)
        context = runtime.corpus.context(cohort, "last", runtime.device)
        total = sum(sum(runtime.corpus.rows[i]["loss_mask"][1:]) for i in cohort)
        examples += len(cohort)
        for batch, _ in runtime.corpus.microbatches(
            cohort, runtime.device, runtime.dtype, training=True
        ):
            share = float(batch["loss_mask"].sum()) / total
            value, count, _, trace = feedback_batch(
                runtime,
                study,
                generator,
                batch,
                context,
                training=True,
                weight=share / len(episodes),
            )
            loss += value * share / len(episodes)
            positions += count
            traces.append({"source_indices": cohort, "prefix_observations": trace})
    gradients = clip_gradients(optimizer, study.max_grad_norm)
    optimizer.step()
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    return {
        "train/drafter_loss": loss,
        "train/response_tokens": positions,
        "train/examples": examples,
        "train/lr": optimizer.param_groups[0]["lr"],
        "train/update_seconds": time.perf_counter() - started,
        **gradients,
    }, traces


@torch.no_grad()
def validation(runtime, study, members, generator):
    runtime.drafter.eval()
    generator.eval()
    categories, losses = {}, {}
    for category, split in sorted(members.items()):
        accumulated = {}
        cohorts = list(groups(split["tune"], study.seed + 9000, 4, 4))
        for number, cohort in enumerate(cohorts):
            upcoming = [i for group in cohorts[number : number + 8] for i in group]
            runtime.corpus.cache.prefetch(
                runtime.corpus.references(upcoming, full=True)
            )
            context = runtime.corpus.context(cohort, "last", runtime.device)
            factors = generator(*context) if not study.feedback else None
            for batch, _ in runtime.corpus.microbatches(
                cohort, runtime.device, runtime.dtype
            ):
                if study.feedback:
                    _, _, metrics, _ = feedback_batch(
                        runtime, study, generator, batch, context, training=False
                    )
                else:
                    _, _, metrics = runtime.forward(batch, factors)
                for key, metric_value in metrics.items():
                    accumulated[key] = accumulated.get(key, 0.0) + float(metric_value)
        normalized = normalize_counted_metrics(accumulated)
        categories[category] = normalized["full_acc_0"]
        for key, value in normalized.items():
            if key.startswith("loss_"):
                losses[key] = losses.get(key, 0.0) + value / len(members)
    return {
        "score": sum(categories.values()) / len(categories),
        "categories": categories,
        "losses": losses,
        "protocol": "original saved-response teacher-forced TTT; full response LK",
    }
