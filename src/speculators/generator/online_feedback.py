"""Current-policy feedback with LK only on target-verified token prefixes.

Rollout never retains an incorrect token in a supervised student input. A gradient
pass predicts accepted tokens and, when present, the first target correction from
its correct prefix. Invalidated suffix positions are excluded from distillation.
"""

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from speculators.bank.artifacts import digest
from speculators.generator.feedback import AdapterExperiment, VerifierFeedback
from speculators.generator.streaming import StreamingDraft, detached_cache
from speculators.generator.target_verifier import FrozenTarget
from speculators.losses import resolve_loss_config


@dataclass
class VerifiedBlock:
    states: torch.Tensor
    shifted_tokens: list[int]
    position: int
    cache: DynamicCache | None
    feedback: torch.Tensor
    verified_tokens: list[int]
    teacher_logits: torch.Tensor


class OnlineFeedbackTrainer(FrozenTarget):
    def __init__(self, runtime, study, generator):
        super().__init__(runtime)
        self.study, self.generator = study, generator
        self.stream = StreamingDraft(runtime)
        self.loss = next(
            iter(resolve_loss_config(study.loss, study.loss_implementation).values())
        )[0]
        self.prompt_futures = {}

    def prompt_key(self, index):
        row = self.runtime.corpus.rows[index]
        return ("feedback-prompt", digest(row["input_ids"][: row["prompt_length"]]))

    def prefetch(self, indices):
        cache = self.runtime.corpus.cache
        for index in indices:
            with cache.ram.lock:
                present = self.prompt_key(index) in cache.ram.full
            if present or index in self.prompt_futures:
                continue
            row = self.runtime.corpus.rows[index]
            self.prompt_futures[index] = cache.pool.submit(
                self.extract, row["input_ids"][: row["prompt_length"]]
            )

    def prompt(self, index):
        cache = self.runtime.corpus.cache
        key = self.prompt_key(index)
        with cache.ram.lock:
            if key in cache.ram.full:
                cache.ram.full.move_to_end(key)
                return cache.ram.full[key]["hidden_states"]
        self.prefetch([index])
        states = self.prompt_futures.pop(index).result()
        cache.ram.put(key, {"hidden_states": states})
        return states

    def factors(self, summary, length, feedback):
        kwargs = (
            {"feedback": feedback.to(self.runtime.device)}
            if self.study.feedback
            else {}
        )
        return self.generator(
            summary[None], torch.tensor([length], device=self.runtime.device), **kwargs
        )

    @torch.no_grad()
    def collect(self, index, *, gradient_blocks=True):  # noqa: C901 -- causal greedy verification and exact budget/EOS handling
        runtime, study = self.runtime, self.study
        runtime.drafter.eval()  # Same dropout-free policy in collection and training.
        self.generator.eval()
        row = runtime.corpus.rows[index]
        prefix = list(row["input_ids"][: row["prompt_length"]])
        states = self.prompt(index)
        summary = states[-1, :-1].flatten().float().to(runtime.device)
        prompt_length = len(prefix)
        limit = min(
            study.rollout_tokens,
            runtime.bank.hidden_states.sequence_length - prompt_length,
        )
        if limit < 1:
            raise ValueError(f"Prompt {index} leaves no space for online continuation")
        feedback = VerifierFeedback(
            AdapterExperiment(refresh="prefix", feedback=study.feedback)
        )
        cache, new_start = DynamicCache(), 0
        blocks, trace, response = [], [], []
        while len(response) < limit:
            bonus = int(self.teacher(states[-1:, -1]).argmax(-1).item())
            if bonus in self.eos:
                response.append(bonus)
                break
            count = min(study.speculative_tokens, limit - len(response) - 1)
            if not count:
                response.append(bonus)
                break
            signal = feedback.tensor(runtime.device)
            factors = self.factors(summary, prompt_length, signal)
            previous_cache = detached_cache(cache) if gradient_blocks else None
            fresh = states[new_start:, :-1].flatten(1).to(runtime.device)
            shifted = prefix[new_start + 1 :] + [bonus]
            proposed, _ = self.stream.propose(
                fresh, shifted, new_start, cache, factors, count=count
            )
            verified = self.extract(prefix + [bonus] + proposed)
            teachers = self.teacher(verified[len(prefix) : len(prefix) + count, -1])
            target_ids = teachers.argmax(-1).tolist()
            accepted = 0
            for actual, expected in zip(proposed, target_ids, strict=True):
                if actual != expected:
                    break
                accepted += 1
                if actual in self.eos:
                    break
            terminal = any(token in self.eos for token in proposed[:accepted])
            # The first mismatch is a prediction on a correct prefix. Its teacher
            # correction is supervised; the wrong proposed token is never fed back.
            valid = accepted if terminal else min(accepted + 1, count)
            targets = teachers[:valid]
            if runtime.drafter.use_draft_vocab:
                targets = targets[:, runtime.drafter.t2d.bool()]
            blocks.append(
                VerifiedBlock(
                    fresh.detach(),
                    shifted,
                    new_start,
                    previous_cache,
                    signal.detach(),
                    target_ids[:valid],
                    targets.detach(),
                )
            )
            trace.append(
                {
                    "prefix_length": len(prefix),
                    "feedback": signal.cpu().tolist(),
                    "proposed": proposed,
                    "accepted": accepted,
                    "proposed_count": count,
                    "supervised_tokens": target_ids[:valid],
                    "supervised_positions": valid,
                    "prefix_sha256": digest(prefix),
                }
            )
            old_length = len(prefix)
            tentative = cache.get_seq_length() - old_length
            if tentative > 0:
                cache.crop(-tentative)
            committed = [bonus] + proposed[:accepted]
            response.extend(committed)
            prefix += committed
            states = verified[: len(prefix)]
            new_start = old_length
            feedback.observe(accepted, count)
            if terminal:
                break
        return (
            summary,
            blocks,
            {
                "source_index": index,
                "source_id": row["primary_id"],
                "response": response,
                "rounds": trace,
                "parent_prompt_only_conditioning": True,
            },
        )

    def train_episode(self, index, divisor):
        summary, blocks, trace = self.collect(index)
        positions = sum(len(block.verified_tokens) for block in blocks)
        total = 0.0
        for block in blocks:
            factors = self.factors(
                summary,
                self.runtime.corpus.rows[index]["prompt_length"],
                block.feedback,
            )
            _, logits = self.stream.propose(
                block.states,
                block.shifted_tokens,
                block.position,
                block.cache,
                factors,
                count=len(block.verified_tokens),
                forced=block.verified_tokens,
            )
            loss = self.loss(logits, block.teacher_logits[None]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite online verified-prefix loss")
            weight = len(block.verified_tokens) / max(positions, 1)
            (loss * weight / divisor).backward()
            total += float(loss.detach()) * weight
        return total, positions, trace

    @torch.no_grad()
    def evaluate_episode(self, index):
        summary, blocks, trace = self.collect(index, gradient_blocks=False)
        # Reconstruct the policy's committed KV while teacher-forcing only valid
        # blocks. This uses real feedback, not a zero-feedback prompt diagnostic.
        cache = DynamicCache()
        total, count, correct = 0.0, 0, 0
        for block in blocks:
            block.cache = cache
            factors = self.factors(
                summary,
                self.runtime.corpus.rows[index]["prompt_length"],
                block.feedback,
            )
            _, logits = self.stream.propose(
                block.states,
                block.shifted_tokens,
                block.position,
                cache,
                factors,
                count=len(block.verified_tokens),
                forced=block.verified_tokens,
            )
            size = len(block.verified_tokens)
            total += float(self.loss(logits, block.teacher_logits[None]).sum())
            correct += int((logits.argmax(-1) == block.teacher_logits.argmax(-1)).sum())
            count += size
            excess = cache.get_seq_length() - (
                block.position + len(block.shifted_tokens)
            )
            if excess > 0:
                cache.crop(-excess)
        return {"loss_sum": total, "positions": count, "correct": correct}, trace
