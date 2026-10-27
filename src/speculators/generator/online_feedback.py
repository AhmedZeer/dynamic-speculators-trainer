"""Current-policy feedback with LK only on target-verified token prefixes.

Rollout never retains an incorrect token in a supervised student input. A gradient
pass predicts accepted tokens and, when present, the first target correction from
its correct prefix. Invalidated suffix positions are excluded from distillation.
"""

import time
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import dataclass

import torch
from transformers import DynamicCache

from speculators.bank.artifacts import digest
from speculators.generator.feedback import AdapterExperiment, VerifierFeedback
from speculators.generator.model import backward_factors, detached_factors
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
    student_logits: torch.Tensor | None = None


class OnlineFeedbackTrainer(FrozenTarget):
    def __init__(self, runtime, study, generator):
        super().__init__(runtime)
        self.study, self.generator = study, generator
        self.stream = StreamingDraft(runtime)
        self.loss = next(
            iter(resolve_loss_config(study.loss, study.loss_implementation).values())
        )[0]
        self.prompt_futures = {}
        self.rollout_concurrency = 4
        self.timings = {
            "collection_seconds": 0.0,
            "backward_seconds": 0.0,
            "generator_calls": 0,
        }

    def prompt_key(self, index):
        row = self.runtime.corpus.rows[index]
        return ("feedback-prompt", digest(row["input_ids"][: row["prompt_length"]]))

    def prefetch(self, indices):
        cache = self.runtime.corpus.cache
        for index in indices:
            key = self.prompt_key(index)
            with cache.ram.lock:
                present = key in cache.ram.full
            if present or key in self.prompt_futures:
                continue
            row = self.runtime.corpus.rows[index]
            self.prompt_futures[key] = cache.pool.submit(
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
        states = self.prompt_futures.pop(key).result()
        cache.ram.put(key, {"hidden_states": states})
        return states

    def factors(self, summary, length, feedback):
        if hasattr(self, "timings"):
            self.timings["generator_calls"] += 1
        kwargs = (
            {"feedback": feedback.to(self.runtime.device)}
            if self.study.feedback
            else {}
        )
        return self.generator(
            summary[None], torch.tensor([length], device=self.runtime.device), **kwargs
        )

    def _rollout(self, index, *, gradient_blocks=True):  # noqa: C901 -- resumable causal verification state machine
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
        previous_key, previous_factors = None, None
        while len(response) < limit:
            bonus = int(self.teacher(states[-1:, -1]).argmax(-1).item())
            if bonus in self.eos:
                response.append(bonus)
                break
            count = min(study.speculative_tokens, limit - len(response) - 1)
            if not count:
                response.append(bonus)
                break
            signal = feedback.tensor("cpu")
            key = tuple(signal.tolist()) if study.feedback else ()
            if previous_factors is None or key != previous_key:
                previous_factors = self.factors(summary, prompt_length, signal)
                previous_key = key
            factors = previous_factors
            previous_cache = detached_cache(cache) if gradient_blocks else None
            fresh = states[new_start:, :-1].flatten(1).to(runtime.device)
            shifted = prefix[new_start + 1 :] + [bonus]
            proposed, student_logits = self.stream.propose(
                fresh, shifted, new_start, cache, factors, count=count
            )
            # Yield only the RPC. GPU work remains on the owning main thread.
            verified = yield prefix + [bonus] + proposed
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
                    student_logits[:, :valid].detach() if not gradient_blocks else None,
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

    def collect_many(self, indices, *, gradient_blocks=True):
        """Interleave RPC waits; retain request-local KV/feedback and input order."""
        self.prefetch(indices)
        results = [None] * len(indices)
        pending, scheduled, next_index = {}, [], 0
        limit = max(1, getattr(self, "rollout_concurrency", 1))
        pool = self.runtime.corpus.cache.pool

        def advance(slot, routine, value=None, *, initial=False):
            with torch.no_grad():
                try:
                    tokens = next(routine) if initial else routine.send(value)
                except StopIteration as complete:
                    results[slot] = complete.value
                    return
            scheduled.append((tokens, slot, routine))

        try:
            while next_index < len(indices) or pending or scheduled:
                while (
                    next_index < len(indices) and len(pending) + len(scheduled) < limit
                ):
                    slot = next_index
                    next_index += 1
                    advance(
                        slot,
                        self._rollout(indices[slot], gradient_blocks=gradient_blocks),
                        initial=True,
                    )
                # Prepare a wave before releasing its verifier RPCs, so the
                # target does not occupy the shared GPU before other proposals.
                for tokens, slot, routine in scheduled:
                    pending[pool.submit(self.extract, tokens)] = (slot, routine)
                scheduled.clear()
                if pending:
                    ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                    # Stable gradient/output order is independent of RPC completion.
                    for future in sorted(ready, key=lambda item: pending[item][0]):
                        slot, routine = pending.pop(future)
                        advance(slot, routine, future.result())
        finally:
            for future, (_, routine) in pending.items():
                future.cancel()
                routine.close()
            for _, _, routine in scheduled:
                routine.close()
        return results

    def collect(self, index, *, gradient_blocks=True):
        return self.collect_many([index], gradient_blocks=gradient_blocks)[0]

    def train_collected(self, index, collected, divisor):
        summary, blocks, trace = collected
        positions = sum(len(block.verified_tokens) for block in blocks)
        total, graphs = 0.0, {}
        for block in blocks:
            # Identical conditioning yields identical factors. Accumulate adapter
            # derivatives, then traverse the large hypernetwork once per context.
            key = tuple(block.feedback.tolist()) if self.study.feedback else ()
            if key not in graphs:
                generated = self.factors(
                    summary,
                    self.runtime.corpus.rows[index]["prompt_length"],
                    block.feedback,
                )
                graphs[key] = generated, detached_factors(generated)
            leaves = graphs[key][1]
            _, logits = self.stream.propose(
                block.states,
                block.shifted_tokens,
                block.position,
                block.cache,
                leaves,
                count=len(block.verified_tokens),
                forced=block.verified_tokens,
            )
            loss = self.loss(logits, block.teacher_logits[None]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite online verified-prefix loss")
            weight = len(block.verified_tokens) / max(positions, 1)
            (loss * weight / divisor).backward()
            total += float(loss.detach()) * weight
        for generated, leaves in graphs.values():
            backward_factors(generated, leaves)
        return total, positions, trace

    def train_episode(self, index, divisor):
        return self.train_collected(index, self.collect(index), divisor)

    def train_batch(self, indices):
        """Collect bounded groups with one unchanged policy, accumulate all grads."""
        results = []
        width = max(1, getattr(self, "rollout_concurrency", 1))
        for offset in range(0, len(indices), width):
            group = indices[offset : offset + width]
            started = time.perf_counter()
            collected = self.collect_many(group)
            self.synchronize()
            self.timings["collection_seconds"] += time.perf_counter() - started
            started = time.perf_counter()
            for index, sample in zip(group, collected, strict=True):
                results.append(self.train_collected(index, sample, len(indices)))
            self.synchronize()
            self.timings["backward_seconds"] += time.perf_counter() - started
        return results

    def synchronize(self):
        if self.runtime.device.type == "cuda":
            torch.cuda.synchronize(self.runtime.device)

    @torch.no_grad()
    def evaluation_values(self, collected):
        _, blocks, trace = collected
        total, count, correct = 0.0, 0, 0
        for block in blocks:
            # Before the first mismatch, these are exactly the logits of the
            # verified-prefix pass. No second generator/draft execution is needed.
            logits = block.student_logits
            if logits is None:
                raise ValueError("Validation collection did not retain policy logits")
            total += float(self.loss(logits, block.teacher_logits[None]).sum())
            correct += int((logits.argmax(-1) == block.teacher_logits.argmax(-1)).sum())
            count += len(block.verified_tokens)
        return {"loss_sum": total, "positions": count, "correct": correct}, trace

    def evaluate_episode(self, index):
        return self.evaluation_values(self.collect(index, gradient_blocks=False))

    def evaluate_many(self, indices):
        return [
            self.evaluation_values(sample)
            for sample in self.collect_many(indices, gradient_blocks=False)
        ]
