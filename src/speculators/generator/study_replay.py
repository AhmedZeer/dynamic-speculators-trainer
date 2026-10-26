"""Bounded target-assisted online rollouts and immediate differentiable replay."""

import time
from pathlib import Path

import torch
from safetensors.torch import load
from transformers import DynamicCache

from speculators.bank.artifacts import digest
from speculators.bank.progress import report
from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    generate_hidden_states,
    wait_for_lock,
)
from speculators.generator.feedback import AdapterExperiment, VerifierFeedback
from speculators.generator.streaming import StreamingDraft, detached_cache
from speculators.losses import resolve_loss_config
from speculators.utils.loading import load_model_layers


class ReplayTrainer:
    def __init__(self, runtime, study, model):
        self.runtime, self.study, self.generator = runtime, study, model
        self.stream = StreamingDraft(runtime)
        report(
            "Replay streaming attention: SDPA with causal prefill "
            "and cached-prefix masks; "
            "ordinary validation/training retain their configured attention backend"
        )
        # Full target head is only used on the few verification positions, to
        # preserve true target greedy verification outside the draft vocabulary.
        loaded = load_model_layers(
            ["lm_head.weight"], runtime.corpus.cache.cfg.online.target_model_path
        )
        if "lm_head.weight" not in loaded:
            raise ValueError("Replay requires the full pinned target LM head")
        self.head = loaded["lm_head.weight"].to(runtime.device, dtype=runtime.dtype)
        self.loss = next(
            iter(resolve_loss_config(study.loss, study.loss_implementation).values())
        )[0]
        self.eos = runtime.drafter.config.transformer_layer_config.eos_token_id
        if not isinstance(self.eos, (list, tuple)):
            self.eos = [self.eos]
        self.trace_records = []
        self.teacher_requests = self.teacher_seconds = self.teacher_tokens = 0

    def extract(self, tokens):
        started = time.perf_counter()
        cache = self.runtime.corpus.cache
        path = Path(
            generate_hidden_states(
                cache.client,
                cache.bank.models.target,
                {"input_ids": tokens},
                timeout=cache.settings.request_timeout,
                max_retries=cache.settings.max_retries,
            )
        ).resolve()
        if not path.is_relative_to(cache.root):
            raise ValueError("Replay extractor returned a file outside shared_dir")
        try:
            if Path(str(path) + ".lock").exists():
                wait_for_lock(
                    str(path) + ".lock", timeout=cache.settings.request_timeout
                )
            payload = load(path.read_bytes())
            check_hidden_states(payload, tokens)
            states = payload["hidden_states"]
            expected = (
                len(tokens),
                len(cache.bank.hidden_states.layer_ids),
                self.head.shape[1],
            )
            if tuple(states.shape) != expected:
                raise ValueError(
                    "Replay hidden-state dimensions differ from the pinned target"
                )
            if states.dtype != self.runtime.dtype:
                raise ValueError(
                    "Replay hidden-state dtype differs from the training recipe"
                )
            self.teacher_requests += 1
            self.teacher_seconds += time.perf_counter() - started
            self.teacher_tokens += len(tokens)
            return states
        finally:
            path.unlink(missing_ok=True)

    @torch.no_grad()
    def teacher(self, states):
        with torch.autocast(
            self.runtime.device.type,
            dtype=self.runtime.dtype,
            enabled=self.runtime.dtype != torch.float32,
        ):
            hidden = self.runtime.drafter.verifier_norm(states.to(self.runtime.device))
            return torch.nn.functional.linear(hidden, self.head).float()

    def episode(self, index, update, episode, divisor):  # noqa: C901 -- causal rollout, verification, backward replay and cache rollback
        runtime, study = self.runtime, self.study
        row = runtime.corpus.rows[index]
        prefix = list(row["input_ids"][: row["prompt_length"]])
        if (
            len(prefix) + study.speculative_tokens + 1
            > runtime.bank.hidden_states.sequence_length
        ):
            raise ValueError("Training prompt leaves no space for a replay block")
        states = self.extract(prefix)
        prompt_summary = states[-1, :-1].flatten().float().to(runtime.device)
        feedback = VerifierFeedback(
            AdapterExperiment(refresh="prefix", feedback=study.feedback)
        )
        cache = DynamicCache()
        old_length, new_start = 0, 0
        trace, total_loss, token_count, accepted_total, rounds = [], 0.0, 0, 0, 0
        for turn in range(study.replay_rounds_per_episode):
            if (
                len(prefix) + study.speculative_tokens + 1
                > runtime.bank.hidden_states.sequence_length
            ):
                break
            bonus = int(self.teacher(states[-1:, -1]).argmax(-1).item())
            if bonus in self.eos:
                break
            summary = (
                prompt_summary
                if study.conditioning == "prompt"
                else states[-1, :-1].flatten().float().to(runtime.device)
            )
            length = (
                row["prompt_length"] if study.conditioning == "prompt" else len(prefix)
            )
            context = (summary[None], torch.tensor([length], device=runtime.device))
            kwargs = (
                {"feedback": feedback.tensor(runtime.device)} if study.feedback else {}
            )
            current = self.generator(*context, **kwargs)
            # Re-generate after each verification for prefix arms. The prompt arm
            # uses the same deterministic factors; no optimizer update occurs
            # inside a request. Historical committed KV is always detached.
            detached = {
                k: tuple(t.detach() for t in pair) for k, pair in current.items()
            }
            initial_cache = detached_cache(cache)
            fresh_states = states[new_start:, :-1].flatten(1).to(runtime.device)
            shifted = prefix[new_start + 1 :] + [bonus]
            runtime.drafter.eval()
            with torch.no_grad():
                proposed, _ = self.stream.propose(
                    fresh_states,
                    shifted,
                    new_start,
                    cache,
                    detached,
                    count=study.speculative_tokens,
                )
            verified = self.extract(prefix + [bonus] + proposed)
            teachers = self.teacher(
                verified[len(prefix) : len(prefix) + study.speculative_tokens + 1, -1]
            )
            target_ids = teachers.argmax(-1).tolist()
            accepted = 0
            for actual, expected in zip(proposed, target_ids, strict=False):
                if actual != expected:
                    break
                accepted += 1
            runtime.drafter.train()
            _, logits = self.stream.propose(
                fresh_states,
                shifted,
                new_start,
                initial_cache,
                current,
                count=study.speculative_tokens,
                forced=proposed,
            )
            target_logits = teachers[: study.speculative_tokens]
            if runtime.drafter.use_draft_vocab:
                target_logits = target_logits[:, runtime.drafter.t2d.bool()]
            loss = self.loss(logits, target_logits[None]).mean()
            if not torch.isfinite(loss):
                raise ValueError("Non-finite replay loss")
            (loss / divisor / study.replay_rounds_per_episode).backward()
            total_loss += float(loss.detach())
            token_count += study.speculative_tokens
            accepted_total += accepted
            rounds += 1
            trace.append(
                {
                    "round": turn,
                    "prefix_tokens": len(prefix),
                    "feedback_input": feedback.tensor("cpu").tolist(),
                    "proposed": proposed,
                    "teacher_greedy": target_ids,
                    "accepted": accepted,
                    "prefix_sha256": digest(prefix),
                    "loss": float(loss.detach()),
                }
            )
            # Drop all tentative draft KV. Newly committed tokens are recomputed
            # from target states in the next round, with the next round's adapter.
            old_length = len(prefix)
            tentative = cache.get_seq_length() - old_length
            if tentative > 0:
                cache.crop(-tentative)
            prefix = prefix + [bonus] + proposed[:accepted]
            states = verified[: len(prefix)]
            new_start = old_length
            feedback.observe(accepted, study.speculative_tokens)
            if any(token in self.eos for token in [bonus, *proposed[:accepted]]):
                break
        self.trace_records.append(
            {
                "update": update,
                "episode": episode,
                "source_index": index,
                "source_id": row["primary_id"],
                "policy": study.conditioning,
                "cache_policy": "causal_stateful",
                "rounds": trace,
            },
        )
        return {
            "loss": total_loss / max(rounds, 1),
            "tokens": token_count,
            "accepted": accepted_total,
            "proposed": token_count,
            "rounds": rounds,
        }
