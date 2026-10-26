"""Serial merged-adapter controls with causal verification-driven refresh."""

import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.feedback import AdapterExperiment, VerifierFeedback
from speculators.generator.serving.controller import DraftAdapterController, elapsed


class ExperimentalAdapterController(DraftAdapterController):
    """Committed KV retains past adapters; new queries use the current adapter.

    Target verification/rollback stays native. Never condition on rejected suffix
    hidden states, and never describe historical KV as recomputed under new weights.
    """

    def __init__(self, bundle, draft):
        super().__init__(bundle, draft)
        serving = json.loads(os.environ["SPECULATORS_SERVER_CONFIG"])
        self.policy = AdapterExperiment.model_validate(serving["adapter_experiment"])
        self.round_root = Path(serving["adapter_rounds_dir"])
        self.feedback_state = VerifierFeedback(self.policy)
        self.last_position = None
        self.proposed = None
        self.current_factors = None
        self.rounds = []
        self.donors = self.donor_metadata = None
        if self.policy.donor_bank:
            path = Path(self.policy.donor_bank)
            self.donor_metadata = json.loads(path.with_suffix(".json").read_text())
            if file_digest(path) != self.donor_metadata["sha256"]:
                raise ValueError("Conditioning donor tensor checksum differs")
            self.donors = load_file(path)
        self.totals.update(
            adapter_refreshes=0,
            feedback_rounds=0,
            feedback_accepted=0,
            feedback_proposed=0,
            scale_sum=0.0,
            scale_observations=0,
        )

    def select_summary(self, tokens, states, length):
        if self.policy.conditioning == "correct":
            return states[-1:].float(), length
        identity = digest(tokens)
        if identity not in self.donor_metadata["queries"]:
            raise ValueError(
                "Prompt is absent from the frozen conditioning-control workload"
            )
        category = self.donor_metadata["queries"][identity]
        if self.policy.conditioning == "cross_category":
            category = {"math": "chat", "chat": "math"}.get(category, "math")
        if self.policy.conditioning == "fixed":
            selected = self.donor_metadata["groups"]["math"][0]
        else:
            choices = [
                row
                for row in self.donor_metadata["groups"][category]
                if row.get("prompt_sha256") != identity
            ]
            selected = random.Random(int(identity[:16], 16)).choice(choices)
        return self.donors[selected["key"]].unsqueeze(0).to(self.device), selected[
            "length"
        ]

    def apply_current(self, summary, length, *, regenerate):
        seconds = 0.0
        if regenerate:
            kwargs = (
                {"feedback": self.feedback_state.tensor(self.device)}
                if self.policy.feedback
                else {}
            )
            self.current_factors, seconds = elapsed(
                self.device,
                lambda: self.generator(
                    summary, torch.tensor([length], device=self.device), **kwargs
                ),
            )
            self.totals["generator_invocations"] += 1
            self.totals["generator_seconds"] += seconds
        self.merger.scale = self.feedback_state.scale
        _, merge = elapsed(self.device, lambda: self.merger.apply(self.current_factors))
        self.totals["merge_seconds"] += merge
        self.record["generator_invocations"] += int(regenerate)
        self.record["generator_seconds"] += seconds
        self.record["merge_seconds"] += merge

    @torch.inference_mode()
    def prepare_proposal(self, ids, lengths, positions, states, kwargs):  # noqa: C901 -- initial prefill, rejection trimming and causal refresh
        if len(ids) != 1:
            raise ValueError("Adapter experiment runtime supports concurrency 1 only")
        request_id = ids[0]
        initial = self.active is None
        if not initial and self.active != request_id:
            raise RuntimeError("New request entered an unfinished serial experiment")
        rejected = kwargs.get("num_rejected_tokens_gpu")
        rejected = int(rejected[0].item()) if rejected is not None else 0
        end = len(positions) - rejected
        if end < 1:
            raise ValueError("No committed target hidden states available")
        position = int(positions[end - 1].item())
        if initial:
            if end != lengths[0] or not torch.equal(
                positions[:end], torch.arange(end, device=positions.device)
            ):
                raise ValueError(
                    "Adapter experiment requires a complete target prefill"
                )
            tokens = kwargs["target_token_ids"][:end].tolist()
            summary, length = self.select_summary(tokens, states[:end], lengths[0])
            self.active = request_id
            self.active_requests = {request_id}
            self.completed_records.pop(request_id, None)
            self.record = {
                "request_id": request_id,
                "prompt_sha256": digest(tokens),
                "generator_invocations": 0,
                "generator_seconds": 0.0,
                "merge_seconds": 0.0,
                "restore_seconds": 0.0,
                "deployment_context_size": 1,
            }
            self.records = {request_id: self.record}
            for key in (
                "prepared_requests",
                "prepared_batches",
                "conditioning_examples",
            ):
                self.totals[key] += 1
            self.record["prepared_sequence_number"] = self.totals["prepared_requests"]
            self.apply_current(summary, length, regenerate=True)
        elif position > self.last_position:
            accepted = position - self.last_position - 1
            self.feedback_state.observe(accepted, self.proposed)
            self.totals["feedback_rounds"] += 1
            self.totals["feedback_accepted"] += accepted
            self.totals["feedback_proposed"] += self.proposed
            previous_scale = self.merger.scale
            refresh = self.policy.refresh == "prefix"
            if refresh or previous_scale != self.feedback_state.scale:
                self.apply_current(
                    states[end - 1 : end].float(), position + 1, regenerate=refresh
                )
                self.totals["adapter_refreshes"] += 1
            if self.policy.record_rounds:
                self.rounds.append(
                    {
                        "round": self.feedback_state.rounds,
                        "prefix_tokens": position + 1,
                        "accepted": accepted,
                        "proposed": self.proposed,
                        "acceptance_ema": self.feedback_state.ema,
                        "next_scale": self.feedback_state.scale,
                    }
                )
        elif position != self.last_position:
            raise RuntimeError(
                "Target prefix moved backwards outside native rejection rollback"
            )
        self.last_position = position
        self.proposed = int(kwargs["num_speculative_tokens"])
        self.totals["scale_sum"] += self.feedback_state.scale
        self.totals["scale_observations"] += 1

    def finish(self, request_id):
        active = request_id in self.active_requests
        result = super().finish(request_id)
        if active:
            if self.policy.record_rounds:
                write_json(
                    self.round_root / (digest(request_id) + ".json"),
                    {
                        "request_id": request_id,
                        "prompt_sha256": result["prompt_sha256"],
                        "prepared_sequence_number": result["prepared_sequence_number"],
                        "policy": self.policy.model_dump(mode="json"),
                        "rounds": self.rounds,
                        "terminal_verification": "not_observed_by_next_proposal_hook",
                    },
                )
            self.feedback_state = VerifierFeedback(self.policy)
            self.last_position = self.proposed = self.current_factors = None
            self.rounds = []
        return result

    @torch.inference_mode()
    def profile(self):
        kwargs = (
            {"feedback": torch.zeros(3, device=self.device)}
            if self.metadata.get("feedback_dim")
            else {}
        )
        factors = self.generator(
            torch.zeros(1, self.metadata["input_dim"], device=self.device),
            torch.ones(1, device=self.device),
            **kwargs,
        )
        try:
            self.merger.apply(factors)
        finally:
            self.merger.restore()
