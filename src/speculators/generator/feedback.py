"""Causal verifier feedback and bounded, explicitly configured adapter strength."""

import math
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class AdapterExperiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    conditioning: Literal["correct", "same_category", "cross_category", "fixed"] = (
        "correct"
    )
    donor_bank: str | None = None
    refresh: Literal["prompt", "prefix"] = "prompt"
    feedback: bool = False
    cache_policy: Literal["causal_stateful"] = "causal_stateful"
    scale: float = Field(default=2.0, ge=0, le=8)
    scale_policy: Literal["fixed", "acceptance_ema"] = "fixed"
    ema_decay: float = Field(default=0.9, ge=0, lt=1)
    scale_min: float = Field(default=0.0, ge=0, le=8)
    scale_max: float = Field(default=4.0, ge=0, le=8)
    scale_step: float = Field(default=0.5, gt=0, le=4)
    low_acceptance: float = Field(default=0.35, ge=0, le=1)
    high_acceptance: float = Field(default=0.75, ge=0, le=1)
    low_action: Literal["decrease", "increase"] = "decrease"
    min_feedback_rounds: int = Field(default=3, ge=1)
    record_rounds: bool = True

    @model_validator(mode="after")
    def compatible(self):
        if self.conditioning != "correct" and not self.donor_bank:
            raise ValueError(
                "Swapped/fixed conditioning requires a prepared donor_bank"
            )
        if self.conditioning != "correct" and self.refresh != "prompt":
            raise ValueError("Conditioning controls keep the adapter fixed per request")
        if self.feedback and self.refresh != "prefix":
            raise ValueError("Generator feedback requires prefix refresh")
        if (
            self.scale_min > self.scale_max
            or not self.scale_min <= self.scale <= self.scale_max
        ):
            raise ValueError("Initial scale must be inside scale bounds")
        if self.low_acceptance >= self.high_acceptance:
            raise ValueError("Acceptance thresholds must be ordered")
        return self


class VerifierFeedback:
    """Only completed verification rounds may update the next proposal's inputs."""

    def __init__(self, policy):
        self.policy = policy
        self.rounds = 0
        self.acceptance = self.ema = 0.0
        self.scale = policy.scale

    def observe(self, accepted, proposed):
        if not 0 <= accepted <= proposed or proposed < 1:
            raise ValueError("Invalid accepted/proposed token counts")
        self.acceptance = accepted / proposed
        self.ema = (
            self.acceptance
            if not self.rounds
            else self.policy.ema_decay * self.ema
            + (1 - self.policy.ema_decay) * self.acceptance
        )
        self.rounds += 1
        if (
            self.policy.scale_policy == "acceptance_ema"
            and self.rounds >= self.policy.min_feedback_rounds
        ):
            direction = (
                -1
                if self.ema < self.policy.low_acceptance
                else (1 if self.ema > self.policy.high_acceptance else 0)
            )
            if self.policy.low_action == "increase":
                direction *= -1
            self.scale = max(
                self.policy.scale_min,
                min(
                    self.policy.scale_max,
                    self.scale + direction * self.policy.scale_step,
                ),
            )
        if not math.isfinite(self.scale):
            raise ValueError("Non-finite adapter scale")

    def tensor(self, device):
        return torch.tensor(
            [self.acceptance, self.ema, float(self.rounds > 0)],
            dtype=torch.float32,
            device=device,
        )
