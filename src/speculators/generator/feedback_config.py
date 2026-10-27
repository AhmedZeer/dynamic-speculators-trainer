"""Independent regular, projector-only and full-feedback continuation configs."""

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from speculators.generator.study_config import GeneratorStudy


class ParentBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    revision: str | None = None
    subfolder: str | None = None
    optimizer_step: int = Field(default=4000, ge=0)

    @model_validator(mode="after")
    def pinned(self):
        from speculators.benchmark.config import Model  # noqa: PLC0415

        Model(
            name="parent",
            type="generator",
            **self.model_dump(exclude={"optimizer_step"}),
        )
        if not Path(self.model).is_dir() and (
            self.revision is None
            or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None
        ):
            raise ValueError("Remote parent bundles require a full revision SHA")
        return self


class FeedbackStudy(GeneratorStudy):
    """Override the legacy replay-only validator for this separate training mode."""

    parent: ParentBundle
    trainable: Literal["generator", "projector", "generator_and_projector"]
    conditioning: Literal["prompt"] = "prompt"
    loss: Literal["lk_hybrid"] = "lk_hybrid"
    rollout_tokens: int = Field(default=64, ge=4, le=8192)

    @model_validator(mode="after")
    def coherent(self):
        if self.training_subset == self.validation_subset:
            raise ValueError("Training and held-out shards must be distinct")
        if (
            self.arm != "generator"
            or self.replay
            or self.initial_checkpoint is not None
        ):
            raise ValueError(
                "Feedback continuation loads a bundle; "
                "replay/LoRA/local-parent modes are disabled"
            )
        if self.feedback != (self.trainable != "generator"):
            raise ValueError(
                "Feedback requires projector or generator_and_projector training; "
                "the regular generator arm disables feedback"
            )
        return self
