"""Single-experiment recipes; no campaign, subprocess chain, or ablation runner."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from speculators.experiment_config import load_experiment_yaml


class StudyTracking(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    project: str = "eagle3-generator-reference-train-v4"
    entity: str | None = None
    group: str = "4k-1000steps"
    mode: Literal["online", "offline"] = "online"


class GeneratorStudy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="inputs", pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    workspace: Path
    campaign_config: Path = Path("examples/eagle3-lora/generator_campaign_online.yaml")
    training_subset: int = Field(default=1, ge=0, le=5)
    validation_subset: int = Field(default=0, ge=0, le=5)
    train_per_category: int = Field(default=1000, ge=1)
    tune_per_category: int = Field(default=50, ge=1)
    test_per_category: int = Field(default=50, ge=1)
    max_output_tokens: int = Field(default=256, ge=1)
    seed: int = 42
    arm: Literal["generator", "lora"] = "generator"
    loss: Literal["lk_hybrid", "kl_div", "ce"] = "lk_hybrid"
    loss_implementation: Literal["fused", "eager"] = "fused"
    steps: int = Field(default=1000, ge=1)
    examples_per_update: int = Field(default=8, ge=1)
    validation_interval: int = Field(default=500, ge=1)
    checkpoint_interval: int = Field(default=500, ge=1)
    hub_interval: int = Field(default=1000, ge=1)
    repo_id: str | None = None
    initial_checkpoint: Path | None = None
    initial_step: int = Field(default=500, ge=1)
    replay: bool = False
    conditioning: Literal["prompt", "prefix"] = "prompt"
    feedback: bool = False
    replay_rounds_per_episode: int = Field(default=4, ge=1)
    speculative_tokens: Literal[3] = 3
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    warmup_steps: int = Field(default=100, ge=0)
    max_grad_norm: float = Field(default=0.85, gt=0)
    wandb: StudyTracking = Field(default_factory=StudyTracking)

    @model_validator(mode="after")
    def coherent(self):
        if self.training_subset == self.validation_subset:
            raise ValueError("Training and held-out shards must be distinct")
        if (self.conditioning == "prefix" or self.feedback) and not self.replay:
            raise ValueError("Prefix/feedback training requires verification replay")
        if self.feedback and self.conditioning != "prefix":
            raise ValueError(
                "Feedback conditioning uses current verified-prefix states"
            )
        if self.replay and (self.arm != "generator" or self.initial_checkpoint is None):
            raise ValueError("Replay requires a generator warm-start checkpoint")
        if self.initial_checkpoint is not None and self.initial_step >= self.steps:
            raise ValueError("The initial step must precede the final training step")
        return self

    @classmethod
    def load(cls, path):
        return cls.model_validate(load_experiment_yaml(path))

    @property
    def output(self):
        return self.workspace / "training" / self.name

    @property
    def inputs(self):
        return self.workspace / "inputs"


def campaign_settings(study):
    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415

    cfg = CampaignConfig.load(study.campaign_config)
    cfg.input_root = study.inputs / "metadata"
    cfg.output_root = study.output
    cfg.seed = study.seed
    cfg.optimization.loss_fn = study.loss
    cfg.optimization.lr = study.lr
    cfg.optimization.weight_decay = study.weight_decay
    cfg.optimization.warmup_updates = study.warmup_steps
    cfg.optimization.max_grad_norm = study.max_grad_norm
    cfg.optimization.episodes_per_update = study.examples_per_update
    cfg.staging_dir = study.workspace / "cache"
    cfg.online.summary_cache_dir = study.workspace / "cache/summaries"
    cfg.online.compiler_cache_dir = study.workspace / "cache/compiler"
    cfg.wandb.enabled = study.wandb.enabled
    cfg.wandb.mode = study.wandb.mode
    cfg.wandb.entity = study.wandb.entity
    cfg.wandb.group = study.wandb.group
    cfg.wandb.pretraining_project = study.wandb.project
    cfg.wandb.layout = "sections"
    return cfg
