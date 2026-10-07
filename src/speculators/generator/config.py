"""Controls for experiments, independent of completed bank configurations."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from speculators.bank.config import Settings

Condition = Literal["last", "last_mean", "projected"]
CONDITIONS = ("last", "last_mean", "projected")


class Architecture(Settings):
    encoder_hidden: int = Field(default=512, ge=1)
    example_dim: int = Field(default=128, ge=1)
    condition_dim: int = Field(default=64, ge=1)
    descriptor_dim: int = Field(default=32, ge=1)
    decoder_width: int = Field(default=512, ge=1)
    residual_blocks: int = Field(default=2, ge=1)


class Optimization(Settings):
    lr: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    max_grad_norm: float = Field(default=0.85, gt=0)
    conditioning_epochs: int = Field(default=5, ge=1)
    pretraining_updates: int = Field(default=1000, ge=1)
    episodes_per_update: int = Field(default=4, ge=1)
    adaptation_epochs: Literal[1] = 1
    checkpoint_interval: int = Field(default=100, ge=1)
    log_interval: int = Field(default=10, ge=1)
    token_budget: int = Field(default=8192, ge=2)


class Context(Settings):
    min_examples: int = Field(default=1, ge=1)
    max_examples: int = Field(default=8, ge=1, le=8)
    evaluation_sizes: list[int] = Field(default_factory=lambda: [1, 4, 8])

    @model_validator(mode="after")
    def compatible(self):
        if self.min_examples > self.max_examples:
            raise ValueError("min_examples exceeds max_examples")
        sizes = self.evaluation_sizes
        if not sizes or len(set(sizes)) != len(sizes):
            raise ValueError("Evaluation context sizes must be nonempty and unique")
        if min(sizes) < self.min_examples or max(sizes) > self.max_examples:
            raise ValueError("Evaluation sizes must be within the context range")
        return self


class Heatmap(Settings):
    strides: list[int] = Field(default_factory=lambda: [10, 20, 30])
    seed_groups: list[list[int]] = Field(
        default_factory=lambda: [[42], [43], [44], [42, 44], [43, 42], [42, 43, 44]]
    )

    @model_validator(mode="after")
    def unique(self):
        if not self.strides or min(self.strides) < 1:
            raise ValueError("Strides must be positive")
        if len(set(self.strides)) != len(self.strides):
            raise ValueError("Duplicate strides")
        groups = [tuple(sorted(g)) for g in self.seed_groups]
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("Seed groups must be nonempty and distinct")
        if any(not g or len(set(g)) != len(g) for g in groups):
            raise ValueError("Each seed group must be nonempty and unique")
        return self


class WandbSettings(Settings):
    enabled: bool = False
    mode: Literal["online", "offline"] = "offline"
    entity: str | None = None
    group: str | None = None
    name_prefix: str = ""
    log_interval: int = Field(default=1, ge=1)
    upload_artifacts: bool = False
    projects: dict[Literal["conditioning", "heatmap", "adaptation"], str] = Field(
        default_factory=lambda: {
            "conditioning": "eagle3-generator-conditioning",
            "heatmap": "eagle3-generator-heatmap",
            "adaptation": "eagle3-generator-adaptation",
        }
    )

    @model_validator(mode="after")
    def distinct_projects(self):
        if set(self.projects) != {"conditioning", "heatmap", "adaptation"}:
            raise ValueError("Provide a W&B project for each of the three experiments")
        names = [name.strip() for name in self.projects.values()]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError("W&B project names must be nonempty and different")
        self.projects = dict(zip(self.projects, names, strict=True))
        return self


class ExperimentConfig(Settings):
    bank_config: Path
    output_root: Path = Path("output/generator-math")
    staging_dir: Path = Path("/tmp/speculators-generator")  # noqa: S108
    conditioning_subset: str = "math-00001"
    bank_subset: str = "math-00000"
    transfer_seed: int = 42
    seed: int = 42
    n_workers: int = Field(default=1, ge=1)
    device: str = "cuda"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    factor_cache_mib: int = Field(default=256, ge=0)
    preprocessing_workers: int = Field(default=4, ge=1)
    architecture: Architecture = Field(default_factory=Architecture)
    optimization: Optimization = Field(default_factory=Optimization)
    context: Context = Field(default_factory=Context)
    heatmap: Heatmap = Field(default_factory=Heatmap)
    wandb: WandbSettings = Field(default_factory=WandbSettings)

    @model_validator(mode="after")
    def separate_subsets(self):
        if self.conditioning_subset == self.bank_subset:
            raise ValueError("Conditioning/adaptation and bank subsets must differ")
        return self

    @classmethod
    def load(cls, path: Path):
        values = yaml.safe_load(path.read_text())
        # Resolve the bank reference relative to the experiment YAML. Output and
        # staging paths retain the repository's working-directory convention.
        bank_path = Path(values["bank_config"])
        if not bank_path.is_absolute():
            values["bank_config"] = path.resolve().parent / bank_path
        return cls.model_validate(values)
