"""Configuration for direct pretraining and unseen adaptation."""

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from speculators.bank.config import Settings
from speculators.generator.config import Context, ExperimentConfig, Optimization

DOMAINS = ("math", "code", "stem", "chat")
SCALES = (1, 2, 4)


class DownloadSettings(Settings):
    capacity_gib: float = Field(default=350, gt=0)
    reserve_gib: float = Field(default=50, ge=0)
    workers: int = Field(default=4, ge=1)
    verification_workers: int = Field(default=2, ge=1)
    lookahead_updates: int = Field(default=8, ge=1)


class PilotSettings(Settings):
    updates: int = Field(default=100, ge=1)
    worker_counts: list[int] = Field(default_factory=lambda: [4, 8])
    max_data_wait_fraction: float = Field(default=0.2, gt=0, lt=1)

    @model_validator(mode="after")
    def valid_workers(self):
        if not self.worker_counts or min(self.worker_counts) < 1:
            raise ValueError("Pilot worker counts must be positive and nonempty")
        if len(set(self.worker_counts)) != len(self.worker_counts):
            raise ValueError("Duplicate pilot worker count")
        return self


class OnlineSettings(Settings):
    summary_cache_dir: Path = Path("output/generator-summary-cache")
    compiler_cache_dir: Path | None = Path("output/generator-compiler-cache")
    lookahead_updates: int = Field(default=2, ge=0)
    ram_cache_gib: float = Field(default=96, ge=0)
    endpoint: str = "http://127.0.0.1:8000/v1"
    shared_dir: Path = Path("/tmp/speculators-online")  # noqa: S108
    concurrency: int = Field(default=16, ge=1)
    buffered_examples: int = Field(default=64, ge=8)
    request_timeout: float = Field(default=600, gt=0)
    max_retries: int = Field(default=3, ge=0)
    reserve_gib: float = Field(default=8, ge=0)
    target_model_path: Path = Path("models/qwen3-8b")
    drafter_model_path: Path = Path("models/qwen3-8b-eagle3")


class CampaignConfig(ExperimentConfig):
    """Three direct pretraining jobs and four unseen-data adaptation jobs."""

    training_recipe: Path | None = None

    bank_revision: str = "2305403f0f3b560fafb6c6d261c370ab3a95759d"
    activation_source: Literal["hub", "vllm"] = "hub"
    data_path: Path | None = None
    trajectories_path: Path | None = None
    bank_release: str = "b0002"
    online: OnlineSettings = Field(default_factory=OnlineSettings)
    input_root: Path = Path("output/generator-campaign-inputs")
    output_root: Path = Path("output/generator-campaign")
    staging_dir: Path = Path("/tmp/speculators-campaign")  # noqa: S108
    adaptation_updates: int = Field(default=1000, ge=1)
    experiment_subsets: list[str] | None = Field(
        default_factory=lambda: [f"{d}-{i:05d}" for d in DOMAINS for i in range(6)]
    )
    expected_train_examples: int = Field(default=1000, ge=8)
    expected_validation_examples: int = Field(default=100, ge=4)
    condition: str = "last"
    context: Context = Field(default_factory=lambda: Context(evaluation_sizes=[4]))
    optimization: Optimization = Field(
        default_factory=lambda: Optimization(
            loss_fn="lk_hybrid",
            pretraining_updates=10000,
            warmup_updates=100,
            checkpoint_interval=500,
        )
    )
    downloads: DownloadSettings = Field(default_factory=DownloadSettings)
    pilot: PilotSettings = Field(default_factory=PilotSettings)

    @model_validator(mode="after")
    def campaign_contract(self):
        if self.activation_source == "vllm" and self.trajectories_path is None:
            raise ValueError("Online campaign requires trajectories_path")
        if self.activation_source == "hub" and not re.fullmatch(
            r"[0-9a-f]{40}", self.bank_revision
        ):
            raise ValueError("Campaign requires a complete bank repository SHA")
        if (
            self.training_recipe is None
            and len(self.additional_bank_configs) != len(DOMAINS) - 1
        ):
            raise ValueError("Campaign requires exactly four category bank configs")
        if self.condition != "last" or self.context.evaluation_sizes != [4]:
            raise ValueError("Campaign uses last conditioning and context-4 validation")
        if self.optimization.loss_fn != "lk_hybrid" or self.n_workers != 1:
            raise ValueError("Campaign requires lk_hybrid and one training worker")
        if self.activation_local_dir is not None:
            raise ValueError(
                "Campaign uses its shared on-demand cache, not staged inputs"
            )
        if self.pretraining_subsets is not None:
            raise ValueError("Campaign derives explicit subset membership per job")
        return self

    @classmethod
    def load(cls, path: Path):
        values = yaml.safe_load(path.read_text())
        if "training_recipe" not in values:
            return super().load(path)
        recipe = Path(values["training_recipe"])
        if not recipe.is_absolute():
            recipe = path.resolve().parent / recipe
        values["training_recipe"] = recipe
        # Shared runtime configuration still uses this internal recipe alias.
        values["bank_config"] = recipe
        return cls.model_validate(values)
