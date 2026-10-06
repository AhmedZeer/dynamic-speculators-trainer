"""External experiment controls, separate from the ordinary trainer defaults."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

CACHED_LAYER_COUNT = 4


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Models(Settings):
    target: str = "Qwen/Qwen3-8B"
    target_revision: str = "main"
    drafter: str = "RedHatAI/Qwen3-8B-speculator.eagle3"
    drafter_revision: str = "main"


class Dataset(Settings):
    source: str = "openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated"
    revision: str = "main"
    configuration: str = "default"
    split: str = "math"
    category: Literal["math", "code", "stem", "chat"] = "math"
    sample_limit: int | None = Field(default=None, ge=1)


class Partition(Settings):
    train_examples: int = Field(default=1000, ge=1)
    validation_examples: int = Field(default=100, ge=1)
    seed: int = 0
    subset_ids: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])

    @model_validator(mode="after")
    def unique_ids(self):
        if not self.subset_ids or min(self.subset_ids) < 0:
            raise ValueError("subset_ids must be nonempty and nonnegative")
        if len(set(self.subset_ids)) != len(self.subset_ids):
            raise ValueError("subset_ids must be unique")
        return self


class Responses(Settings):
    thinking: bool = False
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=1)
    seed: int = 0
    max_tokens: int = Field(default=4096, ge=1)
    context_length: int = Field(default=8192, ge=2)


class HiddenStates(Settings):
    layer_ids: list[int] = Field(default_factory=lambda: [2, 18, 33, 36])
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    sequence_length: int = Field(default=8192, ge=2)


class Lora(Settings):
    rank: int = Field(default=32, ge=1)
    alpha: int = Field(default=64, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ["o_proj", "v_proj"])


class Training(Settings):
    seeds: list[int] = Field(default_factory=lambda: [42, 43, 44])
    warmup_epochs: int = Field(default=1, ge=1)
    collect_epochs: int = Field(default=4, ge=1)
    warmup_lr: float = Field(default=1e-4, gt=0)
    collect_lr: float = Field(default=1e-6, gt=0)
    save_interval: int = Field(default=10, ge=1)
    weight_decay: float = Field(default=0.01, ge=0)
    token_budget: int = Field(default=8192, ge=2)
    noise_std: float = Field(default=0.05, ge=0)
    ttt_steps: int = Field(default=3, ge=1)
    ttt_step_loss_decay: float = Field(default=1.0, gt=0)
    loss_fn: Literal["kl_div", "ce"] = "kl_div"
    log_freq: int = Field(default=10, ge=1)
    gradient_checkpointing: bool = False


class Execution(Settings):
    generation_endpoint: str = "http://localhost:8000/v1/chat/completions"
    extraction_endpoint: str = "http://localhost:8000/v1"
    concurrency: int = Field(default=32, ge=1)
    hidden_state_concurrency: int = Field(default=16, ge=1)
    hidden_state_write_concurrency: int = Field(default=2, ge=1)
    hidden_state_log_interval: float = Field(default=10.0, ge=0.1)
    prompt_chunk_size: int = Field(default=1000, ge=1)
    response_staging_dir: Path | None = Path("/tmp/speculators-responses")  # noqa: S108
    response_sync_interval: int = Field(default=1000, ge=1)
    request_timeout: float = Field(default=600, gt=0)
    max_retries: int = Field(default=3, ge=0)
    n_workers: int = Field(
        default=1, ge=1, description="Maximum independent subset/seed runs in parallel."
    )
    processes: int = Field(default=1, ge=1)
    draft_attn_impl: Literal["simple_flex_attention", "sdpa", "eager"] = (
        "simple_flex_attention"
    )


class Conditioning(Settings):
    max_examples: int = Field(default=32, ge=1)
    states: Literal["prompt"] = "prompt"


class BankConfig(Settings):
    output_root: Path = Path("output/lora-bank-math")
    models: Models = Field(default_factory=Models)
    dataset: Dataset = Field(default_factory=Dataset)
    partition: Partition = Field(default_factory=Partition)
    responses: Responses = Field(default_factory=Responses)
    hidden_states: HiddenStates = Field(default_factory=HiddenStates)
    lora: Lora = Field(default_factory=Lora)
    training: Training = Field(default_factory=Training)
    execution: Execution = Field(default_factory=Execution)
    conditioning: Conditioning = Field(default_factory=Conditioning)

    @model_validator(mode="after")
    def compatible(self):
        if self.training.collect_lr >= self.training.warmup_lr:
            raise ValueError("collect_lr must be lower than warmup_lr")
        if not self.training.seeds or len(set(self.training.seeds)) != len(
            self.training.seeds
        ):
            raise ValueError("Training seeds must be nonempty and unique")
        if set(self.lora.target_modules) != {"o_proj", "v_proj"}:
            raise ValueError("Bank v1 targets exactly o_proj and v_proj")
        if (
            len(self.hidden_states.layer_ids) != CACHED_LAYER_COUNT
            or len(set(self.hidden_states.layer_ids)) != CACHED_LAYER_COUNT
        ):
            raise ValueError(
                "Use three input layers followed by the final target layer"
            )
        if self.hidden_states.sequence_length > self.responses.context_length:
            raise ValueError("Extraction length cannot exceed generation context")
        if self.training.token_budget < self.hidden_states.sequence_length:
            raise ValueError("Token budget must fit a complete prepared example")
        if self.conditioning.max_examples > self.partition.train_examples:
            raise ValueError("Conditioning size exceeds subset training size")
        return self

    @classmethod
    def load(cls, path: Path):
        return cls.model_validate(yaml.safe_load(path.read_text()))
