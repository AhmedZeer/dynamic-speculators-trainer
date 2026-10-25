"""Small declarative model and workload suites with shared measurement settings."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CONCURRENCY = 8


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Model(Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    type: Literal["lora", "generator", "drafter"]
    model: str = Field(min_length=1)
    revision: str | None = None
    subfolder: str | None = None
    conditioning_examples: int | None = Field(default=None, ge=1, le=MAX_CONCURRENCY)
    adapter_policy: Literal["shared", "per_request"] = "shared"
    vanilla_vllm: bool = False
    batching_policy: Literal["fixed_shared_adapter", "continuous"] | None = None

    @model_validator(mode="after")
    def generator_conditioning_only(self):
        if self.conditioning_examples is not None and self.type != "generator":
            raise ValueError("conditioning_examples applies only to generators")
        if self.adapter_policy == "per_request" and (
            self.type != "generator" or self.conditioning_examples not in (None, 1)
        ):
            raise ValueError("per_request requires a singleton-conditioned generator")
        if self.vanilla_vllm and self.type != "drafter":
            raise ValueError("vanilla_vllm is available for original drafters only")
        return self

    @field_validator("subfolder")
    @classmethod
    def relative_subfolder(cls, value):
        if value is not None and (
            Path(value).is_absolute() or ".." in Path(value).parts
        ):
            raise ValueError("subfolder must stay inside the model repository")
        return value


class Benchmark(Strict):
    dataset: Literal["nemotron", "RedHatAI/speculator_benchmarks"]
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    categories: list[str] = Field(default_factory=lambda: ["all"], min_length=1)
    prompt_format: Literal["auto", "raw", "chat"] = "auto"

    @field_validator("categories", mode="before")
    @classmethod
    def category_list(cls, value):
        return (
            [item.strip() for item in value.split(",")]
            if isinstance(value, str)
            else value
        )

    @property
    def label(self):
        return self.name or ("nemotron" if self.dataset == "nemotron" else "specbench")


class Profiling(Strict):
    max_iterations: int = Field(default=30, ge=1, le=200)
    with_stack: bool = False
    record_shapes: bool = False
    with_memory: bool = False


class Settings(Strict):
    campaign_config: Path | None = None
    tokenizer_path: Path | None = None
    max_samples: int = Field(default=50, ge=1)
    max_tokens: int = Field(default=256, ge=1)
    seed: int = 42
    warmups: int = Field(default=3, ge=0)
    concurrency: int = Field(default=1, ge=1, le=8)
    concurrencies: list[int] | None = None
    batch_wait_ms: float = Field(default=50, ge=0)
    batching_policy: Literal["fixed_shared_adapter", "continuous"] = (
        "fixed_shared_adapter"
    )
    port: int = Field(default=8000, ge=1024, le=65535)
    gpu: str | int = 0
    gpu_memory_utilization: float = Field(default=0.85, gt=0, lt=1)
    max_model_len: int = Field(default=8192, ge=2)
    speculative_tokens: int = Field(default=3, ge=1)
    startup_timeout_seconds: float = Field(default=900, gt=0)
    client_timeout_seconds: float = Field(default=3600, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)
    continue_on_error: bool = True
    profiling: Profiling | None = None

    @field_validator("concurrencies")
    @classmethod
    def supported_concurrencies(cls, value):
        if value is not None and (
            not value
            or len(value) != len(set(value))
            or any(not 1 <= item <= MAX_CONCURRENCY for item in value)
        ):
            raise ValueError("concurrencies must contain distinct values from 1 to 8")
        return value

    @property
    def concurrency_values(self):
        return self.concurrencies or [self.concurrency]


class Suite(Strict):
    models: list[Model] = Field(min_length=1)
    benchmarks: list[Benchmark] = Field(min_length=1)
    settings: Settings = Field(default_factory=Settings)

    @model_validator(mode="after")
    def unique_names(self):
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError("Model names must be unique")
        if len({b.label for b in self.benchmarks}) != len(self.benchmarks):
            raise ValueError("Repeated datasets require distinct benchmark names")
        for model in self.models:
            continuous = (
                model.batching_policy or self.settings.batching_policy
            ) == "continuous"
            if model.adapter_policy == "per_request" and not continuous:
                raise ValueError("per_request requires continuous batching")
            if model.vanilla_vllm and not continuous:
                raise ValueError("Vanilla baseline requires continuous batching")
            if (
                continuous
                and model.type != "drafter"
                and model.adapter_policy != "per_request"
            ):
                raise ValueError(
                    "Continuous adapter serving requires a per_request generator"
                )
        return self

    @classmethod
    def load(cls, path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
