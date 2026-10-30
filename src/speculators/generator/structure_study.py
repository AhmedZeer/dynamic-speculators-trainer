"""One fresh generator or static-LoRA run using the GEN-001 protocol."""

from pathlib import Path
from typing import Literal

import yaml
from filelock import FileLock
from huggingface_hub import HfApi
from huggingface_hub.utils import validate_repo_id
from pydantic import Field, model_validator

from speculators.generator.campaign_config import DOMAINS, CampaignConfig


class StructureStudy(CampaignConfig):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    family: Literal["modules", "architecture", "baseline"]
    arm: Literal["generator", "static_lora"] = "generator"
    lora_dtype: Literal["float32", "bfloat16"] = "bfloat16"
    repo_id: str
    provenance_dir: Path
    private: bool = False

    @model_validator(mode="after")
    def fresh_original_protocol(self):
        validate_repo_id(self.repo_id)
        if self.activation_source != "vllm" or self.pretrained_checkpoint is not None:
            raise ValueError(
                "Structure studies train fresh adapters/generators "
                "from saved trajectories"
            )
        if (
            self.training_cohort_size is not None
            or self.training_conditioning_examples is not None
        ):
            raise ValueError("Structure studies use the original random 1–8 cohorts")
        return self

    @classmethod
    def load(cls, path: Path):
        """Resolve a common YAML relative to this file and merge its sections."""
        path = Path(path).resolve()
        values = yaml.safe_load(path.read_text())
        base_path = path
        if "extends" in values:
            base_path = (path.parent / values.pop("extends")).resolve()
            common = yaml.safe_load(base_path.read_text())
            if "extends" in common:
                raise ValueError("Only one common config layer is supported")
            for key, value in values.items():
                if isinstance(value, dict) and isinstance(common.get(key), dict):
                    common[key] = {**common[key], **value}
                else:
                    common[key] = value
            values = common
        recipe = Path(values["training_recipe"])
        if not recipe.is_absolute():
            recipe = base_path.parent / recipe
        values["training_recipe"] = recipe.resolve()
        values["bank_config"] = recipe.resolve()
        return cls.model_validate(values)


def training_job(cfg):
    subsets = [f"{domain}-00000" for domain in DOMAINS]
    job = {
        "kind": "pretraining",
        "stage": "pretraining",
        "condition": "last",
        "label": cfg.name,
        "arm": "fresh_lora" if cfg.arm == "static_lora" else "direct",
        "objective": "direct",
        "subsets": subsets,
        "validation_subsets": subsets,
        "updates": cfg.optimization.pretraining_updates,
        "training_examples": len(DOMAINS) * cfg.expected_train_examples,
        "tracking_group": f"structure-v1/{cfg.family}",
        "hub_upload_interval": cfg.hub_upload_interval,
        "hub_repo": cfg.repo_id,
        "vllm_provenance_dir": str(cfg.provenance_dir.resolve()),
        "drafter_model_path": str(cfg.online.drafter_model_path.resolve()),
    }
    if cfg.arm == "static_lora":
        job["adapter_dtype"] = cfg.lora_dtype
    return job


def train_structure(cfg, *, dry_run=False):
    """Resume locally if present; launch exactly one historical episode schedule."""
    job = training_job(cfg)
    if dry_run:
        return {"job": job, "configuration": cfg.model_dump(mode="json")}
    from speculators.generator.campaign import run_one  # noqa: PLC0415
    from speculators.generator.publish import extraction_provenance  # noqa: PLC0415

    if extraction_provenance(cfg.output_root, cfg.provenance_dir) is None:
        raise FileNotFoundError("Launch the extractor with --provenance-dir first")
    api = HfApi()
    api.whoami()
    api.create_repo(
        repo_id=cfg.repo_id, repo_type="model", exist_ok=True, private=cfg.private
    )
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_root / ".structure.lock"), timeout=0):
        return run_one(cfg, job)
