"""Matched baseline uses the generator's selection and training budget."""

from pathlib import Path

import pytest

from speculators.generator.campaign import jobs
from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.lora_baseline import matched_lora_job, train_matched_lora


@pytest.mark.parametrize("scale", [1, 2, 4])
def test_matched_lora_uses_generator_data_and_budget(tmp_path, scale):
    cfg = CampaignConfig.load(
        Path("examples/eagle3-lora/generator_campaign_online.yaml")
    )
    source = next(job for job in jobs(cfg) if job.get("scale") == scale)
    baseline = matched_lora_job(cfg, scale)
    for key in (
        "subsets",
        "validation_subsets",
        "updates",
        "training_examples",
        "objective",
    ):
        assert baseline[key] == source[key]
    assert baseline["arm"] == "fresh_lora"
    assert baseline["label"] == f"scale-{scale}-lora"
    result = train_matched_lora(cfg, scale, tmp_path / "new", dry_run=True)
    assert result["job"] == baseline
    assert result["configuration"]["output_root"] == str(tmp_path / "new")
    assert not (tmp_path / "new").exists()


def test_matched_lora_invalid_scale():
    cfg = CampaignConfig.load(
        Path("examples/eagle3-lora/generator_campaign_online.yaml")
    )
    with pytest.raises(ValueError, match="1, 2, or 4"):
        matched_lora_job(cfg, 3)
