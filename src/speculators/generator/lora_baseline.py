"""Ordinary LoRA baselines matched to direct-generator scaling jobs."""

from filelock import FileLock

from speculators.generator.campaign import jobs, run_one


def matched_lora_job(cfg, scale):
    generator = next((job for job in jobs(cfg) if job.get("scale") == scale), None)
    if generator is None:
        raise ValueError("LoRA baseline scale must be 1, 2, or 4")
    return {
        **generator,
        "label": f"scale-{scale}-lora",
        "arm": "fresh_lora",
        "matched_generator": generator["label"],
    }


def train_matched_lora(cfg, scale, output_root, *, dry_run=False):
    if cfg.activation_source != "vllm":
        raise ValueError("Matched LoRA baseline requires the online activation config")
    cfg = cfg.model_copy(update={"output_root": output_root})
    job = matched_lora_job(cfg, scale)
    if dry_run:
        return {"job": job, "configuration": cfg.model_dump(mode="json")}
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_root / ".campaign.lock"), timeout=0):
        return run_one(cfg, job)
