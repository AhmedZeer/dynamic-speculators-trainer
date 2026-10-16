"""Training-only campaign using local data and an external vLLM extractor."""

import json

from filelock import FileLock

from speculators.bank.artifacts import write_json
from speculators.generator.campaign import (
    campaign_identity,
    jobs,
    pilot_profile,
    plot_comparisons,
    run_one,
    subset_names,
)
from speculators.generator.local_campaign import load_local_inputs


def online_pilot(cfg, inputs):
    profile = pilot_profile(cfg)
    report = {
        "campaign_identity": campaign_identity(cfg),
        "profile": profile,
        "passed": False,
        "results": [],
        "note": "Training and context-4 offline validation; no decoding benchmark.",
    }
    path = cfg.output_root / "pilot" / "report.json"
    try:
        for arm in ("bank", "direct", "fresh_lora"):
            job = {
                "kind": "pretraining",
                "stage": "pilot",
                "condition": "last",
                "label": f"online-{arm}",
                "arm": arm,
                "objective": "bank" if arm == "bank" else "direct",
                "subsets": subset_names([0]),
                "validation_subsets": subset_names([0]),
                "updates": cfg.pilot.updates,
                "pilot": True,
                "profile": profile,
            }
            report["results"].append(run_one(cfg, job, local_inputs=inputs))
        report["data_wait_fractions"] = {
            r["job"]["arm"]: r["data_wait_seconds"] / max(r["training_seconds"], 1e-9)
            for r in report["results"]
        }
        report["passed"] = True
    except Exception as exc:
        report.update(error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        write_json(path, report)
    return report


def run_online_campaign(cfg, stage, dry_run):  # noqa: C901 -- serial stages
    planned = jobs(cfg)
    manifest = {
        "version": 2,
        "campaign_identity": campaign_identity(cfg),
        "configuration": cfg.model_dump(mode="json"),
        "jobs": planned,
        "activation_source": "external vLLM, transient shared files",
        "metric": "offline agreement at context 4; no decoding benchmark",
    }
    if dry_run:
        return manifest
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_root / ".campaign.lock"), timeout=0):
        inputs = load_local_inputs(cfg)
        manifest["local_inputs"] = inputs.identity
        path = cfg.output_root / "campaign.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if any(
                previous.get(k) != manifest[k]
                for k in ("campaign_identity", "local_inputs")
            ):
                raise ValueError(
                    "Campaign or local data changed; choose a new output_root"
                )
        write_json(path, manifest)
        if stage == "prepare":
            return {**manifest, "trajectory_examples": len(inputs.rows)}
        if stage in ("pilot", "all"):
            report = online_pilot(cfg, inputs)
            if stage == "pilot":
                return report
        else:
            pilot_path = cfg.output_root / "pilot" / "report.json"
            if not pilot_path.is_file():
                raise FileNotFoundError("Run generator campaign --stage pilot first")
            report = json.loads(pilot_path.read_text())
        if (
            not report["passed"]
            or report.get("profile") != pilot_profile(cfg)
            or report["campaign_identity"] != campaign_identity(cfg)
        ):
            raise ValueError(
                "A successful online training pilot is required for these settings"
            )
        results = []
        for phase in ("pretraining", "adaptation"):
            if stage not in (phase, "all"):
                continue
            current = [
                run_one(cfg, job, local_inputs=inputs)
                for job in planned
                if job["stage"] == phase
            ]
            write_json(cfg.output_root / phase / "results.json", current)
            plot_comparisons(cfg, current, phase)
            results.extend(current)
        return {"runs": len(results), "output_root": str(cfg.output_root)}
