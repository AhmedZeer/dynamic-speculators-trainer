"""Allow budget, path, and validation changes while preserving checkpoint identities."""

import copy
import json
import time

from speculators.bank.artifacts import digest, write_json
from speculators.bank.progress import report
from speculators.generator.campaign_config import CampaignConfig


def compatible_campaign(previous, cfg):
    from speculators.generator.campaign import campaign_identity  # noqa: PLC0415

    saved = CampaignConfig.model_validate(previous["configuration"])
    saved.output_root = cfg.output_root
    saved.optimization.pretraining_updates = cfg.optimization.pretraining_updates
    saved.optimization.pretraining_validation_interval = (
        cfg.optimization.pretraining_validation_interval
    )
    saved.adaptation_updates = cfg.adaptation_updates
    return campaign_identity(saved) == campaign_identity(cfg)


def resume_identity(output, current):
    path = output / "resolved_experiment.json"
    if not path.is_file():
        return current
    saved = json.loads(path.read_text())
    if saved == current:
        return current
    adjusted = copy.deepcopy(saved)
    adjusted["configuration"]["output_root"] = current["configuration"]["output_root"]
    adjusted["configuration"]["optimization"]["pretraining_updates"] = current[
        "configuration"
    ]["optimization"]["pretraining_updates"]
    adjusted["job"]["updates"] = current["job"]["updates"]
    for key in ("hub_upload_interval", "save_every_epochs"):
        if key in current["job"]:
            adjusted["job"][key] = current["job"][key]
        else:
            adjusted["job"].pop(key, None)
    if adjusted != current:
        # Let begin_run report an incompatible experiment normally.
        return current
    write_json(
        output / "provenance" / f"resume-{time.time_ns()}" / "configuration.json",
        {
            "checkpoint_identity": digest(saved),
            "effective_identity": current,
            "note": (
                "Budget/output path/upload cadence changed; "
                "saved checkpoint identity retained."
            ),
        },
    )
    report(
        f"Resuming {output.name} with target={current['job']['updates']} updates; "
        "retaining saved checkpoint identity and optimizer progress"
    )
    return saved


def record_validation_change(output, cfg):
    """Archive validation changes separately from the model identity."""
    path = output / "configuration.json"
    if not path.exists():
        return
    previous = json.loads(path.read_text())["optimization"][
        "pretraining_validation_interval"
    ]
    current = cfg.optimization.pretraining_validation_interval
    if previous != current:
        write_json(
            output
            / "provenance"
            / f"validation-change-{time.time_ns()}"
            / "configuration.json",
            {
                "previous_interval": previous,
                "validation_interval": current,
                "configuration": cfg.model_dump(mode="json"),
            },
        )
