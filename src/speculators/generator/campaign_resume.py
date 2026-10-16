"""Allow budget/path changes without rewriting saved checkpoint identities."""

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
    if adjusted != current:
        # Let begin_run report an incompatible experiment normally.
        return current
    write_json(
        output / "provenance" / f"resume-{time.time_ns()}" / "configuration.json",
        {
            "checkpoint_identity": digest(saved),
            "effective_identity": current,
            "note": "Budget/output path changed; saved checkpoint identity retained.",
        },
    )
    report(
        f"Resuming {output.name} with target={current['job']['updates']} updates; "
        "retaining saved checkpoint identity and optimizer progress"
    )
    return saved
