"""Offline conditional LoRA generator experiments."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from speculators.generator.config import ExperimentConfig

app = typer.Typer(
    help="Conditioning ablations, bank heatmaps, and one-epoch adaptation."
)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


class ConditionName(str, Enum):
    last = "last"
    last_mean = "last_mean"
    projected = "projected"


ConditionOption = Annotated[
    ConditionName | None,
    typer.Option("--condition", help="Override automatic conditioning selection."),
]
CheckpointOption = Annotated[
    Path | None,
    typer.Option(
        "--pretrained-checkpoint",
        exists=True,
        dir_okay=False,
        help="Heatmap generator checkpoint for adaptation's pretrained arm.",
    ),
]


def _run(config, stage, condition=None, pretrained_checkpoint=None):
    from speculators.generator import workflow  # noqa: PLC0415

    cfg = ExperimentConfig.load(config)
    if condition is not None:
        cfg.condition = condition.value
    if pretrained_checkpoint is not None:
        cfg.pretrained_checkpoint = pretrained_checkpoint
    getattr(workflow, stage)(cfg)


@app.command()
def prepare(config: ConfigOption):
    """Prepare compact prompt summaries from existing saved hidden states."""
    _run(config, "prepare")


@app.command()
def conditioning(config: ConfigOption):
    """Compare three conditioning representations over five drafter-loss epochs."""
    _run(config, "conditioning")


@app.command()
def heatmap(config: ConfigOption, condition: ConditionOption = None):
    """Train and evaluate a generator per seed-group/stride bank selection."""
    _run(config, "heatmap", condition=condition)


@app.command()
def adaptation(
    config: ConfigOption,
    condition: ConditionOption = None,
    pretrained_checkpoint: CheckpointOption = None,
):
    """Compare pretrained/fresh generators and fresh/transferred ordinary LoRAs."""
    _run(
        config,
        "adaptation",
        condition=condition,
        pretrained_checkpoint=pretrained_checkpoint,
    )
