"""Offline conditional LoRA generator experiments."""

from pathlib import Path
from typing import Annotated

import typer

from speculators.generator.config import ExperimentConfig

app = typer.Typer(
    help="Conditioning ablations, bank heatmaps, and one-epoch adaptation."
)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def _run(config, stage):
    from speculators.generator import workflow  # noqa: PLC0415

    cfg = ExperimentConfig.load(config)
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
def heatmap(config: ConfigOption):
    """Train and evaluate a generator per seed-group/stride bank selection."""
    _run(config, "heatmap")


@app.command()
def adaptation(config: ConfigOption):
    """Compare pretrained/fresh generators and fresh/transferred ordinary LoRAs."""
    _run(config, "adaptation")
