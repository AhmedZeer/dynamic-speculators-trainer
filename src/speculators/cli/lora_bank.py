"""LoRA bank command group."""

import json
from pathlib import Path
from typing import Annotated, Literal

import typer

from speculators.bank import workflow
from speculators.bank.config import BankConfig
from speculators.bank.progress import report, stage_progress

app = typer.Typer(
    help="Prepare, collect, and inspect dataset-specific EAGLE3 LoRA banks."
)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


@app.command()
def prepare(
    config: ConfigOption,
    stage: Literal["select", "responses", "data", "hidden", "all"] = "all",
):
    """Prepare selected target responses and their shared hidden-state cache."""
    report(
        f"prepare --stage {stage}; config={config.resolve()}; code={workflow.__file__}"
    )
    with stage_progress("Loading bank configuration"):
        cfg = BankConfig.load(config)
    result = workflow.prepare(cfg, stage)
    typer.echo(json.dumps({"root": result["root"], "subsets": len(result["subsets"])}))


@app.command()
def train(config: ConfigOption):
    """Train all configured subset/seed runs; resume compatible existing runs."""
    report(f"train; config={config.resolve()}; code={workflow.__file__}")
    with stage_progress("Loading bank configuration"):
        cfg = BankConfig.load(config)
    result = workflow.train(cfg)
    typer.echo(f"Collected {result['candidate_count']} adapter candidates")


@app.command(name="inspect")
def inspect_command(config: ConfigOption):
    """Write bank.json with adapter compatibility and variation diagnostics."""
    result = workflow.inspect_bank(BankConfig.load(config))
    typer.echo(
        json.dumps(
            {"candidate_count": result["candidate_count"], "runs": result["runs"]},
            indent=2,
        )
    )
