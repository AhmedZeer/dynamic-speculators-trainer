"""Launch independent vLLM services and benchmark declarative model suites."""

import json
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="Managed vLLM and GuideLLM benchmark suites.")


@app.command(name="suite")
def suite_command(
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    output_root: Annotated[Path, typer.Option("--output-root")],
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Measure model/category pairs and write a combined throughput table."""
    from speculators.benchmark.suite import run_suite  # noqa: PLC0415

    result = run_suite(config, output_root, resume=resume, dry_run=dry_run)
    typer.echo(json.dumps(result, indent=2))
    if result.get("failed", 0):
        raise typer.Exit(code=1)
