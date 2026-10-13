"""LoRA bank command group."""

import json
import subprocess
import sys
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
    resume_from_hf: Annotated[
        bool,
        typer.Option(
            "--resume-from-hf", help="Restore published inputs before resuming"
        ),
    ] = False,
):
    """Prepare selected target responses and their shared hidden-state cache."""
    report(
        f"prepare --stage {stage}; config={config.resolve()}; code={workflow.__file__}"
    )
    with stage_progress("Loading bank configuration"):
        cfg = BankConfig.load(config)
    result = workflow.prepare(cfg, stage, resume_from_hf=resume_from_hf)
    typer.echo(json.dumps({"root": result["root"], "subsets": len(result["subsets"])}))


@app.command()
def train(config: ConfigOption):
    """Train all configured subset/seed runs; resume compatible existing runs."""
    report(f"train; config={config.resolve()}; code={workflow.__file__}")
    with stage_progress("Loading bank configuration"):
        cfg = BankConfig.load(config)
    result = workflow.train(cfg)
    typer.echo(f"Collected {result['candidate_count']} adapter candidates")


@app.command()
def restore(
    config: ConfigOption,
    revision: Annotated[
        str, typer.Option("--revision", help="Pinned trajectories commit SHA")
    ],
    stage: Literal["select", "responses", "data", "hidden"] = "hidden",
):
    """Restore published artifacts; activation tensors download on demand."""
    from speculators.bank.config import HubArtifacts  # noqa: PLC0415
    from speculators.bank.hub import prepare as prepare_hub  # noqa: PLC0415

    cfg = BankConfig.load(config)
    if cfg.hub is None:
        raise typer.BadParameter("Config must enable Hub artifacts")
    cfg.hub = HubArtifacts.model_validate(
        {**cfg.hub.model_dump(), "mode": "consumer", "trajectories_revision": revision}
    )
    result = prepare_hub(cfg, stage)
    typer.echo(f"Restored {result['root']} through {stage}")


@app.command()
def serve(
    config: ConfigOption,
    mode: Literal["responses", "hidden"] = "responses",
    port: int = 8000,
):
    """Launch the pinned target and preserve server provenance; stop to switch modes."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    from speculators.provenance import find_package_repo  # noqa: PLC0415

    cfg = BankConfig.load(config)
    if not (cfg.output_root / "manifest.json").exists():
        raise typer.BadParameter(
            "Run prepare --stage select before starting the server"
        )
    manifest = workflow.select_data(cfg)
    target = snapshot_download(
        cfg.models.target, revision=manifest["revisions"]["target_sha"]
    )
    script = find_package_repo("speculators") / "scripts" / "launch_vllm.py"
    command = [
        sys.executable,
        str(script),
        "train" if mode == "hidden" else "responses",
        target,
        "--provenance-dir",
        str(cfg.output_root.resolve() / "provenance" / f"server-{mode}"),
        "--served-model-name",
        cfg.models.target,
        "--port",
        str(port),
        "--max-model-len",
        str(cfg.responses.context_length),
    ]
    if mode == "hidden":
        command.extend(
            [
                "--target-layer-ids",
                *map(str, cfg.hidden_states.layer_ids[:-1]),
                "--include-last-layer",
            ]
        )
    subprocess.run(command, check=True)  # noqa: S603 -- argv only


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
