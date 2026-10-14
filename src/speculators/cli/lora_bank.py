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


def cloud_config(config, scratch_root):
    cfg = BankConfig.load(config)
    if scratch_root is not None:
        cfg.output_root = scratch_root / cfg.dataset.category
        if cfg.hub is not None:
            cfg.hub.selection_root = scratch_root / "selection-s0001"
    return cfg


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
def publish_hidden(config: ConfigOption):
    """Publish completed local activations and free their storage; no vLLM requests."""
    cfg = BankConfig.load(config)
    if cfg.hub is None:
        raise typer.BadParameter("Publication requires a Hub-enabled bank config")
    workflow.prepare(cfg, "hidden", publish_only=True)


@app.command()
def compact_hidden(
    config: ConfigOption,
    release: Annotated[str, typer.Option("--release")],
    output_root: Annotated[Path, typer.Option("--output-root")],
):
    """Build training shards in a separate release from published data; no GPU."""
    from speculators.bank.compaction import compact  # noqa: PLC0415

    result = compact(BankConfig.load(config), release, output_root)
    typer.echo(f"Training config: {result}")


@app.command()
def stage_hidden(
    config: ConfigOption,
    local_dir: Annotated[Path, typer.Option("--local-dir")],
    subset: Annotated[list[str] | None, typer.Option("--subset")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 8,
):
    """Download selected subsets in parallel; consume raw or sharded files directly."""
    from speculators.bank.staging import stage_hidden as stage_files  # noqa: PLC0415

    path = stage_files(BankConfig.load(config), local_dir, subset or ["all"], workers)
    typer.echo(f"Training config: {path}")


@app.command()
def train(config: ConfigOption):
    """Train all configured subset/seed runs; resume compatible existing runs."""
    report(f"train; config={config.resolve()}; code={workflow.__file__}")
    with stage_progress("Loading bank configuration"):
        cfg = BankConfig.load(config)
    result = workflow.train(cfg)
    typer.echo(f"Collected {result['candidate_count']} adapter candidates")


@app.command()
def train_cloud(
    config: ConfigOption,
    scratch_root: Annotated[Path | None, typer.Option("--scratch-root")] = None,
    subset: Annotated[list[str] | None, typer.Option("--subset")] = None,
    download_workers: Annotated[int, typer.Option("--download-workers", min=1)] = 8,
    keep_activations: Annotated[bool, typer.Option("--keep-activations")] = False,
):
    """Restore inputs, train on scratch, publish each run, and skip published runs."""
    from speculators.bank.cloud import train_cloud as run_cloud  # noqa: PLC0415

    cfg = cloud_config(config, scratch_root)
    result = run_cloud(cfg, subset, download_workers, keep_activations)
    typer.echo(f"Published bank: {result['repo']}/releases/{result['release']}")


@app.command()
def publish_bank(
    config: ConfigOption,
    scratch_root: Annotated[Path | None, typer.Option("--scratch-root")] = None,
    subset: Annotated[list[str] | None, typer.Option("--subset")] = None,
):
    """Publish completed local LoRA runs, including snapshots and provenance."""
    from speculators.bank.cloud import publish_banks  # noqa: PLC0415

    publish_banks(cloud_config(config, scratch_root), subset)


@app.command()
def restore_bank(
    config: ConfigOption,
    revision: Annotated[str, typer.Option("--revision")],
    scratch_root: Annotated[Path | None, typer.Option("--scratch-root")] = None,
    subset: Annotated[list[str] | None, typer.Option("--subset")] = None,
):
    """Restore completed bank runs from a pinned bank-repository commit."""
    from speculators.bank.cloud import restore_banks  # noqa: PLC0415

    count = restore_banks(cloud_config(config, scratch_root), revision, subset)
    typer.echo(f"Restored {count} completed LoRA runs")


@app.command()
def restore(
    config: ConfigOption,
    revision: Annotated[
        str, typer.Option("--revision", help="Pinned trajectories commit SHA")
    ],
    stage: Literal["select", "responses", "data", "hidden"] = "hidden",
    scratch_root: Annotated[Path | None, typer.Option("--scratch-root")] = None,
):
    """Restore published artifacts; activation tensors download on demand."""
    from speculators.bank.config import HubArtifacts  # noqa: PLC0415
    from speculators.bank.hub import prepare as prepare_hub  # noqa: PLC0415

    cfg = cloud_config(config, scratch_root)
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
    resume_from_hf: Annotated[
        bool,
        typer.Option(
            "--resume-from-hf", help="Restore the published selection before launching"
        ),
    ] = False,
):
    """Launch the pinned target and preserve server provenance; stop to switch modes."""
    from huggingface_hub import HfApi, snapshot_download  # noqa: PLC0415

    from speculators.provenance import find_package_repo  # noqa: PLC0415

    cfg = BankConfig.load(config)
    if resume_from_hf:
        if cfg.hub is None:
            raise typer.BadParameter("--resume-from-hf requires a Hub-enabled config")
        from speculators.bank.hub import prepare as prepare_hub  # noqa: PLC0415

        consumer = cfg.model_copy(deep=True)
        consumer.hub.mode = "consumer"
        if consumer.hub.trajectories_revision == "main":
            consumer.hub.trajectories_revision = (
                HfApi()
                .repo_info(consumer.hub.trajectories_repo, repo_type="dataset")
                .sha
            )
        with stage_progress("Restoring published selection for server startup"):
            prepare_hub(consumer, "select")
    if not (cfg.output_root / "manifest.json").exists():
        raise typer.BadParameter(
            "Run prepare --stage select for new data, or pass --resume-from-hf "
            "to restore an existing published selection"
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
        str(
            max(cfg.responses.context_length, cfg.hidden_states.sequence_length + 1)
            if mode == "hidden"
            else cfg.responses.context_length
        ),
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
