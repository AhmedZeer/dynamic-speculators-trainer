"""Offline conditional LoRA generator experiments."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from speculators.generator.config import ExperimentConfig

app = typer.Typer(
    help="Generator experiments, inference export, and serial vLLM benchmarks."
)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


@app.command()
def campaign(
    config: ConfigOption,
    stage: Annotated[
        str, typer.Option(help="prepare, pilot, pretraining, adaptation, or all")
    ] = "all",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    data_path: Annotated[Path | None, typer.Option("--data-path")] = None,
    trajectories_path: Annotated[
        Path | None, typer.Option("--trajectories-path")
    ] = None,
    vllm_endpoint: Annotated[str | None, typer.Option("--vllm-endpoint")] = None,
):
    """Run direct pretraining and joint adaptation with offline validation."""
    import json  # noqa: PLC0415

    from speculators.generator.campaign import run_campaign  # noqa: PLC0415
    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415

    if stage not in ("prepare", "pilot", "pretraining", "adaptation", "all"):
        raise typer.BadParameter("Unknown campaign stage")
    cfg = CampaignConfig.load(config)
    if data_path is not None:
        cfg.data_path = data_path
    if trajectories_path is not None:
        cfg.trajectories_path = trajectories_path
    if vllm_endpoint is not None:
        cfg.online.endpoint = vllm_endpoint
    result = run_campaign(cfg, stage, dry_run)
    typer.echo(json.dumps(result, indent=2))


@app.command(name="train-lora-baseline")
def train_lora_baseline_command(
    config: ConfigOption,
    output_root: Annotated[Path, typer.Option("--output-root")],
    scale: Annotated[
        int, typer.Option("--scale", help="Match generator scale 1, 2, or 4")
    ] = 1,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Train ordinary LoRA with the generator's data and update budget."""
    import json  # noqa: PLC0415

    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415
    from speculators.generator.lora_baseline import train_matched_lora  # noqa: PLC0415

    result = train_matched_lora(
        CampaignConfig.load(config), scale, output_root, dry_run=dry_run
    )
    typer.echo(json.dumps(result, indent=2))


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
def pretrain(config: ConfigOption, condition: ConditionOption = None):
    """Pretrain a shared generator on selected subsets, including multiple banks."""
    _run(config, "pretrain", condition=condition)


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


@app.command(name="export")
def export_command(
    config: ConfigOption,
    run_dir: Annotated[Path, typer.Option("--run-dir", exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")],
):
    """Export completed generator/ordinary LoRA weights and training provenance."""
    from speculators.generator.serving.bundle import export_run  # noqa: PLC0415

    value = export_run(ExperimentConfig.load(config), run_dir, output)
    typer.echo(f"Exported {value['kind']} to {output}")


@app.command(name="push-to-hub")
def push_to_hub_command(
    run_dir: Annotated[Path, typer.Option("--run-dir", exists=True, file_okay=False)],
    repo_id: Annotated[str, typer.Option("--repo-id")],
    config: Annotated[Path | None, typer.Option("--config", exists=True)] = None,
    drafter_model_path: Annotated[
        Path | None, typer.Option("--drafter-model-path", exists=True, file_okay=False)
    ] = None,
    provenance_dir: Annotated[
        Path | None, typer.Option("--provenance-dir", exists=True, file_okay=False)
    ] = None,
    private: Annotated[bool, typer.Option("--private")] = False,
    include_training_checkpoint: Annotated[
        bool, typer.Option("--include-training-checkpoint")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Upload a completed run's inference weights and provenance to a model repo."""
    import json  # noqa: PLC0415

    from speculators.generator.publish import push_run  # noqa: PLC0415

    result = push_run(
        run_dir,
        repo_id,
        cfg=ExperimentConfig.load(config) if config is not None else None,
        drafter_model_path=drafter_model_path,
        provenance_dir=provenance_dir,
        private=private,
        include_training_checkpoint=include_training_checkpoint,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command(name="export-validation")
def export_validation_command(
    config: ConfigOption,
    output: Annotated[Path, typer.Option("--output")],
    tokenizer_path: Annotated[Path | None, typer.Option("--tokenizer-path")] = None,
    max_tokens: Annotated[int, typer.Option("--max-tokens", min=1)] = 256,
    max_samples: Annotated[
        int,
        typer.Option(
            "--max-samples", min=0, help="Maximum rows per domain; 0 uses all."
        ),
    ] = 0,
    categories: Annotated[
        str,
        typer.Option("--categories", help="Comma-separated Nemotron domains or all."),
    ] = "all",
    seed: Annotated[int, typer.Option("--seed")] = 42,
):
    """Freeze current campaign validation prompts for independent HTTP benchmarks."""
    import json  # noqa: PLC0415

    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415
    from speculators.generator.serving.validation import (  # noqa: PLC0415
        export_validation,
    )

    result = export_validation(
        CampaignConfig.load(config),
        output,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        max_samples=max_samples,
        categories=None
        if categories == "all"
        else [s.strip() for s in categories.split(",")],
        seed=seed,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command(name="check-endpoint")
def check_endpoint_command(
    target: Annotated[str, typer.Option("--target")],
    validation_manifest: Annotated[
        Path, typer.Option("--validation-manifest", exists=True)
    ],
    output: Annotated[Path, typer.Option("--output")],
    reference: Annotated[Path | None, typer.Option("--reference", exists=True)] = None,
    limit: Annotated[int, typer.Option("--limit", min=0)] = 0,
    seed: Annotated[int, typer.Option("--seed")] = 42,
):
    """Check decoding tokens and adapter cleanup against target-only results."""
    import json  # noqa: PLC0415

    from speculators.generator.serving.endpoint import check_endpoint  # noqa: PLC0415

    result = check_endpoint(
        target, validation_manifest, output, reference=reference, limit=limit, seed=seed
    )
    typer.echo(json.dumps(result, indent=2))


@app.command(name="benchmark")
def benchmark_command(
    config: ConfigOption,
    smoke: Annotated[
        bool, typer.Option("--smoke", help="Three prompts, one pass, 32 output tokens.")
    ] = False,
):
    """Compare matched decoding arms in separate, serial vLLM 0.31.0 workers."""
    from speculators.generator.serving.benchmark import (  # noqa: PLC0415
        BenchmarkConfig,
        benchmark,
    )

    results = benchmark(BenchmarkConfig.load(config), smoke=smoke)
    typer.echo(f"Completed {len(results)} decoding arms")
