"""Conditional LoRA generator training, serving and experiment commands."""

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from speculators.generator.config import ExperimentConfig

app = typer.Typer(
    help="Generator experiments, inference export, and serial vLLM benchmarks."
)
ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


@app.command(name="train-structure")
def train_structure_command(
    config: ConfigOption,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Train one module/architecture config with the original saved-response loop."""
    import json  # noqa: PLC0415

    from speculators.generator.structure_study import (  # noqa: PLC0415
        StructureStudy,
        train_structure,
    )

    typer.echo(
        json.dumps(
            train_structure(StructureStudy.load(config), dry_run=dry_run), indent=2
        )
    )


@app.command(name="prepare-experiments")
def prepare_experiments_command(
    config: ConfigOption,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Freeze study data and training-only conditioning donors; no training."""
    import json  # noqa: PLC0415

    from speculators.generator.study_config import GeneratorStudy  # noqa: PLC0415
    from speculators.generator.study_inputs import prepare_study  # noqa: PLC0415

    typer.echo(
        json.dumps(
            prepare_study(GeneratorStudy.load(config), dry_run=dry_run), indent=2
        )
    )


@app.command(name="train-experiment")
def train_experiment_command(
    config: ConfigOption,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Run exactly one configured generator/static-LoRA training experiment."""
    import json  # noqa: PLC0415

    from speculators.generator.study import train_study  # noqa: PLC0415
    from speculators.generator.study_config import GeneratorStudy  # noqa: PLC0415

    typer.echo(
        json.dumps(
            train_study(GeneratorStudy.load(config), resume=resume, dry_run=dry_run),
            indent=2,
        )
    )


@app.command(name="train-feedback")
def train_feedback_command(
    config: ConfigOption,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    extraction_concurrency: Annotated[
        int, typer.Option("--extraction-concurrency", min=0)
    ] = 0,
    ram_cache_gib: Annotated[
        float | None, typer.Option("--ram-cache-gib", min=0.1)
    ] = None,
):
    """Continue original saved-response training, with optional feedback."""
    import json  # noqa: PLC0415

    from speculators.generator.feedback_config import FeedbackStudy  # noqa: PLC0415
    from speculators.generator.feedback_training import train_feedback  # noqa: PLC0415

    typer.echo(
        json.dumps(
            train_feedback(
                FeedbackStudy.load(config),
                resume=resume,
                dry_run=dry_run,
                extraction_concurrency=extraction_concurrency,
                ram_cache_gib=ram_cache_gib,
            ),
            indent=2,
        )
    )


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


class TrainingArm(str, Enum):
    generator = "generator"
    lora = "lora"


@app.command(name="train-full-data")
def train_full_data_command(  # noqa: PLR0917
    config: ConfigOption,
    output_root: Annotated[Path, typer.Option("--output-root")],
    hub_repo: Annotated[str, typer.Option("--hub-repo")],
    provenance_dir: Annotated[Path, typer.Option("--vllm-provenance-dir")],
    epochs: Annotated[int, typer.Option("--epochs", min=1)] = 10,
    max_updates: Annotated[
        int | None,
        typer.Option(
            "--max-updates", min=1, help="Stop after this many optimizer updates"
        ),
    ] = None,
    arm: Annotated[TrainingArm, typer.Option("--arm")] = TrainingArm.generator,
    save_every_epochs: Annotated[
        int, typer.Option("--save-every-epochs", min=1, hidden=True)
    ] = 2,
    hub_upload_interval: Annotated[
        int | None,
        typer.Option(
            "--hub-upload-interval",
            min=1,
            help="Optimizer steps between Hub backups; default from config",
        ),
    ] = None,
    private: Annotated[bool, typer.Option("--private")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
):
    """Train a generator or static LoRA on all local subsets with Hub backups."""
    import json  # noqa: PLC0415

    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415
    from speculators.generator.epoch_training import train_full_data  # noqa: PLC0415

    result = train_full_data(
        CampaignConfig.load(config),
        output_root,
        hub_repo,
        provenance_dir,
        epochs=epochs,
        max_updates=max_updates,
        arm=arm.value,
        save_every_epochs=save_every_epochs,
        hub_upload_interval=hub_upload_interval,
        private=private,
        dry_run=dry_run,
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
        typer.Option("--categories", help="Comma-separated dataset categories or all."),
    ] = "all",
    seed: Annotated[int, typer.Option("--seed")] = 42,
    dataset: Annotated[
        str,
        typer.Option("--dataset", help="nemotron or RedHatAI/speculator_benchmarks"),
    ] = "nemotron",
    prompt_format: Annotated[
        str, typer.Option("--prompt-format", help="Red Hat prompts: auto, raw, or chat")
    ] = "auto",
):
    """Freeze Nemotron validation or Red Hat prompts for HTTP benchmarks."""
    import json  # noqa: PLC0415

    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415
    from speculators.generator.serving.validation import (  # noqa: PLC0415
        export_validation,
    )

    exporter = export_validation
    extra = {}
    if dataset == "RedHatAI/speculator_benchmarks":
        from speculators.generator.serving.external_validation import (  # noqa: PLC0415
            export_external_validation,
        )

        exporter = export_external_validation
        extra["prompt_format"] = prompt_format
    elif dataset != "nemotron":
        raise typer.BadParameter(
            "Dataset must be nemotron or RedHatAI/speculator_benchmarks"
        )
    result = exporter(
        CampaignConfig.load(config),
        output,
        tokenizer_path=tokenizer_path,
        max_tokens=max_tokens,
        max_samples=max_samples,
        categories=None
        if categories == "all"
        else [s.strip() for s in categories.split(",")],
        seed=seed,
        **extra,
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
