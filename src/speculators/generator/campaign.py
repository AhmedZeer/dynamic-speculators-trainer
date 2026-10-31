"""Serial, training-only campaign orchestration. No vLLM entrypoints are used."""

import gc
import json
import os
import shutil
import time

import torch
from filelock import FileLock

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.progress import report
from speculators.generator.campaign_cache import CampaignCache
from speculators.generator.campaign_config import DOMAINS, SCALES
from speculators.generator.campaign_data import CampaignCorpus
from speculators.generator.campaign_inputs import campaign_banks, prepare_inputs
from speculators.generator.campaign_inventory import working_sets
from speculators.generator.campaign_resume import (
    record_validation_change,
    resume_identity,
)
from speculators.generator.campaign_train import run_training
from speculators.generator.config import ExperimentConfig
from speculators.generator.engine import Runtime, begin_run, run_identity
from speculators.generator.tracking import ExperimentTracker, log_completed

COLORS = {
    1: "#174a7e",
    2: "#17633b",
    4: "#9b361e",
}
ADAPTATION_ARMS = ("fresh_lora",)
ADAPTATION_COLORS = {
    "fresh_lora": "#bd6014",
}


def subset_names(numbers):
    return [f"{domain}-{number:05d}" for domain in DOMAINS for number in numbers]


def jobs(cfg):
    planned = []
    for scale in SCALES:
        planned.append(
            {
                "kind": "pretraining",
                "stage": "pretraining",
                "condition": "last",
                "label": f"scale-{scale}-direct",
                "arm": "direct",
                "objective": "direct",
                "scale": scale,
                "subsets": subset_names(range(scale)),
                "validation_subsets": subset_names([0]),
                "updates": cfg.optimization.pretraining_updates,
                "training_examples": len(DOMAINS) * scale * cfg.expected_train_examples,
                "color": COLORS[scale],
            }
        )
    for arm in ADAPTATION_ARMS:
        job = {
            "kind": "pretraining",
            "stage": "adaptation",
            "condition": "last",
            "label": arm,
            "arm": arm,
            "objective": "direct",
            "subsets": subset_names(range(2, 6)),
            "validation_subsets": subset_names(range(2, 6)),
            "updates": cfg.adaptation_updates,
            "training_examples": len(DOMAINS) * 4 * cfg.expected_train_examples,
        }
        planned.append(job)
    return planned


def job_config(cfg, job, *, staging_dir=None):
    values = {key: getattr(cfg, key) for key in ExperimentConfig.model_fields}
    values.update(
        bank_config=cfg.input_root / "bank_math.json",
        additional_bank_configs=[
            cfg.input_root / f"bank_{d}.json" for d in DOMAINS if d != "math"
        ],
        experiment_subsets=list(
            dict.fromkeys(job["subsets"] + job["validation_subsets"])
        ),
        activation_local_dir=None,
        activation_cache_gib=0,
        bank_subset="math-00000",
        conditioning_subset="math-00001",
        staging_dir=staging_dir or cfg.staging_dir,
    )
    values["wandb"] = cfg.wandb.model_copy(deep=True)
    if cfg.wandb.layout == "sections":
        values["wandb"].group = {
            "pilot": "pilot",
            "pretraining": "experiment-1",
            "adaptation": "experiment-2",
        }[job["stage"]]
        if cfg.training_cohort_size is not None and cfg.wandb.group:
            values["wandb"].group = cfg.wandb.group
        return ExperimentConfig.model_validate(values)
    values["wandb"].group = f"{cfg.wandb.group or cfg.output_root.name}-{job['stage']}"
    if job["stage"] == "adaptation":
        values["wandb"].pretraining_project = (
            cfg.wandb.projects["adaptation"] + "-campaign"
        )
    return ExperimentConfig.model_validate(values)


def _source(cfg, job):
    if "source_job" not in job:
        return job
    source = cfg.output_root / "pretraining" / job["source_job"] / "checkpoint.pt"
    result_path = source.parent / "result.json"
    if not result_path.is_file():
        raise ValueError(
            f"Complete direct pretraining before adaptation: {result_path}"
        )
    result = json.loads(result_path.read_text())
    source_identity = json.loads(
        (source.parent / "resolved_experiment.json").read_text()
    )
    expected = subset_names(range(int(job["arm"][-1])))
    if (
        result["job"]["subsets"] != expected
        or source_identity["job"]["objective"] != "direct"
    ):
        raise ValueError("Pretraining source is not the requested direct campaign arm")
    checksum = file_digest(source)
    if checksum != result["checkpoint_sha256"]:
        raise ValueError("Completed source checkpoint checksum changed")
    return {
        **job,
        "pretrained_path": str(source.resolve()),
        "source_sha256": checksum,
        "source_identity": digest(source_identity),
    }


def run_one(  # noqa: C901 -- shared online/offline run lifecycle
    cfg,
    original_job,
    *,
    cache_root=None,
    staging_dir=None,
    download_settings=None,
    local_inputs=None,
):
    started = time.perf_counter()
    job = _source(cfg, original_job)
    local_cfg = job_config(cfg, job, staging_dir=staging_dir)
    if job.get("tracking_group"):
        local_cfg.wandb.group = job["tracking_group"]
    output = cfg.output_root / job["stage"] / job.get("profile", "") / job["label"]
    online = cfg.activation_source == "vllm"
    if online:
        from speculators.generator.local_campaign import (  # noqa: PLC0415
            load_local_inputs,
        )
        from speculators.generator.online_activations import (  # noqa: PLC0415
            ActivationRAM,
            OnlineActivations,
            OnlineCorpus,
        )

        local_inputs = local_inputs or load_local_inputs(cfg)
        bank = local_inputs.banks["math"]
        if local_inputs.ram_cache is None:
            local_inputs.ram_cache = ActivationRAM(
                cfg.online.ram_cache_gib,
                [
                    i
                    for subset in local_inputs.subsets.values()
                    for i in subset["validation_indices"]
                ],
            )
        cache = OnlineActivations(
            cfg, local_inputs.rows, bank, ram=local_inputs.ram_cache
        )
    else:
        bank = BankConfig.load(local_cfg.bank_config)
        cache = CampaignCache(
            cache_root or cfg.staging_dir / "activation-files",
            download_settings or cfg.downloads,
        )
    runtime = None
    try:
        corpus = (
            OnlineCorpus(local_cfg, bank, cache, local_inputs)
            if online
            else CampaignCorpus(local_cfg, bank, cache)
        )
        for name in job["subsets"]:
            if len(corpus.indices(name)) != cfg.expected_train_examples:
                raise ValueError(f"Unexpected training membership size: {name}")
        for name in job["validation_subsets"]:
            if len(corpus.indices(name, True)) != cfg.expected_validation_examples:
                raise ValueError(f"Unexpected validation membership size: {name}")
        identity = run_identity(local_cfg, bank, corpus, job)
        identity["campaign"] = {
            "bank_revision": cfg.bank_revision,
            "trajectory_revision": bank.hub.trajectories_revision,
            "episode_schedule_version": 1,
        }
        if online:
            identity["campaign"] = {
                "trajectory_sources": local_inputs.identity["sources"],
                "activation_source": "vllm",
                "episode_schedule_version": 1,
            }
        identity = resume_identity(output, identity)
        if not begin_run(output, identity):
            result = json.loads((output / "result.json").read_text())
            log_completed(local_cfg, corpus, job, output, identity, result)
            return result
        record_validation_change(output, local_cfg)
        write_json(output / "configuration.json", local_cfg.model_dump(mode="json"))
        if online and cfg.trajectory_recipe is not None:
            for name in ("config.json", "original_config.json", "preparation.json"):
                source = cfg.online.drafter_model_path / name
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Missing native drafter provenance: {source}"
                    )
                write_json(
                    output / "provenance" / "drafter" / name,
                    json.loads(source.read_text()),
                )
        runtime_args = {}
        if online:
            cache.check_server()
            if cfg.online.compiler_cache_dir is not None:
                compiler_cache = cfg.online.compiler_cache_dir.expanduser().resolve()
                compiler_cache.mkdir(parents=True, exist_ok=True)
                # Torch may populate its /tmp default during package import.
                # Apply the configured persistent location before model execution.
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(compiler_cache)
                os.environ["TRITON_CACHE_DIR"] = str(compiler_cache / "triton")
            compiler_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
            report(f"Compiler cache: {compiler_cache or 'PyTorch default'}")
            runtime_args["model_paths"] = {
                "target": cfg.online.target_model_path,
                "drafter": cfg.online.drafter_model_path,
            }
        runtime = Runtime(
            local_cfg, bank, corpus_factory=lambda _cfg, _bank: corpus, **runtime_args
        )
        if online:
            runtime.compiler_cache_dir = compiler_cache
        return run_training(
            runtime,
            job,
            output,
            identity,
            lookahead=cfg.downloads.lookahead_updates,
            started=started,
        )
    finally:
        cache.close()
        del runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def campaign_identity(cfg):
    values = cfg.model_dump(mode="json")
    if values.get("trajectory_recipe") is None:
        values.pop("trajectory_recipe", None)
    if cfg.activation_source == "hub":
        for key in (
            "activation_source",
            "data_path",
            "trajectories_path",
            "bank_release",
            "online",
        ):
            values.pop(key)
    else:
        values.pop("online")
        values.pop("bank_revision")
    # Operational cache/concurrency/W&B settings may change after the pilot.
    for key in (
        "downloads",
        "wandb",
        "staging_dir",
        "activation_prefetch_workers",
        "preprocessing_workers",
        "factor_cache_mib",
        "activation_cache_gib",
        "activation_cache_reserve_gib",
        "heatmap_n_workers",
    ):
        values.pop(key)
    return digest(values)


def pilot_profile(cfg):
    if cfg.activation_source == "vllm":
        return digest(
            {
                "online": cfg.online.model_dump(mode="json"),
                "updates": cfg.pilot.updates,
                "version": 2,
            }
        )[:12]
    return digest(
        {
            "downloads": cfg.downloads.model_dump(exclude={"workers"}),
            "pilot": cfg.pilot.model_dump(),
            "activation_prefetch_workers": cfg.activation_prefetch_workers,
            "preprocessing_workers": cfg.preprocessing_workers,
            "factor_cache_mib": cfg.factor_cache_mib,
            "version": 1,
        }
    )[:12]


def pilot(cfg):
    try:
        return _pilot(cfg)
    except (RuntimeError, OSError, ValueError) as exc:
        write_json(
            cfg.output_root / "pilot" / "report.json",
            {
                "campaign_identity": campaign_identity(cfg),
                "passed": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "note": (
                    "Inspect individual pilot jobs; completed pairs remain resumable."
                ),
            },
        )
        raise


def _pilot(cfg):
    """Cold/warm application-cache training pairs; never launches serving."""
    results = []
    profile = pilot_profile(cfg)
    for workers in cfg.pilot.worker_counts:
        for arm, objective in (
            ("direct", "direct"),
            ("fresh_lora", "direct"),
        ):
            sandbox = cfg.staging_dir / "pilot" / profile / f"workers-{workers}-{arm}"
            for temperature in ("cold", "warm"):
                job = {
                    "kind": "pretraining",
                    "stage": "pilot",
                    "condition": "last",
                    "label": f"workers-{workers}-{arm}-{temperature}",
                    "arm": arm,
                    "objective": objective,
                    "subsets": subset_names([0]),
                    "validation_subsets": subset_names([0]),
                    "updates": cfg.pilot.updates,
                    "pilot": True,
                    "profile": profile,
                    "workers": workers,
                    "cache_temperature": temperature,
                }
                output = cfg.output_root / "pilot" / profile / job["label"]
                if temperature == "cold" and not output.exists() and sandbox.exists():
                    # Only this pilot's owned cache/summary folder is removed.
                    shutil.rmtree(sandbox)
                result = run_one(
                    cfg,
                    job,
                    cache_root=sandbox / "activation-files",
                    staging_dir=sandbox,
                    download_settings=cfg.downloads.model_copy(
                        update={"workers": workers}
                    ),
                )
                results.append(result)
            # Keep reports/checkpoints, release large disposable pilot inputs.
            shutil.rmtree(sandbox, ignore_errors=True)
    direct = [r for r in results if r["job"]["arm"] == "direct"]
    scores = {
        w: sum(r["training_seconds"] for r in direct if r["job"]["workers"] == w)
        for w in cfg.pilot.worker_counts
    }
    selected = min(scores, key=scores.get)
    warm = [r for r in results if r["job"]["cache_temperature"] == "warm"]
    fractions = {
        r["job"]["label"]: r["data_wait_seconds"] / max(r["training_seconds"], 1e-9)
        for r in warm
    }
    report_data = {
        "campaign_identity": campaign_identity(cfg),
        "selected_download_workers": selected,
        "profile": profile,
        "max_data_wait_fraction": cfg.pilot.max_data_wait_fraction,
        "warm_data_wait_fractions": fractions,
        "passed": all(
            v <= cfg.pilot.max_data_wait_fraction
            for k, v in fractions.items()
            if k.startswith(f"workers-{selected}-")
        ),
        "cold_definition": (
            "empty application activation and summary cache; "
            "upstream HF/Xet/OS caches may be warm"
        ),
        "working_sets": working_sets(cfg, jobs(cfg)),
        "results": results,
    }
    projections = {}
    for job in jobs(cfg):
        arm = "fresh_lora" if job["arm"] == "fresh_lora" else job["objective"]
        observed = [
            r
            for r in results
            if r["job"]["arm"] == arm and r["job"]["workers"] == selected
        ]
        projections[job["label"]] = {
            r["job"]["cache_temperature"] + "_seconds": r["training_seconds"]
            / cfg.pilot.updates
            * job["updates"]
            for r in observed
        }
    report_data["projected_training_seconds"] = projections
    report_data["projection_note"] = (
        "Linear pilot extrapolation, excluding model startup, validation, "
        "and checkpoint time. "
        "Larger datasets may exceed the cache and cause additional downloads."
    )
    write_json(cfg.output_root / "pilot" / "report.json", report_data)
    report(
        f"Training-only pilot: passed={report_data['passed']}; "
        f"selected workers={selected}"
    )
    return report_data


def plot_comparisons(cfg, results, stage):
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    root = cfg.output_root / stage
    fig, ax = plt.subplots(figsize=(9, 5))
    rows = []
    for result in results:
        job = result["job"]
        history = result["history"]
        ax.plot(
            [r["update"] for r in history],
            [r["score"] for r in history],
            label=job["label"],
            color=job.get("color", ADAPTATION_COLORS.get(job["arm"])),
        )
        rows.extend(
            {
                "run": job["label"],
                "update": r["update"],
                "offline_agreement": r["score"],
                "color": job.get("color"),
            }
            for r in history
        )
    ax.set(
        xlabel="Optimizer updates",
        ylabel="Offline agreement (full_acc_0), context 4",
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=160)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for result in results:
        recorded = [r for r in result["history"] if "training_seconds" in r]
        job = result["job"]
        ax.plot(
            [r["training_seconds"] for r in recorded],
            [r["score"] for r in recorded],
            label=job["label"],
            color=job.get("color", ADAPTATION_COLORS.get(job["arm"])),
        )
    ax.set(
        xlabel="Measured cumulative training seconds",
        ylabel="Offline agreement (full_acc_0), context 4",
        ylim=(0, 1),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "comparison_by_time.png", dpi=160)
    plt.close(fig)
    write_json(root / "comparison.json", rows)
    if cfg.wandb.enabled:
        job = {
            "kind": "pretraining",
            "stage": stage,
            "condition": "last",
            "label": stage + "-comparison",
            "subsets": [],
        }
        local_cfg = job_config(cfg, results[0]["job"])
        identity = {"rows": rows, "metric": "offline agreement", "stage": stage}
        with ExperimentTracker(
            local_cfg, job, root / "comparison", identity
        ) as tracker:
            tracker.log(
                {
                    "comparison/curves": tracker.module.Image(
                        str(root / "comparison.png")
                    ),
                    "comparison/curves_by_training_seconds": tracker.module.Image(
                        str(root / "comparison_by_time.png")
                    ),
                    "comparison/data": tracker.module.Table(
                        columns=["run", "update", "offline_agreement", "color"],
                        data=[
                            [
                                r[k]
                                for k in ("run", "update", "offline_agreement", "color")
                            ]
                            for r in rows
                        ],
                    ),
                }
            )


def run_campaign(cfg, stage="all", dry_run=False):  # noqa: C901 -- serial stage orchestration
    if cfg.activation_source == "vllm":
        from speculators.generator.online_campaign import (  # noqa: PLC0415
            run_online_campaign,
        )

        return run_online_campaign(cfg, stage, dry_run)
    planned = jobs(cfg)
    manifest = {
        "version": 1,
        "campaign_identity": campaign_identity(cfg),
        "configuration": cfg.model_dump(mode="json"),
        "jobs": planned,
        "metric": "offline agreement at context 4; no decoding inference",
        "activation_source": "pinned Hugging Face dumps, fetched on demand",
    }
    if dry_run:
        return manifest
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cfg.output_root / ".campaign.lock"), timeout=0):
        saved = cfg.output_root / "campaign.json"
        if (
            saved.exists()
            and json.loads(saved.read_text())["campaign_identity"]
            != manifest["campaign_identity"]
        ):
            raise ValueError("Campaign configuration changed; choose a new output_root")
        write_json(saved, manifest)
        if stage in ("prepare", "all"):
            prepare_inputs(cfg)
        else:
            # Fail locally before any model/GPU allocation if preparation is absent.
            campaign_banks(cfg)
            if not all((cfg.input_root / f"bank_{d}.json").is_file() for d in DOMAINS):
                raise FileNotFoundError("Run generator campaign --stage prepare first")
        if stage == "prepare":
            manifest["working_sets"] = working_sets(cfg, planned)
            write_json(cfg.output_root / "campaign.json", manifest)
            return manifest
        if stage in ("pilot", "all"):
            pilot_report = pilot(cfg)
            if stage == "pilot":
                return pilot_report
        else:
            pilot_path = cfg.output_root / "pilot" / "report.json"
            if not pilot_path.exists():
                raise FileNotFoundError(
                    "Run generator campaign --stage pilot before full training"
                )
            pilot_report = json.loads(pilot_path.read_text())
        if (
            pilot_report["campaign_identity"] != campaign_identity(cfg)
            or pilot_report.get("profile") != pilot_profile(cfg)
            or not pilot_report["passed"]
        ):
            raise ValueError(
                "Pilot did not meet this campaign's data-wait target; "
                "inspect pilot/report.json"
            )
        cfg = cfg.model_copy(deep=True)
        cfg.downloads.workers = pilot_report["selected_download_workers"]
        results = []
        for phase in ("pretraining", "adaptation"):
            if stage not in (phase, "all"):
                continue
            current = [run_one(cfg, job) for job in planned if job["stage"] == phase]
            write_json(cfg.output_root / phase / "results.json", current)
            plot_comparisons(cfg, current, phase)
            results.extend(current)
        return {"runs": len(results), "output_root": str(cfg.output_root)}
