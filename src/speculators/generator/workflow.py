"""Ordered conditioning, bank-size, and adaptation experiments."""

import csv
import json
import sys
from pathlib import Path

from speculators.bank.artifacts import digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.progress import report
from speculators.bank.runner import TrainingRun, run_parallel
from speculators.generator.config import CONDITIONS
from speculators.generator.data import Corpus, discover_snapshots, filter_snapshots
from speculators.generator.engine import Runtime, begin_run, run_identity, run_job
from speculators.generator.tracking import (
    log_completed,
    log_heatmap_summary,
    wandb_module,
)

MIDPOINT = 0.5


def prepare(cfg):
    bank = BankConfig.load(cfg.bank_config)
    runtime = Runtime(cfg, bank)
    write_json(
        cfg.output_root / "prepared" / "configuration.json", cfg.model_dump(mode="json")
    )
    report(f"Prepared compact context cache: {runtime.corpus.local}")


def execute(cfg, jobs):
    bank = BankConfig.load(cfg.bank_config)
    corpus = Corpus(cfg, bank)
    pending = []
    if cfg.wandb.enabled:
        wandb_module()
    for job, path in jobs:
        identity = run_identity(cfg, bank, corpus, job)
        if begin_run(path, identity):
            pending.append((job, path))
        else:
            log_completed(
                cfg,
                corpus,
                job,
                path,
                identity,
                json.loads((path / "result.json").read_text()),
            )
    if pending and cfg.n_workers == 1:
        runtime = Runtime(cfg, bank)
        for job, path in pending:
            run_job(runtime, job, path)
    elif pending:
        # Prepare once before spawning; workers only load the compact cache.
        prepare(cfg)
        runs = []
        for number, (job, path) in enumerate(pending, 1):
            request = path / "job.json"
            write_json(
                request,
                {
                    "configuration": cfg.model_dump(mode="json"),
                    "job": job,
                    "output": str(path.resolve()),
                },
            )
            subset = (
                cfg.bank_subset if job["kind"] == "heatmap" else cfg.conditioning_subset
            )
            runs.append(
                TrainingRun(
                    number,
                    len(pending),
                    f"{subset}/{path.name}",
                    cfg.seed,
                    len(corpus.indices(subset)),
                    len(corpus.indices(subset, True)),
                    path,
                    [
                        sys.executable,
                        "-m",
                        "speculators.generator.worker",
                        str(request.resolve()),
                    ],
                )
            )
        run_parallel(runs, cfg.n_workers)
    return [json.loads((path / "result.json").read_text()) for _, path in jobs]


def choose(results):
    # Stable input ordering breaks ties, making reruns reproducible.
    return max(results, key=lambda value: value["score"])


def conditioning(cfg):
    root = cfg.output_root / "conditioning"
    jobs = [
        ({"kind": "conditioning", "condition": name}, root / name)
        for name in CONDITIONS
    ]
    results = execute(cfg, jobs)
    winner = choose(results)
    write_json(
        root / "selection.json",
        {
            "condition": winner["job"]["condition"],
            "score": winner["score"],
            "selection_epoch": cfg.optimization.conditioning_epochs,
            "results": results,
        },
    )
    report(
        f"Selected conditioning: {winner['job']['condition']}; "
        f"score={winner['score']:.6f}"
    )
    return results


def selected_condition(cfg):
    if cfg.condition is not None:
        report(f"Using explicit conditioning representation: {cfg.condition}")
        return cfg.condition
    path = cfg.output_root / "conditioning" / "selection.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No conditioning selection at {path}. Pass --condition "
            "last|last_mean|projected, set condition in the YAML, "
            "or run generator conditioning for automatic selection."
        )
    # Check all upstream identities before allowing a changed config to reuse a winner.
    bank = BankConfig.load(cfg.bank_config)
    corpus = Corpus(cfg, bank)
    value = json.loads(path.read_text())
    for result in value["results"]:
        run_path = Path(result["checkpoint_path"]).parent
        saved = json.loads((run_path / "resolved_experiment.json").read_text())
        if saved != run_identity(cfg, bank, corpus, result["job"]):
            raise ValueError(
                "Conditioning selection belongs to a different configuration"
            )
    return value["condition"]


def heatmap(cfg):
    condition = selected_condition(cfg)
    bank = BankConfig.load(cfg.bank_config)
    entries = discover_snapshots(bank.output_root, cfg.bank_subset)
    seeds = sorted({seed for group in cfg.heatmap.seed_groups for seed in group})
    missing = set(seeds) - {e["seed"] for e in entries}
    if missing:
        raise ValueError(f"Missing bank seeds: {sorted(missing)}")
    start = max(
        min(e["collection_step"] for e in entries if e["seed"] == seed)
        for seed in seeds
    )
    stop = min(
        max(e["collection_step"] for e in entries if e["seed"] == seed)
        for seed in seeds
    )
    window = [e for e in entries if start <= e["collection_step"] <= stop]
    root = cfg.output_root / "heatmap"
    jobs = []
    for stride in cfg.heatmap.strides:
        for group in cfg.heatmap.seed_groups:
            snapshots = filter_snapshots(window, group, stride, (start, stop))
            label = "-".join(str(seed) for seed in group)
            jobs.append(
                (
                    {
                        "kind": "heatmap",
                        "condition": condition,
                        "seeds": group,
                        "stride": stride,
                        "collection_window": [start, stop],
                        "snapshots": snapshots,
                    },
                    root / f"stride-{stride}" / f"seeds-{label}",
                )
            )
    results = execute(cfg, jobs)
    export_heatmaps(root, cfg, results)
    write_json(root / "selection.json", {"winner": choose(results), "results": results})
    log_heatmap_summary(cfg, results, root)
    return results


def export_heatmaps(root, cfg, results):
    columns = [
        " & ".join(f"seed-{s}" for s in group) for group in cfg.heatmap.seed_groups
    ]
    rows = []
    for result in results:
        job = result["job"]
        rows.append(
            {
                "stride": job["stride"],
                "seeds": "+".join(map(str, job["seeds"])),
                "checkpoint_count": len(job["snapshots"]),
                "score": result["score"],
                **{
                    f"full_acc_0_context_{size}": result["history"][-1]["contexts"][
                        str(size)
                    ]["full_acc_0"]
                    for size in cfg.context.evaluation_sizes
                },
            }
        )
    write_json(root / "results.json", rows)
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    try:
        import matplotlib as mpl  # noqa: PLC0415

        mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Heatmap export requires pip install 'speculators[generator]'"
        ) from exc
    for key in (
        "score",
        *(f"full_acc_0_context_{s}" for s in cfg.context.evaluation_sizes),
    ):
        matrix = [
            [
                next(
                    row[key]
                    for row in rows
                    if row["stride"] == stride
                    and row["seeds"] == "+".join(map(str, group))
                )
                for group in cfg.heatmap.seed_groups
            ]
            for stride in cfg.heatmap.strides
        ]
        fig, ax = plt.subplots(figsize=(max(8, len(columns) * 2), 4))
        plot = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(columns)), columns, rotation=25, ha="right")
        ax.set_yticks(range(len(matrix)), cfg.heatmap.strides)
        ax.set_ylabel("Collection stride (optimizer updates)")
        ax.set_title("Mean full_acc_0 over contexts" if key == "score" else key)
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                count = rows[row_index * len(columns) + column_index][
                    "checkpoint_count"
                ]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.4f}\nn={count}",
                    ha="center",
                    va="center",
                    color="white" if value < MIDPOINT else "black",
                )
        fig.colorbar(plot, ax=ax, label="Offline draft agreement")
        fig.tight_layout()
        fig.savefig(root / f"{key}.png", dpi=180)
        fig.savefig(root / f"{key}.pdf")
        plt.close(fig)


def pretrained_source(cfg, bank, corpus, condition):
    if cfg.pretrained_checkpoint is not None:
        return explicit_pretrained_source(cfg, bank, corpus, condition)
    selection = cfg.output_root / "heatmap" / "selection.json"
    if not selection.exists():
        raise FileNotFoundError(
            f"No heatmap selection at {selection}. Pass --pretrained-checkpoint "
            "CHECKPOINT for the pretrained arm, or run generator heatmap."
        )
    winner = json.loads(selection.read_text())["winner"]
    source = Path(winner["checkpoint_path"])
    identity = json.loads((source.parent / "resolved_experiment.json").read_text())
    if identity != run_identity(cfg, bank, corpus, winner["job"]):
        raise ValueError("Heatmap winner belongs to a different configuration")
    source_job = winner["job"]
    if source_job["condition"] != condition:
        raise ValueError("Heatmap winner conditioning differs from --condition")
    return source, source_job, None


def explicit_pretrained_source(cfg, bank, corpus, condition):
    source = cfg.pretrained_checkpoint.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {source}")
    identity = json.loads((source.parent / "resolved_experiment.json").read_text())
    if identity["job"]["kind"] != "heatmap":
        raise ValueError("Pretrained arm requires a bank-pretrained heatmap checkpoint")
    if identity["job"]["condition"] != condition:
        raise ValueError("Pretrained checkpoint conditioning differs from --condition")
    if (
        identity["configuration"]["architecture"]
        != cfg.architecture.model_dump(mode="json")
        or identity["lora"] != bank.lora.model_dump(mode="json")
        or identity["bank_data"]["revisions"] != corpus.identity["revisions"]
    ):
        raise ValueError(
            "Pretrained checkpoint architecture, LoRA, or model revisions "
            "are incompatible"
        )
    stat = source.stat()
    metadata = {
        "identity": digest(identity),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    report(f"Using explicit pretrained generator checkpoint: {source}")
    return source, identity["job"], metadata


def adaptation(cfg):
    condition = selected_condition(cfg)
    bank = BankConfig.load(cfg.bank_config)
    corpus = Corpus(cfg, bank)
    source, source_job, source_metadata = pretrained_source(
        cfg, bank, corpus, condition
    )
    entries = [
        e
        for e in discover_snapshots(bank.output_root, cfg.bank_subset)
        if e["seed"] == cfg.transfer_seed
    ]
    if not entries:
        raise ValueError(f"No adapters for transfer_seed={cfg.transfer_seed}")
    transfer = max(entries, key=lambda e: e["collection_step"])
    jobs = []
    for arm in (
        "pretrained_generator",
        "fresh_generator",
        "fresh_lora",
        "transferred_lora",
    ):
        job = {"kind": "adaptation", "condition": condition, "arm": arm}
        if arm == "pretrained_generator":
            job.update(pretrained_path=str(source), pretrained_job=source_job)
            if source_metadata is not None:
                job["pretrained_source"] = source_metadata
        if arm == "transferred_lora":
            job.update(transfer_path=transfer["path"], transfer_entry=transfer["entry"])
        jobs.append((job, cfg.output_root / "adaptation" / arm))
    results = execute(cfg, jobs)
    write_json(cfg.output_root / "adaptation" / "results.json", results)
    return results
