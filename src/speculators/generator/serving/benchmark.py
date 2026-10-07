"""Matched, serial, fresh-process decoding benchmarks; no hidden-state cache IO."""

import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from datasets import load_from_disk
from huggingface_hub import snapshot_download
from pydantic import Field, model_validator

from speculators.bank.artifacts import digest, file_digest, load_subset, write_json
from speculators.bank.config import BankConfig, Settings
from speculators.bank.progress import report, stage_progress
from speculators.generator.serving.bundle import PROVENANCE, VLLM_COMMIT, read_bundle
from speculators.generator.serving.patch import PATCH_DIR, patch_sources
from speculators.generator.tracking import wandb_module
from speculators.provenance import (
    atomic_write,
    find_package_repo,
    git_diff,
    git_sha,
    package_versions,
)

ARMS = (
    "target_only",
    "eagle3_base",
    "fresh_lora",
    "transferred_lora",
    "fresh_generator",
    "pretrained_generator",
)


class BenchmarkWandb(Settings):
    enabled: bool = False
    mode: str = "offline"
    project: str = "eagle3-generator-vllm"
    entity: str | None = None


class BenchmarkConfig(Settings):
    bank_config: Path
    output_root: Path = Path("output/generator-vllm")
    staging_dir: Path = Path("/tmp/speculators-serving")  # noqa: S108
    bundles: dict[str, Path] = Field(default_factory=dict)
    arms: list[str] = Field(default_factory=lambda: list(ARMS))
    subset: str = "math-00001"
    prompts: int = Field(default=100, ge=1)
    warmups: int = Field(default=3, ge=0)
    repetitions: int = Field(default=3, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_model_len: int | None = Field(default=None, ge=2)
    speculative_tokens: int = Field(default=3, ge=1)
    seed: int = 42
    gpu_memory_utilization: float = Field(default=0.85, gt=0, lt=1)
    output_mismatch: Literal["error", "warn"] = "error"
    wandb: BenchmarkWandb = Field(default_factory=BenchmarkWandb)

    @model_validator(mode="after")
    def valid_arms(self):
        if (
            not self.arms
            or len(set(self.arms)) != len(self.arms)
            or set(self.arms) - set(ARMS)
        ):
            raise ValueError("Choose nonempty, distinct benchmark arms")
        if self.arms[0] != "target_only":
            raise ValueError("target_only must run first for output-agreement checks")
        missing = set(self.arms) - {"target_only", "eagle3_base"} - self.bundles.keys()
        if missing:
            raise ValueError(f"Missing exported arm bundles: {sorted(missing)}")
        if self.wandb.mode not in ("online", "offline", "disabled"):
            raise ValueError("Expected online/offline/disabled W&B mode")
        return self

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        values = yaml.safe_load(path.read_text())
        for key in ("bank_config", "output_root", "staging_dir"):
            if key in values and not Path(values[key]).is_absolute():
                values[key] = path.parent / values[key]
        values["bundles"] = {
            name: path.parent / value if not Path(value).is_absolute() else Path(value)
            for name, value in values.get("bundles", {}).items()
        }
        return cls.model_validate(values)


def benchmark_prompts(cfg, bank, manifest, subset):
    indices = subset["validation_indices"][: cfg.prompts]
    if len(indices) != cfg.prompts:
        raise ValueError("The selected validation subset has too few prompts")
    # Arrow reads only these rows. Full hidden-state tensors are never opened.
    rows = load_from_disk(manifest["data_path"]).select(indices).with_format(None)
    length = cfg.max_model_len or bank.responses.context_length
    cap = cfg.max_tokens or bank.responses.max_tokens
    prompts = []
    for index, row in zip(indices, rows, strict=True):
        boundary = row["prompt_length"]
        tokens = row["input_ids"][:boundary]
        if not 0 < boundary < length or len(tokens) != boundary:
            raise ValueError(f"Invalid or oversized prompt boundary: row {index}")
        prompts.append(
            {
                "row": index,
                "prompt_token_ids": tokens,
                "max_tokens": min(cap, length - boundary),
            }
        )
    return prompts


def scalar_metrics(metrics):
    values = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
        "accepted_per_position": [],
    }
    names = {
        "vllm:spec_decode_num_drafts": "num_drafts",
        "vllm:spec_decode_num_draft_tokens": "num_draft_tokens",
        "vllm:spec_decode_num_accepted_tokens": "num_accepted_tokens",
    }
    for metric in metrics:
        if metric.name in names:
            values[names[metric.name]] += float(metric.value)
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            existing = values["accepted_per_position"]
            existing.extend([0.0] * max(0, len(metric.values) - len(existing)))
            for i, value in enumerate(metric.values):
                existing[i] += float(value)
    return values


def metric_delta(before, after):
    result = {
        key: after[key] - before[key]
        for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    }
    left, right = before["accepted_per_position"], after["accepted_per_position"]
    result["accepted_per_position"] = [
        value - (left[i] if i < len(left) else 0.0) for i, value in enumerate(right)
    ]
    if any(
        result[key] < 0
        for key in ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    ):
        raise ValueError("Speculation counters reset during a measured request")
    return result


def summarize(requests):
    total_seconds = sum(row["request_seconds"] for row in requests)
    output_tokens = sum(len(row["output_token_ids"]) for row in requests)
    drafts = sum(row["num_drafts"] for row in requests)
    proposed = sum(row["num_draft_tokens"] for row in requests)
    accepted = sum(row["num_accepted_tokens"] for row in requests)
    positions = []
    for row in requests:
        positions.extend(
            [0.0] * max(0, len(row["accepted_per_position"]) - len(positions))
        )
        for i, value in enumerate(row["accepted_per_position"]):
            positions[i] += value
    return {
        "requests": len(requests),
        "output_tokens": output_tokens,
        "total_request_seconds": total_seconds,
        "mean_request_seconds": total_seconds / len(requests),
        "output_tokens_per_second": output_tokens / total_seconds,
        "num_drafts": drafts,
        "num_draft_tokens": proposed,
        "num_accepted_tokens": accepted,
        "acceptance_ratio": accepted / proposed if proposed else None,
        "accepted_length": 1 + accepted / drafts if drafts else None,
        "acceptance_per_position": [value / drafts for value in positions]
        if drafts
        else [],
        "generator_seconds": sum(row.get("generator_seconds", 0) for row in requests),
        "merge_seconds": sum(row.get("merge_seconds", 0) for row in requests),
        "restore_seconds": sum(row.get("restore_seconds", 0) for row in requests),
        "generator_invocations": sum(
            row.get("generator_invocations", 0) for row in requests
        ),
        "peak_gpu_allocated_bytes": max(
            row["peak_gpu_allocated_bytes"] for row in requests
        ),
    }


def compare_outputs(reference, result):
    """Compare every request; membership errors always invalidate the benchmark."""
    if len(reference["requests"]) != len(result["requests"]):
        raise ValueError("Benchmark arms have different request counts")
    mismatches = []
    for expected, actual in zip(reference["requests"], result["requests"], strict=True):
        if (expected["row"], expected["repetition"]) != (
            actual["row"],
            actual["repetition"],
        ):
            raise ValueError("Benchmark arms have different prompt membership/order")
        if expected["output_token_ids"] != actual["output_token_ids"]:
            target, draft = expected["output_token_ids"], actual["output_token_ids"]
            position = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(target, draft, strict=False))
                    if a != b
                ),
                min(len(target), len(draft)),
            )
            mismatches.append(
                {
                    "row": actual["row"],
                    "repetition": actual["repetition"],
                    "first_differing_position": position,
                    "target_length": len(target),
                    "arm_length": len(draft),
                    "target_token_id": target[position]
                    if position < len(target)
                    else None,
                    "arm_token_id": draft[position] if position < len(draft) else None,
                    "target_context_ids": target[max(0, position - 5) : position + 6],
                    "arm_context_ids": draft[max(0, position - 5) : position + 6],
                }
            )
    count = len(result["requests"])
    return {
        "status": "unverified" if mismatches else "matched",
        "compared_requests": count,
        "mismatched_requests": len(mismatches),
        "exact_match_fraction": (count - len(mismatches)) / count if count else None,
        "mismatches": mismatches,
    }


def check_outputs(reference, result):
    comparison = compare_outputs(reference, result)
    if comparison["mismatches"]:
        first = comparison["mismatches"][0]
        raise ValueError(
            f"Greedy output disagreement: row={first['row']}, "
            f"repetition={first['repetition']}; inspect before benchmarking"
        )


def record_output_comparison(cfg, arm, reference, result, destination):
    if reference is None:
        comparison = {
            "status": "reference",
            "compared_requests": 0,
            "mismatched_requests": 0,
            "exact_match_fraction": None,
            "mismatches": [],
        }
    else:
        comparison = compare_outputs(reference, result)
    result["output_equivalence"] = comparison
    result["summary"].update(
        output_equivalence=comparison["status"],
        compared_requests=comparison["compared_requests"],
        mismatched_requests=comparison["mismatched_requests"],
        exact_match_fraction=comparison["exact_match_fraction"],
    )
    write_json(destination / "result.json", result)
    diagnostic = destination / "output_disagreement.json"
    if comparison["mismatches"]:
        write_json(diagnostic, {"policy": cfg.output_mismatch, **comparison})
        message = (
            f"Greedy output disagreement: arm={arm}, "
            f"{comparison['mismatched_requests']}/{comparison['compared_requests']} "
            f"requests differ; output equivalence unverified; see {diagnostic}"
        )
        if cfg.output_mismatch == "error":
            raise ValueError(message)
        report(f"WARNING: {message}; continuing exploratory measurements")
    else:
        diagnostic.unlink(missing_ok=True)


def provenance(output, models):
    output.mkdir(parents=True, exist_ok=True)
    root = find_package_repo("speculators")
    header = "\n".join(
        [
            f"# Timestamp: {datetime.now(timezone.utc).isoformat()}",
            f"# Git SHA: {git_sha(root)}",
            *package_versions(),
        ]
    )
    atomic_write(
        output / "eval_command.txt", header + "\n" + shlex.join(sys.argv) + "\n"
    )
    atomic_write(output / "speculators.patch", git_diff(root) + "\n")
    shutil.copy2(PATCH_DIR / "vllm-0.31.0-generator.patch", output / "vllm.patch")
    for role, path in models.items():
        lines = [
            f"{file_digest(file)}  {file.name}"
            for file in sorted(Path(path).glob("*.safetensors"))
        ]
        if not lines:
            raise ValueError(f"No {role} checkpoint weights found at {path}")
        name = (
            "checkpoint_sha256.txt"
            if role == "target"
            else "drafter_checkpoint_sha256.txt"
        )
        atomic_write(output / name, "\n".join(lines) + "\n")


def log_result(cfg, arm, result, identity):
    if not cfg.wandb.enabled:
        return
    local = cfg.staging_dir / "wandb"
    local.mkdir(parents=True, exist_ok=True)
    run = wandb_module().init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        name=f"vllm-{cfg.subset}-{arm}-seed{cfg.seed}"
        + ("-last-ctx1" if arm.endswith("generator") else ""),
        id=digest({"identity": identity, "arm": arm, "project": cfg.wandb.project})[
            :16
        ],
        resume="allow" if cfg.wandb.mode == "online" else None,
        dir=str(local),
        config={"benchmark": cfg.model_dump(mode="json"), "identity": identity},
        job_type="vllm-benchmark",
    )
    try:
        for i, row in enumerate(result["requests"], 1):
            run.log(
                {
                    "request_index": i,
                    **{
                        f"request/{key}": value
                        for key, value in row.items()
                        if isinstance(value, (float, int))
                    },
                }
            )
        run.summary.update(
            {f"benchmark/{key}": value for key, value in result["summary"].items()}
        )
    finally:
        run.finish()


def export_comparison(output, results):
    rows = [{"arm": arm, **result["summary"]} for arm, result in results.items()]
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, key in zip(
        axes.flat,
        (
            "acceptance_ratio",
            "accepted_length",
            "output_tokens_per_second",
            "mean_request_seconds",
        ),
        strict=True,
    ):
        axis.bar(
            range(len(rows)),
            [row[key] if row[key] is not None else float("nan") for row in rows],
        )
        axis.set_xticks(
            range(len(rows)), [row["arm"] for row in rows], rotation=30, ha="right"
        )
        axis.set_title(key.replace("_", " "))
    figure.tight_layout()
    if any(row["output_equivalence"] == "unverified" for row in rows):
        figure.suptitle("Exploratory results — output equivalence unverified")
        figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output / "comparison.png", dpi=160)
    figure.savefig(output / "comparison.pdf")
    plt.close(figure)


def stage_bundle(source, destination):
    """Copy inference weights to local disk once; retain small source provenance."""
    if destination.exists():
        read_bundle(destination)
        return destination
    pending = destination.with_name(destination.name + ".pending")
    if pending.exists():
        shutil.rmtree(pending)
    pending.mkdir(parents=True)
    try:
        for name in ("bundle.json", "model.safetensors"):
            shutil.copyfile(source / name, pending / name)
        read_bundle(pending)
        pending.rename(destination)
    except BaseException:
        shutil.rmtree(pending)
        raise
    return destination


def archive_source(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("bundle.json", *PROVENANCE):
        path = source / name
        if path.is_dir():
            shutil.copytree(path, destination / name, dirs_exist_ok=True)
        elif path.exists():
            shutil.copy2(path, destination / name)


def benchmark(cfg, *, smoke=False):  # noqa: C901 -- serial arm orchestration and resume
    if smoke:
        cfg = cfg.model_copy(
            update={
                "prompts": 3,
                "warmups": 1,
                "repetitions": 1,
                "max_tokens": 32,
                "output_root": cfg.output_root / "smoke",
            }
        )
    status = patch_sources()
    if status["state"] != "patched":
        raise RuntimeError("Apply the vLLM generator patch before benchmarking")
    bank = BankConfig.load(cfg.bank_config)
    manifest, subset = load_subset(bank.output_root / "manifest.json", cfg.subset)
    prompts = benchmark_prompts(cfg, bank, manifest, subset)
    model_refs = {
        role: {
            "id": manifest["cache_spec"]["models"][role],
            "revision": manifest["revisions"][f"{role}_sha"],
        }
        for role in ("target", "drafter")
    }
    bundles = {}
    for arm in cfg.arms:
        if arm in cfg.bundles:
            value = read_bundle(cfg.bundles[arm])
            expected_kind = "generator" if arm.endswith("generator") else "static_lora"
            if value["kind"] != expected_kind or value["models"] != model_refs:
                raise ValueError(f"Bundle kind or frozen-model revisions differ: {arm}")
            if value["source_job"].get("arm") != arm:
                raise ValueError(f"Bundle comes from a different adaptation arm: {arm}")
            bundles[arm] = value
    settings = cfg.model_dump(mode="json")
    settings.pop("wandb")
    source_root = find_package_repo("speculators")
    identity = {
        "configuration": settings,
        "models": model_refs,
        "prompts": digest(prompts),
        "bundle_checksums": {
            arm: value["weights_sha256"] for arm, value in bundles.items()
        },
        "vllm_commit": VLLM_COMMIT,
        "runtime_environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
        "source_git_sha": git_sha(source_root),
        "source_patch_digest": digest(git_diff(source_root)),
    }
    resolved = cfg.output_root / "resolved_benchmark.json"
    if resolved.exists() and json.loads(resolved.read_text()) != identity:
        raise ValueError("Benchmark settings changed; choose another output_root")
    write_json(resolved, identity)
    write_json(cfg.output_root / "prompts.json", prompts)
    models = {}
    for role, reference in model_refs.items():
        with stage_progress(f"Resolving pinned benchmark {role}"):
            models[role] = snapshot_download(
                reference["id"], revision=reference["revision"]
            )
    if not (cfg.output_root / "eval_command.txt").exists():
        with stage_progress("Recording benchmark provenance and checkpoint checksums"):
            provenance(cfg.output_root, models)
    local = cfg.staging_dir / digest(identity)
    local.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in cfg.arms:
        destination = cfg.output_root / arm
        result_path = destination / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            report(f"Reusing completed benchmark arm: {arm}")
        else:
            llm_args = {
                "model": models["target"],
                "dtype": "bfloat16",
                "enforce_eager": True,
                "tensor_parallel_size": 1,
                "max_num_seqs": 1,
                "enable_chunked_prefill": False,
                "enable_prefix_caching": False,
                "async_scheduling": False,
                "max_model_len": cfg.max_model_len or bank.responses.context_length,
                "max_num_batched_tokens": cfg.max_model_len
                or bank.responses.context_length,
                "gpu_memory_utilization": cfg.gpu_memory_utilization,
                "disable_log_stats": False,
                "generation_config": "vllm",
                "seed": cfg.seed,
            }
            if arm != "target_only":
                llm_args["speculative_config"] = {
                    "model": models["drafter"],
                    "method": "eagle3",
                    "num_speculative_tokens": cfg.speculative_tokens,
                    "enforce_eager": True,
                }
                if arm in bundles:
                    staged = stage_bundle(
                        cfg.bundles[arm],
                        cfg.staging_dir / "bundles" / digest(bundles[arm]),
                    )
                    llm_args["speculative_config"]["draft_adapter_path"] = str(
                        staged.resolve()
                    )
                    archive_source(cfg.bundles[arm], destination / "source_provenance")
            request = local / f"{arm}.json"
            write_json(
                request,
                {
                    "arm": arm,
                    "llm_args": llm_args,
                    "prompts": prompts,
                    "warmups": cfg.warmups,
                    "repetitions": cfg.repetitions,
                    "seed": cfg.seed,
                    "output": str(destination.resolve()),
                },
            )
            command = [
                sys.executable,
                "-m",
                "speculators.generator.serving.worker",
                str(request),
            ]
            with stage_progress(
                f"Benchmark arm={arm}; requests={len(prompts) * cfg.repetitions}"
            ):
                subprocess.run(  # noqa: S603 -- argv only
                    command,
                    check=True,
                    env={**os.environ, "VLLM_USE_V2_MODEL_RUNNER": "0"},
                )
            result = json.loads(result_path.read_text())
        record_output_comparison(
            cfg, arm, results.get("target_only"), result, destination
        )
        results[arm] = result
        log_result(cfg, arm, result, identity)
        report(
            f"{arm}: {result['summary']['output_tokens_per_second']:.2f} "
            "output tokens/s"
        )
    write_json(cfg.output_root / "results.json", results)
    export_comparison(cfg.output_root, results)
    return results
