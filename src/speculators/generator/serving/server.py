"""Build matched, independent HTTP servers for target/EAGLE3/adapter bundles."""

import json
import os
import re
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

from speculators.bank.artifacts import digest, write_json
from speculators.generator.serving.bundle import (
    VLLM_VERSION,
    archive_source,
    read_bundle,
)
from speculators.generator.serving.patch import PATCH_DIR, patch_sources
from speculators.provenance import atomic_write, find_package_repo, git_diff, git_sha

MIN_CONTEXT = 2
MAX_CONCURRENCY = 8
WORKER_EXTENSION = "speculators.generator.serving.http.AdapterWorkerExtension"


def pinned_model(model, revision, *, local_only=False):
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Serving requires complete target/drafter revision SHAs")
    if Path(model).is_dir():
        if Path(model).name != revision:
            raise ValueError(
                "Local model directories must be named by their revision SHA"
            )
        return str(Path(model).resolve())
    return (
        snapshot_download(model, revision=revision, local_files_only=True)
        if local_only
        else snapshot_download(model, revision=revision)
    )


def build_server(args, extra):  # noqa: C901 -- pinned launch contract
    if extra:
        raise ValueError(
            "serve uses explicit launch options; unsupported arguments: " + str(extra)
        )
    if not args.provenance_dir or args.no_hash_checkpoints:
        raise ValueError("serve requires --provenance-dir and full checkpoint hashes")
    if args.spec_tokens < 1 or args.max_model_len < MIN_CONTEXT:
        raise ValueError(
            "Speculative tokens must be positive and context length at least 2"
        )
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("GPU memory utilization must be between 0 and 1")
    import importlib.metadata  # noqa: PLC0415

    if importlib.metadata.version("vllm") != VLLM_VERSION:
        raise ValueError(f"Matched serving requires vLLM {VLLM_VERSION}")
    metadata = read_bundle(args.adapter_bundle) if args.adapter_bundle else None
    concurrency = getattr(args, "concurrency", 1)
    batch_wait_ms = getattr(args, "batch_wait_ms", 50)
    conditioning_examples = getattr(args, "conditioning_examples", None)
    batching = getattr(args, "batching_policy", "fixed_shared_adapter")
    adapter_policy = getattr(args, "adapter_policy", "shared")
    vanilla = getattr(args, "vanilla_vllm", False)
    profiler = getattr(args, "profiler_config", None)
    experiment = getattr(args, "adapter_experiment", None)
    if metadata and metadata.get("feedback_dim") and experiment is None:
        raise ValueError(
            "Feedback-trained bundles require an explicit adapter-experiment policy"
        )
    if experiment is not None:
        from speculators.generator.feedback import AdapterExperiment  # noqa: PLC0415

        experiment = AdapterExperiment.model_validate(experiment).model_dump(
            mode="json"
        )
        if (
            concurrency != 1
            or adapter_policy != "shared"
            or not metadata
            or metadata["kind"] != "generator"
        ):
            raise ValueError(
                "Adapter experiments require a merged generator at concurrency 1"
            )
        if (
            metadata.get("source_job", {}).get("conditioning_policy")
            == "fixed_original_prompt"
            and experiment["refresh"] != "prompt"
        ):
            raise ValueError(
                "This continuation was trained with fixed original-prompt conditioning"
            )
        if experiment["feedback"] != bool(metadata.get("feedback_dim", 0)):
            raise ValueError(
                "Serving feedback inputs must match the trained generator bundle"
            )
    if profiler is not None:
        from speculators.benchmark.profiling import validate_profiler_config  # noqa: PLC0415

        validate_profiler_config(profiler)
    if vanilla and (metadata is not None or batching != "continuous"):
        raise ValueError("Vanilla vLLM requires a base model and continuous batching")
    if adapter_policy == "per_request":
        if batching != "continuous" or not metadata or metadata["kind"] != "generator":
            raise ValueError("per_request requires a generator and continuous batching")
        if conditioning_examples not in (None, 1):
            raise ValueError("Per-request generators condition on one prompt")
        conditioning_examples = 1
    elif batching == "continuous" and metadata and metadata["kind"] == "generator":
        raise ValueError(
            "Continuous generator serving requires --adapter-policy per_request"
        )
    if batching == "continuous" and metadata and metadata["kind"] == "static_lora":
        raise ValueError(
            "Continuous mode currently supports base EAGLE3 and per-request generators"
        )
    if not 1 <= concurrency <= MAX_CONCURRENCY or batch_wait_ms < 0:
        raise ValueError("Concurrency must be 1–8 and batch wait must be nonnegative")
    if conditioning_examples is not None and (
        not metadata
        or metadata["kind"] != "generator"
        or not 1 <= conditioning_examples <= MAX_CONCURRENCY
    ):
        raise ValueError(
            "conditioning-examples requires a generator and a value from 1 to 8"
        )
    if (
        metadata
        and metadata["kind"] == "generator"
        and min(concurrency, conditioning_examples or concurrency)
        > metadata["max_examples"]
    ):
        raise ValueError("Requested conditioning size exceeds the generator capacity")
    if metadata is not None:
        if patch_sources()["state"] != "patched":
            raise ValueError(
                "Apply scripts/apply_vllm_generator_patch.py before adapter serving"
            )
        references = metadata["models"]
        for supplied, expected in (
            (args.model, references["target"]["id"]),
            (args.drafter, references["drafter"]["id"]),
        ):
            if supplied is not None and supplied != expected:
                raise ValueError("Bundle serving derives model IDs from bundle.json")
        for supplied, role in (
            (args.revision, "target"),
            (args.drafter_revision, "drafter"),
        ):
            if supplied is not None and supplied != references[role]["revision"]:
                raise ValueError("Supplied model revision differs from bundle.json")
        kind = metadata["kind"]
    else:
        if args.model is None:
            raise ValueError("Supply a target model or --adapter-bundle")
        references = {"target": {"id": args.model, "revision": args.revision}}
        if args.drafter is not None:
            references["drafter"] = {
                "id": args.drafter,
                "revision": args.drafter_revision,
            }
        kind = "eagle3_base" if args.drafter else "target_only"
    models = {
        role: pinned_model(
            ref["id"], ref["revision"], local_only=getattr(args, "dry_run", False)
        )
        for role, ref in references.items()
    }
    args.model = models["target"]
    args.spec_model = models.get("drafter")
    source_root = find_package_repo("speculators")
    source_patch = git_diff(source_root)
    configuration = {
        "speculators_git_sha": git_sha(source_root),
        "speculators_patch_identity": digest(source_patch),
        "api_server_count": 1,
        "renderer_num_workers": 1,
        "worker_extension_cls": WORKER_EXTENSION,
        "version": 1,
        "kind": kind,
        "static_adapter_policy": "merged_once" if kind == "static_lora" else None,
        "models": references,
        "served_model_name": args.served_model_name,
        "vllm_version": VLLM_VERSION,
        "model_runner": "V1",
        "max_num_seqs": concurrency,
        "batching_policy": batching
        if batching == "continuous" or concurrency > 1
        else "serial",
        "adapter_policy": adapter_policy if kind == "generator" else None,
        "vllm_sources": "vanilla" if vanilla else "installed",
        "batch_wait_ms": batch_wait_ms if batching != "continuous" else 0,
        "max_model_len": args.max_model_len,
        "speculative_tokens": args.spec_tokens if args.spec_model else None,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "deployment_context_size": min(
            concurrency, conditioning_examples or concurrency
        )
        if kind == "generator"
        else None,
        "conditioning_examples": conditioning_examples,
        "conditioning_selection": "seeded_uniform_subset"
        if conditioning_examples
        else "all_members",
        "seed": args.seed,
        "bundle_weights_sha256": metadata["weights_sha256"] if metadata else None,
        "source_job": metadata.get("source_job") if metadata else None,
    }
    if profiler is not None:
        configuration["profiler"] = profiler
    if experiment is not None:
        configuration["adapter_experiment"] = experiment
    output = Path(args.provenance_dir)
    if experiment is not None:
        configuration["adapter_rounds_dir"] = str((output / "adapter_rounds").resolve())
    output.mkdir(parents=True, exist_ok=True)
    previous = output / "server.json"
    if previous.exists() and json.loads(previous.read_text()) != configuration:
        raise ValueError("Server configuration changed; choose a new --provenance-dir")
    if metadata:
        archive_source(Path(args.adapter_bundle), output / "source_provenance")
        shutil.copy2(Path(args.adapter_bundle) / "bundle.json", output / "bundle.json")
        shutil.copy2(
            PATCH_DIR / "vllm-0.31.0-generator.patch", output / "generator-vllm.patch"
        )
    write_json(output / "server.json", configuration)
    atomic_write(output / "speculators.patch", source_patch)
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["SPECULATORS_SERVER_CONFIG"] = json.dumps(configuration)
    os.environ["SPECULATORS_TRACE_RANGES"] = "1" if profiler else "0"
    if vanilla:
        from speculators.generator.serving.vanilla import vanilla_source_view  # noqa: PLC0415

        view = vanilla_source_view(output)
        os.environ["PYTHONPATH"] = (
            str(view) + os.pathsep + os.environ.get("PYTHONPATH", "")
        )
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--enforce-eager",
        "--max-num-seqs",
        str(concurrency),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_model_len * concurrency),
        "--tensor-parallel-size",
        "1",
        "--no-async-scheduling",
        "--no-enable-chunked-prefill",
        "--no-enable-prefix-caching",
        "--generation-config",
        "vllm",
        "--seed",
        str(args.seed),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--api-server-count",
        "1",
        "--renderer-num-workers",
        "1",
        "--middleware",
        "speculators.generator.serving.http.AdapterMiddleware",
        "--worker-extension-cls",
        WORKER_EXTENSION,
    ]
    if concurrency > 1 and batching == "fixed_shared_adapter":
        command.extend(
            [
                "--scheduler-cls",
                "speculators.generator.serving.scheduler.FixedBatchScheduler",
                "--scheduling-policy",
                "fcfs",
            ]
        )
    if profiler is not None:
        command.extend(["--profiler-config", json.dumps(profiler)])
    if args.spec_model:
        speculative = {
            "model": args.spec_model,
            "method": "eagle3",
            "num_speculative_tokens": args.spec_tokens,
            "enforce_eager": True,
        }
        if metadata:
            speculative["draft_adapter_path"] = str(Path(args.adapter_bundle).resolve())
        command.extend(["--speculative-config", json.dumps(speculative)])
    write_json(output / "server_identity.json", {"sha256": digest(configuration)})
    return command
