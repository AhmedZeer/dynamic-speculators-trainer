"""Sequential vLLM/GuideLLM suites with immutable workloads and durable progress."""

import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from filelock import FileLock
from huggingface_hub import snapshot_download

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.benchmark.config import Suite
from speculators.benchmark.prepare import local_source, prepare_workload, resolve_model
from speculators.benchmark.processes import (
    ensure_port_free,
    owned_process,
    termination_cleanup,
    wait_ready,
)
from speculators.benchmark.progress import report, report_failure, stage_progress
from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.serving.endpoint import save_provenance
from speculators.generator.serving.validation import load_validation
from speculators.provenance import find_package_repo

DEFAULT_TARGET = {
    "id": "Qwen/Qwen3-8B",
    "revision": "b968826d9c46dd6066d109eabc6255188de91218",
}


def server_command(root, model, target, settings, provenance):
    command = [
        sys.executable,
        str(root / "scripts/launch_vllm.py"),
        "serve",
        "--provenance-dir",
        str(provenance),
        "--port",
        str(settings.port),
        "--max-model-len",
        str(settings.max_model_len),
        "--gpu-memory-utilization",
        str(settings.gpu_memory_utilization),
        "--spec-tokens",
        str(settings.speculative_tokens),
        "--seed",
        str(settings.seed),
        "--concurrency",
        str(settings.concurrency),
        "--batch-wait-ms",
        str(settings.batch_wait_ms),
    ]
    if model["type"] == "drafter":
        command += [
            target["id"],
            "--revision",
            target["revision"],
            "--drafter",
            model["path"],
            "--drafter-revision",
            model["revision"],
        ]
    else:
        command += ["--adapter-bundle", model["path"]]
    if model.get("conditioning_examples") is not None:
        command += ["--conditioning-examples", str(model["conditioning_examples"])]
    command += ["--batching-policy", settings.batching_policy]
    if model.get("adapter_policy") == "per_request":
        command += ["--adapter-policy", "per_request"]
    if model.get("vanilla_vllm"):
        command += ["--vanilla-vllm"]
    if model.get("adapter_experiment") is not None:
        command += ["--adapter-experiment", json.dumps(model["adapter_experiment"])]
    if settings.profiling is not None:
        from speculators.benchmark.profiling import profiler_config  # noqa: PLC0415

        command += [
            "--profiler-config",
            json.dumps(profiler_config(settings, provenance)),
        ]
    return command


def client_command(  # noqa: PLR0917 -- explicit subprocess inputs
    root, task, workload, settings, tokenizer, provenance, output
):
    command = [
        sys.executable,
        str(root / "scripts/evaluate/evaluate.py"),
        "throughput",
        "--target",
        f"http://127.0.0.1:{settings.port}/v1",
        "--validation-manifest",
        workload["path"],
        "--subsets",
        task["category"],
        "--server-provenance",
        str(provenance),
        "--tokenizer-path",
        str(tokenizer),
        "--max-concurrency",
        str(settings.concurrency),
        "--max-requests",
        "0",
        "--warmups",
        str(settings.warmups),
        "--seed",
        str(settings.seed),
        "--output-dir",
        str(output),
    ]
    if settings.profiling is not None:
        command += ["--capture-profile"]
    return command


def model_signature(model):
    return {key: value for key, value in model.items() if key != "path"}


def result_files(output, category):
    paths = [
        output / "results.json",
        output / "benchmark_identity.json",
        output / category / "guidellm.json",
    ]
    if (output / "profile_capture.json").exists():
        from speculators.benchmark.profiling import profile_artifacts  # noqa: PLC0415

        paths += profile_artifacts(output)
    return paths


def verify_result(output, task, workload, model):  # noqa: C901 -- independent result identity checks
    result = json.loads((output / "results.json").read_text())
    identity = json.loads((output / "benchmark_identity.json").read_text())
    if identity["validation"] != workload["identity"]:
        raise ValueError("Client measured the wrong frozen workload")
    if identity.get("concurrency", 1) != model.get("concurrency", 1):
        raise ValueError("Client measured the wrong concurrency")
    if identity["server"].get("conditioning_examples") != model.get(
        "conditioning_examples"
    ):
        raise ValueError("Client measured the wrong conditioning size")
    expected_kind = {
        "lora": "static_lora",
        "generator": "generator",
        "drafter": "eagle3_base",
    }
    if identity["server"]["kind"] != expected_kind[model["type"]]:
        raise ValueError("Client measured the wrong server kind")
    if identity["server"].get("adapter_experiment") != model.get("adapter_experiment"):
        raise ValueError("Client measured a different adapter experiment")
    if (
        model.get("adapter_policy") == "per_request"
        and identity["server"].get("adapter_policy") != "per_request"
    ):
        raise ValueError("Client did not measure per-request adapters")
    if (
        model.get("vanilla_vllm")
        and identity["server"].get("vllm_sources") != "vanilla"
    ):
        raise ValueError("Baseline did not use vanilla vLLM sources")
    if (
        model["type"] != "drafter"
        and identity["server"]["bundle_weights_sha256"] != model["weights_sha256"]
    ):
        raise ValueError("Client measured different adapter weights")
    group = result["groups"][task["category"]]
    expected = workload["groups"][task["category"]]
    if group["dataset_sha256"] != expected["sha256"]:
        raise ValueError("Measured prompt checksum differs")
    if (
        len(group["performance"]) != 1
        or group["performance"][0]["successful_requests"] != expected["rows"]
    ):
        raise ValueError("Client returned an incomplete measurement")
    for path in result_files(output, task["category"]):
        if not path.is_file():
            raise FileNotFoundError(f"Missing native measurement artifact: {path}")
    return group


def valid_completed(output_root, entry):
    if entry.get("status") != "completed" or not entry.get("checksums"):
        return False
    return all(
        (output_root / name).is_file() and file_digest(output_root / name) == sha
        for name, sha in entry["checksums"].items()
    )


def validate_resume_request(previous, config):
    """Allow added models/concurrencies while preserving existing measurements."""
    saved = Suite.model_validate(previous)
    old, new = saved.model_dump(mode="json"), config.model_dump(mode="json")
    for request in (old, new):
        request["settings"].pop("concurrency")
        request["settings"].pop("concurrencies")
    old_models = {model["name"]: model for model in old.pop("models")}
    new_models = {model["name"]: model for model in new.pop("models")}
    models_preserved = all(
        new_models.get(name) == model for name, model in old_models.items()
    )
    if (
        old != new
        or not models_preserved
        or not set(saved.settings.concurrency_values)
        <= set(config.settings.concurrency_values)
    ):
        raise ValueError(
            "Suite configuration changed; resume only permits adding concurrency "
            "values or models, with identical existing models, datasets, "
            "and other settings"
        )


def migrate_serial_tasks(state, models, previous_models):
    """Relabel legacy C=1 tasks; retain original artifact paths and checksums."""
    legacy_names = {
        model["name"]
        for model in previous_models
        if model.get("concurrency", 1) == 1
        and model["name"] == model.get("base_name", model["name"])
    }
    renames = {
        model["base_name"]: model["name"]
        for model in models
        if model["concurrency"] == 1
        and model["name"] != model["base_name"]
        and model["base_name"] in legacy_names
    }
    updated = {}
    for old_key, saved_entry in state["tasks"].items():
        entry = saved_entry
        name, _, suffix = old_key.partition("/")
        new_key = old_key
        if name in renames and not entry.get("reused_from_task"):
            name = renames[name]
            new_key = name + "/" + suffix
            entry = {
                **entry,
                "key": new_key,
                "model": name,
                "reused_from_task": old_key,
            }
            report(f"Reusing serial task {old_key} as {new_key}; artifacts unchanged")
        if new_key in updated:
            raise ValueError(f"Ambiguous saved serial tasks: {old_key}, {new_key}")
        updated[new_key] = entry
    state["tasks"] = updated


def write_summary(output, plan, state):
    if plan.get("configuration", {}).get("settings", {}).get("profiling") is not None:
        from speculators.benchmark.profiling import write_profile_summary  # noqa: PLC0415

        write_profile_summary(output, plan, state)
        return
    columns = []
    for benchmark, workload in plan["workloads"].items():
        # Keep the familiar Nemotron order; other groups retain stable file order.
        names = list(workload["groups"])
        preferred = ["nemo-math", "nemo-code", "nemo-stem", "nemo-chat"]
        names.sort(
            key=lambda name: (preferred.index(name) if name in preferred else 4, name)
        )
        columns += [(benchmark, name) for name in names]
    rows, details = [], {}
    table = [
        "# Benchmark suite: output tokens/s",
        "",
        (
            "Aggregate output token throughput includes prefill and serving overhead. "
            "Correctness was not checked."
        ),
        "",
        "| Model | " + " | ".join(f"{bench}/{cat}" for bench, cat in columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    if any(model.get("concurrency", 1) > 1 for model in plan["models"]):
        continuous = (
            plan.get("configuration", {}).get("settings", {}).get("batching_policy")
            == "continuous"
        )
        table[4:4] = [
            "-cN denotes concurrency N; native continuous admission is enabled."
            if continuous
            else "-cN denotes concurrency N; fixed batches never refill active groups.",
            "",
        ]
    for model in plan["models"]:
        row = {
            "model": model["name"],
            "type": model["type"],
            "concurrency": model.get("concurrency", 1),
        }
        cells = []
        details[model["name"]] = {}
        for benchmark, category in columns:
            key = f"{model['name']}/{benchmark}/{category}"
            entry = state["tasks"].get(key, {})
            value = None
            if entry.get("status") == "completed":
                result = json.loads(
                    (output / entry["output"] / "results.json").read_text()
                )
                group = result["groups"][category]
                value = group["performance"][0]["output_tokens_per_second"]
                details[model["name"]][f"{benchmark}/{category}"] = group
            row[f"{benchmark}/{category}"] = value
            cells.append(
                f"{value:.2f}" if value is not None else entry.get("status", "pending")
            )
        rows.append(row)
        table.append("| " + model["name"] + " | " + " | ".join(cells) + " |")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.md").write_text("\n".join(table) + "\n")
    write_json(
        output / "summary.json",
        {
            "suite_identity": plan["identity"],
            "rows": rows,
            "results": details,
            "correctness_status": "not_checked",
        },
    )
    write_json(
        output / "failures.json",
        {
            key: value
            for key, value in state["tasks"].items()
            if value.get("status") != "completed"
        },
    )


def _measure_model(root, output, model, plan, state, settings, tokenizer):  # noqa: C901, PLR0917 -- owns sequential server/client lifecycle
    tasks = []
    skipped = 0
    for benchmark, workload in plan["workloads"].items():
        for category in workload["groups"]:
            task = {
                "key": f"{model['name']}/{benchmark}/{category}",
                "model": model["name"],
                "benchmark": benchmark,
                "category": category,
            }
            if not valid_completed(output, state["tasks"].get(task["key"], {})):
                tasks.append(task)
            else:
                skipped += 1
    report(
        f"Model {model['name']}: type={model['type']}; "
        f"pending={len(tasks)}; completed/skipped={skipped}"
    )
    target = f"http://127.0.0.1:{settings.port}/v1"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(settings.gpu)}
    while tasks:
        ensure_port_free(settings.port)
        try:
            state["server_attempts"] = state.get("server_attempts", 0) + 1
            provenance = (
                output
                / "servers"
                / model["name"]
                / f"attempt-{state['server_attempts']:04d}"
            )
            write_json(output / "state.json", state)
            command = server_command(root, model, plan["target"], settings, provenance)
            write_json(provenance / "launch_argv.json", command)
            report(
                f"Suite: starting {model['name']}; {len(tasks)} measurements pending; "
                f"attempt={state['server_attempts']}; GPU={settings.gpu}; "
                f"endpoint={target}; server_log={provenance / 'server.log'}"
            )
            with owned_process(
                command,
                provenance / "server.log",
                cwd=root,
                env=env,
                timeout=settings.shutdown_timeout_seconds,
            ) as process:
                report(f"Server {model['name']}: launched pid={process.pid}")
                manifest = json.loads(
                    Path(plan["workloads"][tasks[0]["benchmark"]]["path"]).read_text()
                )
                with stage_progress(f"Server {model['name']}: waiting for readiness"):
                    wait_ready(
                        process,
                        target,
                        model,
                        manifest,
                        settings.startup_timeout_seconds,
                    )
                report(f"Server {model['name']}: ready; model identity verified")
                while tasks:
                    task = tasks.pop(0)
                    previous = state["tasks"].get(task["key"], {})
                    attempt = previous.get("attempt", 0) + 1
                    destination = (
                        output / "measurements" / task["key"] / f"attempt-{attempt:04d}"
                    )
                    entry = {
                        **task,
                        "attempt": attempt,
                        "status": "running",
                        "output": str(destination.relative_to(output)),
                        "server_provenance": str(provenance.relative_to(output)),
                    }
                    state["tasks"][task["key"]] = entry
                    write_json(output / "state.json", state)
                    workload = plan["workloads"][task["benchmark"]]
                    client = client_command(
                        root,
                        task,
                        workload,
                        settings,
                        tokenizer,
                        provenance,
                        destination,
                    )
                    write_json(destination / "client_argv.json", client)
                    report(
                        f"Suite: measuring {task['key']}; attempt={attempt}; "
                        f"requests={workload['groups'][task['category']]['rows']}; "
                        f"warmups={settings.warmups}; remaining_after={len(tasks)}; "
                        f"client_log={destination / 'client.log'}"
                    )
                    started = time.monotonic()
                    try:
                        with (
                            stage_progress(f"Client {task['key']}"),
                            owned_process(
                                client,
                                destination / "client.log",
                                cwd=root,
                                env=env,
                                timeout=settings.shutdown_timeout_seconds,
                            ) as worker,
                        ):
                            code = worker.wait(timeout=settings.client_timeout_seconds)
                            if code:
                                raise RuntimeError(
                                    f"Benchmark client exited with code {code}; "
                                    f"see {destination / 'client.log'}"
                                )
                        group = verify_result(destination, task, workload, model)
                        if (
                            settings.profiling is not None
                            and not (destination / "profile_summary.json").is_file()
                        ):
                            raise ValueError(
                                "Profiling client did not produce a trace summary"
                            )
                        entry.update(
                            status="completed",
                            checksums={
                                str(path.relative_to(output)): file_digest(path)
                                for path in result_files(destination, task["category"])
                            },
                        )
                        performance = group["performance"][0]
                        report(
                            f"COMPLETED {task['key']}: "
                            f"{performance['output_tokens_per_second']:.2f} tok/s; "
                            f"requests={performance['successful_requests']}; "
                            f"wall={time.monotonic() - started:.2f}s"
                        )
                    except KeyboardInterrupt:
                        entry["status"] = "interrupted"
                        report(
                            f"INTERRUPTED {task['key']}; "
                            f"client_log={destination / 'client.log'}"
                        )
                        raise
                    except Exception as exc:
                        entry.update(
                            status="failed", error=f"{type(exc).__name__}: {exc}"
                        )
                        report_failure(task["key"], exc, destination / "client.log")
                        if not settings.continue_on_error:
                            raise
                        # Recycle the server to clear outstanding work.
                        report(
                            f"Recycling server {model['name']} after client failure; "
                            f"{len(tasks)} measurements remain"
                        )
                        break
                    finally:
                        entry["wall_seconds"] = time.monotonic() - started
                        write_json(output / "state.json", state)
                        write_summary(output, plan, state)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            report_failure(
                f"Server attempt {model['name']}", exc, provenance / "server.log"
            )
            if not settings.continue_on_error:
                raise
            for task in tasks:
                state["tasks"][task["key"]] = {
                    **state["tasks"].get(task["key"], {}),
                    **task,
                    "status": "failed",
                    "phase": "server_start",
                    "server_provenance": str(provenance.relative_to(output)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            tasks.clear()
            write_json(output / "state.json", state)
        finally:
            report(f"Server {model['name']}: attempt ended; owned processes cleaned up")
        # Give terminated workers time to release the listener before the next launch.
        deadline = time.monotonic() + settings.shutdown_timeout_seconds
        while time.monotonic() < deadline:
            try:
                ensure_port_free(settings.port)
                break
            except RuntimeError:
                time.sleep(0.2)


def run_suite(  # noqa: C901 -- resolve, launch and resume the complete suite
    config_path, output_root, *, resume=False, dry_run=False, profile_mode=None
):
    session_started = time.monotonic()
    config_path = Path(config_path).resolve()
    config = Suite.load(config_path)
    if profile_mode is None and config.settings.profiling is not None:
        raise ValueError("Use speculators benchmark profile for profiling configs")
    if profile_mode is not None:
        from speculators.benchmark.config import Profiling  # noqa: PLC0415

        config.settings.profiling = (
            (config.settings.profiling or Profiling()) if profile_mode else None
        )
    if dry_run:
        return {
            "dry_run": True,
            "configuration": config.model_dump(mode="json"),
            "note": "No model downloads, workload exports, or processes started",
        }
    root = find_package_repo("speculators")
    if root is None or not (root / "scripts/launch_vllm.py").exists():
        raise FileNotFoundError(
            "Suite execution requires the editable repository checkout"
        )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report(
        f"Suite: config={config_path}; output={output}; resume={resume}; "
        f"models={len(config.models)}; datasets={len(config.benchmarks)}"
    )
    port_lock = (
        Path(tempfile.gettempdir())
        / f"speculators-suite-{os.getuid()}-{config.settings.port}.lock"
    )
    with (
        FileLock(str(output / ".suite.lock"), timeout=0),
        FileLock(str(port_lock), timeout=0),
        termination_cleanup(),
    ):
        request = config.model_dump(mode="json")
        request_path = output / "suite_request.json"
        previous = None
        if request_path.exists():
            if not resume:
                raise FileExistsError(
                    "Suite output exists; use --resume or a new output root"
                )
            previous = json.loads(request_path.read_text())
            validate_resume_request(previous, config)
        attempt_dir = output / "suite_attempts" / str(time.time_ns())
        attempt_dir.mkdir(parents=True)
        if previous is not None and previous != request:
            for name in (
                "suite_request.json",
                "suite.json",
                "state.json",
                "campaign_configuration.json",
            ):
                if (output / name).exists():
                    shutil.copy2(output / name, attempt_dir / ("previous_" + name))
            report(f"Preserved previous suite configuration/state in {attempt_dir}")
        write_json(request_path, request)
        save_provenance(attempt_dir)
        settings = config.settings
        campaign_path = (
            local_source(settings.campaign_config, config_path.parent)
            if settings.campaign_config
            else root / "examples/eagle3-lora/generator_campaign_online.yaml"
        )
        cfg = CampaignConfig.load(campaign_path)
        cfg.trajectories_path = local_source(cfg.trajectories_path, root)
        cfg.online.target_model_path = local_source(cfg.online.target_model_path, root)
        pins_path = output / "resolved_models.json"
        pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}
        models = []
        for spec in config.models:
            previous = pins.get(spec.name)
            with stage_progress(
                f"Resolve {spec.name}: type={spec.type}; source={spec.model}"
            ):
                model = resolve_model(
                    spec,
                    config_path.parent,
                    output / "local_models",
                    pinned=previous["revision"] if previous else None,
                )
            if spec.conditioning_examples is not None:
                model["conditioning_examples"] = spec.conditioning_examples
            report(f"Resolved {spec.name}: revision={model['revision']}")
            if previous and model_signature(model) != model_signature(previous):
                raise ValueError(f"Model {spec.name} changed since the saved suite")
            pins[spec.name] = model
            models.append(model)
            write_json(pins_path, pins)
        references = [m["models"]["target"] for m in models if "models" in m]
        models = [
            {
                **model,
                "base_name": model["name"],
                "name": f"{model['name']}-c{concurrency}"
                if len(settings.concurrency_values) > 1 or concurrency != 1
                else model["name"],
                "concurrency": concurrency,
            }
            for model in models
            for concurrency in settings.concurrency_values
        ]
        if len({model["name"] for model in models}) != len(models):
            raise ValueError("Model names collide after adding concurrency suffixes")
        report(
            f"Concurrency sweep={settings.concurrency_values}; "
            f"server variants={len(models)}"
        )
        target = references[0] if references else DEFAULT_TARGET
        if any(reference != target for reference in references):
            raise ValueError("All models in a suite must share one pinned target")
        tokenizer = (
            local_source(settings.tokenizer_path, config_path.parent)
            if settings.tokenizer_path
            else cfg.online.target_model_path
        )
        if not tokenizer.is_dir():
            tokenizer = Path(
                snapshot_download(
                    target["id"],
                    revision=target["revision"],
                    allow_patterns=[
                        "tokenizer*",
                        "config.json",
                        "special_tokens_map.json",
                        "vocab.json",
                        "merges.txt",
                        "added_tokens.json",
                        "chat_template*",
                    ],
                )
            )
        workloads = {}
        for benchmark in config.benchmarks:
            with stage_progress(f"Prepare workload {benchmark.label}"):
                workload = prepare_workload(
                    benchmark,
                    cfg,
                    output / "workloads" / benchmark.label,
                    settings,
                    tokenizer,
                )
            report(
                f"Workload {benchmark.label}: "
                + ", ".join(
                    f"{name}={info['rows']} prompts"
                    for name, info in workload["groups"].items()
                )
            )
            if workload["models"]["target"] != target:
                raise ValueError("Workload target differs from the model bundles")
            _, groups = load_validation(Path(workload["path"]))
            if any(
                len(row["prompt_token_ids"]) + row["output_tokens_count"]
                > settings.max_model_len
                for rows in groups.values()
                for row in rows
            ):
                raise ValueError("Selected prompts exceed the serving context length")
            workloads[benchmark.label] = workload
        plan = {
            "configuration": request,
            "models": models,
            "target": target,
            "workloads": workloads,
        }
        plan["identity"] = digest(plan)
        prior_plan = (
            json.loads((output / "suite.json").read_text())
            if (output / "suite.json").exists()
            else {"models": []}
        )
        state_path = output / "state.json"
        state = (
            json.loads(state_path.read_text()) if state_path.exists() else {"tasks": {}}
        )
        migrate_serial_tasks(state, models, prior_plan["models"])
        write_json(state_path, state)
        write_json(output / "suite.json", plan)
        write_json(output / "campaign_configuration.json", cfg.model_dump(mode="json"))
        for model in models:
            for benchmark, workload in workloads.items():
                for category in workload["groups"]:
                    key = f"{model['name']}/{benchmark}/{category}"
                    entry = state["tasks"].setdefault(key, {"status": "pending"})
                    if not valid_completed(output, entry):
                        entry["status"] = "pending"
        try:
            for model in models:
                _measure_model(
                    root,
                    output,
                    model,
                    plan,
                    state,
                    settings.model_copy(
                        update={
                            "concurrency": model["concurrency"],
                            "batching_policy": model.get(
                                "batching_policy", settings.batching_policy
                            ),
                        }
                    ),
                    tokenizer,
                )
        finally:
            write_json(state_path, state)
            write_summary(output, plan, state)
        from speculators.benchmark.tracking import publish_completed  # noqa: PLC0415

        publish_completed(output, plan, state)
        failed = sum(
            entry["status"] != "completed" for entry in state["tasks"].values()
        )
        report(
            f"Suite finished: completed={len(state['tasks']) - failed}; "
            f"failed/incomplete={failed}; "
            f"wall={time.monotonic() - session_started:.2f}s; "
            f"summary={output / 'summary.md'}; failures={output / 'failures.json'}"
        )
        return {
            "output": str(output),
            "completed": len(state["tasks"]) - failed,
            "failed": failed,
            "summary": str(output / "summary.md"),
        }
