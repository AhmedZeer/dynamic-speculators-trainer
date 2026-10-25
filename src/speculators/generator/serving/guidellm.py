"""GuideLLM measurements against existing servers; never constructs a model."""

import csv
import json
import shutil
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import snapshot_download

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.serving.endpoint import (
    check_tokenization,
    endpoint_root,
    http_json,
    row_identity,
    save_provenance,
    selected_rows,
    server_status,
    validate_adapter_delta,
    validate_status,
)
from speculators.generator.serving.validation import load_validation

MAPPER = {
    "kind": "generative_column_mapper",
    "column_mappings": {
        "text_column": "prompt",
        "prompt_tokens_count_column": "prompt_tokens_count",
        "output_tokens_count_column": "output_tokens_count",
    },
}


def copy_server_provenance(source, output):
    """Copy launch evidence without duplicating installed packages or old traces."""
    shutil.copytree(
        source, output, ignore=shutil.ignore_patterns("traces", "vanilla-vllm")
    )
    manifest = source / "vanilla_source.json"
    if manifest.is_file():
        originals = json.loads(manifest.read_text())["original_sha256"]
        for relative, expected in originals.items():
            path = source / "vanilla-vllm" / relative
            if file_digest(path) != expected:
                raise ValueError(f"Baseline original source changed: {relative}")
            destination = output / "vanilla-vllm" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def native_summary(path, expected_requests):
    raw = json.loads(Path(path).read_text())
    benchmarks = raw.get("benchmarks", [])
    if not benchmarks:
        raise ValueError("GuideLLM did not produce any benchmark results")
    rows = []
    for benchmark in benchmarks:
        metrics = benchmark["metrics"]
        counts = metrics["request_totals"]
        if (
            counts["successful"] != expected_requests
            or counts["errored"]
            or counts["incomplete"]
        ):
            raise ValueError(f"Incomplete or failed GuideLLM workload: {counts}")
        row = {
            "strategy": benchmark["config"]["strategy"]["type_"],
            "successful_requests": counts["successful"],
            "total_output_tokens": metrics["output_token_count"]["successful"][
                "total_sum"
            ],
            "target_rate": benchmark["config"]["strategy"].get("rate", ""),
            "rps_median": metrics["requests_per_second"]["successful"]["median"],
        }
        for metric, label in (
            ("request_latency", "latency_median_s"),
            ("time_to_first_token_ms", "ttft_median_ms"),
            ("inter_token_latency_ms", "itl_median_ms"),
            ("output_tokens_per_second", "output_tps_median"),
        ):
            row[label] = metrics[metric]["successful"]["median"]
        scheduler = benchmark["scheduler_metrics"]
        seconds = scheduler["measure_end_time"] - scheduler["measure_start_time"]
        if seconds <= 0:
            raise ValueError("GuideLLM measured a nonpositive duration")
        row["measured_seconds"] = seconds
        row["output_tokens_per_second"] = row["total_output_tokens"] / seconds
        rows.append(row)
    return rows


def read_metrics(target):
    with urlopen(endpoint_root(target) + "/metrics", timeout=30) as response:  # noqa: S310
        return response.read().decode()


def check_report(args, manifest, groups, status):
    """Validate an optional report without requiring extra generation requests."""
    if not args.correctness_report:
        return None, False
    correctness = json.loads(Path(args.correctness_report).read_text())
    overridden = (
        getattr(args, "allow_output_mismatch", False)
        and correctness["status"] == "unverified"
        and bool(correctness.get("mismatches"))
    )
    if correctness["validation_identity"] != manifest["identity"] or not (
        correctness["status"] in ("reference", "matched") or overridden
    ):
        raise ValueError(
            "Correctness report is unverified or belongs to different validation data"
        )
    if correctness["server_configuration"] != (
        status["configuration"] if status else None
    ):
        raise ValueError("Correctness report belongs to a different server/checkpoint")
    for rows in groups.values():
        for row in rows:
            checked = correctness["rows"].get(row["row_id"])
            if checked is None or checked["identity"] != row_identity(row, args.seed):
                raise ValueError(
                    "Correctness report does not cover this workload and seed"
                )
    return correctness, overridden


def benchmark_endpoint(args, utils):  # noqa: C901 -- explicit measurement boundaries
    if version("guidellm") != "0.7.1":
        raise ValueError("Validation benchmarking requires guidellm==0.7.1")
    if args.gen_kwargs:
        raise ValueError(
            "Validation mode fixes greedy generation settings; omit --gen-kwargs"
        )
    if args.warmups < 0:
        raise ValueError("Warmup count must be nonnegative")
    manifest, groups = load_validation(args.validation_manifest)
    categories = (
        None if args.subsets == "all" else [s.strip() for s in args.subsets.split(",")]
    )
    groups = selected_rows(groups, args.max_requests or 0, categories)
    if getattr(args, "validation_grouping", "category") == "subset":
        rows = [row for group in groups.values() for row in group]
        groups = {
            name: [row for row in rows if row["subset"] == name]
            for name in sorted({row["subset"] for row in rows})
        }
    status = server_status(args.target)
    validate_status(status, manifest)
    capture = getattr(args, "capture_profile", False)
    if capture and (
        len(groups) != 1
        or status is None
        or not status["configuration"].get("profiler")
        or args.mode != "throughput"
    ):
        raise ValueError(
            "Trace capture requires throughput mode, one category "
            "and a profiler-enabled server"
        )
    if not capture and status and status["configuration"].get("profiler"):
        raise ValueError("Use --capture-profile or launch a server without profiling")
    capacity = (
        status["configuration"].get("max_num_seqs", 1)
        if status
        else args.max_concurrency
    )
    if not 1 <= args.max_concurrency <= capacity:
        raise ValueError("Client concurrency must not exceed the server's max_num_seqs")
    correctness, overridden = check_report(args, manifest, groups, status)
    correctness_status = correctness["status"] if correctness else "not_checked"
    tokenizer_path = (
        str(args.tokenizer_path)
        if args.tokenizer_path
        else snapshot_download(
            manifest["models"]["target"]["id"],
            revision=manifest["models"]["target"]["revision"],
            allow_patterns=[
                "tokenizer*",
                "special_tokens_map.json",
                "config.json",
                "vocab.json",
                "merges.txt",
                "added_tokens.json",
                "chat_template*",
            ],
        )
    )
    output = Path(args.output_dir)
    if (output / "benchmark_identity.json").exists():
        raise FileExistsError("Choose a new measurement output directory")
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "measurement_kind": "profiled_diagnostic" if capture else "benchmark",
        "validation": manifest["identity"],
        "guidellm_version": "0.7.1",
        "target": args.target,
        "server": status["configuration"] if status else None,
        "mode": args.mode,
        "categories": list(groups),
        "grouping": getattr(args, "validation_grouping", "category"),
        "limit": args.max_requests,
        "seed": args.seed,
        "concurrency": args.max_concurrency,
        "warmups": args.warmups,
        "sweep_size": args.sweep_rate,
        "correctness_sha256": file_digest(Path(args.correctness_report))
        if correctness
        else None,
        "correctness_status": correctness_status,
        "output_mismatch_override": overridden,
        "mismatched_prompts": len(correctness.get("mismatches", []))
        if correctness
        else None,
        "tokenizer_path": tokenizer_path,
    }
    write_json(output / "benchmark_identity.json", identity)
    validation_dir = output / "validation"
    write_json(validation_dir / "manifest.json", manifest)
    for info in manifest["groups"].values():
        shutil.copy2(
            Path(args.validation_manifest).parent / info["file"],
            validation_dir / info["file"],
        )
    if correctness:
        shutil.copy2(args.correctness_report, output / "correctness.json")
    if args.server_provenance:
        source = Path(args.server_provenance).resolve()
        saved = json.loads((source / "server.json").read_text())
        if status is None or saved != status["configuration"]:
            raise ValueError("Server provenance does not describe the connected server")
        if source == output.resolve() or source in output.resolve().parents:
            raise ValueError("Use distinct server-provenance and benchmark directories")
        copy_server_provenance(source, output / "server_provenance")
    save_provenance(output)
    model = http_json(endpoint_root(args.target) + "/v1/models")["data"][0]["id"]
    all_results, performance = {}, []
    for category, rows in groups.items():
        directory = output / category
        directory.mkdir()
        dataset = directory / "validation.jsonl"
        dataset.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        check_tokenization(args.target, model, rows)
        # Explicit warmups happen before both HTTP and adapter counter snapshots.
        for index in range(args.warmups):
            row = rows[index % len(rows)]
            http_json(
                endpoint_root(args.target) + "/v1/completions",
                {
                    "model": model,
                    "prompt": row["prompt_token_ids"],
                    "max_tokens": row["output_tokens_count"],
                    "temperature": 0,
                    "seed": args.seed,
                    "add_special_tokens": False,
                    "ignore_eos": False,
                },
            )
        before_text = read_metrics(args.target)
        before = server_status(args.target)
        validate_status(before, manifest)
        native_output = directory / "guidellm.json"
        from speculators.benchmark.profiling import capture_profile  # noqa: PLC0415

        tracing = (
            capture_profile(args.target, output, status["configuration"]["profiler"])
            if capture
            else nullcontext()
        )
        with tracing:
            utils.run_guidellm(
                target=args.target,
                dataset=str(dataset),
                subset=None,
                data_column_mapper=json.dumps(MAPPER),
                profile=args.mode,
                rate=args.sweep_rate,
                max_requests=len(rows),
                max_concurrency=args.max_concurrency,
                output_path=native_output,
                max_tokens=manifest["max_tokens"],
                model=model,
                tokenizer_path=tokenizer_path,
                request_format="/v1/completions",
                deterministic_seed=args.seed,
                command_path=directory / "guidellm_command.json",
                gen_kwargs={
                    "temperature": 0,
                    "seed": args.seed,
                    "add_special_tokens": False,
                    "ignore_eos": False,
                },
            )
        after_text = read_metrics(args.target)
        after = server_status(args.target)
        validate_status(after, manifest)
        adapter = validate_adapter_delta(before, after)
        (directory / "metrics_before.prom").write_text(before_text)
        (directory / "metrics_after.prom").write_text(after_text)
        write_json(directory / "server_before.json", before)
        write_json(directory / "server_after.json", after)
        acceptance = utils.extract_spec_decode_metrics(
            utils.parse_prometheus_metrics(after_text),
            baseline_metrics=utils.parse_prometheus_metrics(before_text),
        )
        kind = status["configuration"]["kind"] if status else None
        if kind == "target_only":
            acceptance = None
        summaries = native_summary(native_output, len(rows))
        performance.extend(
            {
                "subset": category,
                "correctness_status": correctness_status,
                "output_mismatch_override": overridden,
                **row,
            }
            for row in summaries
        )
        all_results[category] = {
            "measurement_kind": identity["measurement_kind"],
            "correctness_status": correctness_status,
            "output_mismatch_override": overridden,
            "performance": summaries,
            "acceptance": acceptance,
            "acceptance_scope": "aggregate across all sweep points in this invocation",
            "adapter_metrics": adapter,
            "dataset_sha256": file_digest(dataset),
            "subsets": sorted({row["subset"] for row in rows}),
        }
        write_json(directory / "result.json", all_results[category])
    with (output / "perf_results.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(performance[0]))
        writer.writeheader()
        writer.writerows(performance)
    for filename, key in (
        ("acceptance.csv", "acceptance"),
        ("adapter_performance.csv", "adapter_metrics"),
    ):
        rows = [
            {"subset": name, "scope": "whole GuideLLM invocation", **value[key]}
            for name, value in all_results.items()
            if value[key]
        ]
        if rows:
            fields = [
                "subset",
                "scope",
                *sorted(
                    set().union(*(row.keys() for row in rows)) - {"subset", "scope"}
                ),
            ]
            with (output / filename).open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    write_json(
        output / "results.json",
        {
            "measurement_kind": identity["measurement_kind"],
            "identity": digest(identity),
            "correctness_status": correctness_status,
            "output_mismatch_override": overridden,
            "groups": all_results,
        },
    )
    return all_results
