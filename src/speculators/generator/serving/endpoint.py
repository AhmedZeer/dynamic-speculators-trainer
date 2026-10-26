"""Token-exact correctness checks against an independently running HTTP server."""

import json
import shlex
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.serving.validation import load_validation, nemotron_domain
from speculators.provenance import (
    atomic_write,
    find_package_repo,
    git_diff,
    git_sha,
    package_versions,
)


def http_json(url, body=None):
    request = Request(  # noqa: S310 -- explicit HTTP endpoint
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=600) as response:  # noqa: S310
        return json.load(response)


def endpoint_root(target):
    return target.rstrip("/").removesuffix("/v1")


def server_status(target):
    try:
        return http_json(endpoint_root(target) + "/speculators/status")
    except HTTPError as exc:
        if exc.code == HTTPStatus.NOT_FOUND:
            return None
        raise


def validate_status(status, manifest):  # noqa: C901 -- validate runner, model and baseline source identity
    if status is None:
        return
    configuration = status["configuration"]
    if configuration["models"]["target"] != manifest["models"]["target"]:
        raise ValueError("Server target revision differs from validation manifest")
    if (
        "drafter" in configuration["models"]
        and "drafter" in manifest["models"]
        and configuration["models"]["drafter"] != manifest["models"]["drafter"]
    ):
        raise ValueError("Server drafter revision differs from validation manifest")
    workers = status["workers"]
    if configuration.get("vllm_sources") == "vanilla":
        from speculators.generator.serving.patch import PATCH_DIR  # noqa: PLC0415

        original = json.loads((PATCH_DIR / "manifest.json").read_text())["files"]
        for worker in workers:
            actual = worker.get("vllm_source_sha256", {})
            for relative, record in original.items():
                name = relative.removesuffix(".py").replace("/", ".")
                if actual.get(name) != record["before_sha256"]:
                    raise ValueError(f"Vanilla baseline worker source differs: {name}")
    adapter = configuration["kind"] in ("generator", "static_lora")
    if (
        len(workers) != 1
        or workers[0]["use_v2_model_runner"]
        or (adapter and not workers[0]["adapter_controller_present"])
    ):
        raise ValueError(
            "Server has an unsupported runner or missing adapter controller"
        )
    if adapter and (
        workers[0]["bundle_weights_sha256"] != configuration["bundle_weights_sha256"]
        or workers[0]["adapter_kind"] != configuration["kind"]
    ):
        raise ValueError(
            "Worker loaded a different adapter than the server configuration"
        )
    if any(worker.get("active_request_id") is not None for worker in workers):
        raise ValueError("Benchmark requires an idle server with no other clients")


def validate_adapter_delta(before, after):  # noqa: C901 -- historical/shared and per-request adapter accounting
    if before is not None and (
        after is None or before["configuration"] != after["configuration"]
    ):
        raise ValueError("Connected server changed during measurement")
    if before is None or before["configuration"]["kind"] not in (
        "generator",
        "static_lora",
    ):
        return {}
    left = before["workers"][0]["adapter_totals"]
    right = after["workers"][0]["adapter_totals"]
    delta = {key: right[key] - value for key, value in left.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError("Server adapter counters reset during the run")
    if (
        delta["prepared_requests"] < 1
        or delta["prepared_requests"] != delta["completed_requests"]
    ):
        raise ValueError(
            "Adapters were not executed or were not restored for all prepared requests"
        )
    experiment = before["configuration"].get("adapter_experiment") or {}
    expected = delta["prepared_requests"] + delta.get("adapter_refreshes", 0)
    if experiment.get("refresh") == "prefix":
        if delta["generator_invocations"] != expected:
            raise ValueError(
                "Generator refresh count differs from observed adapter updates"
            )
    elif before["configuration"]["kind"] == "generator" and delta[
        "generator_invocations"
    ] != (
        delta["prepared_requests"]
        if before["configuration"].get("adapter_policy") == "per_request"
        else delta.get("prepared_batches", delta["prepared_requests"])
    ):
        raise ValueError("Generator must execute exactly once per prepared adapter")
    if "prepared_batches" in delta:
        if delta["prepared_batches"] != delta["completed_batches"]:
            raise ValueError("A shared-adapter batch was not completely finished")
        delta["mean_batch_size"] = (
            delta["prepared_requests"] / delta["prepared_batches"]
        )
        if (
            before["configuration"]["kind"] == "generator"
            and "conditioning_examples" in delta
        ):
            delta["mean_conditioning_size"] = (
                delta["conditioning_examples"] / delta["prepared_batches"]
            )
    if before["configuration"].get("adapter_policy") == "per_request":
        # A prepared unit is now one request's adapter, not a scheduled GPU batch.
        delta.pop("mean_batch_size", None)
        delta["mean_requests_per_adapter"] = 1.0
        if delta.get("draft_forward_calls", 0):
            delta["mean_draft_batch_size"] = (
                delta["draft_forward_requests"] / delta["draft_forward_calls"]
            )
    if delta.get("scale_observations", 0):
        delta["mean_effective_scale"] = delta["scale_sum"] / delta["scale_observations"]
    if delta.get("generator_invocations", 0):
        delta["generator_ms_per_call"] = (
            1000 * delta["generator_seconds"] / delta["generator_invocations"]
        )
    return delta


def selected_rows(groups, limit=0, categories=None):
    if limit < 0:
        raise ValueError("Request limit must be nonnegative; zero selects all rows")
    names = categories or list(groups)
    # Accept domain shorthand for new exports and prefixed names for old ones.
    resolved = []
    for requested in names:
        name = requested
        if name not in groups:
            domain = nemotron_domain(name)
            name = f"nemo-{domain}" if f"nemo-{domain}" in groups else domain
        if name not in resolved:
            resolved.append(name)
    names = resolved
    if set(names) - groups.keys():
        raise ValueError("Unknown validation categories")
    return {
        name: rows[:limit] if limit else rows
        for name in names
        for rows in [groups[name]]
    }


def row_identity(row, seed):
    return digest(
        {
            "tokens": row["prompt_token_ids"],
            "max_tokens": row["output_tokens_count"],
            "seed": seed,
        }
    )


def check_tokenization(target, model, rows):
    for row in rows:
        response = http_json(
            endpoint_root(target) + "/tokenize",
            {
                "model": model,
                "prompt": row["prompt"],
                "add_special_tokens": False,
            },
        )
        if response["tokens"] != row["prompt_token_ids"]:
            raise ValueError(
                f"Server tokenization differs from saved prompt: {row['row_id']}"
            )


def save_provenance(output):
    root = find_package_repo("speculators")
    atomic_write(
        output / "eval_command.txt",
        "\n".join(
            [
                f"# Timestamp: {datetime.now(timezone.utc).isoformat()}",
                f"# Git SHA: {git_sha(root)}",
                *package_versions(),
                shlex.join(sys.argv),
                "",
            ]
        ),
    )
    atomic_write(output / "speculators.patch", git_diff(root))


def check_endpoint(  # noqa: C901 -- explicit correctness checks
    target, validation_manifest, output, *, reference=None, limit=0, seed=42
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new correctness output: {output}")
    manifest, groups = load_validation(validation_manifest)
    chosen = selected_rows(groups, limit)
    rows = [row for group in chosen.values() for row in group]
    before = server_status(target)
    validate_status(before, manifest)
    model = http_json(endpoint_root(target) + "/v1/models")["data"][0]["id"]
    baseline = json.loads(Path(reference).read_text()) if reference else None
    if baseline is not None:
        if (
            baseline["status"] != "reference"
            or baseline["validation_identity"] != manifest["identity"]
        ):
            raise ValueError(
                "Reference must use target-only decoding and this validation manifest"
            )
        if baseline.get("target_model") != manifest["models"]["target"]:
            raise ValueError("Reference target model revision differs")
    elif before is not None and before["configuration"]["kind"] != "target_only":
        raise ValueError(
            "Adapter/EAGLE3 checks require --reference from target-only decoding"
        )
    check_tokenization(target, model, rows)
    results = {}
    mismatches = []
    for row in rows:
        fingerprint = row_identity(row, seed)
        expected = baseline["rows"].get(row["row_id"]) if baseline else None
        if baseline and (expected is None or expected["identity"] != fingerprint):
            raise ValueError(
                "Reference prompt membership or generation settings differ"
            )
        reply = http_json(
            endpoint_root(target) + "/v1/completions",
            {
                "model": model,
                "prompt": row["prompt_token_ids"],
                "max_tokens": row["output_tokens_count"],
                "temperature": 0,
                "seed": seed,
                "stream": False,
                "add_special_tokens": False,
                "return_token_ids": True,
                "ignore_eos": False,
            },
        )
        tokens = reply["choices"][0].get("token_ids")
        if not isinstance(tokens, list):
            raise ValueError("Correctness checks require vLLM return_token_ids support")
        results[row["row_id"]] = {
            "identity": fingerprint,
            "output_token_ids": tokens,
            "category": row["category"],
            "subset": row["subset"],
        }
        if expected is not None and tokens != expected["output_token_ids"]:
            mismatches.append(row["row_id"])
    after = server_status(target)
    validate_status(after, manifest)
    diagnostics = validate_adapter_delta(before, after)
    report = {
        "version": 1,
        "status": "unverified"
        if mismatches
        else "matched"
        if baseline
        else "reference",
        "validation_identity": manifest["identity"],
        "target_model": manifest["models"]["target"],
        "server_configuration": before["configuration"] if before else None,
        "reference_sha256": file_digest(Path(reference)) if reference else None,
        "rows": results,
        "mismatches": mismatches,
        "adapter_metrics": diagnostics,
    }
    write_json(output, report)
    save_provenance(output.parent)
    if mismatches:
        raise ValueError(
            f"Output token disagreement for {len(mismatches)} prompts; see {output}"
        )
    return {"status": report["status"], "requests": len(results), "output": str(output)}
