# ruff: noqa: INP001
# Standalone documentation artifact, not an importable package.
"""Validate the supplied aggregate suite and regenerate detailed_analysis.json.

Run with Python 3 from any directory. No model execution or third-party packages.
CUDA-event durations are not a wall-time partition or isolated drafter timings.
"""

import hashlib
import json
import math
from pathlib import Path

MODEL_COUNT = 5
REQUEST_COUNT = 50


def require(condition):
    if not condition:
        raise ValueError("Benchmark consistency check failed")


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "suite_summary_detailed.json"
summary = json.loads(SOURCE.read_text())
prior = json.loads((ROOT / "metrics.json").read_text())
results = summary["results"]
rows = {row["model"]: row for row in summary["rows"]}
require(len(results) == len(rows) == MODEL_COUNT)
require(summary["correctness_status"] == "not_checked")


def close(left, right):
    require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12))


def measurement(model, domain):
    return results[model]["nemotron/nemo-" + domain]


hashes = {}
subsets = {}
measurements = []
for model in rows:
    require(rows[model]["concurrency"] == 1)
    for domain in ("math", "chat"):
        result = measurement(model, domain)
        acceptance, adapter = result["acceptance"], result["adapter_metrics"]
        require(len(result["performance"]) == 1)
        perf = result["performance"][0]
        require(result["correctness_status"] == "not_checked")
        require(result["measurement_kind"] == "benchmark")
        require(result["output_mismatch_override"] is False)
        require(perf["successful_requests"] == REQUEST_COUNT)
        for key in (
            "prepared_requests",
            "completed_requests",
            "prepared_batches",
            "completed_batches",
            "conditioning_examples",
        ):
            require(adapter[key] == REQUEST_COUNT)
        close(adapter["mean_batch_size"], 1.0)
        expected_calls = 50 if rows[model]["type"] == "generator" else 0
        require(adapter["generator_invocations"] == expected_calls)
        if expected_calls:
            close(adapter["mean_conditioning_size"], 1.0)
            close(
                adapter["generator_ms_per_call"],
                adapter["generator_seconds"] * 1000 / expected_calls,
            )
        proposed, accepted, drafts = (
            acceptance[key]
            for key in ("num_draft_tokens", "num_accepted_tokens", "num_drafts")
        )
        close(proposed, 3 * drafts)
        close(acceptance["draft_acceptance_rate"], accepted / proposed)
        close(acceptance["accepted_prefix_length"], accepted / drafts)
        close(acceptance["acceptance_length"], 1 + accepted / drafts)
        close(
            sum(acceptance["acceptance_at_pos_" + str(i)] for i in range(3)),
            accepted / drafts,
        )
        close(perf["total_output_tokens"], 12800 if domain == "math" else 11032)
        close(
            perf["output_tokens_per_second"],
            perf["total_output_tokens"] / perf["measured_seconds"],
        )
        close(rows[model]["nemotron/nemo-" + domain], perf["output_tokens_per_second"])
        old_row = next(row for row in prior["rows"] if row["model"] == model)
        require(
            round(perf["output_tokens_per_second"], 2)
            == old_row[domain + "_output_tokens_per_second"]
        )
        hashes.setdefault(domain, result["dataset_sha256"])
        subsets.setdefault(domain, result["subsets"])
        require(hashes[domain] == result["dataset_sha256"])
        require(subsets[domain] == result["subsets"])
        times = {
            name + "_ms_per_request": adapter[name + "_seconds"] * 1000 / 50
            for name in ("generator", "merge", "restore")
        }
        total_seconds = sum(
            adapter[name + "_seconds"] for name in ("generator", "merge", "restore")
        )
        measurements.append(
            dict(
                model=model,
                domain=domain,
                draft_acceptance_percent=100 * acceptance["draft_acceptance_rate"],
                acceptance_length=acceptance["acceptance_length"],
                **times,
                timed_adapter_ms_per_request=1000 * total_seconds / 50,
                timed_adapter_percent_of_measured_wall=100
                * total_seconds
                / perf["measured_seconds"],
                ttft_median_ms=perf["ttft_median_ms"],
                tpot_median_ms=perf["tpot_median_ms"],
            )
        )

pairs = json.loads((ROOT / "analysis.json").read_text())["comparisons"]
comparisons = []
for pair in pairs:
    candidate, reference = pair["candidate"], pair["reference"]
    delta = {}
    for domain in ("math", "chat"):
        c, r = measurement(candidate, domain), measurement(reference, domain)
        ca, ra = c["acceptance"], r["acceptance"]
        cp, rp = c["performance"][0], r["performance"][0]
        delta[domain] = {
            "throughput_change_percent": 100
            * (cp["output_tokens_per_second"] / rp["output_tokens_per_second"] - 1),
            "draft_acceptance_change_percentage_points": 100
            * (ca["draft_acceptance_rate"] - ra["draft_acceptance_rate"]),
            "acceptance_length_change_percent": 100
            * (ca["acceptance_length"] / ra["acceptance_length"] - 1),
            "ttft_median_change_ms": cp["ttft_median_ms"] - rp["ttft_median_ms"],
            "tpot_median_change_ms": cp["tpot_median_ms"] - rp["tpot_median_ms"],
        }
        if (
            c["adapter_metrics"]["generator_invocations"]
            and r["adapter_metrics"]["generator_invocations"]
        ):
            delta[domain]["generator_ms_per_call_change_percent"] = 100 * (
                c["adapter_metrics"]["generator_ms_per_call"]
                / r["adapter_metrics"]["generator_ms_per_call"]
                - 1
            )
    comparisons.append(
        {"candidate": candidate, "reference": reference, "changes": delta}
    )
analysis = {
    "source": SOURCE.name,
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "suite_identity": summary["suite_identity"],
    "source_precision": (
        "Full-precision numbers in user-supplied detailed suite summary"
    ),
    "validation": (
        "All ten aggregate measurements pass counts, acceptance identities, "
        "throughput identities, row/rounded-table agreement and shared "
        "workload checks."
    ),
    "dataset_sha256": hashes,
    "measurements": measurements,
    "comparisons": comparisons,
    "limitations": [
        "One measurement per model/domain; no repeated-trial uncertainty.",
        (
            "Correctness not checked; equal output-token counts do not imply "
            "identical token sequences."
        ),
        (
            "CUDA-event adapter durations are not a wall-time partition; host, "
            "synchronization and persistent-memory effects are not isolated."
        ),
        "No isolated drafter-forward or verifier timings in supplied summary.",
        (
            "Checkpoint hashes, executed hardware/code provenance and "
            "request-level results not supplied."
        ),
    ],
}
(ROOT / "detailed_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
print(analysis["validation"])  # noqa: T201 -- CLI validation result
