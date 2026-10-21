"""Compare saved HTTP measurements without launching servers or loading weights."""

import argparse
import csv
import json
from pathlib import Path


def compare(baseline, candidate):  # noqa: C901 -- explicit comparison compatibility checks
    baseline, candidate = Path(baseline), Path(candidate)
    identities = [
        json.loads((path / "benchmark_identity.json").read_text())
        for path in (baseline, candidate)
    ]
    for key in (
        "validation",
        "guidellm_version",
        "mode",
        "categories",
        "grouping",
        "limit",
        "seed",
        "concurrency",
        "warmups",
        "sweep_size",
    ):
        if identities[0].get(key) != identities[1].get(key):
            raise ValueError(f"Benchmark settings differ: {key}")
    if identities[0]["mode"] != "throughput":
        raise ValueError("Speedup comparison currently requires throughput mode")
    servers = [identity["server"] for identity in identities]
    if not all(servers):
        raise ValueError(
            "Comparison requires recorded Speculators server configurations"
        )
    if servers[0]["models"]["target"] != servers[1]["models"]["target"]:
        raise ValueError("Target models differ")
    for key in ("max_model_len", "max_num_seqs", "model_runner", "vllm_version"):
        if servers[0].get(key) != servers[1].get(key):
            raise ValueError(f"Server settings differ: {key}")
    results = [
        json.loads((path / "results.json").read_text())["groups"]
        for path in (baseline, candidate)
    ]
    if results[0].keys() != results[1].keys():
        raise ValueError("Measured groups differ")
    rows = []
    for group, base in results[0].items():
        other = results[1][group]
        if base["dataset_sha256"] != other["dataset_sha256"]:
            raise ValueError(f"Measured prompts differ: {group}")
        if len(base["performance"]) != 1 or len(other["performance"]) != 1:
            raise ValueError("Expected one throughput measurement per group")
        b, c = base["performance"][0], other["performance"][0]
        if b["successful_requests"] != c["successful_requests"]:
            raise ValueError(f"Request counts differ: {group}")
        if min(b["output_tokens_per_second"], c["latency_median_s"]) <= 0:
            raise ValueError("Comparison requires positive throughput and latency")
        rows.append(
            {
                "group": group,
                "baseline_directory": str(baseline),
                "candidate_directory": str(candidate),
                "baseline_kind": servers[0]["kind"],
                "candidate_kind": servers[1]["kind"],
                "baseline_output_tokens_per_second": b["output_tokens_per_second"],
                "candidate_output_tokens_per_second": c["output_tokens_per_second"],
                "throughput_speedup": c["output_tokens_per_second"]
                / b["output_tokens_per_second"],
                "median_latency_speedup": b["latency_median_s"] / c["latency_median_s"],
                "baseline_output_tokens": b["total_output_tokens"],
                "candidate_output_tokens": c["total_output_tokens"],
                "baseline_correctness_status": identities[0].get(
                    "correctness_status", "unknown"
                ),
                "candidate_correctness_status": identities[1].get(
                    "correctness_status", "unknown"
                ),
                "exploratory": any(
                    i.get("output_mismatch_override", False) for i in identities
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = compare(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
