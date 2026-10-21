"""Speedup calculations require the same measured workload and settings."""

import json

import pytest

from scripts.evaluate.compare_http import compare


def write_measurement(path, *, kind, throughput, latency, failed=False):
    path.mkdir()
    identity = {
        "validation": "same-validation",
        "mode": "throughput",
        "server": {
            "kind": kind,
            "models": {"target": {"id": "target", "revision": "fixed"}},
        },
        "correctness_status": "unverified" if failed else "reference",
        "output_mismatch_override": failed,
    }
    group = {
        "dataset_sha256": "same-workload",
        "performance": [
            {
                "successful_requests": 50,
                "output_tokens_per_second": throughput,
                "latency_median_s": latency,
                "total_output_tokens": 12800,
            }
        ],
    }
    (path / "benchmark_identity.json").write_text(json.dumps(identity))
    (path / "results.json").write_text(json.dumps({"groups": {"nemo-math": group}}))


def test_speedup_and_unverified_label(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "generator"
    write_measurement(base, kind="target_only", throughput=100, latency=4)
    write_measurement(
        candidate, kind="generator", throughput=150, latency=2, failed=True
    )
    result = compare(base, candidate)[0]
    assert result["throughput_speedup"] == 1.5
    assert result["median_latency_speedup"] == 2
    assert result["exploratory"] is True
    assert result["candidate_correctness_status"] == "unverified"
    data = json.loads((candidate / "results.json").read_text())
    data["groups"]["nemo-math"]["dataset_sha256"] = "different"
    (candidate / "results.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="prompts differ"):
        compare(base, candidate)


def test_speedup_rejects_different_budget(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "generator"
    write_measurement(base, kind="eagle3_base", throughput=100, latency=4)
    write_measurement(candidate, kind="generator", throughput=150, latency=2)
    data = json.loads((candidate / "benchmark_identity.json").read_text())
    data["concurrency"] = 2
    (candidate / "benchmark_identity.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="concurrency"):
        compare(base, candidate)


def test_unchecked_speedup_remains_exploratory(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "generator"
    write_measurement(base, kind="target_only", throughput=100, latency=4)
    write_measurement(candidate, kind="generator", throughput=150, latency=2)
    path = candidate / "benchmark_identity.json"
    identity = json.loads(path.read_text())
    identity["correctness_status"] = "not_checked"
    path.write_text(json.dumps(identity))
    result = compare(base, candidate)[0]
    assert result["exploratory"] is True
    assert result["candidate_correctness_status"] == "not_checked"
