"""Frozen checkpoint coverage and paired acceptance deltas for scaling studies."""

import csv
import json
import signal
from types import SimpleNamespace

import pytest

from scripts import benchmark_attention_scaling as study
from speculators.benchmark.config import Suite


@pytest.mark.parametrize("ttt", [3, 7])
def test_pinned_suite_matrix_and_shared_prompts(tmp_path, ttt):
    counts = {}
    manifests = set()
    for policy in study.PHASES["all"]:
        config = Suite.model_validate(study.suite_config(tmp_path, policy, ttt, "2", 8000))
        counts[policy] = len(config.models) * 3
        assert config.settings.speculative_tokens == ttt
        assert config.settings.concurrency == 1
        assert config.benchmarks[0].categories == ["math", "stem", "chat"]
        manifests.add(config.benchmarks[0].validation_manifest)
        assert all(m.revision in (study.GENERATOR[1], study.STATIC[1]) for m in config.models)
        if policy == "static":
            assert all(m.type == "lora" and m.conditioning_examples is None for m in config.models)
        else:
            assert all(m.adapter_experiment.conditioning == policy for m in config.models)
            assert all(m.adapter_experiment.scale == 2 for m in config.models)
    assert counts == {"correct": 18, "static": 15, "fixed": 18, "cross_category": 18}
    assert len(manifests) == 1


def write_summaries(root, *, mismatch=False):
    for policy in study.PHASES["all"]:
        results = {}
        rate = {"correct": .6, "static": .5, "fixed": .4, "cross_category": .3}[policy]
        for model in study.suite_config(root, policy, 3, "0", 8000)["models"]:
            results[model["name"]] = {
                f"nemotron/nemo-{c}": {
                    "dataset_sha256": "different" if mismatch and policy == "fixed" else c,
                    "performance": [{"output_tokens_per_second": 100}],
                    "acceptance": {"draft_acceptance_rate": rate},
                } for c in study.CATEGORIES
            }
        path = root / f"suites/ttt3/{policy}/summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"results": results}))


def test_collection_keeps_all_steps_categories_and_signed_pp_deltas(tmp_path):
    write_summaries(tmp_path)
    study.collect(tmp_path, 3, "all")
    with (tmp_path / "tables/scaling_ttt3.csv").open() as f:
        scaling = list(csv.DictReader(f))
    with (tmp_path / "tables/dynamism_ttt3.csv").open() as f:
        dynamism = list(csv.DictReader(f))
    assert len(scaling) == 30  # 5 steps x 2 methods x 3 categories.
    assert {r["steps"] for r in scaling} == {"1000", "2000", "3000", "4000", "5000"}
    assert len(dynamism) == 18
    assert {r["steps"] for r in dynamism} == {str(s * 1000) for s in range(1, 7)}
    assert all(float(r["correct_minus_fixed_pp"]) == pytest.approx(20) for r in dynamism)
    assert all(float(r["cross_category_minus_fixed_pp"]) == pytest.approx(-10) for r in dynamism)


def test_collection_rejects_unpaired_workloads(tmp_path):
    write_summaries(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="hashes differ"):
        study.collect(tmp_path, 3, "all")


def test_frozen_config_cannot_change_gpu_or_checkpoint_pin(tmp_path):
    path = tmp_path / "suite.yaml"
    config = study.suite_config(tmp_path, "correct", 3, "0", 8000)
    study.frozen(path, config)
    study.frozen(path, config)
    config["models"][0]["revision"] = "changed"
    with pytest.raises(ValueError, match="Configuration changed"):
        study.frozen(path, config)


def test_interrupted_cli_receives_termination_and_cleanup(monkeypatch):
    calls = []
    waits = [KeyboardInterrupt(), None]

    def wait(**kwargs):
        result = waits.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    process = SimpleNamespace(pid=12345, wait=wait, poll=lambda: None)
    monkeypatch.setattr(study.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(study.os, "killpg", lambda *a: calls.append(a))
    with pytest.raises(KeyboardInterrupt):
        study.run(["child"], {})
    assert calls == [(12345, signal.SIGTERM)]
