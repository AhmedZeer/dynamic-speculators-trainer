"""Closed-cohort scheduling and one generated adapter per packed prefill group."""

import json
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from speculators.generator.serving import cohort
from speculators.generator.serving.controller import (
    DraftAdapterController,
    prepare_proposer,
    validate_runtime,
)
from speculators.generator.serving.endpoint import validate_adapter_delta
from speculators.generator.serving.http import finish_http_request
from tests.unit.generator.test_serving import draft, valid_config, write_bundle


@pytest.mark.parametrize("kind", ["generator", "static_lora"])
def test_shared_adapter_is_generated_once_and_restored_only_after_last_member(
    tmp_path, monkeypatch, kind
):
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG",
        json.dumps(
            {
                "batching_policy": "fixed_shared_adapter",
                "static_adapter_policy": "merged_once",
            }
        ),
    )
    write_bundle(tmp_path / "bundle", kind)
    model = draft()
    controller = DraftAdapterController(tmp_path / "bundle", model)
    baseline = {
        name: value.clone() for name, value in controller.merger.weights.items()
    }
    states = torch.randn(5, 12)
    calls = []
    if controller.generator:
        controller.generator.register_forward_pre_hook(
            lambda _, args: calls.append(args)
        )
    controller.prepare_batch(
        ["one-1234abcd", "two-5678abcd"],
        [2, 3],
        torch.tensor([0, 1, 0, 1, 2]),
        states,
        [0, 2, 5],
    )
    if kind == "generator":
        assert len(calls) == 1
        torch.testing.assert_close(calls[0][0], states[[1, 4]])
        assert calls[0][1].tolist() == [2, 3]
    merged = {name: value.clone() for name, value in controller.merger.weights.items()}
    controller.prepare_batch(
        ["two-5678abcd"], [3], torch.tensor([8]), states[:1], [0, 1]
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(drafter=SimpleNamespace(draft_adapter=controller))
    )
    first = finish_http_request(worker, "one")
    assert first["deployment_context_size"] == 2
    assert controller.active_requests == {"two-5678abcd"}
    for name, value in controller.merger.weights.items():
        torch.testing.assert_close(value, merged[name])
    with pytest.raises(RuntimeError, match="unfinished fixed batch"):
        controller.prepare_batch(["new"], [1], torch.tensor([0]), states[:1], [0, 1])
    controller.finish("two-5678abcd")
    assert controller.active is None
    for name, value in controller.merger.weights.items():
        torch.testing.assert_close(value, baseline[name])
    assert controller.totals["prepared_requests"] == 2
    assert controller.totals["completed_requests"] == 2
    assert controller.totals["prepared_batches"] == 1
    assert controller.totals["completed_batches"] == 1
    assert controller.totals["generator_invocations"] == int(kind == "generator")
    assert finish_http_request(worker, "one") == first  # delayed cleanup is idempotent
    controller.prepare_batch(
        ["new-aaaa1234"], [2], torch.arange(2), states[:2] + 1, [0, 2]
    )
    current = {name: value.clone() for name, value in controller.merger.weights.items()}
    finish_http_request(worker, "one")
    assert controller.active_requests == {"new-aaaa1234"}
    for name, value in controller.merger.weights.items():
        torch.testing.assert_close(value, current[name])
    controller.finish("new-aaaa1234")


def test_packed_proposer_uses_attention_boundaries_and_keeps_context_fixed(tmp_path):
    write_bundle(tmp_path / "bundle")
    controller = DraftAdapterController(tmp_path / "bundle", draft())
    proposer = SimpleNamespace(
        draft_adapter=controller,
        adapter_runner=SimpleNamespace(
            input_batch=SimpleNamespace(req_ids=["a", "b"]),
            requests={
                "a": SimpleNamespace(num_prompt_tokens=2),
                "b": SimpleNamespace(num_prompt_tokens=3),
            },
        ),
    )
    metadata = SimpleNamespace(query_start_loc_cpu=torch.tensor([0, 2, 5]))
    prepare_proposer(
        proposer, torch.tensor([0, 1, 0, 1, 2]), torch.randn(5, 12), metadata
    )
    prepare_proposer(proposer, torch.tensor([2, 3]), torch.randn(2, 12), metadata)
    assert controller.totals["generator_invocations"] == 1
    assert controller.totals["conditioning_examples"] == 2
    controller.finish("a")
    controller.finish("b")
    with pytest.raises(ValueError, match="complete target prefills"):
        controller.prepare_batch(
            ["a", "b"],
            [2, 3],
            torch.tensor([1, 2, 0, 1, 2]),
            torch.randn(5, 12),
            [0, 2, 5],
        )


def test_batch_runtime_requires_fixed_admission_and_reports_batch_counts(
    tmp_path, monkeypatch
):
    metadata, _ = write_bundle(tmp_path / "bundle")
    config = valid_config(metadata)
    config.scheduler_config.max_num_seqs = 4
    monkeypatch.setenv("SPECULATORS_SERVER_CONFIG", "{}")
    with pytest.raises(ValueError, match="fixed batch scheduler"):
        validate_runtime(config, metadata)
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG", '{"batching_policy":"fixed_shared_adapter"}'
    )
    with pytest.raises(ValueError, match="fixed batch scheduler"):
        validate_runtime(config, metadata)
    config.scheduler_config.scheduler_cls = (
        "speculators.generator.serving.scheduler.FixedBatchScheduler"
    )
    validate_runtime(config, metadata)
    before = {
        "configuration": {"kind": "generator"},
        "workers": [
            {
                "adapter_totals": {
                    "prepared_requests": 0,
                    "completed_requests": 0,
                    "prepared_batches": 0,
                    "completed_batches": 0,
                    "generator_invocations": 0,
                }
            }
        ],
    }
    after = {
        "configuration": before["configuration"],
        "workers": [
            {
                "adapter_totals": {
                    "prepared_requests": 6,
                    "completed_requests": 6,
                    "prepared_batches": 2,
                    "completed_batches": 2,
                    "generator_invocations": 2,
                }
            }
        ],
    }
    assert validate_adapter_delta(before, after)["mean_batch_size"] == 3
    after["workers"][0]["adapter_totals"]["completed_batches"] = 1
    with pytest.raises(ValueError, match="not completely finished"):
        validate_adapter_delta(before, after)


class Queue(deque):
    def pop_request(self):
        return self.popleft()

    def add_request(self, request):
        self.append(request)

    def prepend_request(self, request):
        self.appendleft(request)


class ParentScheduler:
    def schedule(self, *args, **kwargs):
        selected = list(self.waiting)
        self.waiting.clear()
        self.running.extend(selected)
        return SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=req.request_id) for req in selected
            ],
            preempted_req_ids=set(),
        )


class Scheduler(cohort.FixedBatchMixin, ParentScheduler):
    def __init__(self):
        self.waiting = Queue()
        self.kv_holding_waiting = Queue()
        self.running = []


def test_scheduler_collects_partial_batches_and_does_not_refill_vacant_slots(
    monkeypatch,
):
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG", '{"max_num_seqs":2,"batch_wait_ms":50}'
    )
    now = [0.0]
    monkeypatch.setattr(cohort.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(cohort.time, "sleep", lambda *_: None)
    scheduler = Scheduler()
    a, b, c = [SimpleNamespace(request_id=name) for name in ("a", "b", "c")]
    scheduler.waiting.add_request(a)
    assert not scheduler.schedule().scheduled_new_reqs
    scheduler.waiting.add_request(b)
    assert len(scheduler.schedule().scheduled_new_reqs) == 2
    scheduler.running.remove(a)
    scheduler.waiting.add_request(c)
    assert not scheduler.schedule().scheduled_new_reqs
    assert list(scheduler.waiting) == [c]
    scheduler.running.clear()
    assert not scheduler.schedule().scheduled_new_reqs
    now[0] = 0.06
    assert len(scheduler.schedule().scheduled_new_reqs) == 1
    assert scheduler.running == [c]


def test_scheduler_rejects_preemption_and_incomplete_admission(monkeypatch):
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG", '{"max_num_seqs":2,"batch_wait_ms":0}'
    )
    scheduler = Scheduler()
    scheduler.waiting.add_request(SimpleNamespace(request_id="a"))
    monkeypatch.setattr(
        ParentScheduler,
        "schedule",
        lambda *a, **kw: SimpleNamespace(
            scheduled_new_reqs=[],
            preempted_req_ids=set(),
        ),
    )
    with pytest.raises(RuntimeError, match="Entire fixed batch"):
        scheduler.schedule()
    monkeypatch.setattr(
        ParentScheduler,
        "schedule",
        lambda *a, **kw: SimpleNamespace(
            scheduled_new_reqs=[],
            preempted_req_ids={"a"},
        ),
    )
    with pytest.raises(RuntimeError, match="preemption"):
        scheduler.schedule()
