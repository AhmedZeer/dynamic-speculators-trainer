"""Component GPU intervals are collected without synchronizing each operation."""

from types import SimpleNamespace

import pytest
import torch

from speculators.generator.campaign_train import UpdateTimings
from speculators.generator.tracking_layout import sectioned_metrics


def test_cuda_component_intervals_are_resolved_after_collection(monkeypatch):
    records, resolutions = [], []

    class Event:
        def __init__(self, *, enable_timing):
            assert enable_timing

        def record(self):
            records.append(self)

        def elapsed_time(self, end):
            assert self in records
            assert end in records
            resolutions.append((self, end))
            return 3.0

    def no_sync(*_):
        pytest.fail("A component timer synchronized CUDA")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", no_sync)
    timers = UpdateTimings(SimpleNamespace(device=SimpleNamespace(type="cuda")))
    with timers.measure("generator_forward"):
        pass
    with timers.measure("drafter_forward"):
        pass
    with timers.measure("drafter_forward"):
        pass
    assert len(records) == 6
    assert not resolutions
    seconds = timers.seconds()
    assert seconds["generator_forward"] == pytest.approx(0.003)
    assert seconds["drafter_forward"] == pytest.approx(0.006)
    assert seconds["generator_backward"] == 0
    assert len(resolutions) == 3


def test_training_component_metrics_use_existing_wandb_section():
    values = {f"train/{name}_seconds": 0.2 for name in UpdateTimings.COMPONENTS}
    tracked = sectioned_metrics(values)
    assert tracked == {
        f"training_performance/{name}_seconds": 0.2 for name in UpdateTimings.COMPONENTS
    }
