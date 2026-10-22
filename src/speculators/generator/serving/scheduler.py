"""Loaded only inside the pinned vLLM engine process."""

from vllm.v1.core.sched.scheduler import Scheduler

from speculators.generator.serving.cohort import FixedBatchMixin


class FixedBatchScheduler(FixedBatchMixin, Scheduler):
    """FCFS fixed batches sharing one adapter until their final request ends."""
