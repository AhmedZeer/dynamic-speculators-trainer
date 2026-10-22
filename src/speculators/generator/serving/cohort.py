"""Closed cohorts: never admit a replacement request into an active batch."""

import json
import os
import time


class FixedBatchMixin:
    """Wrap vLLM 0.31's FCFS scheduler without changing its KV allocation policy."""

    def schedule(self, *args, **kwargs):  # noqa: C901 -- closed admission and KV guards
        config = json.loads(os.environ["SPECULATORS_SERVER_CONFIG"])
        capacity = config["max_num_seqs"]
        wait_seconds = config.get("batch_wait_ms", 50) / 1000
        active = getattr(self, "_cohort_ids", set())
        active.intersection_update(request.request_id for request in self.running)
        waiting = self.waiting
        selected = []
        if self.kv_holding_waiting:
            raise RuntimeError("Fixed batches do not support external KV transfers")
        if not active and waiting:
            since = getattr(self, "_collect_since", None)
            if since is None:
                self._collect_since = since = time.monotonic()
            if len(waiting) >= capacity or time.monotonic() - since >= wait_seconds:
                selected = [
                    waiting.pop_request() for _ in range(min(capacity, len(waiting)))
                ]
                self._collect_since = None
        elif not waiting:
            self._collect_since = None
        # Calls during decoding see no waiting requests, so vacant slots stay
        # empty until the last member finishes. Newly arriving work stays queued.
        self.waiting = type(waiting)()
        for request in selected:
            self.waiting.add_request(request)
        try:
            output = super().schedule(*args, **kwargs)
        finally:
            leftovers = list(self.waiting)
            for request in reversed(leftovers):
                waiting.prepend_request(request)
            self.waiting = waiting
        if output.preempted_req_ids:
            raise RuntimeError(
                "Fixed batch KV preemption is unsupported; "
                "reduce concurrency or context length"
            )
        if selected:
            expected = {request.request_id for request in selected}
            actual = {request.req_id for request in output.scheduled_new_reqs}
            if expected != actual:
                raise RuntimeError(
                    "Entire fixed batch must fit in one prefill; "
                    "reduce concurrency or context length"
                )
            active = expected
        self._cohort_ids = active
        if not active and waiting and not selected:
            # EngineCore checks new inputs between schedule calls. A short yield
            # avoids spinning while collecting a partial cohort's final members.
            time.sleep(0.001)
        return output
