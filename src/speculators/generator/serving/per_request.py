"""One fixed, singleton-conditioned draft adapter per continuous-batched request."""

import logging
from contextlib import contextmanager

import torch

from speculators.generator.serving.bundle import bundle_state, load_generator
from speculators.generator.serving.controller import (
    FINISHED_RECORD_LIMIT,
    AdapterMerger,
    DraftAdapterController,
    elapsed,
)
from speculators.generator.serving.multilora import NativeDraftLoRA
from speculators.generator.serving.ranges import attach_ranges, trace_range, traced

logger = logging.getLogger(__name__)


class PerRequestAdapterController(DraftAdapterController):
    continuous = True

    def __init__(self, bundle, draft, config, *, backend_factory=NativeDraftLoRA):
        self.metadata, state = bundle_state(bundle)
        if self.metadata["kind"] != "generator":
            raise ValueError("Per-request adapters require a generator bundle")
        # Reuse the established shape checks before native wrappers replace layers.
        checked = AdapterMerger(draft, self.metadata)
        self.device = checked.device
        del checked
        self.generator = load_generator(self.metadata, state).to(self.device)
        self.backend = backend_factory(draft, self.metadata, config)
        self.setup_merge_seconds = 0.0
        self._initialize_tracking()
        self.totals.update(
            draft_forward_calls=0, draft_forward_requests=0, draft_forward_tokens=0
        )
        self.adapters = {}
        self.next_id = 1
        self.scheduled_ids = []

    def _generate(self, summary, length):
        if summary.shape[1] != self.metadata["input_dim"]:
            raise ValueError("Auxiliary layer order/width differs from training")
        factors = self.generator(summary, torch.tensor([length], device=self.device))
        retained = {}
        for name, (a, b) in factors.items():
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Non-finite generated adapter")
            with trace_range("speculators.adapter_to_cpu"):
                retained[name] = (a.detach().cpu(), b.detach().cpu())
        return retained

    @torch.inference_mode()
    @traced("speculators.adapter_prepare")
    def prepare_scheduled(self, ids, lengths, positions, states, offsets):
        """Generate only for new requests; cached adapters survive KV recomputation."""
        count = len(ids)
        if (
            not count
            or len(set(ids)) != count
            or len(lengths) != count
            or len(offsets) != count + 1
            or offsets[0] != 0
            or offsets[-1] != states.shape[0]
            or positions.numel() != states.shape[0]
            or any(
                end <= start for start, end in zip(offsets, offsets[1:], strict=False)
            )
        ):
            raise ValueError("Invalid continuous-batch request/token boundaries")
        for index, request_id in enumerate(ids):
            if request_id in self.adapters:
                continue
            start, end = offsets[index : index + 2]
            length = lengths[index]
            if end - start != length or not torch.equal(
                positions[start:end], torch.arange(length, device=positions.device)
            ):
                raise ValueError("New request requires complete target prefill")
            summary = states[end - 1 : end].float()
            factors, seconds = elapsed(
                self.device,
                lambda summary=summary, length=length: self._generate(summary, length),
            )
            self.adapters[request_id] = (self.next_id, factors)
            self.next_id += 1
            self.active_requests.add(request_id)
            self.completed_records.pop(request_id, None)
            self.records[request_id] = {
                "request_id": request_id,
                "adapter_id": self.adapters[request_id][0],
                "generator_invocations": 1,
                "generator_seconds": seconds,
                "merge_seconds": 0.0,
                "restore_seconds": 0.0,
                "deployment_context_size": 1,
            }
            for key in (
                "prepared_requests",
                "prepared_batches",
                "conditioning_examples",
                "generator_invocations",
            ):
                self.totals[key] += 1
            self.totals["generator_seconds"] += seconds
            logger.info(
                "Request adapter ready: request=%s; generation=%.4fs",
                request_id,
                seconds,
            )
        self.scheduled_ids = list(ids)
        # Ordinary decode/reordering only needs a new token mapping. Avoid a
        # synchronized slot-loading boundary on every proposal when IDs are stable.
        if set(ids) != set(self.backend.resident):
            _, seconds = elapsed(
                self.device,
                lambda: self.backend.activate(
                    {name: self.adapters[name] for name in ids}
                ),
            )
            self.totals["adapter_load_seconds"] += seconds
        self.active = next(iter(self.active_requests), None)
        self.record = self.records.get(self.active, {})

    @traced("speculators.token_mapping_build")
    def map_forward(self, metadata):
        """Map each forward's final query layout, including later draft steps."""
        offsets = metadata.query_start_loc_cpu.tolist()
        if len(offsets) != len(self.scheduled_ids) + 1 or offsets[0] != 0:
            raise ValueError("Draft forward request order/boundaries differ")
        ids = [self.adapters[name][0] for name in self.scheduled_ids]
        tokens = []
        for adapter_id, start, end in zip(ids, offsets, offsets[1:], strict=False):
            if end <= start:
                raise ValueError("Empty or reversed draft query")
            tokens.extend([adapter_id] * (end - start))
        self.backend.map_tokens(tokens, ids)
        self.totals["draft_forward_calls"] += 1
        self.totals["draft_forward_requests"] += len(ids)
        self.totals["draft_forward_tokens"] += len(tokens)

    @traced("speculators.adapter_cleanup")
    def finish(self, request_id):
        if request_id in self.adapters:
            self.backend.release(request_id)
            del self.adapters[request_id]
            self.active_requests.remove(request_id)
            record = self.records.pop(request_id)
            self.totals["completed_requests"] += 1
            self.totals["completed_batches"] += 1
            self.last_record = dict(record)
            self.completed_records[request_id] = dict(record)
            while len(self.completed_records) > FINISHED_RECORD_LIMIT:
                self.completed_records.pop(next(iter(self.completed_records)))
            self.active = next(iter(self.active_requests), None)
            self.record = self.records.get(self.active, {})
        return dict(self.completed_records.get(request_id, {}))

    @contextmanager
    def profiling(self, num_tokens):
        """Exercise actual multi-LoRA kernels during vLLM memory profiling."""
        if self.active_requests:
            raise RuntimeError("Cannot profile while request adapters are active")
        with torch.inference_mode():
            factors = self.generator(
                torch.zeros(1, self.metadata["input_dim"], device=self.device),
                torch.ones(1, device=self.device),
            )
            factors = {
                name: tuple(t.cpu() for t in pair) for name, pair in factors.items()
            }
        count = min(num_tokens, self.backend.capacity)
        dummy = {f"profile-{i}": (self.next_id + i, factors) for i in range(count)}
        try:
            self.backend.activate(dummy)
            ids = [entry[0] for entry in dummy.values()]
            self.backend.map_tokens([ids[i % count] for i in range(num_tokens)], ids)
            yield
        finally:
            for name in dummy:
                self.backend.release(name)
            self.backend.map_tokens([], [])


def create_controller(proposer):
    import json  # noqa: PLC0415
    import os  # noqa: PLC0415

    bundle = proposer.speculative_config.draft_adapter_path
    serving = json.loads(os.environ.get("SPECULATORS_SERVER_CONFIG", "{}"))
    if serving.get("adapter_experiment") is not None:
        from speculators.generator.serving.experiments import (  # noqa: PLC0415
            ExperimentalAdapterController,
        )

        return attach_ranges(
            ExperimentalAdapterController(bundle, proposer.model), proposer.model
        )
    if serving.get("adapter_policy") == "per_request":
        return attach_ranges(
            PerRequestAdapterController(bundle, proposer.model, proposer.vllm_config),
            proposer.model,
        )
    from speculators.generator.serving import controller  # noqa: PLC0415

    return attach_ranges(
        controller.DraftAdapterController(bundle, proposer.model), proposer.model
    )
