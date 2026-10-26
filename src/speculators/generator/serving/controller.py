"""Generated request adapters and permanent static LoRAs for the vLLM hook."""

import json
import logging
import os
import random
import time
from pathlib import Path

import torch

from speculators.generator.serving.bundle import (
    VLLM_VERSION,
    bundle_state,
    load_generator,
    read_bundle,
)
from speculators.generator.serving.ranges import traced

logger = logging.getLogger(__name__)
MATRIX_NDIM = 2
REQUEST_ID_SUFFIX_LENGTH = 8
FINISHED_RECORD_LIMIT = 128
MAX_BATCH_SIZE = 8


def elapsed(device, operation):
    """Synchronize only at the one-time generation/merge/restoration boundaries."""
    if device.type == "cuda":
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        return result, start.elapsed_time(end) / 1000
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


def validate_runtime(config, metadata):
    """Fail before loading GPU weights for unsupported scheduling/precision."""
    spec = config.speculative_config
    scheduler = config.scheduler_config
    parallel = config.parallel_config
    serving = json.loads(os.environ.get("SPECULATORS_SERVER_CONFIG", "{}"))
    conditioning_limit = min(
        scheduler.max_num_seqs,
        serving.get("conditioning_examples") or scheduler.max_num_seqs,
    )
    continuous = serving.get("adapter_policy") == "per_request"
    if continuous:
        conditioning_limit = 1
    checks = {
        "V1 model runner": not getattr(config, "use_v2_model_runner", False),
        "method=eagle3": spec.method == "eagle3",
        "supported batch capacity": 1 <= scheduler.max_num_seqs <= MAX_BATCH_SIZE,
        "generator conditioning capacity": metadata["kind"] != "generator"
        or conditioning_limit <= metadata["max_examples"],
        "fixed batch scheduler": (
            continuous
            and getattr(scheduler, "scheduler_cls", None)
            in (None, "vllm.v1.core.sched.scheduler.Scheduler")
        )
        or scheduler.max_num_seqs == 1
        or (
            getattr(scheduler, "scheduler_cls", None)
            == "speculators.generator.serving.scheduler.FixedBatchScheduler"
            and json.loads(os.environ.get("SPECULATORS_SERVER_CONFIG", "{}")).get(
                "batching_policy"
            )
            == "fixed_shared_adapter"
        ),
        "unchunked prefill": not scheduler.enable_chunked_prefill,
        "synchronous scheduling": scheduler.async_scheduling is False,
        "prefix caching disabled": not config.cache_config.enable_prefix_caching,
        "eager target and draft": config.model_config.enforce_eager
        and (spec.enforce_eager is not False),
        "TP/PP/DP=1": all(
            getattr(parallel, key) == 1
            for key in (
                "tensor_parallel_size",
                "pipeline_parallel_size",
                "data_parallel_size",
            )
        )
        and spec.draft_tensor_parallel_size in (None, 1),
        "BF16 target": config.model_config.dtype == torch.bfloat16,
        "BF16 draft": spec.draft_model_config.dtype == torch.bfloat16,
        "unquantized draft": spec.quantization is None
        and not getattr(spec.draft_model_config.hf_config, "quantization_config", None),
        "sequential drafting": not spec.parallel_drafting,
        "per-request singleton conditioning": not continuous
        or (
            metadata["kind"] == "generator"
            and serving.get("conditioning_examples") == 1
        ),
        "padded draft batches": not continuous
        or not getattr(spec, "disable_padded_drafter_batch", False),
        "target without LoRA": not continuous
        or getattr(config, "lora_config", None) is None,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("Conditional LoRA prototype requires: " + ", ".join(failed))
    for role, model_config in (
        ("target", config.model_config),
        ("drafter", spec.draft_model_config),
    ):
        expected = metadata["models"][role]
        source = Path(model_config.model)
        pinned_local = source.is_dir() and source.name == expected["revision"]
        pinned_remote = model_config.model == expected["id"] and (
            model_config.revision == expected["revision"]
        )
        if not (pinned_local or pinned_remote):
            raise ValueError(f"{role} must use the bundle's pinned model revision")
    hf = spec.draft_model_config.hf_config
    for key in ("norm_before_residual", "norm_before_fc", "fc_norm", "norm_output"):
        if bool(getattr(hf, key, False)) != bool(
            metadata["draft_config"].get(key, False)
        ):
            raise ValueError(f"Frozen drafter normalization differs: {key}")
    hf.eagle_aux_hidden_state_layer_ids = metadata["layer_ids"]


class AdapterMerger:
    """Merge V only in packed QKV; never alter Q/K or accumulate updates."""

    def __init__(self, draft, metadata):
        inner = draft.model
        if len(inner.layers) != 1:
            raise ValueError("The prototype supports exactly one draft layer")
        attention = inner.layers[0].self_attn
        qkv = attention.qkv_proj.weight
        output = attention.o_proj.weight
        inputs, values = metadata["shapes"]["v_proj"]
        if (
            qkv.ndim != MATRIX_NDIM
            or qkv.shape[1] != inputs
            or qkv.shape[0] <= 2 * values
        ):
            raise ValueError("Packed QKV does not match exported V dimensions")
        self.weights = {"v_proj": qkv[-values:], "o_proj": output}
        for name, weight in self.weights.items():
            in_features, out_features = metadata["shapes"][name]
            if tuple(weight.shape) != (out_features, in_features):
                raise ValueError(f"Draft projection shape differs: {name}")
        self.base = {
            name: value.detach().clone() for name, value in self.weights.items()
        }
        self.rank = metadata["rank"]
        self.scale = metadata["alpha"] / self.rank
        self.device = output.device

    @torch.inference_mode()
    @traced("speculators.adapter_merge")
    def apply(self, factors):
        for name, weight in self.weights.items():
            a, b = factors[name]
            if a.shape != (self.rank, weight.shape[1]) or b.shape != (
                weight.shape[0],
                self.rank,
            ):
                raise ValueError(f"Invalid generated factors: {name}")
            update = b.float() @ a.float()
            update.mul_(self.scale).add_(self.base[name].float())
            merged = update.to(weight.dtype)
            if not torch.isfinite(merged).all():
                self.restore()
                raise ValueError("Non-finite merged adapter")
            weight.copy_(merged)

    @torch.inference_mode()
    @traced("speculators.adapter_restore")
    def restore(self):
        for name, weight in self.weights.items():
            weight.copy_(self.base[name])


class DraftAdapterController:
    def __init__(self, bundle, draft):
        self.metadata, state = bundle_state(bundle)
        self.merger = AdapterMerger(draft, self.metadata)
        self.device = self.merger.device
        self.generator = None
        self.factors = None
        if self.metadata["kind"] == "generator":
            self.generator = load_generator(self.metadata, state).to(self.device)
        else:
            self.factors = {
                name: tuple(
                    state[f"{name}.{suffix}"].to(self.device) for suffix in ("A", "B")
                )
                for name in self.metadata["shapes"]
            }
        serving = json.loads(os.environ.get("SPECULATORS_SERVER_CONFIG", "{}"))
        self.conditioning_examples = serving.get("conditioning_examples")
        self.conditioning_seed = serving.get("seed", 42)
        self.permanent_static = (
            self.generator is None
            and serving.get("static_adapter_policy") == "merged_once"
        )
        self.setup_merge_seconds = 0.0
        if self.permanent_static:
            _, self.setup_merge_seconds = elapsed(
                self.device, lambda: self.merger.apply(self.factors)
            )
        self._initialize_tracking()

    def _initialize_tracking(self):
        self.active = None
        self.active_requests = set()
        self.records = {}
        self.completed_records = {}
        self.record = {}
        self.last_record = {}
        self.totals = {
            "prepared_requests": 0,
            "completed_requests": 0,
            "generator_invocations": 0,
            "generator_seconds": 0.0,
            "merge_seconds": 0.0,
            "restore_seconds": 0.0,
            "prepared_batches": 0,
            "completed_batches": 0,
            "conditioning_examples": 0,
            "adapter_load_seconds": 0.0,
        }

    @torch.inference_mode()
    def prepare(self, request_id, prompt_length, positions, states):
        if self.active == request_id:
            return
        if self.active is not None:
            raise RuntimeError("A new request started before its predecessor finished")
        # Full prefill is mandatory: otherwise draft KV may already contain base V.
        if (
            positions.ndim != 1
            or positions.numel() != prompt_length
            or not torch.equal(
                positions, torch.arange(prompt_length, device=positions.device)
            )
        ):
            raise ValueError(
                "Adapter initialization requires a complete target prefill"
            )
        if states.shape[0] != prompt_length:
            raise ValueError("Target states do not match the prompt boundary")
        factors, generation_seconds = self.factors, 0.0
        if self.generator is not None:
            summary = states[-1:].float()
            if summary.shape[1] != self.metadata["input_dim"]:
                raise ValueError("Auxiliary layer order/width differs from training")
            factors, generation_seconds = elapsed(
                self.device,
                lambda: self.generator(
                    summary, torch.tensor([prompt_length], device=self.device)
                ),
            )
        try:
            merge_seconds = 0.0
            if not self.permanent_static:
                _, merge_seconds = elapsed(
                    self.device, lambda: self.merger.apply(factors)
                )
        except BaseException:
            self.merger.restore()
            raise
        self.active = request_id
        self.active_requests = {request_id}
        self.last_record = {}
        self.record = {
            "request_id": request_id,
            "generator_invocations": int(self.generator is not None),
            "generator_seconds": generation_seconds,
            "merge_seconds": merge_seconds,
            "restore_seconds": 0.0,
            "deployment_context_size": 1,
        }
        self.totals["prepared_requests"] += 1
        self.totals["prepared_batches"] += 1
        self.totals["conditioning_examples"] += 1
        self.records = {request_id: self.record}
        self.completed_records.pop(request_id, None)
        for key in ("generator_invocations", "generator_seconds", "merge_seconds"):
            self.totals[key] += self.record[key]
        logger.info(
            "Draft adapter ready: request=%s, generator=%.4fs, merge=%.4fs",
            request_id,
            generation_seconds,
            merge_seconds,
        )

    @torch.inference_mode()
    def prepare_batch(self, request_ids, prompt_lengths, positions, states, offsets):  # noqa: C901 -- validate all packed prefills before merging
        """Generate once from all prompt summaries before any draft prefill."""
        if self.active is not None:
            if not set(request_ids) <= self.active_requests:
                raise RuntimeError("New request entered an unfinished fixed batch")
            return
        count = len(request_ids)
        if not 1 <= count <= MAX_BATCH_SIZE:
            raise ValueError("Unsupported fixed batch size")
        conditioning_count = min(count, self.conditioning_examples or count)
        if (
            self.generator is not None
            and conditioning_count > self.metadata["max_examples"]
        ):
            raise ValueError("Batch exceeds the generator's conditioning capacity")
        if (
            len(set(request_ids)) != count
            or len(offsets) != count + 1
            or len(prompt_lengths) != count
        ):
            raise ValueError("Invalid fixed batch request boundaries")
        if (
            offsets[0] != 0
            or offsets[-1] != states.shape[0]
            or positions.numel() != states.shape[0]
        ):
            raise ValueError("Fixed batch token boundaries differ from target prefill")
        summaries = []
        for index, length in enumerate(prompt_lengths):
            start, end = offsets[index : index + 2]
            if end - start != length or not torch.equal(
                positions[start:end], torch.arange(length, device=positions.device)
            ):
                raise ValueError("Fixed batch requires complete target prefills")
            summaries.append(states[end - 1])
        factors, generation_seconds = self.factors, 0.0
        if self.generator is not None:
            selected = list(range(count))
            if conditioning_count < count:
                selected = sorted(
                    random.Random(
                        self.conditioning_seed + self.totals["prepared_batches"]
                    ).sample(selected, conditioning_count)
                )
            summary = torch.stack([summaries[index] for index in selected]).float()
            selected_lengths = [prompt_lengths[index] for index in selected]
            if summary.shape[1] != self.metadata["input_dim"]:
                raise ValueError("Auxiliary layer order/width differs from training")
            factors, generation_seconds = elapsed(
                self.device,
                lambda: self.generator(
                    summary, torch.tensor(selected_lengths, device=self.device)
                ),
            )
        merge_seconds = 0.0
        if not self.permanent_static:
            _, merge_seconds = elapsed(self.device, lambda: self.merger.apply(factors))
        self.active_requests = set(request_ids)
        self.active = request_ids[0]
        self.records = {}
        for index, request_id in enumerate(request_ids):
            self.completed_records.pop(request_id, None)
            self.records[request_id] = {
                "request_id": request_id,
                "generator_invocations": int(index == 0 and self.generator is not None),
                "generator_seconds": generation_seconds if index == 0 else 0.0,
                "merge_seconds": merge_seconds if index == 0 else 0.0,
                "restore_seconds": 0.0,
                "deployment_context_size": conditioning_count,
                "batch_size": count,
            }
        self.record = self.records[self.active]
        self.last_record = {}
        self.totals["prepared_requests"] += count
        self.totals["prepared_batches"] += 1
        self.totals["conditioning_examples"] += conditioning_count
        self.totals["generator_invocations"] += int(self.generator is not None)
        self.totals["generator_seconds"] += generation_seconds
        self.totals["merge_seconds"] += merge_seconds
        logger.info(
            "Shared adapter ready: batch_size=%d, requests=%s", count, request_ids
        )

    def finish(self, request_id):
        if request_id in self.active_requests:
            self.active_requests.remove(request_id)
            seconds = 0.0
            if not self.active_requests and not self.permanent_static:
                _, seconds = elapsed(self.device, self.merger.restore)
            record = self.records.pop(request_id)
            record["restore_seconds"] = seconds
            self.totals["completed_requests"] += 1
            self.totals["restore_seconds"] += seconds
            self.last_record = dict(record)
            self.completed_records[request_id] = dict(record)
            while len(self.completed_records) > FINISHED_RECORD_LIMIT:
                self.completed_records.pop(next(iter(self.completed_records)))
            if not self.active_requests:
                self.totals["completed_batches"] += 1
            self.active = next(iter(self.active_requests), None)
            self.record = self.records[self.active] if self.active else {}
        return dict(self.completed_records.get(request_id, {}))

    def finish_external(self, request_id):
        """Resolve external IDs against active and recently completed members.

        InputProcessor appends '-' plus eight UUID hex characters internally.
        Lifecycle hooks use that internal ID; RequestOutput exposes the external
        ID. Never fall back to an unrelated active or completed request.
        """
        for internal in self.active_requests | self.completed_records.keys():
            external, separator, suffix = internal.rpartition("-")
            if internal == request_id or (
                separator
                and external == request_id
                and len(suffix) == REQUEST_ID_SUFFIX_LENGTH
                and all(character in "0123456789abcdef" for character in suffix)
            ):
                return self.finish(internal)
        return {}

    @torch.inference_mode()
    def profile(self):
        """Exercise temporary generation/merge memory inside vLLM's profiling run."""
        if self.active is not None or self.permanent_static:
            return
        factors = self.factors
        if self.generator is not None:
            factors = self.generator(
                torch.zeros(
                    1,
                    self.metadata["input_dim"],
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.ones(1, device=self.device),
            )
        try:
            self.merger.apply(factors)
        finally:
            self.merger.restore()


def initialize_proposer(proposer, runner):
    """Called only when patched speculative_config.draft_adapter_path is set."""
    import vllm  # noqa: PLC0415

    if vllm.__version__ != VLLM_VERSION:
        raise ValueError(f"Draft adapter runtime requires vLLM {VLLM_VERSION}")
    proposer.adapter_runner = runner
    metadata = read_bundle(proposer.speculative_config.draft_adapter_path)
    validate_runtime(proposer.vllm_config, metadata)


def prepare_proposer(
    proposer, positions, states, attention_metadata=None, proposal_kwargs=None
):
    runner = proposer.adapter_runner
    ids = runner.input_batch.req_ids
    if hasattr(proposer.draft_adapter, "prepare_proposal"):
        proposer.draft_adapter.prepare_proposal(
            list(ids),
            [runner.requests[name].num_prompt_tokens for name in ids],
            positions,
            states,
            proposal_kwargs,
        )
        return
    if getattr(proposer.draft_adapter, "continuous", False):
        if attention_metadata is None:
            raise ValueError("Continuous drafting requires attention token boundaries")
        proposer.draft_adapter.prepare_scheduled(
            list(ids),
            [runner.requests[name].num_prompt_tokens for name in ids],
            positions,
            states,
            attention_metadata.query_start_loc_cpu.tolist(),
        )
        return
    if proposer.draft_adapter.active is not None:
        if not set(ids) <= proposer.draft_adapter.active_requests:
            raise RuntimeError("New request entered an unfinished fixed batch")
        return
    if len(ids) != 1:
        if attention_metadata is None:
            raise ValueError("Batched drafting requires attention token boundaries")
        offsets = attention_metadata.query_start_loc_cpu.tolist()
        proposer.draft_adapter.prepare_batch(
            list(ids),
            [runner.requests[name].num_prompt_tokens for name in ids],
            positions,
            states,
            offsets,
        )
        return
    request = runner.requests[ids[0]]
    proposer.draft_adapter.prepare(ids[0], request.num_prompt_tokens, positions, states)
