"""Request-lifetime adapters; imported by the optional vLLM proposer hook."""

import logging
import time
from pathlib import Path

import torch

from speculators.generator.serving.bundle import (
    VLLM_VERSION,
    bundle_state,
    load_generator,
    read_bundle,
)

logger = logging.getLogger(__name__)
MATRIX_NDIM = 2
REQUEST_ID_SUFFIX_LENGTH = 8


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
    checks = {
        "V1 model runner": not getattr(config, "use_v2_model_runner", False),
        "method=eagle3": spec.method == "eagle3",
        "one active sequence": scheduler.max_num_seqs == 1,
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
        self.active = None
        self.record = {}
        self.last_record = {}

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
            _, merge_seconds = elapsed(self.device, lambda: self.merger.apply(factors))
        except BaseException:
            self.merger.restore()
            raise
        self.active = request_id
        self.last_record = {}
        self.record = {
            "request_id": request_id,
            "generator_invocations": int(self.generator is not None),
            "generator_seconds": generation_seconds,
            "merge_seconds": merge_seconds,
            "restore_seconds": 0.0,
            "deployment_context_size": 1,
        }
        logger.info(
            "Draft adapter ready: request=%s, generator=%.4fs, merge=%.4fs",
            request_id,
            generation_seconds,
            merge_seconds,
        )

    def finish(self, request_id):
        if self.active == request_id:
            _, seconds = elapsed(self.device, self.merger.restore)
            self.record["restore_seconds"] = seconds
            self.last_record = dict(self.record)
            self.active = None
            self.record = {}
        return (
            dict(self.last_record)
            if self.last_record.get("request_id") == request_id
            else {}
        )

    def finish_external(self, request_id):
        """Resolve vLLM 0.31's external ID against this serial request's record.

        InputProcessor appends '-' plus eight UUID hex characters internally.
        Lifecycle hooks use that internal ID; RequestOutput exposes the external
        ID. Never fall back to an unrelated active or completed request.
        """
        internal = self.active or self.last_record.get("request_id")
        if internal is None:
            return {}
        if internal == request_id:
            return self.finish(internal)
        external, separator, suffix = internal.rpartition("-")
        if (
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
        if self.active is not None:
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


def prepare_proposer(proposer, positions, states):
    runner = proposer.adapter_runner
    ids = runner.input_batch.req_ids
    if len(ids) != 1:
        raise ValueError("Draft adapter runtime requires exactly one active request")
    request = runner.requests[ids[0]]
    proposer.draft_adapter.prepare(ids[0], request.num_prompt_tokens, positions, states)
