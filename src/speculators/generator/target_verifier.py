"""Frozen full-vocabulary target verification using transient vLLM activations."""

import time
from pathlib import Path
from threading import Lock

import torch
from safetensors.torch import load_file

from speculators.data_generation.offline import check_hidden_states
from speculators.data_generation.vllm_client import (
    generate_hidden_states,
    wait_for_lock,
)
from speculators.utils.loading import load_model_layers


class FrozenTarget:
    def __init__(self, runtime, *, loader=None, extractor=None):
        self.runtime = runtime
        self.extractor = extractor or generate_hidden_states
        loaded = (loader or load_model_layers)(
            ["lm_head.weight"], runtime.corpus.cache.cfg.online.target_model_path
        )
        if "lm_head.weight" not in loaded:
            raise ValueError(
                "Full-vocabulary verification requires the pinned target LM head"
            )
        self.head = loaded["lm_head.weight"].to(runtime.device, dtype=runtime.dtype)
        eos = runtime.drafter.config.transformer_layer_config.eos_token_id
        self.eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        self.teacher_requests = self.teacher_seconds = self.teacher_tokens = 0
        self.stats_lock = Lock()
        self.teacher_rpc_seconds = self.teacher_file_seconds = 0.0
        self.teacher_file_bytes = 0

    def extract(self, tokens):
        started = time.perf_counter()
        cache = self.runtime.corpus.cache
        path = Path(
            self.extractor(
                cache.client,
                cache.bank.models.target,
                {"input_ids": tokens},
                timeout=cache.settings.request_timeout,
                max_retries=cache.settings.max_retries,
            )
        ).resolve()
        rpc_seconds = time.perf_counter() - started
        file_started = time.perf_counter()
        if not path.is_relative_to(cache.root):
            raise ValueError("Target extractor returned a file outside shared_dir")
        try:
            if Path(str(path) + ".lock").exists():
                wait_for_lock(
                    str(path) + ".lock", timeout=cache.settings.request_timeout
                )
            file_bytes = path.stat().st_size
            # Tensor storage keeps memory mappings alive after unlink, avoiding
            # a full-prefix Python bytes buffer and its deserialization copy.
            payload = load_file(path, device="cpu")
            check_hidden_states(payload, tokens)
            states = payload["hidden_states"]
            expected = (
                len(tokens),
                len(cache.bank.hidden_states.layer_ids),
                self.head.shape[1],
            )
            if tuple(states.shape) != expected:
                raise ValueError(
                    "Target hidden-state dimensions differ from the pinned target"
                )
            if states.dtype != self.runtime.dtype:
                raise ValueError(
                    "Target hidden-state dtype differs from the training recipe"
                )
            with self.stats_lock:
                self.teacher_requests += 1
                self.teacher_seconds += time.perf_counter() - started
                self.teacher_tokens += len(tokens)
                self.teacher_rpc_seconds += rpc_seconds
                self.teacher_file_seconds += time.perf_counter() - file_started
                self.teacher_file_bytes += file_bytes
            return states
        finally:
            path.unlink(missing_ok=True)

    @torch.no_grad()
    def teacher(self, states):
        with torch.autocast(
            self.runtime.device.type,
            dtype=self.runtime.dtype,
            enabled=self.runtime.dtype != torch.float32,
        ):
            hidden = self.runtime.drafter.verifier_norm(states.to(self.runtime.device))
            return torch.nn.functional.linear(hidden, self.head).float()
