"""Opt-in CPU scopes linked to CUDA launches by the native torch profiler."""

import os
from contextlib import nullcontext
from functools import wraps

import torch

ENABLED = os.environ.get("SPECULATORS_TRACE_RANGES") == "1"


def trace_range(name):
    return torch.profiler.record_function(name) if ENABLED else nullcontext()


def traced(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if not ENABLED:
                return function(*args, **kwargs)
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def attach_ranges(controller, draft):
    if not ENABLED:
        return controller
    if controller.generator is not None:
        controller.generator.forward = traced("speculators.generator")(
            controller.generator.forward
        )
    draft.forward = traced("speculators.draft_forward")(draft.forward)
    if getattr(controller, "continuous", False):
        attention = draft.model.layers[0].self_attn
        for name in ("qkv_proj", "o_proj"):
            module = getattr(attention, name)
            module.forward = traced(f"speculators.{name}_with_lora")(module.forward)
    return controller
