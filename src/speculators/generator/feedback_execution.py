"""Execution-only resource controls, independent of experiment/checkpoint identity."""

import os
from pathlib import Path


def memory_limit_bytes():
    limits = [os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")]
    for name in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ):
        try:
            value = Path(name).read_text().strip()
            if value != "max" and int(value) > 0:
                limits.append(int(value))
        except (OSError, ValueError):
            continue
    return min(limits)


def execution_settings(runtime, *, concurrency=0, ram_cache_gib=None):
    gpu_bytes = 0
    if runtime.device.type == "cuda":
        import torch  # noqa: PLC0415

        gpu_bytes = torch.cuda.get_device_properties(runtime.device).total_memory
    if concurrency < 0 or (ram_cache_gib is not None and ram_cache_gib <= 0):
        raise ValueError("Concurrency must be nonnegative and RAM cache positive")
    # The user confirmed eight requests fit on the 40 GB Colab GPU. Smaller
    # devices retain the conservative setting; explicit overrides remain valid.
    actual = concurrency or (8 if gpu_bytes >= 35 * 2**30 else 2)
    physical = memory_limit_bytes()
    requested = runtime.corpus.cache.cfg.online.ram_cache_gib
    cap = min(
        requested, ram_cache_gib if ram_cache_gib is not None else physical / 2 / 2**30
    )
    if cap * 2**30 >= physical:
        raise ValueError("RAM cache must leave system memory for the models and server")
    runtime.corpus.cache.ram.limit = int(cap * 2**30)
    return {
        "rollout_concurrency": actual,
        "ram_cache_gib": cap,
        "system_memory_bytes": physical,
        "gpu_memory_bytes": gpu_bytes,
        "validation_logits": "reuse_collected_policy_logits",
        "generator_backward": "once_per_distinct_conditioning_input",
        "draft_logits": "last_position_only",
        "forced_token_selection": "supplied_tokens_without_argmax",
        "activation_scan": "new_positions_only_with_complete_token_alignment",
    }
