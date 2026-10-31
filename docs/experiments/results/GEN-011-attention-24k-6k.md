# GEN-011 — Attention generator on 24K rows through 6,000 updates

Date: 2026-10-31. Status: configs and launch commands prepared; schemas and
expected-membership schedule validated locally. Training, HF uploads and serving
benchmarks have not been run here. No new measured conclusion is recorded.

## Purpose and protocol

The [GEN-010](GEN-010-structure-serving.md) table mixes 4K attention generators
with historical 24K V/O generators. Train a fresh Q/K/V/O generator on the
original 24K saved-trajectory pool, then compare its 2,000/4,000/6,000 snapshots.
This is a data-aligned follow-up, not a continuation of the historical model.

Use [train.yaml](../../../examples/eagle3-lora/attention_24k_6k_v1/train.yaml)
and the [setup/training/benchmark commands](../../../examples/eagle3-lora/attention_24k_6k_v1/README.md).
All 24 original subsets contribute 1K training rows each (6K/domain). The
original shuffled epoch schedule uses cohorts of 1–8, four cohorts/update,
seed 42; it is capped at 6,000 optimizer updates. The expected-release schedule
matches the historical full-data config exactly before truncation. Architecture
matches the earlier attention generator, including decoder width 512. BF16
parameters/optimizer state and batched generator execution are retained.

Offline context-4 validation runs every 250 updates, initially and finally;
it is separate from serving acceptance. Local saves and synchronous HF uploads
occur at 2,000/4,000/6,000 updates in
`azeeer/eagle3-generator-attention-24k-6k-v1`. W&B training project remains
`eagle3-generator-ablations-v3`, group `attention-24k-6k-v1`.

## Planned serving comparison

[Three-checkpoint suite](../../../examples/eagle3-lora/attention_24k_6k_v1/benchmarks/math_chat_c1.yaml):
math/chat, concurrency 1, 50 prompts/category, 256 output tokens, three draft
tokens, three warmups, seed 42, prompt-conditioned shared/merged adapter.
The suite requires all three exact snapshot folders. Collect output throughput,
draft acceptance and acceptance length, generator/merge/restore durations,
TTFT/TPOT, completed-request counts and workload/checkpoint identities.

Keep executed training, extraction and evaluation provenance with the results.
Data-pool matching does not establish equal hardware/runtime or numerical
execution: the historical generator used FP32, while this run uses the current
BF16/batched implementation. Historical comparisons retain those limitations.
A same-runtime V/O rerun and repeated trials would strengthen causal attribution.
