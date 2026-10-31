# GEN-012 — Generator on frozen DeepSeek EAGLE3 ttt7: 4K / 1,500 updates

Date: 2026-10-31. Status: config, compatibility support and commands prepared;
44 focused/affected CPU checks and the CLI dry run pass. Full-size GPU training
and generated-adapter serving measurements are pending.

Baseline: the external drafter in [GEN-010](GEN-010-structure-serving.md) reports
82.50 math / 54.10 chat output tok/s, with 74.04% / 37.13% draft acceptance,
using three draft tokens at concurrency 1. Train a fresh generator on this
frozen drafter, not on the older Red Hat drafter.

[Training config](../../../examples/eagle3-lora/deepseek_ttt7_generator_v1/train.yaml)
and [preparation/extraction/training commands](../../../examples/eagle3-lora/deepseek_ttt7_generator_v1/README.md)
retain the original 4K saved-response protocol: 1K training rows per domain,
separate validation membership, random 1–8 cohorts, four cohorts/update,
1,500 optimizer updates, original three-step LK loss, Q/K/V/O adapters and
decoder width 512. BF16 generator parameters and cohort batching are enabled.
Validate every 250 updates; local saves and HF backups every 500 updates.
No response-rollout, replay or feedback objective is introduced.

The pinned external checkpoint uses five auxiliary target outputs, so its
20,480-wide feature input and Qwen3 Q/K normalization must be retained. The
preparation helper writes native training metadata, checks every source tensor
name/shape using a meta-device model and preserves all source tensor bytes.
Verifier head/norm weights are loaded separately from the frozen local target.
Original and native configs and preparation hashes are copied into run provenance.

An explicit `trajectory_recipe` retains the original saved-token source recipe.
The existing loader validates its manifests and membership before constructing
new runtime extraction metadata from `training_recipe`. Original cache/model
lineage remains in `trajectory_source`; target revision and dataset changes are
rejected. No source trajectory files are modified. New feature-layer/model
identities keep old three-feature caches separate. Five-feature extraction uses
indices `[2,10,18,26,34]` plus final verifier state 36.

HF destination:
`azeeer/eagle3-generator-deepseek-ttt7-attention-4k-1500`.
W&B training project remains `eagle3-generator-ablations-v3`.

The completed CPU checks cover the real saved-response loader, retained
membership/provenance, target mismatch rejection, checkpoint weight preservation,
legacy training/resume/upload regressions and the serving metadata path used
by the measured external baseline. The actual published header matches every
native parameter shape except the two intentionally verifier-owned tensors.
These checks do not establish full-size GPU numerical parity or an acceptance
improvement. Future serving must use the same math/chat C1 workload and three
draft tokens before extending the baseline comparison.


A [standalone step-1500 math/chat C1 benchmark config](../../../examples/eagle3-lora/deepseek_ttt7_generator_v1/benchmarks/math_chat_c1.yaml)
is prepared, with the [command in the runbook](../../../examples/eagle3-lora/deepseek_ttt7_generator_v1/README.md).
It measures only the trained generator and matches the baseline's benchmark
settings, including three draft tokens. Config/schema and actual CLI dry-run
validation pass; executed checkpoint availability and GPU results remain pending.


Benchmark expansion: the user added STEM and code to the original math/chat
category list. The same config now includes the pinned frozen DeepSeek baseline
alongside the step-1500 generator: two models × four categories at concurrency 1,
with identical per-category workloads and three draft tokens. Use the new output
root in the runbook rather than resuming a math/chat-only suite. Compare acceptance
and throughput within the new paired measurements; results remain pending.
