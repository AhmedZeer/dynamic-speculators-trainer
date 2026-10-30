# GEN-010 — Structure generators and static LoRAs: math/chat C1

Status: suite prepared; new GPU throughput/acceptance measurements pending.
The requested comparison uses step-1500 checkpoints and shared/merged adapters
at concurrency 1. See [suite, readiness and command](../../../examples/eagle3-lora/structure_benchmarks_v1/README.md).
On 2026-10-30 the attention static-LoRA repository initially had only step 500
and reached step 1000 on the latest lookup. The user requested waiting for step
1500, not substituting either shorter-budget snapshot.

## Throughput table to extend

Historical rows are copied from [GEN-003](GEN-003-generator-suite-throughput.md)
and retain their original evidence limits. They are not fresh measurements
from the new suite. The four new models use a different training budget/data
pool from the historical 24K generators. Hardware/runtime equivalence and
prompt hashes must be checked before claiming relative speedups.

| Model | Nemo math output tok/s | Nemo chat output tok/s |
| --- | ---: | ---: |
| EAGLE3 base — historical | 64.34 | 54.97 |
| Trained v/o LoRA — historical | 72.50 | 56.35 |
| Generator, 24K / 2,000 steps — historical | 77.58 | 57.58 |
| Generator, 24K / 4,000 steps — historical | 79.11 | 56.49 |
| Attention static LoRA, 4K / 1,500 steps | pending | pending |
| All-module static LoRA, 4K / 1,500 steps | pending | pending |
| Attention generator, 4K / 1,500 steps | pending | pending |
| All-module generator, 4K / 1,500 steps | pending | pending |

Do not fill new cells from offline agreement curves or expected acceptance
improvements. Aggregate output throughput includes prefill and serving overhead.
Correctness is not checked in the timing suite.

## Acceptance measurements to add

Retain math/chat draft acceptance rate, per-position acceptance, acceptance
length, accepted/proposed token counts and draft cycles from `summary.json`.
Record generator/merge/restore timing, TTFT, TPOT, model/weight hashes, selected
step/revision, workload hashes, server/client logs and reproducibility artifacts.
There are no new measured values in this record yet.

The questions are whether the module coverage gains in
[GEN-009](GEN-009-generator-structure.md) translate to serving acceptance and
throughput, and whether generators outperform static LoRAs at matched target
sets and training budgets. These remain open until the suite is run.
