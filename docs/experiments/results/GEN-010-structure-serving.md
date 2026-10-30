# GEN-010 — Structure generators and static LoRAs: math/chat C1

Status: suite prepared; new GPU throughput/acceptance measurements pending.
The requested comparison uses step-1500 checkpoints and shared/merged adapters
at concurrency 1. See [suite, readiness and command](../../../examples/eagle3-lora/structure_benchmarks_v1/README.md).
The attention static LoRA now has step 1500, verified at revision
`dfb78b270a39f625bc7a39ef75cfaaab5868eb54`. The attention-only decoder-width-256
generator is now included as a fifth model. Its latest verified revision
`afd0df1af02dfb6f379eb4f0d529029a1abdc075` has only step 500/1000; the suite waits
for step 1500 to become available and never substitutes a shorter snapshot.

## Throughput table to extend

Historical rows are copied from [GEN-003](GEN-003-generator-suite-throughput.md)
and retain their original evidence limits. They are not fresh measurements
from the new suite. The five new models use a different training budget/data
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
| Attention generator, decoder 256, 4K / 1,500 steps | pending | pending |

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

## Reported all-module/chat incomplete workload

The user supplied a failed `generator-all-4k-1500/nemotron/nemo-chat` first
attempt: expected 50 requests; GuideLLM measured 49 successful, zero errored,
zero incomplete, total 49. The client failed its completeness check, followed
by owned-server cleanup. No throughput row from this attempt is accepted.
This is not evidence by itself of a model crash or a specific GuideLLM bug.
The native result and complete client log have not been supplied for inspection.

Real GuideLLM 0.7.1 tests against a local HTTP endpoint completed 50-request
workloads at C1 and C2, including the existing short-response path; empty-output
requests were also tested. They did not reproduce this failure. Diagnostic
reporting now preserves scheduler/measurement distinctions on count mismatch,
and the unchanged suite resume path preserves completed measurements.
