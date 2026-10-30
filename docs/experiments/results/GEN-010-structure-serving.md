# GEN-010 — Structure generators and static LoRAs: math/chat C1

Status: preliminary throughput received for all five models; acceptance and
native request-level evidence remain unavailable. One supplied result per cell,
no repeated-trial uncertainty. Correctness was not checked.
The requested comparison uses step-1500 checkpoints and shared/merged adapters
at concurrency 1. See [suite, readiness and command](../../../examples/eagle3-lora/structure_benchmarks_v1/README.md).
During preparation, attention static LoRA step 1500 was verified at revision
`dfb78b270a39f625bc7a39ef75cfaaab5868eb54`. The attention-only decoder-width-256
generator was included as a fifth model. Its pre-run revision
`afd0df1af02dfb6f379eb4f0d529029a1abdc075` had only step 500/1000, so the suite
required waiting for step 1500. The subsequently supplied five-row table is
archived below. Executed bundle/checkpoint hashes and final Hub pins were not
supplied, so the named budgets are reported/configured rather than independently
verified from the final native run artifacts.

## Supplied measurements

Receipt date: 2026-10-30; execution date and per-run hardware provenance not
provided with the table. [Original supplied Markdown](benchmarks/2026-10-30-structure-math-chat-c1/summary.md),
[transcribed metrics](benchmarks/2026-10-30-structure-math-chat-c1/metrics.json), and
[derived comparisons](benchmarks/2026-10-30-structure-math-chat-c1/analysis.json).
The metrics JSON is an explicit transcription, not a native suite export.

## Extended throughput table

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
| Attention static LoRA, 4K / 1,500 steps | 75.72 | 56.48 |
| All-module static LoRA, 4K / 1,500 steps | 77.90 | 57.26 |
| Attention generator, 4K / 1,500 steps | 78.18 | 55.41 |
| All-module generator, 4K / 1,500 steps | 76.72 | 55.05 |
| Attention generator, decoder 256, 4K / 1,500 steps | 77.19 | 55.57 |

The five new rows come directly from the supplied table, not offline agreement
curves or inferred acceptance improvements. Aggregate output throughput includes
prefill and serving overhead.
Correctness is not checked in the timing suite.

## Matched comparisons and interpretation

Changes below are calculated from the supplied, two-decimal rounded values;
positive means higher output throughput. They are descriptive differences,
not statistically established wins.

| Candidate vs reference | Math | Chat |
| --- | ---: | ---: |
| Attention generator vs attention static LoRA | +3.25% | −1.89% |
| All-module generator vs all-module static LoRA | −1.51% | −3.86% |
| Attention decoder-256 generator vs decoder-512 generator | −1.27% | +0.29% |
| Attention decoder-256 generator vs attention static LoRA | +1.94% | −1.61% |
| All-module static LoRA vs attention static LoRA | +2.88% | +1.38% |
| All-module generator vs attention generator | −1.87% | −0.65% |

- **Conditional generation has mixed throughput evidence.** The original
  attention generator beats its matched static LoRA on math, but loses on chat.
  The all-module generator loses to its matched static LoRA on both categories.
  Prompt-dependent adaptation is not established as an overall throughput win.
- **Coverage gains depend on the adaptation method.** All-module static LoRA
  is faster than attention-only static LoRA in both categories, whereas the
  all-module generator is slower than the attention generator in both. Broad
  module coverage is not a universal end-to-end speed improvement.
- **Best displayed values are close on math.** Attention generator is highest
  at 78.18 tok/s, only 0.36% above all-module static LoRA at 77.90. All-module
  static LoRA is highest on chat at 57.26. Repeats are needed to assess small gaps.
- **The compact generator preserves similar throughput with fewer parameters.**
  Its math result is slightly lower and chat result slightly higher than the
  larger attention generator. Calculated configured parameter counts are
  326,709,344 versus 646,297,952 (about 49.45% fewer). This supports a compact
  candidate for parameter efficiency; it does not measure a generator-latency,
  peak-memory or statistically significant throughput improvement.

The all-module generator's stronger offline validation agreement in GEN-009
did not coincide with the strongest serving throughput here. This does not
show that its verifier acceptance regressed: acceptance counters are absent.
Higher generation/merge cost, different actual acceptance, response-length
differences, and run variability are possible explanations, not diagnoses.

Working decision: retain attention generation and the compact attention variant
as candidates, with all-module static LoRA as a strong throughput control. Do
not favor maximum generated weight count solely from offline agreement. Collect
native acceptance and overhead metrics before redesigning the hypernetwork or
attributing these gaps to infrastructure.

## Acceptance measurements to add

Retain math/chat draft acceptance rate, per-position acceptance, acceptance
length, accepted/proposed token counts and draft cycles from `summary.json`.
Record generator/merge/restore timing, TTFT, TPOT, model/weight hashes, selected
step/revision, workload hashes, server/client logs and reproducibility artifacts.
Those fields were not included in the supplied Markdown and cannot be inferred
from its throughput values. Native results also establish workload hashes,
response-token totals and request counts needed for more controlled comparisons.

The questions are whether the module coverage gains in
[GEN-009](GEN-009-generator-structure.md) translate to serving acceptance and
throughput, and whether generators outperform static LoRAs at matched target
sets and training budgets. Throughput now has the mixed evidence above;
the acceptance/cost mechanisms remain open.

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

The later supplied summary includes a numeric all-module/chat row. The native
files and retry history were not supplied, so it does not retrospectively
establish the cause of the first failed attempt or validate accepting 49/50.
