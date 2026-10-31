# GEN-011 — Attention generator on 24K rows through 6,000 updates

Date: 2026-10-31. Status: user-supplied serving summary received for three
checkpoints across four categories at C1 (12 measurements). Aggregate counters
and workload identities validate. Correctness is unchecked; repeated trials,
executed checkpoint hashes and hardware/runtime provenance remain unavailable.

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

## Serving protocol

[Three-checkpoint suite](../../../examples/eagle3-lora/attention_24k_6k_v1/benchmarks/math_chat_c1.yaml):
math/chat plus the user-added code and STEM categories, concurrency 1,
50 prompts/category, 256 output tokens, three draft
tokens, three warmups, seed 42, prompt-conditioned shared/merged adapter.
The suite requires all three exact snapshot folders. Collect output throughput,
draft acceptance and acceptance length, generator/merge/restore durations,
TTFT/TPOT, completed-request counts and workload/checkpoint identities.

Keep executed training, extraction and evaluation provenance with the results.
Data-pool matching does not establish equal hardware/runtime or numerical
execution: the historical generator used FP32, while this run uses the current
BF16/batched implementation. Historical comparisons retain those limitations.
A same-runtime V/O rerun and repeated trials would strengthen causal attribution.


## Supplied measurements

[Detailed summary](benchmarks/2026-10-31-attention-24k-6k-c1/summary.json),
[throughput table](benchmarks/2026-10-31-attention-24k-6k-c1/summary.md),
[validated comparisons](benchmarks/2026-10-31-attention-24k-6k-c1/analysis.json).
Suite identity: `7bbe505379c8768498a4f1a3de611ef1168e69fdfe6ca9c4103ee68e10bca6c9`.
These are user-supplied measurements, not locally executed training/benchmarks.
Configured checkpoint names/budgets are reported; final executed HF pins and
training provenance were not included in the summary. This suite is the original
Red Hat drafter's attention generator, not the DeepSeek generator in GEN-012.

| Checkpoint | Math output tok/s | Code output tok/s | STEM output tok/s | Chat output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 2,000 | 82.80 | 73.14 | 70.71 | 59.77 |
| 4,000 | 81.17 | 71.64 | 69.40 | 59.44 |
| 6,000 | 81.89 | 71.81 | 69.41 | 59.16 |

| Checkpoint | Math draft acceptance | Code draft acceptance | STEM draft acceptance | Chat draft acceptance |
| --- | ---: | ---: | ---: | ---: |
| 2,000 | 66.85% | 55.26% | 51.97% | 39.61% |
| 4,000 | 67.93% | 55.75% | 52.49% | 40.45% |
| 6,000 | 68.84% | 55.87% | 52.44% | 40.89% |

| Checkpoint | Math acceptance length | Code acceptance length | STEM acceptance length | Chat acceptance length |
| --- | ---: | ---: | ---: | ---: |
| 2,000 | 3.005 | 2.658 | 2.559 | 2.188 |
| 4,000 | 3.038 | 2.672 | 2.575 | 2.214 |
| 6,000 | 3.065 | 2.676 | 2.573 | 2.227 |

Acceptance length includes one verifier/bonus token; draft acceptance is
accepted/proposed tokens. Neither measures task accuracy. All twelve cells have
50 successful requests, 50 prepared/completed adapters and 50 generator calls:
one generation per request, mean conditioning size 1. Counters satisfy
`proposed = 3 * cycles`, `rate = accepted / proposed`, and
`length = 1 + accepted / cycles`. Throughput equals output tokens / measured
seconds. The same category hashes, subset lists and output totals appear across
all three checkpoints: math 12,800, code 12,785, STEM 12,751, chat 11,032 tokens.
Math/chat hashes also match the earlier structure and DeepSeek summaries.
Equal totals/hashes do not establish identical output sequences or correctness.

## Working findings and limits

- **More updates improve acceptance relative to step 2,000.** At 6,000 the
  gains are +1.99 percentage points on math, +0.61 on code, +0.47 on STEM and
  +1.28 on chat. Math/code/chat improve at every displayed checkpoint; STEM
  peaks at 4,000 and then declines by only 0.05 points. This does not establish
  overfitting or uniform benefit at every intermediate training step.
- **Step 2,000 has the highest measured throughput in all four categories.**
  At 6,000 throughput is lower by 1.09% on math, 1.81% on code, 1.85% on STEM
  and 1.03% on chat. Thus the strongest acceptance checkpoint is not the
  strongest throughput checkpoint in this invocation. These small timing gaps
  need repeated, preferably interleaved trials before selecting a durable winner.
- **Recorded one-time adapter cost is effectively flat.** Across all twelve
  cells, generator calls take 3.110–3.161 ms. Generation + merge + restore total
  6.530–6.601 ms/request, 0.177–0.211% of benchmark wall time. These CUDA-event
  durations are not a full wall-time partition. Their tiny variation does not
  explain multi-second category-duration differences. Drafter/verifier timings,
  host work, run order, GPU clocks/thermal state and other execution differences
  remain unisolated possibilities, not diagnosed causes.
- **Historical same-data/step V/O comparisons favor attention generation in
  displayed throughput.** At 2,000 steps: math +6.72%, chat +3.80%; at 4,000:
  math +2.61%, chat +5.22%. Those references are two-decimal historical values
  from GEN-010, not fresh matched-runtime measurements. BF16/cohort batching
  differs from historical FP32 execution, so target coverage alone cannot be
  assigned the entire timing gain.
- **Versus 4K / 1,500 attention generation, both data and training steps change.**
  The 24K / 2,000 result has +5.90% math and +7.86% chat throughput, but only
  +0.16 and +0.87 points of draft acceptance. This cannot isolate a data-size
  effect or explain the throughput gap from acceptance alone.
- **The new step-2,000 math throughput is effectively level with DeepSeek's
  external baseline in these separate runs.** 82.80 versus 82.50 tok/s is only
  +0.36%, while draft acceptance is lower by 7.19 points (66.85% versus 74.04%).
  Chat throughput is +10.49% and acceptance +2.47 points. The drafter architectures,
  training histories and unverified hardware/runtime conditions differ; these
  are descriptive comparisons, not architecture-controlled wins.

Working choice: retain step 2,000 as the throughput candidate and step 6,000 as
the strongest math/code/chat acceptance candidate. Repeat their serving comparison
under controlled conditions; do not discard later checkpoints as failed training
or infer that more training universally reduces serving speed. Use the paired
DeepSeek baseline/generator experiment in GEN-012 to study whether generated
adapters also help that different frozen drafter.
