# GEN-011 — Attention generator on 24K rows through 6,000 updates

Date: 2026-10-31. Status: user-supplied serving summary received for three
checkpoints across seven categories at C1 (21 measurements after OOD resume).
The original 12-cell ID summary is retained separately. Aggregate counters
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

The [historical all-category table in GEN-003](GEN-003-generator-suite-throughput.md)
is extended with these rows. A [combined seven-category view](benchmarks/2026-10-31-attention-24k-6k-c1/extended_historical_table.md)
shows the historical models, the attention checkpoints and the later DeepSeek
pair. All attention-generator and paired DeepSeek OOD cells are now filled.

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


## OOD benchmark extension — supplied results

The same checkpoint suite now appends `RedHatAI/speculator_benchmarks` with
HumanEval, math reasoning and summarization. Rerun the existing command/output
root with `--resume` after pulling the updated code. Safe additive-dataset resume
keeps the twelve valid completed Nemotron results, frozen workloads and checkpoint
pins, and runs nine new OOD cells. Existing workload/settings changes remain
rejected. Previous suite metadata/state is archived on extension.

The user then supplied the completed [21-cell resumed summary](benchmarks/2026-10-31-attention-24k-6k-c1/summary_with_ood.json),
[seven-category table](benchmarks/2026-10-31-attention-24k-6k-c1/summary_with_ood.md),
and [validated OOD comparisons](benchmarks/2026-10-31-attention-24k-6k-c1/analysis_with_ood.json).
New suite identity: `e226cec297e90a717284f9b9940f2bdddda849f1ff8b4d0808c1f44c85f075ba`.
The twelve original Nemotron result objects are unchanged in the combined archive;
all 21 cells pass 50-request/adapter counts, three-token acceptance identities,
adapter timing means and aggregate throughput identities. The OOD cells share
per-category workload hashes, subsets and 12,800 output tokens across checkpoints.

| Checkpoint | HumanEval output tok/s | Math reasoning output tok/s | Summarization output tok/s |
| --- | ---: | ---: | ---: |
| 2,000 | 71.26 | 69.62 | 59.94 |
| 4,000 | **72.32** | **70.69** | **60.65** |
| 6,000 | 71.96 | 70.11 | 60.04 |

| Checkpoint | HumanEval draft acceptance | Math reasoning draft acceptance | Summarization draft acceptance |
| --- | ---: | ---: | ---: |
| 2,000 | 55.00% | 52.77% | **41.43%** |
| 4,000 | **55.21%** | **53.07%** | 41.22% |
| 6,000 | 54.86% | 52.99% | 41.06% |

Working update:

- Step 4,000 has the highest measured OOD throughput in all three categories,
  1.19–1.53% above step 2,000. Step 2,000 remains the highest measured Nemotron
  throughput among these checkpoints. No single checkpoint leads every domain.
- OOD acceptance changes are small: 4,000 versus 2,000 gives +0.21 points on
  HumanEval, +0.29 on math reasoning, and −0.22 on summarization. Summarization
  throughput improves despite lower acceptance. Later training does not improve
  OOD acceptance uniformly, and the data do not isolate the timing mechanism.
- Step 6,000 does not exceed step 4,000 in OOD throughput or acceptance here.
  This is preliminary plateau/mixed evidence, not a demonstrated general
  overfitting effect or a universal checkpoint ranking; gaps need repeats.
- Versus historical V/O generation at the reported same 24K/4K budget, the new
  4K attention checkpoint is +4.14% HumanEval, +3.47% math reasoning and +3.35%
  summarization throughput. Historical workload hashes, precision and runtime
  equivalence remain unverified; no current frozen-drafter baseline was included
  for these OOD cells. Do not infer a controlled OOD adaptation gain from this alone.
- Timed adapter totals remain 6.53–6.60 ms/request. These are CUDA-event counters,
  not an isolated drafter/verifier or complete host/GPU wall-time partition.

Keep 2,000 as the ID-throughput candidate, 4,000 as the OOD-throughput candidate,
and 6,000 for its stronger math/code/chat ID acceptance. Correctness, statistical
uncertainty and executed checkpoint/hardware provenance remain unchecked. The
DeepSeek pair
in GEN-012 subsequently supplied its OOD measurements; those now fill the
remaining cells of the combined table.

## Follow-up: published training provenance inspected

The step-2000 training and extraction artifacts were later retrieved from model
revision `40d7f6cc9c72a8872377c49418e8b42f071b8f15`. The recorded extractor uses
vLLM 0.31.0, 16 active sequences, a 0.50 GPU budget and two API server processes;
the runbook's 128/0.40 launch example is not the executed command. Timing logs
record 35,918 sample presentations and 39,001,722 response tokens at step 2,000.
The deterministic saved membership/schedule reconstructs 24,000 distinct rows
and 11,918 repeated presentations. Training and benchmark provenance are
separate: this follow-up does not establish the executed serving environment.
See the [source artifacts and audit](benchmarks/2026-11-01-attention-100k-2k-c1/training_audit/README.md)
and [GEN-013 comparison](GEN-013-attention-100k-2k.md#follow-up-actual-training-exposure-and-extraction-provenance).
