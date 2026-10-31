# GEN-010 — Structure generators and static LoRAs: math/chat C1

Status: preliminary aggregate throughput, acceptance and adapter timings received
for all five models. One supplied result per model/category, no repeated-trial
uncertainty. Correctness was not checked. Request-level results and executed
checkpoint/hardware/code provenance remain unavailable.

A later external DeepSeek EAGLE3 ttt7 measurement extends the table below; its
training budget is not matched to the five structure models.

The requested comparison uses step-1500 checkpoints and shared/merged adapters
at concurrency 1. See [suite, readiness and command](../../../examples/eagle3-lora/structure_benchmarks_v1/README.md).
During preparation, attention static LoRA step 1500 was verified at revision
`dfb78b270a39f625bc7a39ef75cfaaab5868eb54`. The attention-only decoder-width-256
generator was included as a fifth model. Its pre-run revision
`afd0df1af02dfb6f379eb4f0d529029a1abdc075` had only step 500/1000, so the suite
required waiting for step 1500. Executed bundle/checkpoint hashes and final Hub
pins were not supplied; the named budgets are configured/reported rather than
independently verified from final native run artifacts.

## Working summary retained for future experiments — 2026-10-31

These are the corrected working findings from the discussion, scoped to the
experiments measured so far:

- **Generate all attention projections rather than only V/O.** Expanding the
  generator target set from V/O to Q/K/V/O improves acceptance under the same
  training data and update budget. The user explicitly confirmed the matched
  data/step budget on 2026-10-31; do not dismiss this comparison merely because
  the latest five-model serving table omits the earlier V/O row. See the
  earlier module ablation in [GEN-009](GEN-009-generator-structure.md).
- **Generator versus static LoRA is domain dependent.** Generated attention
  and all-module adapters improve math acceptance over their matched static
  LoRAs by 5.58 and 4.07 percentage points. Chat shows no consistent advantage;
  the evidence does not support saying generators always beat static LoRA.
- **Generation/merge latency is a cost, not an established explanation for the
  entire throughput gap.** Recorded generation, merge and restore durations
  total less than 0.5% of benchmark wall time. Optimized CUDA kernels may reduce
  those operations, but cannot yet be claimed to solve the throughput gap.
  Separate drafter/verifier profiling and wall-time accounting remain needed.
- **Smaller decoder, lower generation latency, similar acceptance.** Decoder
  256 reduces timed generation latency by 16–17%. End-to-end throughput is mixed:
  decoder 512 leads on math (78.18 versus 77.19 tok/s), while decoder 256 leads
  slightly on chat (55.57 versus 55.41). Neither is always faster.
- **All-module generation is feasible, but its extra coverage has not yielded
  a throughput advantage here.** It completed serving and attained the highest
  math draft acceptance among the five structure models (67.79%).
  Attention-only generation is the current
  promising cost–acceptance tradeoff; efficient broader generation remains an
  open direction, rather than a demonstrated infeasibility.

Correctness remains unchecked and repeated-trial uncertainty is unavailable.
Target-module coverage has a clearer observed benefit than increasing the
hypernetwork's internal size; these findings do not establish architecture
failure or universal rankings.

## Supplied measurements and checks

Receipt date: 2026-10-30; execution date and per-run hardware provenance were
not provided. The initial [Markdown throughput table](benchmarks/2026-10-30-structure-math-chat-c1/summary.md),
[rounded transcription](benchmarks/2026-10-30-structure-math-chat-c1/metrics.json)
and [rounded-value analysis](benchmarks/2026-10-30-structure-math-chat-c1/analysis.json)
are preserved as the original evidence. Their missing-evidence statements
describe that first receipt. The subsequently supplied
[detailed suite summary](benchmarks/2026-10-30-structure-math-chat-c1/suite_summary_detailed.json)
adds full-precision throughput, acceptance counters, workload hashes and adapter
timings; it is archived from the user's JSON, not retrieved from their runtime.
Current comparisons use the [full-precision analysis](benchmarks/2026-10-30-structure-math-chat-c1/detailed_analysis.json),
reproducible with [analyze.py](benchmarks/2026-10-30-structure-math-chat-c1/analyze.py).

Suite identity: `ff2115d88b483658a62c2a6ba0791c3f042ed126dbbd309606afbafbb3124fab`.
All ten measurements report 50 successful requests and 50 prepared/completed
adapters/batches, mean batch size 1, with no output-mismatch override. Each
category has the same dataset hash across all five models:

- Math: `d2990b90cdc231a932afe725035f9b403a82f483d390062888beed4358d93b28`;
  12,800 output tokens per measurement.
- Chat: `4462e1e17606a424d791e4d34ad18558f8d327aa646597dd3362e336afd0f91f`;
  11,032 output tokens per measurement.

Acceptance identities and output tokens / measured seconds match the summary,
and all rounded throughput values match the initial table. Equal hashes and
output-token totals strengthen workload comparability; they do not establish
identical output sequences or correctness. Native request files, server/client
logs and reproducibility artifacts are still absent.

## Extended throughput table

Historical rows are copied from [GEN-003](GEN-003-generator-suite-throughput.md)
and retain their original evidence limits. They are not fresh measurements
from this suite. The five new models use a different training budget/data pool
from the historical 24K generators. Hardware/runtime equivalence and historical
workload identity must be checked before claiming historical relative speedups.

| Model | Nemo math output tok/s | Nemo chat output tok/s |
| --- | ---: | ---: |
| EAGLE3 base — historical | 64.34 | 54.97 |
| Trained v/o LoRA 4K / 1,500 steps — historical | 72.50 | 56.35 |
| Generator, 24K / 2,000 steps — historical | 77.58 | 57.58 |
| Generator, 24K / 4,000 steps — historical | 79.11 | 56.49 |
| Attention static LoRA, 4K / 1,500 steps | 75.72 | 56.48 |
| All-module static LoRA, 4K / 1,500 steps | 77.90 | 57.26 |
| Attention generator, 4K / 1,500 steps | 78.18 | 55.41 |
| All-module generator, 4K / 1,500 steps | 76.72 | 55.05 |
| Attention generator, decoder 256, 4K / 1,500 steps | 77.19 | 55.57 |
| DeepSeek EAGLE3 ttt7 — external pretrained drafter | 82.50 | 54.10 |

Aggregate output throughput includes prefill and serving overhead. It differs
from GuideLLM's median per-request output token rate.

## External DeepSeek EAGLE3 ttt7 follow-up — 2026-10-31

The later [supplied full summary](benchmarks/2026-10-31-deepseek-ttt7-math-chat-c1/summary.json)
and [derived comparisons](benchmarks/2026-10-31-deepseek-ttt7-math-chat-c1/analysis.json)
add one external drafter, `deepseek-ai/eagle3_qwen3_8b_ttt7`, at concurrency 1.
The [standalone config/runbook](../../../examples/eagle3-lora/structure_benchmarks_v1/deepseek_ttt7.md)
pins revision `f6485ba8d21e11942958617dbe7e71b467f38f38` and prepares its
metadata for vLLM 0.31.0 without changing tensor bytes. The supplied runtime
summary does not include executed preparation/checkpoint hashes, so the pin and
translation describe configured intent rather than independently verified
execution provenance. The earlier dry-run-only status is superseded by these
supplied GPU-serving measurements; correctness/forward parity remain unchecked.

Suite identity: `16c13e472f51ae40d9cbe310b9846998e62fc11cabf3c380d1058f10e288d6e4`.
Both categories report 50 successful requests, no mismatch override, the same
subsets, workload hashes and output-token totals as all five structure models.
Acceptance and throughput identities pass validation. Adapter metrics are empty;
they provide no isolated drafter/verifier timings.

| Metric | Math | Chat |
| --- | ---: | ---: |
| Aggregate output tok/s | 82.50 | 54.10 |
| Draft acceptance rate | 74.04% | 37.13% |
| Acceptance length (includes one verifier/bonus token) | 3.221 | 2.114 |
| Acceptance at position 0 / 1 / 2 | 84.92% / 73.18% / 64.01% | 57.11% / 33.66% / 20.63% |
| Median TTFT ms | 45.31 | 46.35 |
| Median TPOT ms | 11.647 | 18.451 |

- **Math:** highest reported throughput in the extended table and higher draft
  acceptance than all five structure models. Versus the attention generator,
  throughput is 5.52% higher and draft
  acceptance is 7.36 percentage points higher. Versus the all-module generator,
  acceptance is 6.25 points higher and throughput is 7.53% higher. The gain is
  visible through the third draft position (64.01% versus 54.19% for all-module
  generation); the checkpoint's name does not change this suite's three-token
  speculation setting.
- **Chat:** throughput is 2.38% below attention generation and 5.53% below
  all-module static LoRA. Draft acceptance is 1.60 and 2.16 points lower,
  respectively. This external checkpoint is not a universal improvement across
  the two categories.
- **Interpretation:** retain it as a strong external math baseline. Its five
  auxiliary target features, Qwen3 drafter architecture and external training
  data/budget differ from the models adapted in our experiments. This is not a
  matched-budget comparison or evidence that any single design choice caused
  the math gain. Equal workload hashes improve comparability but do not prove
  equal runtime/hardware or correctness. Repeated trials and executed provenance
  are still needed.

The acceptance and timing tables below retain the original five-model matched
structure comparison; the external checkpoint is measured separately above.

## Acceptance and matched comparisons

Acceptance rate is accepted draft tokens / proposed draft tokens. With three
proposals per cycle, the reported acceptance length equals
`1 + accepted_tokens / draft_cycles = 1 + 3 * draft_acceptance_rate`.
It includes one verifier/bonus token; accepted draft prefix length excludes that
one. These are aggregate speculative acceptance measurements, not task accuracy
or offline teacher-forced agreement. Their scope is the whole invocation; here
each model/category has one concurrency point.

| Model | Math draft acceptance | Chat draft acceptance | Math acceptance length | Chat acceptance length |
| --- | ---: | ---: | ---: | ---: |
| Attention static LoRA | 61.11% | 38.83% | 2.833 | 2.165 |
| All-module static LoRA | 63.72% | 39.30% | 2.911 | 2.179 |
| Attention generator | 66.68% | 38.74% | 3.000 | 2.162 |
| All-module generator | 67.79% | 38.80% | 3.034 | 2.164 |
| Attention generator, decoder 256 | 66.52% | 39.32% | 2.996 | 2.179 |

Changes below use full-precision values; percentage points (pp) describe absolute
acceptance-rate changes. They are descriptive differences without trial-level
uncertainty estimates.

| Candidate vs reference | Math throughput | Chat throughput | Math acceptance change | Chat acceptance change |
| --- | ---: | ---: | ---: | ---: |
| Attention generator vs attention static LoRA | +3.25% | −1.88% | +5.58 pp | −0.09 pp |
| All-module generator vs all-module static LoRA | −1.51% | −3.86% | +4.07 pp | −0.49 pp |
| Attention decoder-256 vs decoder-512 generator | −1.27% | +0.29% | −0.16 pp | +0.58 pp |
| Attention decoder-256 generator vs attention static LoRA | +1.94% | −1.60% | +5.41 pp | +0.49 pp |
| All-module static LoRA vs attention static LoRA | +2.87% | +1.39% | +2.61 pp | +0.47 pp |
| All-module generator vs attention generator | −1.87% | −0.65% | +1.10 pp | +0.07 pp |

The detailed evidence updates the earlier throughput-only interpretation:

- **Generation improves math acceptance over matched static LoRA.** Attention
  increases acceptance length by 5.90%, while all modules increase it by 4.20%.
  The attention gain accompanies higher throughput; the all-module gain does
  not. The all-module generator has the highest math acceptance in this set.
- **Chat has no consistent acceptance advantage from generation.** The two
  original generators are slightly below their matched static LoRAs. The compact
  attention generator is effectively level with all-module static LoRA
  (39.316% versus 39.298%). This workload does not show a clear conditional
  adaptation benefit on chat.
- **Module coverage improves math acceptance, without guaranteeing throughput.**
  All modules beat attention-only acceptance for both adaptation methods on math.
  On chat the changes are small. The all-module generator is slower than the
  attention generator despite better math acceptance. GEN-009's offline coverage
  result therefore has some serving-acceptance support, but throughput needs a
  separate explanation.
- **The compact generator retains similar acceptance and takes less generation
  time in this run.** Configured parameter counts are 326,709,344 versus
  646,297,952, about 49.45% fewer (calculated from pinned projection shapes, not
  measured from submitted checkpoints). The timed generator call is 16–17%
  shorter, as detailed below. Peak memory and repeatability remain unmeasured.

## Adapter timing and the unresolved throughput gap

The counters report 50 generator invocations for 50 completed requests for each
generator, mean conditioning size 1. The adapter is generated once per prompt
and merged for that request; these measurements do not regenerate LoRA at each
decoding step. Static LoRA has zero generator calls and zero per-request
merge/restore timing because its weights are merged permanently before serving.
Zero per-request cost does not mean model loading/setup has no cost.

The runtime uses CUDA events for generation, merge and restore
([elapsed](../../../src/speculators/generator/serving/controller.py)); the table
uses arithmetic means, not latency medians or percentiles. Each aggregate duration
is divided by 50 completed requests. Adapter total is the sum of those three
reported durations; its wall fraction divides by benchmark `measured_seconds`.

| Generator | Category | Generate ms/request | Merge ms/request | Restore ms/request | Timed adapter total ms/request | Timed total / measured wall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Attention, decoder 512 | Math | 3.152 | 3.181 | 0.256 | 6.590 | 0.201% |
| Attention, decoder 512 | Chat | 3.209 | 3.192 | 0.260 | 6.662 | 0.167% |
| All modules | Math | 5.668 | 9.702 | 0.714 | 16.084 | 0.482% |
| All modules | Chat | 5.735 | 9.709 | 0.709 | 16.153 | 0.403% |
| Attention, decoder 256 | Math | 2.604 | 3.186 | 0.270 | 6.060 | 0.183% |
| Attention, decoder 256 | Chat | 2.692 | 3.206 | 0.271 | 6.169 | 0.155% |

Decoder 256 reduces generator-call duration by 17.40% on math and 16.11% on chat;
merge cost remains similar. All-module generation takes about 79–80% longer
than attention-only, while merging takes about three times as long.

**These recorded one-time durations alone do not explain the full throughput
gaps.** Their summed fraction is below 0.5%, including for all-module generation,
whose throughput is 1.51% below its static counterpart on math and 3.86% below
on chat. CUDA-event durations are not a complete wall-time partition. Host-side
work, synchronization, persistent memory effects, prefill/decoding interactions
and run variability are not isolated by these counters. They remain hypotheses,
not diagnosed causes.

| Model | Math median TTFT ms | Chat median TTFT ms | Math median TPOT ms | Chat median TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| Attention static LoRA | 43.48 | 45.10 | 12.906 | 16.753 |
| All-module static LoRA | 43.46 | 44.56 | 12.408 | 16.787 |
| Attention generator | 50.89 | 52.11 | 12.187 | 17.667 |
| All-module generator | 61.72 | 62.92 | 12.691 | 17.761 |
| Attention generator, decoder 256 | 50.99 | 52.05 | 12.467 | 17.535 |

Compared with matched static LoRA, median TTFT is about 7 ms higher for attention
generation and 18 ms higher for all-module generation. All-module math TPOT is
also higher despite better acceptance, so setup alone is not an adequate causal
account. Medians are separate request statistics and cannot be subtracted from
aggregate durations to construct an exact time budget.

There is **no isolated drafter-forward latency or verifier latency** in the
supplied summary. End-to-end TPOT contains drafting, verification and serving
work; subtracting generator seconds from benchmark wall time would not isolate
the drafter. Profiling those components under matched runtime conditions and
repeating trials is needed to locate the remaining gap.

## Working interpretation and open evidence

Keep the attention generator and its compact variant as candidates, with
all-module static LoRA as a strong throughput control. Attention generation has
a math acceptance advantage over matched static LoRA; all-module generation
has the strongest math acceptance within that original five-model set. Neither
establishes a general throughput win. Within the original five-model set, best
math throughput is attention generation at 78.18 tok/s, only 0.37% above
all-module static LoRA at 77.90; best chat throughput is all-module static
LoRA at 57.26. Small rankings need repeated trials.

These results do not establish that the hypernetwork architecture is inadequate.
They support compact generation and a domain-dependent acceptance benefit.
They also do not justify maximizing generated weight count from offline
agreement alone. Before architectural redesign, collect matched drafter/verifier
profiling and repeated serving measurements. Preserve request-level results,
selected checkpoint/revision and weight hashes, hardware/package/code provenance,
server/client logs, and training/eval/vLLM command and patch artifacts.

## Reported all-module/chat incomplete workload

The user previously supplied a failed `generator-all-4k-1500/nemotron/nemo-chat`
first attempt: expected 50 requests; GuideLLM measured 49 successful, zero
errored, zero incomplete, total 49. The client failed its completeness check,
followed by owned-server cleanup. No throughput row from that incomplete attempt
is accepted. This is not evidence by itself of a model crash or a specific
GuideLLM bug. The failed attempt's native result and complete client log have
not been supplied for inspection.

Real GuideLLM 0.7.1 tests against a local HTTP endpoint completed 50-request
workloads at C1 and C2, including short-response and empty-output paths. They did
not reproduce the failure. Diagnostic reporting now preserves scheduler/measurement
distinctions on count mismatch; suite resume preserves completed measurements.

The subsequent detailed all-module/chat summary reports 50 successful and 50
completed requests. Its aggregate completeness checks pass. This establishes
completeness for the supplied later measurement, without retrospectively
establishing the earlier failure's cause or validating acceptance of 49/50.
