# GEN-012 — Generator on frozen DeepSeek EAGLE3 ttt7: 4K / 1,500 updates

Date: 2026-10-31. Status: supplied resumed paired serving summary, 14 C1
measurements across four Nemotron and three Red Hat OOD categories. The eight
original ID cells are retained unchanged. Aggregate acceptance and throughput identities
validate. Correctness remains unchecked; repeated trials and executed
checkpoint/config/preparation/hardware provenance were not supplied.

## Configured protocol and preparation history

[Training config](../../../examples/eagle3-lora/deepseek_ttt7_generator_v1/train.yaml)
and [preparation/extraction/training/benchmark commands](../../../examples/eagle3-lora/deepseek_ttt7_generator_v1/README.md)
use a fresh Q/K/V/O generator on frozen `deepseek-ai/eagle3_qwen3_8b_ttt7`,
pinned source revision `f6485ba8d21e11942958617dbe7e71b467f38f38`.
This is not the Red Hat drafter experiment in GEN-011.

The original 4K saved-response protocol is retained: 1K training rows/domain,
separate validation membership, random 1–8 cohorts, four cohorts/update,
1,500 optimizer updates, three-step LK loss, rank 32/alpha 64, decoder width 512.
BF16 generator parameters and cohort batching are enabled. Offline validation
every 250 updates and local/HF saves every 500. No rollout, replay or feedback
objective is introduced. These describe the checked-in plan and reported model
label; actual training membership/provenance has not been independently checked
from the supplied serving summary.

The external checkpoint uses five auxiliary target outputs: extraction indices
`[2,10,18,26,34]`, plus final verifier state 36 for supervision. Generator input
width is 20,480; Qwen3 Q/K normalization and all source tensor bytes are retained.
The preparation helper checks source names/shapes on the meta device and writes
native training metadata; only verifier-owned head/norm tensors load separately
from the local target. Original/native configs and checksums are copied into
training provenance. `trajectory_recipe` separately retains the source saved-token
recipe and validates original membership before new runtime extraction metadata
is constructed. Target/dataset changes are rejected; source lineage remains in
`trajectory_source`, and old feature caches remain separate.

Initial implementation validation comprised 44 focused/affected CPU checks and
an actual CLI dry run. The original benchmark was generator-only math/chat;
the user then added STEM/code and the frozen baseline. Subsequent supplied
measurements below supersede the previous GPU-serving-pending status. CPU checks
and these unchecked timing measurements do not establish full-size numerical
parity or token-level correctness.

HF destination: `azeeer/eagle3-generator-deepseek-ttt7-attention-4k-1500`.
Training W&B project: `eagle3-generator-ablations-v3`.

## Supplied paired ID measurements

[Full summary](benchmarks/2026-10-31-deepseek-generator-4k-c1/summary.json),
[throughput table](benchmarks/2026-10-31-deepseek-generator-4k-c1/summary.md),
[validated comparisons](benchmarks/2026-10-31-deepseek-generator-4k-c1/analysis.json).
Suite identity: `302891f36a1c9a6b8b34a25f98400aec00ab30c2c1be45a1084bd637a9e048fc`.
All eight cells report 50 successful requests and no output-mismatch override.
The generator has 50 completed adapters/calls per category, mean conditioning
size 1: one adapter generation per request, not per draft step.

| Category | Paired baseline output tok/s | Generator output tok/s | Throughput gain |
| --- | ---: | ---: | ---: |
| Math | 86.58 | 92.26 | +6.56% |
| Code | 75.40 | 83.73 | +11.04% |
| STEM | 67.20 | 78.01 | +16.08% |
| Chat | 56.68 | 65.46 | +15.50% |

| Category | Baseline draft acceptance | Generator draft acceptance | Absolute gain | Acceptance length: baseline → generator |
| --- | ---: | ---: | ---: | ---: |
| Math | 74.04% | 81.74% | +7.71 pp | 3.221 → 3.452 |
| Code | 60.70% | 70.65% | +9.95 pp | 2.821 → 3.119 |
| STEM | 49.93% | 62.85% | +12.92 pp | 2.498 → 2.885 |
| Chat | 37.13% | 47.80% | +10.67 pp | 2.114 → 2.434 |

Rate is accepted/proposed draft tokens; acceptance length includes one verifier/
bonus token. Counters satisfy three proposals/cycle and all acceptance identities.
Output tokens / measured seconds matches reported aggregate throughput. Both
models have identical category hashes/subsets/output totals: math 12,800, code
12,785, STEM 12,751, chat 11,032 output tokens. Hashes also match GEN-011's
four-domain workload. Equal totals and hashes do not prove equal output sequences.

## Adapter and request latency

| Category | Generate ms/request | Merge ms/request | Restore ms/request | Total timed adapter ms/request | Timed total / generator wall | Median TTFT increase ms | Median TPOT change ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Math | 3.115 | 3.172 | 0.263 | 6.550 | 0.236% | +6.89 | −0.558 |
| Code | 3.123 | 3.173 | 0.266 | 6.561 | 0.215% | +7.40 | −1.398 |
| STEM | 3.105 | 3.169 | 0.261 | 6.535 | 0.200% | +6.41 | −2.149 |
| Chat | 3.131 | 3.179 | 0.258 | 6.568 | 0.195% | +6.56 | −2.673 |

Adapter timings are CUDA-event arithmetic means; TTFT/TPOT are separate
per-request medians. They cannot be combined into an exact wall-time partition.
The TTFT increase is consistent in scale with one-time setup, while lower TPOT
and fewer draft cycles accompany improved acceptance. Draft-cycle counts fall
6.66–13.45% across the categories. Baseline adapter metrics are empty; the summary
does not isolate baseline/generator drafter-forward or verifier-forward latency.

## Working findings and evidence limits

- **Generated attention adaptation helps this strong drafter across all four
  measured domains.** Acceptance rises by 7.71–12.92 percentage points and
  throughput by 6.56–16.08% versus the frozen baseline in the same supplied suite.
  Chat and STEM improve as well as math/code, unlike the inconsistent chat
  advantage in the earlier Red Hat generator/static-LoRA comparison.
- **The acceptance benefit translates into a measured end-to-end benefit here.**
  Fewer verification cycles and lower median TPOT accompany higher throughput;
  the one-time adapter cost is small in these measurements. This does not
  establish a general timing decomposition or guarantee the same outcome at
  higher concurrency, longer/shorter responses or on other GPUs.
- **Use the paired baseline, not its earlier separate timing.** The original
  DeepSeek baseline was 82.50 math / 54.10 chat tok/s. This suite's baseline is
  86.58 / 56.68, 4.94% / 4.77% higher, despite identical aggregate acceptance
  dictionaries on those categories. Using the earlier denominator would inflate
  the reported generator improvement. The summaries do not diagnose whether
  hardware/runtime conditions, run variability or another cause explains this
  timing shift. Both runs remain archived separately.
- **This establishes benefit over the frozen drafter, not superiority to matched
  ordinary LoRA or proof that prompt conditioning causes the gain.** There is
  no DeepSeek static-LoRA arm, fixed/shuffled-conditioning control or repeated
  trial in this suite. The earlier Red Hat static-LoRA results are not a matched
  control for this different drafter. Task accuracy remains unchecked. The
  subsequent OOD serving measurements for this pair are recorded below; GEN-011
  remains a different frozen-drafter experiment.

Working direction: retain the DeepSeek-based attention generator as a strong
current candidate and compare against its paired baseline. A matched static-LoRA
arm, conditioning controls and repeat measurements would test which part of
training/generation supplies the advantage. Preserve training membership,
checkpoint/config/preparation hashes, actual hardware/package/code provenance,
request-level results and launch/eval command/patch artifacts with future runs.


## Paired OOD extension — supplied results

The existing paired config now appends `RedHatAI/speculator_benchmarks` with
HumanEval, math reasoning and summarization, using the same OOD settings as
GEN-011. Both the frozen DeepSeek baseline and trained step-1500 generator run
these categories at C1 with three draft tokens, 50 prompts/category and seed 42.
No existing Nemotron workload, model definition, W&B group or serving setting
changes. Pull updated code and rerun the runbook command in the **same paired-suite
output root** with `--resume`: eight valid completed ID cells are retained and
six new OOD cells run. Schema/resume checks and the actual CLI dry run passed.

The user subsequently supplied the [complete 14-cell summary](benchmarks/2026-10-31-deepseek-generator-4k-c1/summary_with_ood.json),
[seven-category table](benchmarks/2026-10-31-deepseek-generator-4k-c1/summary_with_ood.md),
and [validated paired comparisons](benchmarks/2026-10-31-deepseek-generator-4k-c1/analysis_with_ood.json).
Suite identity: `4c8ff44062d89196d9d242dc8033616e65f6d23cbe3dbfc9a0aea10e28854419`.
The original eight ID result objects are unchanged. All 14 cells report 50
successful requests and no mismatch override, and satisfy three-token acceptance
and throughput identities. All seven workload hashes, subsets and output-token
totals match both models and the full GEN-011 suite. Each OOD cell produces
12,800 output tokens. Exact-output correctness remains unchecked.

| OOD category | Baseline output tok/s | Generator output tok/s | Throughput gain | Baseline → generator draft acceptance | Acceptance gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 61.23 | 62.03 | +1.32% | 41.36% → 44.96% | +3.60 pp |
| Math reasoning | 73.54 | 76.74 | +4.36% | 56.76% → 63.92% | +7.16 pp |
| Summarization | 59.62 | 62.80 | +5.32% | 39.98% → 46.41% | +6.43 pp |

| OOD category | Acceptance length: baseline → generator | Timed adapter ms/request | Timed total / generator wall | Median TTFT increase ms | Median TPOT change ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| HumanEval | 2.241 → 2.349 | 6.566 | 0.159% | +8.45 | −0.460 |
| Math reasoning | 2.703 → 2.918 | 6.562 | 0.197% | +7.91 | −0.678 |
| Summarization | 2.199 → 2.392 | 6.578 | 0.161% | +8.12 | −0.799 |

Working update:

- Generated attention adaptation improves measured acceptance and throughput
  over the paired frozen drafter in all seven reported categories. OOD gains
  are smaller (+1.32–5.32% throughput, +3.60–7.16 acceptance points) than the
  Nemotron gains (+6.56–16.08% throughput, +7.71–12.92 points). This supplies
  positive serving evidence on these OOD prompt sets, not task-accuracy evidence
  or proof of universal generalization.
- HumanEval's +1.32% throughput difference particularly needs repeats. Its
  acceptance-length increase is 4.82%, which does not translate one-for-one
  into throughput. The measured setup accounts for only 0.159% of wall; the
  remaining timing behavior is not diagnosed by isolated drafter/verifier
  profiling. Do not attribute the entire difference to generator/merge latency.
- In the combined descriptive table, the DeepSeek generator has highest
  displayed throughput in six of seven categories. HumanEval remains higher
  for the Red Hat-drafter attention generator at 24K/4K updates (72.32 versus
  62.03 tok/s). Training budgets, base architecture and unverified cross-run
  runtime equivalence prevent calling this a controlled architecture ranking.
- Native preparation/weight/checkpoint hashes, actual training membership,
  hardware/code/package provenance, per-request records and repeated trials
  remain unavailable. A matched static-LoRA DeepSeek arm and conditioning
  controls are still needed to attribute the gain specifically to conditional
  generation rather than adaptation in general.

The [complete all-category comparison table](benchmarks/2026-10-31-attention-24k-6k-c1/extended_historical_table.md)
now has measured values for every displayed row and category. Earlier standalone
DeepSeek and paired ID archives remain unchanged; all adaptation gains use the
corresponding paired baseline.
