# BATCH-001 — Shared-adapter concurrency sweep

## Record and evidence

- Recorded: 2026-10-24; execution dates not supplied.
- Status: measured throughput supplied; preliminary, no repeat uncertainty or correctness verification.
- Evidence: completed user-supplied `comparison-v2` summary, four models × four concurrencies × seven categories = 112 values.
- [Full transcribed table](benchmarks/2026-10-24-comparison-v2-concurrency/summary.md), [structured measurements](benchmarks/2026-10-24-comparison-v2-concurrency/metrics.json), and [derived comparisons](benchmarks/2026-10-24-comparison-v2-concurrency/derived_comparisons.json).
- Concurrency-1 values are reused from [GEN-003](GEN-003-generator-suite-throughput.md); all 28 match exactly. They are not new repeat measurements.
- Initially, only the throughput table was supplied. The detailed suite summary received on 2026-10-24 now supplies workload hashes, aggregate acceptance, adapter counters/timings, and median latency. Native request-level GuideLLM files, exact checkpoint hashes, hardware identity, and full command/patch provenance remain unavailable.

The initial failed LoRA rows were startup failures, not throughput observations.
The reported `KeyError: 'max_examples'` came from applying generator-only metadata
requirements to a static bundle. After runtime fix `c96b851`, the user supplied
the completed LoRA rows recorded here. Successful results were retained on resume.
This identifies the reported failure and fix, not every execution's code revision.

## Protocol and what is being measured

Numbers are aggregate output tokens/s, including prefill and serving overhead.
The intended suite protocol uses BF16, eager V1 execution and fixed cohorts:
concurrency caps of 1/2/4/8, one generated shared adapter per generator cohort,
no refilling vacant slots before the last member finishes, and the same admission
policy for original EAGLE3 and static LoRA. See the
[runbook](../eagle3_benchmark_suite.md#shared-adapter-concurrency-sweeps).
Actual sizes can be below the cap; summary tables cannot establish them.

Nemotron math/code/STEM/chat are treated as in-distribution. The selected Red Hat
HumanEval/math reasoning/summarization workloads are treated as out-of-distribution.
Current exporter defaults use raw prompts for HumanEval and chat formatting for
the other Red Hat files; category and formatting shifts are therefore confounded.

Current full-data training code forms contexts of 1–8 examples within one subset
([EpochSchedule](../../../src/speculators/generator/epoch_training.py)). The suite
benchmarks categories separately. Randomly mixed-domain serving has not been tested.

Checkpoint names are labels, not verified update counts. The example YAML maps
`generator-24k-2000step` to `epoch-0002`; an earlier archive of an epoch-2 checkpoint
recorded 2,676 updates. Confirm source checkpoint metadata before interpreting the
two labels as precisely 2,000 versus 4,000 updates or a doubling of training.
The recorded static-LoRA recipe is a 4K/1,500-update baseline; the generator labels
describe 24K training. Exact matched budgets are unverified from this summary.

## Matched-concurrency comparisons

Percentage change is `100 × (candidate tok/s ÷ reference tok/s − 1)`.
Group means below are **unweighted means of per-category percentage changes**;
they are not combined-workload throughput or pooled speedup estimates.

| Concurrency | G4 vs EAGLE3: ID mean | G4 vs EAGLE3: OOD mean | G4 vs LoRA: ID mean | G4 vs LoRA: OOD mean |
| --- | ---: | ---: | ---: | ---: |
| 1 | +13.73% | +1.66% | +5.72% | +1.87% |
| 2 | +14.65% | +1.79% | +2.98% | +4.09% |
| 4 | +14.89% | −0.82% | +4.34% | +0.61% |
| 8 | +13.08% | +0.20% | +1.08% | −1.20% |

G4 denotes the checkpoint labeled `generator-4000step`. G2 denotes
`generator-24k-2000step`.

| Category, concurrency 8 | G2 vs EAGLE3 | G4 vs EAGLE3 | G4 vs LoRA |
| --- | ---: | ---: | ---: |
| Nemo math | +20.97% | +21.92% | +1.04% |
| Nemo code | +15.38% | +14.40% | +1.18% |
| Nemo STEM | +10.77% | +10.86% | +0.15% |
| Nemo chat | +3.07% | +5.13% | +1.93% |
| HumanEval | −6.10% | −4.68% | +0.35% |
| Math reasoning | +0.60% | +1.88% | −1.30% |
| Summarization | +4.46% | +3.39% | −2.64% |

## Hypotheses supported and not established

1. **Adaptation improves Nemotron throughput over original EAGLE3: supported descriptively.** Both generators beat EAGLE3 in all four ID categories at every concurrency. For G4, math gains remain about 22–23%; chat gains are much smaller. This is the strongest consistent pattern.
2. **Dynamic conditioning beats a static learned adapter: partial support.** G4 has positive ID category-mean gains over LoRA at all concurrencies, but only +1.08% at C8. Its C8 STEM advantage is 0.61 tok/s (+0.15%). Such small single-run gaps cannot establish a robust deployment advantage.
3. **General OOD acceleration: not supported as a general claim.** At C4, G4 is slower than EAGLE3 in all three OOD categories. At C8, EAGLE3 leads HumanEval, while static LoRA leads math reasoning and summarization. A near-zero OOD mean hides opposing category effects; it is not evidence of equivalence.
4. **Larger conditioning groups improve adapter quality: not identified.** Concurrency, conditioning count, GPU batching, overhead amortization, and cohort tail effects change together. Higher aggregate throughput alone cannot separate them.
5. **More training consistently improves serving: limited, diminishing evidence.** G4 exceeds G2 in 22/28 cells, but some differences are tiny. ID mean changes are +1.19%, +1.57%, +2.84%, +0.51% at C1/C2/C4/C8. OOD changes are +0.19%, +2.88%, +0.28%, +0.59%. Code at C8 and summarization at C1/C8 favor G2. This is not a clean overfitting result or a verified learning curve in exact optimizer steps.
6. **The system wins under ordinary vLLM serving: open.** Baselines were deliberately constrained to the same fixed-group scheduler. Original EAGLE3 and a permanent static LoRA could benefit from continuous batching. That practical baseline has not been measured here.

Large C8/C1 throughput ratios should not be presented as the generator's
incremental speedup. Nemo math scales 5.71× for G4 and 5.76× for EAGLE3. Most of
that increase is available through batching in either system; the relevant C8
G4/EAGLE3 ratio is 1.219×, and G4/LoRA is 1.010×.

The static adapter's own HumanEval regression suggests that generic adaptation
or distribution mismatch may contribute to this failure; it cannot be assigned
solely to set pooling. Neither compute saturation, acceptance changes, nor batch
stragglers are diagnosed by these throughput summaries alone.

## Relation to earlier training observations

[GEN-001](GEN-001-direct-data-scaling.md) found closely tracking offline validation
curves across 4K/8K/16K pools at a fixed update budget. It does not establish that
24K training is unnecessary for OOD or serving performance, and offline agreement
is not serving acceptance. No retired generator row is added to this comparison.

[GEN-004](GEN-004-bank-warmup-adaptation.md) supports the operational decision to
retire bank warmup: the fresh generator caught up in the displayed adaptation
losses. That study did not isolate equal total compute or establish universal
warmup uselessness. No additional bank experiments are proposed.

The generator has roughly 293M trainable parameters and a conditional inference
step, whereas ordinary LoRA is much smaller. Unequal training data, update/token
budgets, parameterization, and tuning can explain part of the current gap. The
existing comparison does not isolate conditional information as its cause.

## Prioritized next experiments

1. **Inspect already-produced artifacts before more GPU work.** Recover per-category accepted draft length/acceptance by position, prepared and completed batches, actual mean batch size, generator/merge/restore time per request, and request latency. Link them to workload and checkpoint hashes. They distinguish candidate explanations without repeating inference. The Markdown summary cannot supply these measurements.
2. **Repeat the small margins with matched cohorts.** At C8, start with math, STEM, HumanEval, and math reasoning, using EAGLE3, static LoRA, and G4. Run 3–5 paired repetitions with randomized model order and fixed request groups/lengths. Report uncertainty across complete runs or cohorts, not independent-token confidence intervals. C1 reuse and later LoRA retries span different execution periods, so they do not control runtime drift.
3. **Decouple conditioning count K from serving concurrency B.** Fix B=8, the same checkpoint, and the same cohort members; generate one shared adapter using K=1/2/4/8 selected prompt summaries. Randomize which member supplies K=1. All prompts still get normal target prefill, and every variant generates once per cohort. Compare acceptance and timing. This directly tests whether multiple conditioning examples help.
4. **Test conditioning information, not merely generator execution.** Compare own-cohort summaries with summaries from another fixed cohort, first within the same category and then across categories, keeping timing costs comparable. Permuting summaries inside the same cohort is not an ablation: mean/std pooling is permutation invariant. A same-architecture constant-conditioning model trained with the same budget is a stronger capacity/parameterization control than a deliberately mismatched input to the trained conditional model.
5. **Strengthen the static and serving baselines.** Train ordinary LoRA on the same 24K pool at a verified matched update/token budget and the same rank, loss and frozen models; give its learning rate a small fair tuning budget. Separately add standard continuous-batching evaluation for EAGLE3 and permanently merged LoRA. Report both matched-policy scientific comparisons and best available serving baselines.
6. **Probe transfer failures and heterogeneous groups.** Replay homogeneous and mixed-category cohorts at B=8 with comparable prompt/output lengths. Include HumanEval raw versus appropriately reformatted code prompts, and split short/long prompt and generation-length regimes. Formatting variants change the workload and need separate manifests. Use these tests to locate a failure before changing training data or architecture.

A cheap domain-specific/static-adapter routing control is also useful if
same-category mismatched conditioning performs like correct conditioning: the
generator may primarily identify domain-level adapter choices rather than
request-specific behavior. Oracle category routing must be labeled as such.

## Evidence limits and current interpretation

This is evidence for useful ID adaptation under the tested fixed-cohort setup,
with a smaller incremental throughput advantage over static LoRA at high concurrency.
The detailed summary below additionally establishes aggregate acceptance gains on ID.
It is not yet evidence of robust OOD superiority, improved task quality,
optimal production throughput, or verified lossless acceleration.
[GEN-002](GEN-002-http-token-equality.md) records unresolved output equality and
repeatability issues. Continue exploratory timing as requested; restore a small
controlled correctness/quality check before making lossless or quality claims.

## Updates

- 2026-10-24: Archived the complete supplied sweep, matched-concurrency ratios,
  hypothesis assessment, unresolved confounders, and proposed next controls.

### 2026-10-24 — Detailed suite summary and experiment selection

The supplied `/home/azeer/Downloads/summary.json` is archived unchanged as
[suite_summary_detailed.json](benchmarks/2026-10-24-comparison-v2-concurrency/suite_summary_detailed.json),
SHA256 `7b39e0f62c995a0d383107764ddb938523651092ce96a0ec68508f4b1ed01614`.
[detail_analysis.json](benchmarks/2026-10-24-comparison-v2-concurrency/detail_analysis.json)
records derived timing quantities and the source hash. All 112 measurements
contain 50 successful requests; each category's workload checksum matches across
all 16 model/concurrency variants. This verifies prompt-workload identity, not
output equality or identical execution environments.

The reported acceptance-length metric is `1 + accepted_tokens / draft_iterations`;
it includes a baseline token and is not an acceptance probability.

| Nemo category at C8 | EAGLE3 length | LoRA length | G4 length | G4 vs LoRA: length | G4 vs LoRA: throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| Math | 2.419 | 2.810 | 2.944 | +4.75% | +1.04% |
| Code | 2.305 | 2.520 | 2.618 | +3.85% | +1.18% |
| STEM | 2.233 | 2.471 | 2.537 | +2.66% | +0.15% |
| Chat | 2.087 | 2.162 | 2.198 | +1.68% | +1.93% |

The generator's aggregate drafting-quality advantage over static LoRA is clearer
than its throughput margin suggests. Its timed generation/merge/restoration
cost does not explain the remaining throughput discrepancy: summed recorded GPU
times account for only **0.070–0.105% of wall time** across the supplied generator
measurements. G4 generator execution is about 1.94–2.04 ms per invocation at C1,
and 2.09–2.92 ms at C8 (about 0.29–0.41 ms per request at C8). These timers exclude
uninstrumented CPU, scheduling, synchronization outside the measured operations,
and any other overhead; they do not prove zero serving overhead.

At C2/C4/C8, all supplied generator/static adapter records contain 25/13/7
completed batches for 50 requests, respectively: the minimum possible batch
counts. Mean C8 size is 50/7 = 7.143. Frequent extra partial cohorts are therefore
not evident, but these totals do not reveal within-cohort vacant-slot time,
individual cohort composition, or straggler effects.

HumanEval is not simply a case of lower mean acceptance: G4's C8 length is 2.579
versus EAGLE3's 2.541 (+1.53%), while throughput is 4.68% lower. Recorded adapter
GPU work is about 0.074% of wall time there. Scheduling/cohort composition, cost
per iteration, and execution variation remain possible explanations; these
aggregates cannot identify the cause. Output token totals also vary slightly
on Nemo chat/code/STEM across variants, so matching prompts do not establish
matching output sequences. Correctness remains unchecked.

The owner elected not to perform the repeated-run experiment. The next prepared
study trains four separate fixed-conditioning generators (K=1/2/4/8) and a
matched static LoRA on the 24K pool for 1,000 updates per arm, with fixed 8-example
loss cohorts and an ID-only serving comparison at B=8. This is a combined
training/serving K-policy comparison, rather than the originally proposed
single-checkpoint inference-only ablation. See
[BATCH-002 protocol and commands](../eagle3_conditioning_24k_1k.md).
