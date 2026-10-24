# BATCH-001 — Shared-adapter concurrency sweep

## Record and evidence

- Recorded: 2026-10-24; execution dates not supplied.
- Status: measured throughput supplied; preliminary, no repeat uncertainty or correctness verification.
- Evidence: completed user-supplied `comparison-v2` summary, four models × four concurrencies × seven categories = 112 values.
- [Full transcribed table](benchmarks/2026-10-24-comparison-v2-concurrency/summary.md), [structured measurements](benchmarks/2026-10-24-comparison-v2-concurrency/metrics.json), and [derived comparisons](benchmarks/2026-10-24-comparison-v2-concurrency/derived_comparisons.json).
- Concurrency-1 values are reused from [GEN-003](GEN-003-generator-suite-throughput.md); all 28 match exactly. They are not new repeat measurements.
- Native GuideLLM results, actual cohort sizes, acceptance, timing breakdowns, exact checkpoint hashes, hardware identity, and per-run command/patch provenance have not been supplied for this record.

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
with a smaller incremental advantage over static LoRA at high concurrency.
It is not yet evidence of robust OOD superiority, better acceptance, improved
task quality, optimal production throughput, or verified lossless acceleration.
[GEN-002](GEN-002-http-token-equality.md) records unresolved output equality and
repeatability issues. Continue exploratory timing as requested; restore a small
controlled correctness/quality check before making lossless or quality claims.

## Updates

- 2026-10-24: Archived the complete supplied sweep, matched-concurrency ratios,
  hypothesis assessment, unresolved confounders, and proposed next controls.
