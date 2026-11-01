# GEN-013 — Attention generator: 100K eligible pool, 1K/2K updates

Date: 2026-11-01. Status: supplied 14-cell C1 benchmark summary; correctness and
repeatability unchecked. These are user-supplied measurements, not locally
executed GPU benchmarks.

## Evidence and protocol

- [Supplied detailed metrics](benchmarks/2026-11-01-attention-100k-2k-c1/summary.json)
- [Throughput table](benchmarks/2026-11-01-attention-100k-2k-c1/summary.md)
- [Validated counters and comparisons](benchmarks/2026-11-01-attention-100k-2k-c1/analysis.json)
- [Available/missing provenance](benchmarks/2026-11-01-attention-100k-2k-c1/provenance.json)
- [Extended seven-category historical table](benchmarks/2026-10-31-attention-24k-6k-c1/extended_historical_table.md)
- [Matching draft acceptance-rate table](benchmarks/2026-10-31-attention-24k-6k-c1/extended_acceptance_table.md)

Suite identity: `2a70c89bad91a8e365577b5b0f1b383eb12532b51ccc19ba78c370bf0e8c6e2a`.
The [archived benchmark config](benchmarks/2026-11-01-attention-100k-2k-c1/benchmark_config.yaml)
requests `azeeer/eagle3-generator-attention-100k-2k-v1` at commit
`cf09002ae6fbca1fe0e9139ab8a0eee3b149a124`, folders `step-00001000` and
`step-00002000`. Those configured pins are not an executed model-checksum record.

Serving settings match GEN-011: concurrency 1, 50 requests/category, 256 output
tokens, three draft tokens, three warmups, seed 42, shared/merged adapter, one
conditioning prompt. Seven workloads cover four Nemotron categories and three
Red Hat OOD categories. All supplied workload hashes, subset lists, and output
totals match the GEN-011 reference. All 14 cells pass 50-request counts,
three-proposal acceptance identities, generator timing means, and
`output_tokens / measured_seconds` throughput checks.

The intended training protocol is a fresh original-RedHat-drafter Q/K/V/O
generator with the earlier attention architecture, BF16/cohort batching, and
four same-subset cohorts/update, sizes 1–8. The eligible pool has 100K training
rows (25K/category), with a deterministic 1K offline validation view. The full
run caps at 2,000 optimizer updates, validates every 250, and saves/uploads every
500. It does not imply a complete pass over 100K examples. The executed training
command, actual examples seen, runtime/checkpoint hashes and timing artifacts
were not supplied with these metrics; training setup follows the checked-in
[runbook](../../../examples/eagle3-lora/attention_100k_2k_v1/README.md).

## Measurements

| Model | Nemo math | Nemo code | Nemo STEM | Nemo chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Attention generator, 100K / 1,000 | 76.49 | 67.85 | 66.14 | 56.64 | 68.17 | 66.56 | 58.11 |
| Attention generator, 100K / 2,000 | 79.53 | 69.79 | 67.75 | 58.17 | 70.11 | 68.68 | 59.24 |

Draft acceptance is accepted proposed tokens divided by proposed tokens:

| Checkpoint | Nemo math | Nemo code | Nemo STEM | Nemo chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1000 | 65.15% | 54.08% | 51.68% | 39.73% | 54.47% | 52.69% | 41.52% |
| 2000 | 67.43% | 55.10% | 51.99% | 40.26% | 55.09% | 52.87% | 41.75% |

Observed comparisons use unrounded source values:

| Category | 2K vs 1K tok/s | 100K/2K vs 24K/2K tok/s | 100K/2K vs 24K/2K acceptance |
| --- | ---: | ---: | ---: |
| Nemo math | +3.98% | -3.94% | +0.58 pp |
| Nemo code | +2.86% | -4.58% | -0.17 pp |
| Nemo STEM | +2.43% | -4.19% | +0.02 pp |
| Nemo chat | +2.70% | -2.68% | +0.65 pp |
| HumanEval | +2.84% | -1.62% | +0.09 pp |
| Math reasoning | +3.19% | -1.35% | +0.10 pp |
| Summarization | +1.94% | -1.17% | +0.32 pp |

## Preliminary observations and open questions

1. From step 1K to 2K, throughput rises in all seven categories (+1.94–3.98%).
   Draft acceptance also rises in every category (+0.18–2.28 percentage points).
2. At the same 2K-update checkpoint, the 100K checkpoint has lower
   measured throughput than GEN-011's 24K/2K checkpoint in every category
   (−1.17% to −4.58%). Acceptance is slightly higher in six categories and
   slightly lower in code (−0.17 pp); math/chat gain approximately 0.58/0.65 pp.
   The result therefore does not show a uniform acceptance loss from the
   expanded eligible pool, nor a measured throughput gain from expansion.
3. Reported generator+merge timing is 6.32–6.40 ms/request, 0.145–0.197% of
   aggregate measured wall time. These counters do not isolate drafter or
   verifier latency and do not explain the entire throughput difference.
4. Matched workload hashes do not establish matched hardware/runtime. The
   24K and 100K pools also have different composition and coverage at a fixed
   update budget. Equal steps are not equal completed epochs. Without executed
   provenance, repeat trials and examples-seen counters, this comparison cannot
   attribute the throughput difference to pool size or model architecture.

Retain both 100K checkpoints in the working table. A same-runtime repeated
24K/2K versus 100K/2K comparison, with drafter/verifier timings and training
coverage recorded, would separate execution variation from a data-scaling
signal. Longer 100K training is unmeasured here; these observations are not a
final conclusion that additional data helps or hurts.

## Follow-up: actual training exposure and extraction provenance

The original summary lacked execution artifacts. A subsequent audit retrieved
the **published step-2000 training artifacts** for both models; see
[the audit and immutable source hashes](benchmarks/2026-11-01-attention-100k-2k-c1/training_audit/audit.json)
and [archived commands/configs](benchmarks/2026-11-01-attention-100k-2k-c1/training_audit/README.md).
Benchmark execution provenance and repeated serving runs remain unavailable.

| At update 2,000 | 24K attention | 100K attention |
| --- | ---: | ---: |
| Eligible training rows | 24,000 | 100,000 |
| Logged sample presentations | 35,918 | 35,790 |
| Reconstructed distinct training rows | 24,000 | 35,790 |
| Repeated presentations | 11,918 | 0 |
| Eligible pool visited | 100% | 35.79% |
| Logged response tokens | 39,001,722 | 38,501,176 |
| Updates in first complete epoch | 1,338 | 5,552 |
| Offline validation rows | 400 | 1,000 |

The two runs expose similar numbers of examples/response tokens at 2K updates;
100K denotes the eligible pool, not 100K rows processed by the optimizer. The
expanded run trades repeated exposure for more distinct examples. The old 24K
training IDs remain in the 100K pool; old validation IDs do not enter its training
split. This is an ID-level check, not a new text/semantic decontamination result.

Published configs agree on Q/K/V/O targets, rank 32/alpha 64/dropout 0.05,
architecture, BF16 generator/drafter execution, generator batching, `lk_hybrid`,
four cohorts/update and 8,192-token microbatches. Logged learning rates agree at
every update: 100-update linear warmup to 1e-4, then constant. Although the 24K
snapshot comes from a run planned through 6K updates, it has no different
cosine/terminal learning-rate decay at step 2K. The checked-in core forward/loss/
update files have no diff between the two recorded training commits.

**Actual extractor commands differ from the checked-in 24K runbook defaults:**

| Recorded extractor setting | 24K | 100K |
| --- | ---: | ---: |
| vLLM version | 0.31.0 | 0.32.0 |
| Active sequences | 16 | 128 |
| GPU memory utilization | 0.50 | 0.40 |
| API server processes | 2 | 4 |

Both use the same target weight-shard hashes, BF16, layers 2/18/33/36,
8,193-token context/batch limits and eager extraction without prefix caching or
chunked prefill. Training checkpoint/upload intervals also differ (2,000 versus
500). Runtime matching from the YAML/runbook alone was therefore too strong;
the recorded command is the stronger evidence. These are extraction settings,
not proof of which environment executed the serving benchmarks.

The serving discrepancy is more specific than "larger pool has lower
acceptance": at step 2K, 100K acceptance exceeds 24K in six categories, yet
throughput is lower in all seven. For math, measured wall time per counted
speculation round is 36.23 ms (24K) versus 37.94 ms (100K), approximately +4.72%,
while the 100K model needs fewer rounds (4,242 versus 4,267). This wall-time proxy
includes prefill/client/serving overhead and cannot isolate drafter/verifier
kernels. Generator+merge costs alone are too small to account for this gap.

Interpretation remains open: more diverse data with fewer repetitions can
plausibly yield similar acceptance at a fixed update budget; these results do
not diagnose hypernetwork collapse or insufficient capacity. A repeated,
alternating 24K/2K versus 100K/2K serving comparison under one recorded GPU/
software setup, with a common base drafter control, is the most direct next check
for the throughput discrepancy. For data-coverage scaling, roughly 8.3K updates
would give the 100K pool the same approximate presentation-count/pool-size ratio
as 24K at 2K; improved acceptance is not guaranteed by matching this ratio.
