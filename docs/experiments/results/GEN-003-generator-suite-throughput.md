# GEN-003 — Generator suite throughput on in- and out-of-distribution categories

## Record

- Received: 2026-10-22; execution date not supplied.
- Status: preliminary historical suite plus subsequent GEN-011 attention rows;
  correctness not checked.
- Evidence: user-supplied `summary.md` for suite `comparison-v2`.
- Original location: `/content/drive/MyDrive/dyn-spec-bench-v1/suites/comparison-v2/summary.md`.
- Structured measurements: [metrics.json](benchmarks/2026-10-22-comparison-v2/metrics.json).
- Runbook: [benchmark suites](../eagle3_benchmark_suite.md).

## Setup and metric

Compare original EAGLE3, a trained ordinary LoRA, and generator checkpoints
labeled 2,000 and 4,000 steps. The original suite includes these four models. A later supplied 24K attention
generator study extends the Nemotron table with three checkpoints below; the
original archived measurements remain unchanged.
Nemotron is the in-distribution evaluation; the selected Red Hat benchmark
categories are the out-of-distribution evaluation.

Cells report **aggregate output tokens per measured wall second**, including
prefill and serving overhead. Higher is better. These are serving throughput
measurements, not offline agreement, acceptance rates, or isolated decode kernel
speeds. The generator checkpoints are described as trained on 24K examples;
checkpoint steps and data size follow the supplied labels/conversation and have
not been verified against archived checkpoint provenance for this suite.

The summary does not establish hardware, effective serving settings, prompt
hashes, sample counts, seeds, repeated-run uncertainty, or exact checkpoint
identities. Those remain unverified until the native suite artifacts are added.
Do not infer them from current configuration defaults or earlier benchmarks.

## In-distribution — Nemotron (output tokens/s)

| Model | Math | Code | STEM | Chat |
| --- | ---: | ---: | ---: | ---: |
| EAGLE3 base — historical | 64.34 | 60.93 | 59.28 | 54.97 |
| Trained V/O LoRA — historical | 72.50 | 64.79 | 64.23 | 56.35 |
| V/O generator · 24K · 2,000 steps — historical | 77.58 | 68.11 | 66.50 | 57.58 |
| V/O generator · 24K · 4,000 steps — historical | 79.11 | 69.83 | 67.92 | 56.49 |
| Attention generator · 24K · 2,000 steps | 82.80 | 73.14 | 70.71 | 59.77 |
| Attention generator · 24K · 4,000 steps | 81.17 | 71.64 | 69.40 | 59.44 |
| Attention generator · 24K · 6,000 steps | 81.89 | 71.81 | 69.41 | 59.16 |
| DeepSeek EAGLE3 ttt7 — paired-suite baseline | 86.58 | 75.40 | 67.20 | 56.68 |
| DeepSeek attention generator · 4K · 1,500 steps | **92.26** | **83.73** | **78.01** | **65.46** |

The new attention rows come from [GEN-011](GEN-011-attention-24k-6k.md),
three checkpoints × four Nemotron categories at C1. The
[combined all-seven-category table](benchmarks/2026-10-31-attention-24k-6k-c1/extended_historical_table.md)
retains the historical Red Hat results and marks new out-of-distribution cells
as unmeasured. No new HumanEval, math-reasoning or summarization results were
supplied. Historical timing/precision/runtime equivalence is unverified, so the
extended table is descriptive rather than a controlled speedup experiment.

The subsequent paired baseline/generator rows come from [GEN-012](GEN-012-deepseek-generator-4k.md).
They use a different frozen drafter from the earlier models; compare adaptation
gains against their same-suite baseline. The historical earlier DeepSeek math/chat
baseline is preserved separately in GEN-010. No new OOD cells have been supplied
for the DeepSeek pair.

Follow-up: [GEN-008](GEN-008-coefficient-sweep.md) archives the supplied fixed
coefficient 0.5/4/8 results for the step-4000 generator, shared/merged execution,
math/chat only, C1. Its extended table preserves coefficient 2 as a historical
reference rather than presenting it as a same-session measurement.

## Out-of-distribution — Red Hat (output tokens/s)

| Model | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: |
| EAGLE3 base | 68.32 | 66.55 | 58.29 |
| Trained LoRA | 67.24 | 67.19 | 58.30 |
| Generator · 24K · 2,000 steps | 69.24 | 67.99 | **58.81** |
| Generator · 24K · 4,000 steps | **69.44** | **68.32** | 58.69 |

## Interpretation and limits

Within the original four-model suite, both generator checkpoints have higher
reported throughput than EAGLE3 in all
seven measured categories. Gains are larger on Nemotron math/code/STEM and
smaller on the measured Red Hat categories. The 4,000-step checkpoint leads in
five categories; the 2,000-step checkpoint leads in chat and summarization.
The trained LoRA is slightly slower than EAGLE3 on HumanEval in this measurement.

These observations do not establish statistical significance, output quality,
exact-output correctness, speedup over target-only decoding, or performance at
multiple concurrent requests. Small differences need repeated matched runs.

## Decision and next evidence

Retain the original four-model archive and its in-/out-of-distribution grouping.
The extended Nemotron table now includes the three GEN-011 attention checkpoints;
the Red Hat table remains historical pending new out-of-distribution measurements.
Attach native results, workload identities, checkpoint hashes, server/evaluation
commands, and patches when available; their absence is explicit in the structured
record. Existing historical records are not merged into these tables.

## Updates

- 2026-10-22: Recorded the supplied four-model suite summary and category split.

- 2026-10-31: Extended Nemotron with attention-generator 24K checkpoints at
  2,000/4,000/6,000 updates. Added a combined seven-category table with missing
  new Red Hat cells explicit; source archives and historical limits preserved.
