# Extended historical comparison: all categories

Aggregate output tokens/s, including prefill and serving overhead; correctness
not checked. The first four rows are historical, followed by the supplied
24K attention checkpoints, the paired DeepSeek baseline/generator suite, and
the supplied 100K attention checkpoints.
New measurements use concurrency 1. Separate runs retain
hardware/runtime/precision and repeatability limitations.

| Model | Nemo math | Nemo code | Nemo STEM | Nemo chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 base — historical | 64.34 | 60.93 | 59.28 | 54.97 | 68.32 | 66.55 | 58.29 |
| Trained V/O LoRA — historical | 72.50 | 64.79 | 64.23 | 56.35 | 67.24 | 67.19 | 58.30 |
| V/O generator, 24K / 2,000 — historical | 77.58 | 68.11 | 66.50 | 57.58 | 69.24 | 67.99 | 58.81 |
| V/O generator, 24K / 4,000 — historical | 79.11 | 69.83 | 67.92 | 56.49 | 69.44 | 68.32 | 58.69 |
| Attention generator, 24K / 2,000 | 82.80 | 73.14 | 70.71 | 59.77 | 71.26 | 69.62 | 59.94 |
| Attention generator, 24K / 4,000 | 81.17 | 71.64 | 69.40 | 59.44 | 72.32 | 70.69 | 60.65 |
| Attention generator, 24K / 6,000 | 81.89 | 71.81 | 69.41 | 59.16 | 71.96 | 70.11 | 60.04 |
| DeepSeek EAGLE3 ttt7 — paired-suite baseline | 86.58 | 75.40 | 67.20 | 56.68 | 61.23 | 73.54 | 59.62 |
| DeepSeek attention generator, 4K / 1,500 | 92.26 | 83.73 | 78.01 | 65.46 | 62.03 | 76.74 | 62.80 |
| Attention generator, 100K / 1,000 | 76.49 | 67.85 | 66.14 | 56.64 | 68.17 | 66.56 | 58.11 |
| Attention generator, 100K / 2,000 | 79.53 | 69.79 | 67.75 | 58.17 | 70.11 | 68.68 | 59.24 |

All displayed rows now have measurements for all seven categories. HumanEval,
math reasoning and summarization are the Red Hat out-of-distribution workloads;
Nemotron math/code/STEM/chat are the in-distribution workloads.

Sources: [historical four-model metrics](../2026-10-22-comparison-v2/metrics.json),
[complete resumed 21-cell summary](summary_with_ood.json),
[paired DeepSeek summary](../2026-10-31-deepseek-generator-4k-c1/summary_with_ood.json), and
[100K step-1000/2000 summary](../2026-11-01-attention-100k-2k-c1/summary.json). See
[GEN-003](../../GEN-003-generator-suite-throughput.md),
[GEN-011](../../GEN-011-attention-24k-6k.md),
[GEN-012](../../GEN-012-deepseek-generator-4k.md), and
[GEN-013](../../GEN-013-attention-100k-2k.md) for their evidence limits.

DeepSeek adaptation gains must use the paired baseline row. Its earlier separate
math/chat run reported 82.50/54.10 tok/s and is retained in GEN-010; substituting
that earlier denominator would inflate the training gain.
