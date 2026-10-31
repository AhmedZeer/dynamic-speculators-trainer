# Extended historical comparison: all categories

Aggregate output tokens/s, including prefill and serving overhead; correctness
not checked. The first four rows are historical, followed by the supplied
24K attention checkpoints and the later paired DeepSeek baseline/generator suite.
New measurements use concurrency 1. Separate runs retain
hardware/runtime/precision and repeatability limitations.

| Model | Nemo math | Nemo code | Nemo STEM | Nemo chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 base — historical | 64.34 | 60.93 | 59.28 | 54.97 | 68.32 | 66.55 | 58.29 |
| Trained V/O LoRA — historical | 72.50 | 64.79 | 64.23 | 56.35 | 67.24 | 67.19 | 58.30 |
| V/O generator, 24K / 2,000 — historical | 77.58 | 68.11 | 66.50 | 57.58 | 69.24 | 67.99 | 58.81 |
| V/O generator, 24K / 4,000 — historical | 79.11 | 69.83 | 67.92 | 56.49 | 69.44 | 68.32 | 58.69 |
| Attention generator, 24K / 2,000 | 82.80 | 73.14 | 70.71 | 59.77 | — | — | — |
| Attention generator, 24K / 4,000 | 81.17 | 71.64 | 69.40 | 59.44 | — | — | — |
| Attention generator, 24K / 6,000 | 81.89 | 71.81 | 69.41 | 59.16 | — | — | — |
| DeepSeek EAGLE3 ttt7 — paired-suite baseline | 86.58 | 75.40 | 67.20 | 56.68 | — | — | — |
| DeepSeek attention generator, 4K / 1,500 | 92.26 | 83.73 | 78.01 | 65.46 | — | — | — |

— means no result was supplied for that new checkpoint/category. HumanEval,
math reasoning and summarization are the Red Hat out-of-distribution workloads;
new Nemotron code/STEM values must not be substituted into those columns.

Sources: [historical four-model metrics](../2026-10-22-comparison-v2/metrics.json),
[new twelve-cell summary](summary.json), and
[paired DeepSeek summary](../2026-10-31-deepseek-generator-4k-c1/summary.json). See
[GEN-003](../../GEN-003-generator-suite-throughput.md) and
[GEN-011](../../GEN-011-attention-24k-6k.md), and
[GEN-012](../../GEN-012-deepseek-generator-4k.md) for their evidence limits.

DeepSeek adaptation gains must use the paired baseline row. Its earlier separate
math/chat run reported 82.50/54.10 tok/s and is retained in GEN-010; substituting
that earlier denominator would inflate the training gain.
