# Extended historical comparison: draft acceptance rate

Accepted proposed draft tokens divided by all proposed draft tokens, shown as
percentages. Rows with detailed summaries use concurrency 1 and three draft
tokens. Correctness remains unchecked; cross-run runtime/repeatability limits
from the [throughput table](extended_historical_table.md) still apply.

| Model | Nemo math | Nemo code | Nemo STEM | Nemo chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 base — historical | — | — | — | — | — | — | — |
| Trained V/O LoRA — historical | — | — | — | — | — | — | — |
| V/O generator, 24K / 2,000 — historical | — | — | — | — | — | — | — |
| V/O generator, 24K / 4,000 — historical | — | — | — | — | — | — | — |
| Attention generator, 24K / 2,000 | 66.85% | 55.26% | 51.97% | 39.61% | 55.00% | 52.77% | 41.43% |
| Attention generator, 24K / 4,000 | 67.93% | 55.75% | 52.49% | 40.45% | 55.21% | 53.07% | 41.22% |
| Attention generator, 24K / 6,000 | 68.84% | 55.87% | 52.44% | 40.89% | 54.86% | 52.99% | 41.06% |
| DeepSeek EAGLE3 ttt7 — paired baseline | 74.04% | 60.70% | 49.93% | 37.13% | 41.36% | 56.76% | 39.98% |
| DeepSeek attention generator, 4K / 1,500 | 81.74% | 70.65% | 62.85% | 47.80% | 44.96% | 63.92% | 46.41% |
| Attention generator, 100K / 1,000 | 65.15% | 54.08% | 51.68% | 39.73% | 54.47% | 52.69% | 41.52% |
| Attention generator, 100K / 2,000 | 67.43% | 55.10% | 51.99% | 40.26% | 55.09% | 52.87% | 41.75% |

**— means unavailable.** The four matching historical rows were supplied as
throughput-only summaries. Acceptance is not inferred from tokens/s, and values
from separate concurrency/runtime runs are not substituted into those rows.

Sources: [historical throughput-only archive](../2026-10-22-comparison-v2/metrics.json),
[24K attention checkpoints](summary_with_ood.json),
[paired DeepSeek baseline/generator](../2026-10-31-deepseek-generator-4k-c1/summary_with_ood.json),
and [100K attention checkpoints](../2026-11-01-attention-100k-2k-c1/summary.json).
[Machine-readable rates and sources](../2026-11-01-attention-100k-2k-c1/extended_acceptance_table.json)
retain unrounded ratios and explicit nulls for missing measurements.
