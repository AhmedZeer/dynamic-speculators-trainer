# Extended historical comparison: draft acceptance rate

Accepted proposed draft tokens divided by all proposed draft tokens, shown as
percentages. Rows with detailed summaries use concurrency 1 and three draft
tokens. Correctness remains unchecked; cross-run runtime/repeatability limits
from the [throughput table](extended_historical_table.md) still apply.

| Model | Nemo math | Nemo code | Nemo STEM | Nemo chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 base — historical¹ | 47.42% | 43.28% | 40.85% | 36.07% | 52.16% | 50.93% | 40.08% |
| Trained V/O LoRA — historical¹ | 59.67% | 50.60% | 48.54% | 38.77% | 52.30% | 51.72% | 41.33% |
| V/O generator, 24K / 2,000 — historical¹ | 64.86% | 52.88% | 50.85% | 39.64% | 53.38% | 52.50% | 41.06% |
| V/O generator, 24K / 4,000 — historical¹ | 65.34% | 54.16% | 51.03% | 39.65% | 53.32% | 52.19% | 40.60% |
| Attention generator, 24K / 2,000 | 66.85% | 55.26% | 51.97% | 39.61% | 55.00% | 52.77% | 41.43% |
| Attention generator, 24K / 4,000 | 67.93% | 55.75% | 52.49% | 40.45% | 55.21% | 53.07% | 41.22% |
| Attention generator, 24K / 6,000 | 68.84% | 55.87% | 52.44% | 40.89% | 54.86% | 52.99% | 41.06% |
| DeepSeek EAGLE3 ttt7 — paired baseline | 74.04% | 60.70% | 49.93% | 37.13% | 41.36% | 56.76% | 39.98% |
| DeepSeek attention generator, 4K / 1,500 | 81.74% | 70.65% | 62.85% | 47.80% | 44.96% | 63.92% | 46.41% |
| Attention generator, 100K / 1,000 | 65.15% | 54.08% | 51.68% | 39.73% | 54.47% | 52.69% | 41.52% |
| Attention generator, 100K / 2,000 | 67.43% | 55.10% | 51.99% | 40.26% | 55.09% | 52.87% | 41.75% |

**— means unavailable.**

¹ These four rows were originally supplied as throughput-only summaries
(`—` for acceptance, through 2026-10-31). On 2026-11-10 a separate suite
re-run supplied acceptance alongside throughput for the same four models,
all seven categories, concurrencies 1/2/4/8; see
[`../2026-11-10-matched-comparison-v2-acceptance/summary.md`](../2026-11-10-matched-comparison-v2-acceptance/summary.md).
The re-run's throughput matches these historical values to two decimals in
every cell, supporting that it is the same suite rather than a different one,
but it is still a separate invocation: hardware/runtime equivalence and
repeatability are not established, and acceptance values are not retroactively
assigned to the original historical measurement.

Sources: [historical throughput-only archive](../2026-10-22-comparison-v2/metrics.json),
[24K attention checkpoints](summary_with_ood.json),
[paired DeepSeek baseline/generator](../2026-10-31-deepseek-generator-4k-c1/summary_with_ood.json),
and [100K attention checkpoints](../2026-11-01-attention-100k-2k-c1/summary.json).
[Machine-readable rates and sources](../2026-11-01-attention-100k-2k-c1/extended_acceptance_table.json)
retain unrounded ratios and explicit nulls for missing measurements.
