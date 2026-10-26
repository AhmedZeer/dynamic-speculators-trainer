# GEN-005 — Generator rank, adapter dynamism and serving performance

## Record

- Receipt date: 2026-10-25; execution date unavailable.
- Status: preliminary; three supplied screenshots, transcribed at displayed precision.
- Evidence: [adapter geometry](assets/GEN-005-rank-dynamism.png),
  [acceptance](assets/GEN-005-rank-acceptance.png),
  [tokens/s](assets/GEN-005-rank-throughput.png).
- Structured values, derived comparisons and image hashes:
  [metrics.json](benchmarks/2026-10-25-rank-sweep/metrics.json).

## Question and setup

Does increasing generator LoRA rank increase input-dependent adapter variation
and improve acceptance and serving token rate? The screenshots compare ranks
8, 16, 32 and 64. The geometry summary is labelled “modules averaged.” The serving
plots label a “base model,” whose exact checkpoint and serving mode are unavailable.
Its displayed acceptance is 53.53% and token rate is 36.1 tokens/s.

Training data, update budget, optimizer, seeds, checkpoint revisions, evaluation
membership, hardware, concurrency and serving configuration were not supplied.
The three screenshots are recorded together as the supplied rank study; identical
checkpoint identities across geometry and serving measurements are unverified.

## Measurements

### Adapter geometry

Column names preserve the supplied analysis labels. Their formulas, normalization,
category classifier protocol and module aggregation weights are unavailable.
In particular, `eff_dim` and `erank` are separate reported metrics.

| Rank | dyn_ratio | cos_within | cos_between | Within − between | eff_dim | erank | cat_acc | max_cdist |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.3424 | 0.9961 | 0.9586 | 0.0375 | 1.01 | 7.11 | 0.735 | 0.8866 |
| 16 | 0.3921 | 0.9941 | 0.9102 | 0.0839 | 1.02 | 12.04 | 0.805 | 1.0687 |
| 32 | 0.4639 | 0.9850 | 0.8521 | 0.1329 | 1.13 | 18.02 | 0.900 | 1.1781 |
| 64 | 0.5158 | 0.9717 | 0.8145 | 0.1572 | 1.30 | 24.26 | 0.905 | 1.2255 |

| Rank | dyn[chat] | dyn[code] | dyn[math] | dyn[reading] |
| --- | ---: | ---: | ---: | ---: |
| 8 | 0.3688 | 0.2932 | 0.5191 | 0.1884 |
| 16 | 0.4398 | 0.3350 | 0.6110 | 0.1827 |
| 32 | 0.4986 | 0.4208 | 0.6879 | 0.2485 |
| 64 | 0.5191 | 0.4652 | 0.7387 | 0.3402 |

### Acceptance and token rate

Acceptance gains are percentage-point differences; token-rate gains are
relative to 36.1 tokens/s. The acceptance denominator and token-rate aggregation
are not specified, so these are retained as plotted metrics rather than assumed
to match the exact metric definitions in earlier serving studies.

| Model / rank | Acceptance | Gain vs. base | Tokens/s | Token-rate gain vs. base |
| --- | ---: | ---: | ---: | ---: |
| Base model | 53.53% | — | 36.1 | — |
| Generator rank 8 | 59.63% | +6.10 pp | 39.8 | +10.25% |
| Generator rank 16 | 60.33% | +6.80 pp | 40.3 | +11.63% |
| Generator rank 32 | 60.92% | +7.39 pp | 40.6 | +12.47% |
| Generator rank 64 | 61.40% | +7.87 pp | 41.0 | +13.57% |

## Interpretation and limits

The reported overall dynamism increases monotonically from 0.3424 to 0.5158.
Chat, code and math dynamism also increase at every rank; reading dips at rank 16
before increasing. Math has the largest reported dynamism at each rank.

Cosine similarity **decreases**, rather than increases, with rank: both within
categories and between categories. Between-category similarity falls more,
increasing the within-minus-between gap from 0.0375 to 0.1572. This is consistent
with stronger category separation while within-category similarity stays high.
The increasing `cat_acc` and `max_cdist` are consistent with that interpretation,
subject to their missing metric definitions. These observations do not by
themselves establish that useful adapter variation causes improved acceptance.

All four ranks exceed the displayed serving baseline, and acceptance and token
rate increase monotonically. Rank 8 to 64 adds 1.77 acceptance percentage points
and 1.2 tokens/s (3.02% relative to rank 8). The successive acceptance increments
are 0.70, 0.59 and 0.48 pp, suggesting diminishing increments in this supplied
series. Without repeated runs, the small adjacent differences are not established
as reliable. There is no matched static-LoRA rank sweep or rank-dependent resource
measurement here, so an optimal rank or benefit specific to dynamic conditioning
cannot be determined.

Both plots use interrupted vertical axes. The numeric tables preserve the absolute
values and baseline differences without relying on visual line separation.

## Decision and next check

Retain this as preliminary evidence that higher rank accompanies more varied,
more category-separated adapters and higher acceptance/token rate in the supplied
study. Keep it separate from earlier benchmark suites until workload and runtime
identities are available. The next useful evidence is the original analysis script,
raw metrics and configurations linking each rank to its checkpoint, followed by
matched repeated serving measurements and resource costs.

## Updates

- 2026-10-25: Archived the supplied geometry and serving figures and their values.
