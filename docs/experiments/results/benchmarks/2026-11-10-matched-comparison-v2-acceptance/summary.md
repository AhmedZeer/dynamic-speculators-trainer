# Matched comparison-v2 suite with draft acceptance — C1

Derived from `summary.json` in this directory (user-supplied, received
2026-11-10; `correctness_status: "not_checked"`; `suite_identity:
99467dbf4363b3dd31e0083f4b170d85085e0ce23834ce957f96f5fe23c9f1e2`). The suite
re-runs the four `comparison-v2` models from
[GEN-003](../../GEN-003-generator-suite-throughput.md) (EAGLE3 base, trained
V/O LoRA, and the V/O generator at 24K/2,000 and 24K/4,000 steps) across all
seven categories at concurrencies 1/2/4/8 and this time records draft
acceptance alongside throughput. Only C1 is tabulated below; C2/C4/C8 are in
`summary.json` (`-c2`/`-c4`/`-c8` suffixes on each model key) but not yet
transcribed.

Throughput below matches the historical `comparison-v2` values in GEN-003 and
the extended historical table to two decimals, confirming this is a re-run of
the same suite rather than a different one. **This fills the "unavailable"
acceptance cells for these four models in
[`extended_acceptance_table.md`](../2026-10-31-attention-24k-6k-c1/extended_acceptance_table.md).**

All cells: 50 successful requests, 3 draft tokens/cycle, 50 prompts/category.
Acceptance = accepted ÷ proposed draft tokens, from the suite's own
`acceptance` block per category (not re-derived from throughput).

## Draft acceptance rate (%)

| Model | Math | Code | STEM | Chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 base | 47.42% | 43.28% | 40.85% | 36.07% | 52.16% | 50.93% | 40.08% |
| Trained V/O LoRA | 59.67% | 50.60% | 48.54% | 38.77% | 52.30% | 51.72% | 41.33% |
| V/O generator, 24K / 2,000 | 64.86% | 52.88% | 50.85% | 39.64% | 53.38% | 52.50% | 41.06% |
| V/O generator, 24K / 4,000 | 65.34% | 54.16% | 51.03% | 39.65% | 53.32% | 52.19% | 40.60% |

## Output tokens/s

| Model | Math | Code | STEM | Chat | HumanEval | Math reasoning | Summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EAGLE3 base | 64.34 | 60.93 | 59.28 | 54.97 | 68.32 | 66.55 | 58.29 |
| Trained V/O LoRA | 72.50 | 64.79 | 64.23 | 56.35 | 67.24 | 67.19 | 58.30 |
| V/O generator, 24K / 2,000 | 77.58 | 68.11 | 66.50 | 57.58 | 69.24 | 67.99 | 58.81 |
| V/O generator, 24K / 4,000 | 79.11 | 69.83 | 67.92 | 56.49 | 69.44 | 68.32 | 58.69 |

## Limits

As with the original suite: hardware, effective serving settings, prompt
hashes, seeds, checkpoint identities, and repeated-run uncertainty are not
supplied with this summary. Acceptance length and per-position acceptance are
present in `summary.json` but not transcribed here. This is a single run per
cell with no confidence interval.
