# GEN-017 — Grouped speedup/acceptance figures vs vanilla Eagle3

## Record

- Built: 2026-11-10. Status: derived figures combining measurements from
  four separate records; not a new benchmark run. Correctness/repeatability
  limits of each source record apply.
- Sources: [GEN-003](GEN-003-generator-suite-throughput.md) (vanilla Eagle3,
  static LoRA V/O, V/O generator 24K/4K — throughput),
  [GEN-015](GEN-015-matched-comparison-acceptance.md) (same four models'
  acceptance), [GEN-011](GEN-011-attention-24k-6k.md) (attention generator
  24K/4K), [GEN-016](GEN-016-attention-100k-ttt3-scaling-dynamism.md)
  (attention generator 100K/6K, no-drafter baseline, today's vanilla Eagle3
  on the frozen TTT3 workload).
- Files: [`benchmarks/2026-11-10-grouped-comparison-eagle3/`](benchmarks/2026-11-10-grouped-comparison-eagle3/)
  — `grouped_speedup_vs_eagle3.{png,pdf,csv}`,
  `grouped_acceptance_vs_eagle3.{png,pdf,csv}`, and the two generating
  scripts (`make_speedup_figure.py`, `make_acceptance_figure.py`; reference
  copies — see each script's header for how to point them at the sibling
  benchmark directories to rerun).

## Purpose

Two bar charts put five drafting arms side by side on the same axis: vanilla
Eagle3, static LoRA (V/O), V/O generator (24K/4,000 steps), attention
generator (24K/4,000 steps), and attention generator (100K/6,000 steps).
**Each arm's bar in a given chart is drawn from a different source record**
(listed above); this is a cross-run comparison, not a matched suite.

## Figure 1 — `grouped_speedup_vs_eagle3`

Bars are each arm's output tok/s divided by **vanilla Eagle3 from the same
source run**, per category (math/code/STEM/chat), y-axis from 0.9. The
100K/6K bars divide by today's own vanilla-Eagle3 measurement (not GEN-003's),
since it ran on the frozen TTT3 workload rather than the historical
`comparison-v2` prompts; the figure's caption states this explicitly. Code is
unmeasured for the 100K/6K arm (today's run covered math/STEM/chat only).

| | Math | Code | STEM | Chat |
|---|---|---|---|---|
| Vanilla Eagle3 | 1.00× | 1.00× | 1.00× | 1.00× |
| Static LoRA (V/O) | 1.13× | 1.06× | 1.08× | 1.03× |
| V/O generator, 24K/4K | 1.23× | 1.15× | 1.15× | 1.03× |
| Attention generator, 24K/4K | 1.26× | 1.18× | 1.17× | 1.08× |
| Attention generator, 100K/6K | 1.22× | not measured | 1.13× | 1.03× |

## Figure 2 — `grouped_acceptance_vs_eagle3`

Bars are draft acceptance rate (%) per category (math/STEM/chat only — Code
is not available for the two attention-generator arms in their source
records). Values:

| | Math | STEM | Chat |
|---|---|---|---|
| Vanilla Eagle3 (today, original drafter) | 47.02% | 40.09% | 36.05% |
| Static LoRA (V/O) | 59.67% | 48.54% | 38.77% |
| V/O generator, 24K/4K | 65.34% | 51.03% | 39.65% |
| Attention generator, 24K/4K | 67.93% | 52.49% | 40.45% |
| Attention generator, 100K/6K | 68.65% | 52.15% | 42.20% |

Note the vanilla-Eagle3 bar here is **today's** original-drafter measurement
(from GEN-016, on the frozen TTT3 workload), not GEN-003's/GEN-015's
`eagle3-base` (47.42%/40.85%/36.07% on their own, different, historical
workload) — close but not identical, consistent with separate-run variance.

## Interpretation and limits

- **Not a controlled comparison.** Four different runs, at least two
  different prompt workloads (GEN-003/015/011's historical `comparison-v2`
  shards vs GEN-016's frozen TTT3 split), and unverified hardware/runtime
  equivalence across all of them. Treat these as a descriptive cross-run
  snapshot, not a matched ablation.
- **Training-data size and steps both change** between the three generator
  bars (24K/4K vs 24K/4K-attention vs 100K/6K), so a gap cannot be attributed
  to either factor alone.
- Static LoRA's training-set size is not recorded in its source suite.
- Each bar is a single run; no confidence intervals are shown on these two
  figures (per-cell CIs exist separately for the 100K/6K source in GEN-016's
  `ci_ttt3.md`).
- On the acceptance figure, 24K/4K and 100K/6K are close on math/STEM/chat
  despite the different data pool and step count; this does not establish
  that data-pool size has no effect, only that this particular pair of
  checkpoints lands close together on these three categories.

## Decision and next check

These figures are for the preprint's qualitative "where do we stand" view,
not for a quantitative data-size or step-count claim. A same-workload,
same-day rerun of all five arms (vanilla Eagle3, static LoRA, V/O generator,
and both attention generators, on one frozen prompt set, one VM session)
would let this comparison carry confidence intervals and a controlled
reading; none of the current source records are matched closely enough for
that.

## Updates

- 2026-11-10: Initial record.
