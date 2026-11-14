# GEN-015 — Matched comparison-v2 suite re-run with draft acceptance

## Record

- Received: 2026-11-10; execution date not supplied.
- Status: supplied suite summary (28 cells transcribed: 4 models × 7
  categories at C1); C2/C4/C8 present in the raw file but not transcribed;
  correctness not checked.
- Evidence: user-supplied `summary.json`.
- Location: [`benchmarks/2026-11-10-matched-comparison-v2-acceptance/summary.json`](benchmarks/2026-11-10-matched-comparison-v2-acceptance/summary.json),
  [derived table](benchmarks/2026-11-10-matched-comparison-v2-acceptance/summary.md).
- Suite identity: `99467dbf4363b3dd31e0083f4b170d85085e0ce23834ce957f96f5fe23c9f1e2`.
- Related: [GEN-003](GEN-003-generator-suite-throughput.md) (original
  throughput-only `comparison-v2` suite), its
  [`extended_acceptance_table.md`](benchmarks/2026-10-31-attention-24k-6k-c1/extended_acceptance_table.md)
  (the gap this fills).

## Question and hypothesis

GEN-003's four-model `comparison-v2` suite (EAGLE3 base, trained V/O LoRA, V/O
generator at 24K/2,000 and 24K/4,000 steps) reported throughput only; its
draft acceptance was recorded as unavailable in the extended acceptance table.
This re-run asks whether the same suite, executed again, reproduces the
historical throughput and additionally exposes acceptance.

## Setup

| Arm | Changed variable | Training data | Update budget | Categories | Concurrency |
| --- | --- | --- | --- | --- | --- |
| EAGLE3 base | none (unmodified drafter) | — | — | 7 (4 Nemotron + 3 Red Hat) | 1, 2, 4, 8 |
| Trained V/O LoRA | static V/O LoRA adapter | historical, unrecorded here | — | 7 | 1, 2, 4, 8 |
| V/O generator | generated V/O adapter | 24K | 2,000 | 7 | 1, 2, 4, 8 |
| V/O generator | generated V/O adapter | 24K | 4,000 | 7 | 1, 2, 4, 8 |

Three draft tokens/cycle, 50 prompts/category, 50 successful
requests/cell. Hardware, seeds, and checkpoint identities are not supplied.

## Evidence and measurements

See [`summary.md`](benchmarks/2026-11-10-matched-comparison-v2-acceptance/summary.md)
for the full C1 throughput and acceptance tables. Throughput matches the
historical GEN-003/extended-table values to two decimals in all 28 cells,
e.g. EAGLE3 base math 64.34 tok/s and V/O generator 24K/4,000 math 79.11
tok/s in both the original and this re-run. This supports that the same
suite was re-executed rather than a different one, without establishing
hardware or runtime equivalence.

Draft acceptance, newly available: EAGLE3 base 36.07–52.16% across
categories; trained V/O LoRA 38.77–59.67%; V/O generator 24K/4,000
39.65–65.34%. Acceptance rises from EAGLE3 base → LoRA → generator/2,000 →
generator/4,000 in every one of the 7 categories, tracking the existing
throughput ordering.

## Interpretation and limits

This closes the specific "throughput supplied, acceptance unavailable" gap
for these four historical rows, using a later re-run rather than the
original invocation. It does not establish that the re-run shares hardware,
runtime, or exact checkpoint identity with the original; it does not add a
repeated-trial uncertainty estimate (each cell is one run); and C2/C4/C8
acceptance is present in the raw file but not yet transcribed into a table.

## Decision and next check

Use this table for acceptance wherever the historical `comparison-v2`
throughput rows are cited. Transcribing C2/C4/C8 acceptance alongside the
existing concurrency throughput work (BATCH-001) would extend this at no
new GPU cost, since the data is already supplied.

## Updates

- 2026-11-10: Initial record.
