# GEN-016 — Attention generator (100K) scaling and dynamism, serving TTT3

## Record

- Execution date: 2026-11-10, on a Colab A100 (40 GB), via
  `examples/eagle3-lora/attention_100k_2k_v1/benchmarks/run_scaling_dynamism.sh`
  (runbook: [`SCALING_DYNAMISM.md`](../../examples/eagle3-lora/attention_100k_2k_v1/benchmarks/SCALING_DYNAMISM.md)).
- Status: all 69 planned measurements completed; two cells needed a second
  attempt after an incomplete GuideLLM workload / an infrastructure stop (see
  Evidence quality below). Correctness not checked. Single run per cell — no
  repeated-trial uncertainty, only prompt-resampling confidence intervals.
- Checkpoints: generator `azeeer/eagle3-generator-attention-100k-2k-v1`
  (commit `3edcce5e4d2fbfc56dbfa71ae0cd96fb8dcc1cba`, steps 1K–6K), static
  attention LoRA `azeeer/eagle3-static-lora-attention-100k-10k-v1` (commit
  `8b5ef5792fcd5e3ca7e49a4959edc0b347b496e6`, steps 1K–5K). See the runbook
  for the frozen-workload/donor-bank protocol and checkpoint coverage.
- Raw outputs, per-cell provenance (`vllm_command.txt`, `vllm.patch`,
  checkpoint hashes, `server.log`, native GuideLLM JSON) and the suite's own
  `summary.json`/`.md` remain on Google Drive under
  `dynspec-bench-2/attention-scaling-dynamism-v1/suites/ttt3/`; they are not
  copied into the repository (too large / not yet pulled). This record and
  [`benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/)
  hold the derived tables, figures and analysis scripts produced from them.

## Question and hypothesis

Does the attention generator's draft acceptance scale with training steps
(1K–6K) and does acceptance stay elevated when conditioned on a prompt from
the correct category versus a fixed or cross-category donor, for the 100K
training pool on serving length (TTT) 3? Separately: how do the generator,
the static attention LoRA, and the unmodified Eagle3 drafter compare to a
no-drafter baseline on the same frozen prompts?

## Setup

All suites: Qwen3-8B target, concurrency 1, 50 measured prompts/category
(math/STEM/chat; a new shared frozen test split, not the historical
validation shards used elsewhere in this archive — see the runbook), 256
max output tokens, 3 draft tokens, 3 warmups, seed 42, scale 2 (rank
32/alpha 64), singleton generator conditioning.

| Arm | Conditioning | Steps | Measurements |
| --- | --- | --- | --- |
| Generator | correct-category prompt | 1K–6K | 18 |
| Generator | fixed donor (a math training prompt) | 1K–6K | 18 |
| Generator | cross-category donor (deterministic by prompt hash) | 1K–6K | 18 |
| Static attention LoRA | prompt-independent | 1K–5K | 15 |
| No-drafter baseline (target only) | — | — | 3 |
| Original Eagle3 drafter (no adapter) | — | — | 3 |

The no-drafter baseline and original-drafter rows ran on a separate A100
session the same day (13:50–15:03 UTC) from the 69-cell suite (04:57–11:55
UTC), using the suite's own vLLM launch flags and the same frozen 50
prompts/category.

## Evidence quality

- One static-LoRA cell (`static_lora_3k_static`/math) recorded 49/50
  successful requests on its first attempt (0 errored, 1 still processing
  when GuideLLM closed the benchmark); the suite refuses partial results and
  reran it to completion on resume.
- One generator cell (`generator_6k_fixed`/math) was cut short mid-run by an
  infrastructure fault unrelated to the benchmark (the auto-stop watcher
  wrongly treated a transient Colab connection loss as a dead VM); rerun to
  completion on resume.
- Both reruns kept the earlier attempt's partial data alongside the accepted
  one; neither is evidence of a benchmark or model defect.

## Evidence and measurements

- [`scaling_ttt3.csv`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/scaling_ttt3.csv) —
  30 rows: 5 steps × 2 methods (generator/correct, static LoRA) × 3
  categories; `tokens_per_second`, `acceptance_percent`.
- [`dynamism_ttt3.csv`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/dynamism_ttt3.csv) /
  [`dynamism_ttt3_with_correct_minus_cross.csv`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/dynamism_ttt3_with_correct_minus_cross.csv) —
  18 rows: 6 steps × 3 categories; acceptance under correct/fixed/cross-category
  conditioning and the pairwise percentage-point differences
  (`correct_minus_fixed_pp`, `cross_category_minus_fixed_pp`,
  `correct_minus_cross_pp`).
- [`ci_ttt3.md`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/ci_ttt3.md) —
  95% bootstrap confidence intervals (5,000 resamples over the 50 prompts/cell)
  for tok/s and acceptance per cell, plus paired bootstrap differences
  (generator-vs-static, correct-vs-fixed, correct-vs-cross,
  cross-vs-fixed) on matched prompts. **CIs cover prompt-sampling
  uncertainty only — each cell is still a single run, so run-to-run timing
  variation is not included**, which matters most for the tok/s intervals.
- [`latency_ttt3.md`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/latency_ttt3.md) /
  `.csv` — per-request median/p90 time-to-first-token, inter-token latency,
  time-per-output-token, and request latency, per cell.
- [`speedup_ttt3.md`](benchmarks/2026-11-10-attention-scaling-dynamism-v1-ttt3/speedup_ttt3.md) /
  `.csv` — every arm's tok/s expressed as a paired-bootstrap speedup over the
  no-drafter baseline on the same prompts, alongside the original Eagle3
  drafter's speedup for reference.
- `figures/`: `scaling_acceptance_ttt3` (generator vs static LoRA acceptance
  by step, one panel/category), `dynamism_policies_acceptance_ttt3`
  (correct/fixed/cross-category acceptance by step), and
  `dynamism_correct_vs_controls_ttt3` (correct−fixed and correct−cross
  percentage-point differences by step). PNG + PDF.

### Headline numbers

At serving TTT3, generator vs static LoRA (correct conditioning), acceptance
at step 5K: math 68.08% vs 64.45%, STEM 52.04% vs 50.16%, chat 42.46% vs
40.65% (full curve in `scaling_ttt3.csv`). Generator-vs-static paired
differences are positive with lower CI bound above 0 in every one of the 15
matched cells (steps 1K–5K × 3 categories) in `paired_differences_ttt3.csv`.

Dynamism (generator, step 6K): correct-vs-fixed is +8.28 pp (STEM) and +6.35
pp (chat), but only −0.10 pp on math, because the fixed donor is itself a
math-category prompt (not a neutral control there). Correct-vs-cross-category
shows the same pattern on STEM/chat and a positive +13.4 pp on math, where
cross-category is a non-math donor. Both gaps widen with more training steps
on STEM and chat; math's correct-vs-fixed gap stays near zero across all
checkpoints.

Speedup over no-drafter (29.16–29.31 tok/s depending on category): original
Eagle3 drafter 1.91–2.20×; static LoRA 1.94–2.56×; generator/correct
1.98–2.79× (peaks near step 3K on math/STEM in this run); generator/fixed and
generator/cross-category are sometimes close to or above generator/correct in
raw tok/s despite lower acceptance (e.g. fixed donor reaches 2.78× at math
6K) — flagged in `speedup_ttt3.md` as likely cross-run timing drift rather
than a real throughput advantage, since the no-drafter baseline/original
drafter ran on a different VM session than the 69 suite cells.

## Interpretation and limits

- Single run per cell: CIs reflect only which 50 prompts were sampled, not
  repeated-trial or cross-session timing variance. Treat small (≤0.05×)
  tok/s differences between arms with caution, especially across the
  baseline/suite VM-session boundary.
- The fixed donor is a math-category training prompt, so correct-vs-fixed is
  not a neutral comparison on math; use correct-vs-cross-category for math
  instead (see runbook).
- This workload is a new frozen 50-prompt/category split, distinct from the
  historical validation shards used in GEN-003/010/011/013/014; do not splice
  these acceptance/throughput numbers into the same curve as those without
  rerunning the matched controls.
- No per-prompt correctness check; acceptance proxy in `speedup_ttt3.md`
  (output tokens ÷ streamed decode iterations) tracks the suite's native
  acceptance within about 0.8 pp across all 69 cells but is not identical to
  it.

## Decision and next check

Use `scaling_ttt3`/`dynamism_ttt3` (native values) for the acceptance claims
in the preprint; use `speedup_ttt3` for throughput claims, flagging the
cross-session caveat above. The serving-length-7 repeat
(`--ttt 3 7` in the runbook) and a code/extra-category pass for the
no-drafter baseline (currently math/STEM/chat only) remain open follow-ups,
as does pulling the full per-cell provenance from Drive into the repository
if the preprint needs it archived here.

## Updates

- 2026-11-10: Initial record.
