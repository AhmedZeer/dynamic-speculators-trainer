# GEN-007 — Feedback added to original-pipeline continuation

## Record

- Observation date: 2026-10-28; execution date unavailable.
- Status: preliminary, one displayed seed (42), screenshot evidence through
  additional optimizer step 500.
- Evidence: [training and validation losses](assets/GEN-007-original-feedback-v3-losses.png).
- [Structured observations and image hash](benchmarks/2026-10-28-original-feedback-v3/metrics.json).
- [Runbook](../eagle3_original_pipeline_continuation_v3.md) and
  [architecture and tensor axes](../eagle3_generator_architecture.md).
- Raw metric exports, W&B run IDs, executed config/code snapshots and resulting
  checkpoint hashes were not supplied. Configured setup below is taken from the
  repository; its exact correspondence with the executed jobs is unverified.

## Question and setup

Does adding current-model saved-prefix agreement feedback improve the continuation
of the existing generator while retaining its original saved-response objective?

| Arm shown | Trainable parameters | Feedback |
| --- | --- | --- |
| `original-regular-from4000-v3-gseed42` | Whole original generator | None |
| `original-feedback-from4000-v3-gseed42` | Whole generator plus 192-parameter projector | Preceding teacher-forced prefix agreement, EMA, observed flag |

Configured common source: generator step 4000 from
`azeeer/eagle3-generator-all24k-10ep-v1`, revision
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`. Both arms restore parent Adam/RNG state
and use the original saved-response EAGLE3 forward and LK loss. The target and
drafter stay frozen. There is no new continuation generation or replay trainer.

The configured pool is 4K training responses: 1K/category for math, code, STEM and
chat from shard 00001. Four episodes/update use original 1–8-example conditioning
cohorts. Validation uses context-4 cohorts on 50 tuning examples/category from
held-out shard 00000; reserved test rows remain unused. The planned stage budget
is 1,000 additional updates, with validation every 125. The screenshot contains
validations at 0, 125, 250, 375 and 500; it does not establish completion at 1,000.

The feedback arm partitions supervised positions into the first 64, the next 64,
and the remaining positions of each packed microbatch. Earlier parts provide the
next part's three-scalar signal. All original response positions remain supervised
with the original per-depth denominator. This is a teacher-forced agreement proxy,
not live-serving speculative acceptance. It is shared over each packed microbatch,
and can cover multiple saved responses when they are packed together.

## Evidence and measurements

Qualitative observations only; exact numerical values are not reconstructed from
the plot:

| Displayed metric | Observation |
| --- | --- |
| `training_losses/drafter_loss` | Noisy per-update loss; smoothed arms overlap closely, without a clear separation |
| `validation_losses/loss_0` | Both improve in the middle evaluations, then rise at 500; feedback is slightly higher at 500 |
| `validation_losses/loss_1` | Similar trajectory and crossover; feedback is slightly higher at 500 |
| `validation_losses/loss_2` | Similar trajectory and crossover; feedback is slightly higher at 500 |

Feedback is slightly lower at some intermediate evaluations. Its sign of advantage
does not persist through the latest displayed validation. The training curves
extend beyond 500, but no later validation is visible in the supplied evidence.

**Working finding: this feedback variant has not demonstrated a consistent or
practically clear validation-loss improvement over regular continuation through
step 500.** Both arms show some validation-loss reduction relative to their start.

## Interpretation and limits

This is a negative result for the proposed feedback addition on the supplied
validation-loss evidence. It is not a result that ordinary generator training
failed, nor a measurement of serving speed or acceptance. `loss_0/1/2` are the
original three teacher-forced prediction-depth losses, not three datasets or
three separate models.

The plot does not show projector norm, gradients, adapter changes under alternate
signals, or feedback statistics. It cannot establish that the projector remained
zero or that the generator completely ignored feedback. No repeat-seed variance
or exported values are available, so overlapping curves do not establish
statistical equivalence. The remaining planned 500 updates and validation-agreement
metrics were not supplied.

The tested signal is coarse: three shared scalars, at most two observations before
the long remaining tail, and a fixed original-prompt summary. This does not test
all possible feedback representations or continuous per-round serving updates.

## Decision and open checks

Record the feedback addition as unsuccessful in producing a visible advantage
so far. Do not claim an acceptance or throughput gain from these plots. Preserve
the original generator baseline as the demonstrated reference pipeline.

If more artifacts are later supplied, append their exact values, run/checkpoint
identities and remaining validation steps. Projector norm and factor sensitivity
would distinguish an inactive feedback path from an active but unhelpful one.
No additional training or configuration change is introduced by this record.

## Updates

- 2026-10-28: Initial screenshot-based record and current architecture audit.
