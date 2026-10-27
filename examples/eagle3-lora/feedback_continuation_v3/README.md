# Original pipeline continuation

These are the active configs. Both continue step 4000 using the original saved
responses, cached target activations, teacher-forced EAGLE3 forward and LK loss.
There is no speculative rollout or replay trainer in the active command.

- `regular.yaml`: original generator training.
- `feedback.yaml`: train the complete generator plus a zero-initialized projector,
  using current-model agreement on earlier saved-prefix positions.

Both restore the original generator's Adam moments/RNG, use only 4K training rows
(1K/category), train for 1,000 additional updates, and validate every 125 steps.
The source's four episodes/update and 1–8-example conditioning cohorts are restored.
Validation uses context size 4, including its short final cohort.

W&B: `eagle3-generator-original-feedback-v3`.

See [setup and the two commands](../../../docs/experiments/eagle3_original_pipeline_continuation_v3.md).
