# Full-generator online feedback comparison

Run each config independently:

| Config | Updated parameters | Feedback |
| --- | --- | --- |
| `regular.yaml` | Whole original generator | Disabled |
| `feedback.yaml` | Whole generator plus new feedback projector | Current-policy acceptance |

Both start from the pinned step-4000 generator with fresh stage optimizers, use
4K prompts (1K/category), and run 1,000 additional updates with eight prompts per
update. Validation runs at 0, 125, 250, …, 1,000. The feedback projector starts at
zero, preserving the initial parent function. The drafter and target stay frozen.

W&B project: `eagle3-generator-online-feedback-v2`.

Use a new workspace: set `SPECULATORS_FEEDBACK_ROOT` to
`/content/drive/MyDrive/dyn-spec-online-feedback-v2`. Existing v1 runs are preserved;
v1 projector-only optimizer state is incompatible with this new feedback arm.
Use `--resume` to recover the corresponding v2 run's own checkpoint.

See [the runbook](../../../docs/experiments/eagle3_online_feedback_continuation_v2.md)
for setup, the two commands, restored optimizations and the measured bottleneck.
