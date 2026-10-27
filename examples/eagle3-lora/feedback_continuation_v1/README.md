# Current-policy online feedback continuation

Two independent training configs starting from the pinned generator step 4000:

- `regular.yaml`: train the original generator for 1,000 additional updates.
- `projector_only.yaml`: freeze it and train only the 192-parameter feedback projection.

Both use exactly 4K training prompts (1K/category), causal current-policy feedback,
verified-prefix LK supervision and the new W&B project
`eagle3-generator-online-feedback-v1`. `common.yaml` holds shared settings.

See [the runbook](../../../docs/experiments/eagle3_online_feedback_continuation_v1.md)
for the extraction server and the two separate training commands.
