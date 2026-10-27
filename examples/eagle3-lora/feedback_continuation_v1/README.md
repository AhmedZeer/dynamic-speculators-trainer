# Current-policy online feedback continuation

Two independent training configs starting from the pinned generator step 4000:

- `regular.yaml`: train the original generator for 1,000 additional updates.
- `projector_only.yaml`: freeze it and train only the 192-parameter feedback projection.

Both use exactly 4K training prompts (1K/category), causal current-policy feedback,
verified-prefix LK supervision and the new W&B project
`eagle3-generator-online-feedback-v1`. `common.yaml` holds shared settings.

See [the runbook](../../../docs/experiments/eagle3_online_feedback_continuation_v1.md)
for the extraction server and the two separate training commands.

Execution flags `--rollout-concurrency` and `--ram-cache-gib` can change on resume.
Use concurrency 8 on the observed 40 GB GPU or a larger GPU; automatic mode uses
8 with at least 35 GiB GPU memory and 2 below that. RAM is capped automatically to
leave memory for the models and extraction server. Training concurrency is bounded
by the eight examples in each optimizer update; a higher flag only helps validation.
