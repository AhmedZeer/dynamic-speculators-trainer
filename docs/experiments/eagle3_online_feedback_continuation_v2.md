# Full-generator online feedback continuation v2

Status: implemented with local numerical/recovery checks. GPU training and results
for this comparison have not been measured.

## Matched comparison

| Config | Trainable parameters | Acceptance feedback |
| --- | --- | --- |
| `feedback_continuation_v2/regular.yaml` | Whole original generator | Disabled |
| `feedback_continuation_v2/feedback.yaml` | Whole generator plus 3→64 projector | Current policy, causal |

The feedback arm now trains the original generator too. Both arms warm-start from
`azeeer/eagle3-generator-all24k-10ep-v1`, revision
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`, subfolder `step-00004000`.
The projector has zero weights and no bias, preserving initial parent factors even
for nonzero feedback. Both arms start fresh stage AdamW optimizers; resume restores
the corresponding stage's own optimizer and RNG. Target and drafter stay frozen.

Shared settings match the original continuation recipe:

- Exactly 4K training prompts: 1K each from math, code, stem and chat shard 00001.
- Eight prompts/update, 1,000 additional updates, 64-token online horizon and up to
  three proposals/verification round. This is 8,000 presentations, two passes over
  the 4K prompt pool. Same verified-prefix LK, learning rate, warmup and seed.
- Fixed original-prompt summary/length. Feedback is latest acceptance ratio,
  acceptance EMA (decay 0.9), and observed flag. It resets per request and affects
  only the next round. All eight requests use pre-update parameters; rollouts are
  freshly collected and not reused after an optimizer update.
- Supervision only covers accepted-prefix and first-correction predictions;
  incorrect proposals never enter supervised student inputs.

Validation uses the same 50 tuning prompts/category from held-out shard 00000,
with current-policy online acceptance and verified-prefix LK. Reserved test
examples remain unused. It runs at **0, 125, 250, 375, 500, 625, 750, 875 and 1,000
additional updates**. More frequent validation adds wall time; memberships, rollout
horizon and training budgets are not shortened to compensate.

The comparison tests whether acceptance-conditioned adaptation with a trainable
complete generator improves held-out acceptance. It adds feedback and a small
projector, and does not isolate each feedback component's contribution. Supervised
positions, verifier work and wall time can vary with acceptance and are logged
rather than assumed equal.

## Setup and separate commands

Use a new workspace to preserve the v1 regular/projector-only runs. Output names,
Hub repositories and W&B project are distinct too. No v1 checkpoint is rewritten.

```bash
export SPECULATORS_FEEDBACK_ROOT=/content/drive/MyDrive/dyn-spec-online-feedback-v2
python -m pip install -e '.[generator]'
```

If the existing extractor still runs with the pinned target, required layer IDs,
and `/tmp/speculators-online`, reuse it and copy its provenance:

```bash
mkdir -p "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor"
cp -a /content/drive/MyDrive/dyn-spec-online-feedback-v1/provenance/vllm-extractor/. \
  "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor/"
```

Otherwise follow [the original extractor setup](eagle3_online_feedback_continuation_v1.md#setup)
after setting the v2 workspace. Always pass `--provenance-dir` to the launch script.
Training automatically freezes membership and resolves the parent; there is no
offline feedback-preparation command.

Train the regular arm:

```bash
speculators generator train-feedback \
  --config examples/eagle3-lora/feedback_continuation_v2/regular.yaml \
  --rollout-concurrency 8 --resume
```

Train the complete feedback-aware generator:

```bash
speculators generator train-feedback \
  --config examples/eagle3-lora/feedback_continuation_v2/feedback.yaml \
  --rollout-concurrency 8 --resume
```

Run independently, sequentially on one GPU. Set the workspace in each shell/cell.
`--resume` also works on a new run and restores that v2 arm's latest saved state
after interruption. `--dry-run` resolves the config without model downloads, server
calls or GPU work. The v1 projector-only config remains a historical experiment.

## Tracking, checkpoints and restored execution

W&B project: **`eagle3-generator-online-feedback-v2`**, group
`step4000-4k-plus1000-full-generator`. Run names:
`regular-full-from4000-v2-gseed42` and `feedback-full-from4000-v2-gseed42`.

Full stage checkpoints and portable bundles are saved at additional steps 500 and
1,000; Hub upload remains at 1,000, with optimizer/RNG and reproducibility artifacts.
Validation every 125 steps does not alter checkpoint frequency. Repositories:

- `azeeer/eagle3-online-regular-full-4k-plus1k-v2`
- `azeeer/eagle3-online-feedback-full-4k-plus1k-v2`

The previously reverted optimization pass is restored: final-position-only draft
vocabulary projection, direct supplied-token selection for teacher forcing, CPU
validation of known prompt lengths, and suffix-only finiteness scans with complete
token alignment/shape/dtype checks. Attention still processes the full prompt;
initial prompt activations are scanned in full. Architecture and loss are unchanged.

Parallel verification, separate request KV/feedback, generator derivative reuse,
validation-logit reuse and memory-mapped loading also remain. Automatic concurrency
allows eight requests on the observed 40 GB GPU (at least 35 GiB detected VRAM).
Execution settings may change on resume and are saved in provenance. Training
concurrency is bounded by the eight-example optimizer batch. Distinct feedback
inputs can require more generator backward traversals than the regular arm.

## Measured bottleneck and hardware implications

User-supplied timings from the previous regular arm before the restored pass:

| Update | Total seconds | Collection seconds | Backward seconds |
| --- | ---: | ---: | ---: |
| 1 | 26.10 | 12.79 | 13.05 |
| 10 | 19.32 | 15.83 | 3.36 |
| 20 | 12.22 | 8.98 | 3.16 |

These are individual updates, not ten-step averages or measurements of the new
feedback arm. Collection takes about 74% at update 20. It includes draft GPU work,
target full-prefix recomputation, activation export/read/scan, host/device copies,
HTTP waits and host coordination. Phase timings do not separate GPU time from the
other costs. File/RPC timers can overlap and must not be added to collection wall
time; lazy mmap faults can occur outside the file-opening timer.

Eight requests already fit, so extra VRAM alone does not remove the bottleneck or
enlarge the eight-example optimizer batch. Higher BF16 throughput/bandwidth can
accelerate kernels, but overall benefit depends on actual device-time fraction.
For illustration, halving only the 3.16-second backward phase changes 12.22 seconds
to about 10.64 seconds. This does not bound gains from accelerating target GPU work
inside collection, and is not a prediction for a particular GPU.

To judge an upgrade, compare warm update times and inspect CPU/CUDA activity for
both trainer and extraction server. The
[PyTorch profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
distinguishes host operators and device kernels; a trainer-only trace does not
include kernels running in the separate vLLM process.

Persistent target KV and suffix-only activation export remain the larger backend
opportunity. The current extractor recomputes and emits the growing prefix every
round. This pass reduces client/draft work but does not implement incremental
target verification. No GPU speedup factor or purchase recommendation follows
from these phase timings alone.

## Local verification scope

Checks cover matched configs and 125-step cadence, unchanged initial parent
factors with a zero projector, full generator/projector trainability, frozen draft
parameters, gradients, causal feedback, bundle export, optimizer/RNG recovery, and
rejection of projector-only state for a full-feedback run. Restored execution
checks compare logits and adapter gradients with the full core model. GPU memory
fit and speed for the new arm remain unmeasured.

The focused regression suite passed **149 tests**, with one CUDA-only test skipped.
Both configs also pass CLI dry-run resolution without downloading or training.
