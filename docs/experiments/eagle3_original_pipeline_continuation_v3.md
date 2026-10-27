# Restore the original pipeline: continuation and feedback

Status: implementation and local numerical/recovery checks. No new GPU training
result or speedup is claimed. The online-rollout continuation direction was dropped
at the user's request. Earlier v1/v2 documents are historical and not the active
training recipe.

## The two arms

| Config | Training |
| --- | --- |
| `feedback_continuation_v3/regular.yaml` | Original generator continuation |
| `feedback_continuation_v3/feedback.yaml` | Same saved-response supervision, with the whole generator and feedback projector trainable |

Both start from `azeeer/eagle3-generator-all24k-10ep-v1`, pinned revision
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`, subfolder `step-00004000`.
They restore parent weights **and original Adam moments/RNG**, with no new warmup.
The projector is zero-initialized and gets new optimizer slots; existing generator
moments are mapped by parameter name. The original checkpoint predates explicit
parameter names, so its ordered model state supplies the original optimizer order.
The downloaded checkpoint's SHA256 must match the pinned serving bundle. Missing
or mismatching state is an error, never a silent fresh-optimizer fallback.

The pinned parent includes `step-00004000/checkpoint.pt` (about 3.52 GB), which the
command downloads automatically once through the Hub cache. A local parent folder
can also supply this file. Each arm's later resume uses its own saved optimizer,
RNG, progress and identity.

## What is restored

The regular arm directly invokes `campaign_train.train_update`, the implementation
used by the original successful training pipeline. It retains saved token
trajectories, bounded on-demand activation extraction/RAM caching, conditioning,
teacher forcing, noise/dropout, the complete response loss mask, three TTT steps,
and original LK/decay/normalization. It does not call `OnlineFeedbackTrainer`,
`StreamingDraft` or a per-round target verifier.

The original four episodes/update and 1–8-example conditioning sets are restored.
`EpochSchedule` shuffles/partitions the explicitly selected training membership
with the original cohort mechanism. Conditioning still uses the original last
prompt-state summaries and set encoder. Actual examples/update vary with sampled
cohort sizes and are logged, rather than pretending every update has eight rows.

Membership is **4K saved training responses**, 1K each from math, code, stem and
chat shard 00001. The stage runs for 1,000 additional updates (lineage 4000→5000).
It is a restricted-data continuation, not a continuation of the original full-24K
data order. No training rows outside the selected membership enter the schedule.

Validation uses the original teacher-forced agreement diagnostic and context-4
cohorts, including a short last cohort, on 50 tuning examples/category from held-out
shard 00000. Reserved test examples remain unused. Validation runs at 0, 125, 250,
375, 500, 625, 750, 875 and 1,000. These are training agreement measurements, not
live-serving speculative acceptance measurements.

## Feedback without generated continuations

Feedback is computed from the current generator on earlier positions of the same
**saved, teacher-forced** packed training sequence. No fresh response, rejected
prefix replay, donor policy or cached historical acceptance label is introduced.

For the feedback arm the supervised response positions are partitioned into the
first 64 positions, the next 64, and the remaining positions (short responses use
only the available parts). Prefix parts receive only their preceding context;
future tokens/features are padded out. Earlier completed parts provide the
feedback signal for the next part. All positions remain supervised exactly once.
The original full-response denominator is retained separately for every TTT step,
so partitioning does not overweight the short prefix parts or discard the tail.

The signal is `[latest ratio, EMA, observed flag]`, resetting for each packed
microbatch, with EMA decay 0.9. The ratio sums the existing cumulative-correct TTT
counts over eligible draft-slot counts in the preceding part. This is explicitly a
**teacher-forced prefix-agreement proxy**, not a claim that an actual speculative
continuation was generated/accepted. Full-vocabulary serving acceptance and the
effect of this proxy at inference remain benchmarking questions.

Generator conditioning uses the same original cohort prompt summaries throughout
the episode. Only completed earlier parts affect later feedback; future response
labels cannot influence earlier inputs. All episode gradients accumulate before
one optimizer step. Target and drafter weights stay frozen. Extra prefix forwards
in the feedback arm read the same cached activations, with no per-prefix vLLM RPC.
The regular arm needs only its original full-sequence forward.

## Setup and commands

Use a new workspace so earlier rollout experiments cannot be resumed accidentally:

```bash
export SPECULATORS_FEEDBACK_ROOT=/content/drive/MyDrive/dyn-spec-original-feedback-v3
mkdir -p "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor"
cp -a /content/drive/MyDrive/dyn-spec-online-feedback-v1/provenance/vllm-extractor/. \
  "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor/"
```

This reuses the already-running extractor with the same target, required layer
IDs and shared directory. If starting it again, use the established
[extractor setup](eagle3_online_feedback_continuation_v1.md#setup) after exporting
the v3 workspace, always passing `--provenance-dir`. It extracts activations for
saved complete trajectories, with the original cache, not per verification round.

Run the regular continuation:

```bash
speculators generator train-feedback \
  --config examples/eagle3-lora/feedback_continuation_v3/regular.yaml \
  --extraction-concurrency 16 --resume
```

Run full-generator feedback continuation separately:

```bash
speculators generator train-feedback \
  --config examples/eagle3-lora/feedback_continuation_v3/feedback.yaml \
  --extraction-concurrency 16 --resume
```

`--extraction-concurrency` controls the original cache's extraction workers.
The old `--rollout-concurrency` flag is removed; the active trainer has no rollout
path. RAM is automatically bounded; `--ram-cache-gib` is an
optional execution-only override. Add `--dry-run` to inspect without downloads or
GPU work. No feedback preparation or multi-experiment launcher is needed.

## Outputs and tracking

W&B project: **`eagle3-generator-original-feedback-v3`**, group
`original-pipeline-step4000-4k-plus1000`. Runs and Hub destinations are distinct:

- `original-regular-from4000-v3`: `azeeer/eagle3-original-regular-4k-plus1k-v3`.
- `original-feedback-from4000-v3`: `azeeer/eagle3-original-feedback-4k-plus1k-v3`.

Checkpoint/bundle saving remains every 500 additional steps; final Hub upload is
at 1,000, including recovery state and provenance. Validation every 125 does not
alter checkpoint frequency. Parent-optimizer checksum/mapping is recorded in
`parent_training_state.json`. Membership, resolved recipe, source hashes, train
and extractor commands/patches, execution settings, metrics and feedback-prefix
observations are retained. Old rollout checkpoints have a different training-mode
identity and cannot resume this pipeline.

## Verification and limits

Tests cover direct use of the original regular-arm update, the original cohort
sampler restricted to selected members, original loss and generator-gradient
equivalence with a zero projector (including packed documents), complete response
supervision, lack of future-label leakage, absence of generated-trajectory target
requests, parent Adam mapping, new-projector slots, stage resume and exports.
GPU fit, wall time and improved validation/serving acceptance remain unmeasured.

The broader focused suite passed 193 tests with two CUDA-only tests skipped.
After the final original-cohort, packed-response and selected-summary changes,
all 27 direct continuation tests passed again. The source optimizer mapping was
also exercised through a complete synthetic stage: original Adam counters advance
4000→4002 while the newly introduced projector advances 0→2.
