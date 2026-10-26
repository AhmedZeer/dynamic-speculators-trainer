# Generator dynamism experiments, v4

Status: implementation and CPU verification; GPU measurements have not been run.
Each leaf YAML is one independently runnable experiment. There is no ablation
runner. YAML `extends` only merges configuration; it never launches another run.
Run commands from the repository root.

## Shared protocol

- Training: 1,000 saved trajectories from shard `00001` in each of math, code,
  stem and chat: **4,000 examples jointly**. No LoRA bank is read.
- Ordinary training: 1,500 optimizer updates, eight independent request examples
  per update, one conditioning example per generated adapter. This gives 12,000
  example presentations, or three passes over the 4K pool. The static LoRA uses
  the same examples but stops at 1,000 updates (8,000 presentations, two passes).
  Its budget therefore differs from the 1,500-update generators. Backbone, vocabulary, LoRA rank/targets and
  three speculative tokens remain those of the existing recipe.
- Replay branches restore the prompt-LK checkpoint at update 500, including Adam
  moments and RNG; they add 1,000 updates to reach 1,500 total. The feedback
  projection starts at zero, preserving the parent generator's initial function.
- Replay collects up to four actual greedy speculative verification blocks for
  each of the eight requests per update. It immediately replays the same proposed
  prefixes with gradients, using the current generator. Full target-vocabulary
  verification determines acceptance; the frozen draft vocabulary is unchanged.
  Ordinary runs supervise saved response trajectories; replay runs supervise
  drafted positions. **Updates are matched, supervised tokens and teacher compute
  are not.** Teacher requests, input tokens, replay positions and time are logged.
- Membership is frozen before training. Held shard `00000` supplies 100 validation
  examples per category: seed 42 divides these into 50 tuning and 50 test examples.
  Training, tuning and test row memberships are disjoint; this is not a content
  decontamination claim. Conditioning control donors come only from training.
- Training validation runs at updates 0, 500, 1,000 and 1,500. It is a singleton,
  prompt-conditioned, teacher-forced probe on the **tuning** split. Feedback is
  zero in that probe. Its agreement score does not measure online refresh quality;
  inference benchmarks do. Continuation runs start validation at their restored step.
- Inference: math and chat only, 50 test prompts each, concurrency 1, greedy
  generation, at most 256 output tokens, three warmups. The vanilla EAGLE3
  baseline uses unmodified vLLM sources. Other runs use explicit experimental
  adapter policies and the same target/drafter revisions.
- Acceptance comes from native verifier counters: `draft_acceptance_rate` is
  accepted draft tokens / proposed draft tokens. `accepted_prefix_length` is
  accepted draft tokens / verification rounds. Existing `acceptance_length`
  includes the bonus token. Throughput includes prefill and serving overhead.
  Correctness is recorded as unchecked; speed measurements do not establish it.

## Setup and input preparation

Use the existing pinned local target/drafter model directories and saved token
trajectory download described in [the online campaign guide](eagle3_generator_campaign_online.md).
No activation dataset download is needed. All commands below are manual.

```bash
export SPECULATORS_STUDY_ROOT=/content/drive/MyDrive/dyn-spec-ablations-v4
CFG=examples/eagle3-lora/generator_ablations_v4
python -m pip install -e '.[generator]'
wandb login
hf auth login
python scripts/apply_vllm_generator_patch.py --apply
```

Use the repository's supported vLLM 0.31.0 environment. Reapply the patch after
updating the code and restart existing servers; the proposal hook now forwards
verification metadata. The vanilla reference is launched from pristine sources.

Start the activation extractor in a separate terminal/cell, with the same absolute
shared-directory path visible to both processes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 18 33 --include-last-layer \
  --hidden-states-backend file --hidden-states-path /tmp/speculators-online \
  --provenance-dir "$SPECULATORS_STUDY_ROOT/provenance/vllm-extractor" \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 16 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

In the training terminal/cell, export the same workspace variable, then:

```bash
speculators generator prepare-experiments --config "$CFG/prepare.yaml"
```

This freezes row memberships and tokenized tuning/test manifests, and extracts
up to 64 training-only prompt summaries per category for conditioning controls.
It does not launch training or download activation dumps. Preparation is reusable.
All training configurations inherit local paths and extraction settings from
`examples/eagle3-lora/generator_campaign_online.yaml`: default extraction
concurrency 16, buffer 64 and full CPU activation cache 96 GiB. Summary and compiler
caches persist under the workspace; transient extractor files are consumed and
removed. Replay additionally requests target states for its own drafted prefixes.
The 40% extractor GPU reservation is a starting point on an A100 80 GB; actual fit
and timings must be measured, especially for replay's full target head.

## Independently trained configurations

Run `prompt_lk` first; the three replay continuations need its step-500 checkpoint.
All other commands can be selected and ordered manually.

| Configuration | Training | New updates | W&B project |
| --- | --- | ---: | --- |
| `prompt_lk` | Generator, prompt states, existing LK objective | 1500 | `eagle3-generator-objectives-train-v4` |
| `static_lora_lk` | Static LoRA, same data and LK objective; shorter budget | 1000 | `eagle3-generator-objectives-train-v4` |
| `prompt_kl` | Generator, prompt states, KL objective | 1500 | `eagle3-generator-objectives-train-v4` |
| `prompt_ce` | Generator, prompt states, CE objective | 1500 | `eagle3-generator-objectives-train-v4` |
| `replay_prompt_lk` | Prompt-only generator with on-policy replay | 1000 | `eagle3-generator-replay-train-v4` |
| `replay_prefix_lk` | Verified-prefix refresh with replay | 1000 | `eagle3-generator-refresh-train-v4` |
| `replay_prefix_feedback_lk` | Verified-prefix refresh plus previous acceptance feedback | 1000 | `eagle3-generator-refresh-train-v4` |

```bash
speculators generator train-experiment --config "$CFG/training/prompt_lk.yaml" --resume
```

```bash
speculators generator train-experiment --config "$CFG/training/static_lora_lk.yaml" --resume
```

```bash
speculators generator train-experiment --config "$CFG/training/prompt_kl.yaml" --resume
```

```bash
speculators generator train-experiment --config "$CFG/training/prompt_ce.yaml" --resume
```

```bash
speculators generator train-experiment --config "$CFG/training/replay_prompt_lk.yaml" --resume
```

```bash
speculators generator train-experiment --config "$CFG/training/replay_prefix_lk.yaml" --resume
```

```bash
speculators generator train-experiment --config "$CFG/training/replay_prefix_feedback_lk.yaml" --resume
```

Every run saves full optimizer/RNG checkpoints and a serving bundle every 500
updates. The configured distinct Hugging Face repository receives step 1,000 and
the final step, including the full training checkpoint and provenance. The static
LoRA ends at 1,000 updates and uses `azeeer/eagle3-static-lora-lk-4k-1000-v4`;
generators end at 1,500. Change `repo_id`
in an individual YAML before starting if a different destination is wanted.
`--resume` restores local durable checkpoints and skips completed runs; without it
an existing output directory is refused. Put the workspace on persistent Drive in
Colab. Hugging Face upload alone does not automatically restore a deleted workspace;
restore the published checkpoint/provenance before resuming. Unuploaded updates can
be lost after a runtime reset. Training identity checks reject changed experimental
settings in an existing run directory.

## Inference studies

Stop your extraction server before running these suites: the suite owns its own
vLLM server on the same GPU/port, launches the client, records provenance and stops
its owned processes. Each command below measures only one model configuration,
on math and chat. `--resume` reuses completed measurements and retries W&B publication
without repeating GPU work. Use a distinct output directory per configuration.

The first conditioning study uses the existing step-4000 remote checkpoint, so it
can run immediately after preparation, before new training. Same-category donors
are deterministic training prompts different from the evaluated prompt;
cross-category donors swap math and chat; fixed uses one training donor throughout.
All other studies use the newly trained step-1500 local generator bundles; the
static-LoRA reference uses its final step-1000 bundle.

| Family | Question | W&B project |
| --- | --- | --- |
| references | Vanilla EAGLE3, matched static LoRA, prompt generator | `eagle3-generator-reference-bench-v4` |
| conditioning | Correct, same-category swap, cross-category swap, fixed donor | `eagle3-generator-conditioning-bench-v4` |
| objectives | LK versus KL versus CE | `eagle3-generator-objectives-bench-v4` |
| replay | Ordinary LK versus prompt-only on-policy replay | `eagle3-generator-replay-bench-v4` |
| refresh | Prompt-once, untrained prefix-refresh probe, matched replay controls and trained refresh | `eagle3-generator-refresh-bench-v4` |
| scale | Effective LoRA scale 0, 0.5, 1, 2, 4 | `eagle3-generator-scale-bench-v4` |
| calibration | Tune controller direction on tuning prompts only | `eagle3-generator-scale-calibration-v4` |
| feedback | Fixed/adaptive scale × prefix generator without/with feedback, plus prompt scale control | `eagle3-generator-feedback-bench-v4` |

### References

```bash
speculators benchmark suite --config "$CFG/inference/references/prompt_lk.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/references-prompt_lk" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/references/static_lora.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/references-static_lora" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/references/vanilla_eagle3.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/references-vanilla_eagle3" --resume
```

### Conditioning

```bash
speculators benchmark suite --config "$CFG/inference/conditioning/correct.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/conditioning-correct" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/conditioning/cross_category.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/conditioning-cross_category" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/conditioning/fixed.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/conditioning-fixed" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/conditioning/same_category.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/conditioning-same_category" --resume
```

### Objectives

```bash
speculators benchmark suite --config "$CFG/inference/objectives/ce.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/objectives-ce" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/objectives/kl.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/objectives-kl" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/objectives/lk.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/objectives-lk" --resume
```

### Replay

```bash
speculators benchmark suite --config "$CFG/inference/replay/ordinary.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/replay-ordinary" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/replay/replay.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/replay-replay" --resume
```

### Refresh

```bash
speculators benchmark suite --config "$CFG/inference/refresh/prefix_feedback_trained.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/refresh-prefix_feedback_trained" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/refresh/prefix_probe.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/refresh-prefix_probe" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/refresh/prefix_trained.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/refresh-prefix_trained" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/refresh/prompt_once.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/refresh-prompt_once" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/refresh/prompt_replay.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/refresh-prompt_replay" --resume
```

### Scale

```bash
speculators benchmark suite --config "$CFG/inference/scale/s0.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/scale-s0" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/scale/s05.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/scale-s05" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/scale/s1.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/scale-s1" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/scale/s2.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/scale-s2" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/scale/s4.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/scale-s4" --resume
```

### Feedback

```bash
speculators benchmark suite --config "$CFG/inference/feedback/feedback_fixed.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/feedback-feedback_fixed" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/feedback/feedback_scale.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/feedback-feedback_scale" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/feedback/prefix_fixed.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/feedback-prefix_fixed" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/feedback/prefix_scale.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/feedback-prefix_scale" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/feedback/prompt_scale.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/feedback-prompt_scale" --resume
```

## Calibrate acceptance-based scaling before its test runs

The initial controller is a hypothesis, not a learned or validated policy. Run
these two configurations on the **tuning** manifest before `prompt_scale`,
`prefix_scale` and `feedback_scale` test measurements:

```bash
speculators benchmark suite --config "$CFG/inference/feedback/calibrate_decrease.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/calibrate_decrease" --resume
```

```bash
speculators benchmark suite --config "$CFG/inference/feedback/calibrate_increase.yaml" \
  --output-root "$SPECULATORS_STUDY_ROOT/benchmarks/calibrate_increase" --resume
```

Compare tuning throughput, accepted-prefix length and adapter overhead. Select
one direction, then freeze `low_action` in `common/scale_controller.yaml` **before**
running adaptive policies on test prompts. Other tunable fields are in that file.
Use a new output root if calibration settings change. Do not pick the best policy
using test measurements. The two included calibration configs deliberately
compare opposite direction hypotheses, not a claim that either is optimal.

The default starts at effective scale 2, maintains an acceptance EMA with decay
0.9, waits for three observed rounds, then moves by 0.5 inside [0,4] when the EMA
is below 0.35 or above 0.75. `low_action: decrease` reduces scale on low acceptance;
`increase` reverses that direction. Effective scale replaces the old alpha/r
multiplier; it is not multiplied by the old value of 2 again. Each request resets
the controller. Adaptive prompt scaling reuses the initial factors and changes
only their multiplier, separating scale feedback from generating new factors.

## Refresh and feedback semantics

Refresh observes only committed target hidden states and **previous** verification
outcomes; rejected speculative positions are excluded. Feedback inputs are the
last accepted/proposed ratio, its EMA, and an observed flag. The first round uses
zero feedback. A feedback-trained generator gets the extra zero-initialized linear
projection; ordinary generators retain their old checkpoint structure.

All refresh experiments explicitly use `causal_stateful` draft KV semantics:
historical draft states keep the adapters used when those positions were committed.
Native rejection rollback still applies; newly committed query slots are recomputed
with the new adapter. We do not rebuild the entire prefix KV on every refresh or
claim equivalence to a fixed-adapter prefix cache. This policy is shared by replay
and serving and is included in provenance.

Feedback counters in the proposal hook cover verification outcomes observed before
a subsequent proposal. The terminal verification has no next proposal callback;
use native vLLM verifier metrics for overall acceptance. Per-request round logs
record prompt hashes, observed acceptance and scale. W&B artifacts include only
measured-request round logs, excluding warmup requests.

## Outputs and W&B layout

```text
$SPECULATORS_STUDY_ROOT/
  inputs/{membership.json,conditioning.safetensors,conditioning.json,tune/,test/}
  provenance/vllm-extractor/
  cache/{summaries/,compiler/}
  training/<name>/
    study.json, identity.json, checkpoint.pt, metrics.jsonl, validation.json
    provenance/, wandb/, replay/update-*.json
    checkpoints/step-00000500/{checkpoint.pt,bundle/,provenance/...}
    checkpoints/step-00001000/{checkpoint.pt,bundle/,hub.json,...}
    checkpoints/step-00001500/{checkpoint.pt,bundle/,hub.json,...}
  benchmarks/<family-configuration>/
    summary.md, summary.json, state.json, plan.json
    servers/, measurements/
```

Training W&B sections separate `training_losses`, `validation_losses`, `validation_scores`,
`training_performance`, `validation_performance`, `preparation_performance`,
`cache_performance`, `replay_scores`, and `replay_performance`.
Benchmark W&B sections separate `throughput`, `acceptance`, `latency`, and
`adapter_performance`. There is one benchmark W&B run per model/category, grouped
under `math-chat-c1`, with frozen config, resolved revisions, results and provenance
as artifacts. Each study family has its own project; new results cannot mix with
v3 history. Training step is the optimizer-update axis; benchmark runs are scalar
comparisons. Local JSON and Markdown remain the primary evidence if tracking is
unavailable.

## Verification and limits

Verification: 121 focused CPU tests passed; all 41 YAML files resolve, including
seven training and 29 individual inference configurations. CLI preparation/training
dry runs and the pinned vLLM proposal-hook checksum also pass.

CPU tests exercise config inheritance, exact example scheduling, bounded causal
feedback, rejected-state exclusion, streaming draft parity and differentiable
replay, optimizer warm-start mapping, checkpoint/export recovery, W&B publication
retry, and warmup exclusion. The supported vLLM patch is pinned and checksum-checked.
Live A100/vLLM execution, memory fit, refresh quality and controller usefulness
remain experiments to run, not measured results in this document.
