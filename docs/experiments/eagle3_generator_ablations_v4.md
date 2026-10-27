# Generator dynamism experiments, v4

Status: implementation and CPU verification; user-supplied GPU measurements for
correct, fixed-donor and cross-category conditioning are recorded below. Other
GPU experiments remain unverified in this record.
Each leaf YAML is one independently runnable experiment. There is no ablation
runner. YAML `extends` only merges configuration; it never launches another run.
Run commands from the repository root.

## Shared protocol

- Training: 1,000 saved trajectories from shard `00001` in each of math, code,
  stem and chat: **4,000 examples jointly**. No LoRA bank is read.
- All training configurations stop at **1,000 total optimizer updates**.
  Ordinary generators and the static LoRA use eight independent request examples
  per update: 8,000 presentations, or two passes over the 4K pool. Each generated
  adapter conditions on one example. Backbone, vocabulary, LoRA rank/targets and
  three speculative tokens remain those of the existing recipe.
- Replay branches restore the prompt-LK checkpoint at update 500, including Adam
  moments and RNG; they add 500 updates to reach 1,000 total. The feedback
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
- Training validation runs at updates 0, 500 and 1,000. It is a singleton,
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
| `prompt_lk` | Generator, prompt states, existing LK objective | 1000 | `eagle3-generator-objectives-train-v4` |
| `static_lora_lk` | Static LoRA, matched data, updates and LK objective | 1000 | `eagle3-generator-objectives-train-v4` |
| `prompt_kl` | Generator, prompt states, KL objective | 1000 | `eagle3-generator-objectives-train-v4` |
| `prompt_ce` | Generator, prompt states, CE objective | 1000 | `eagle3-generator-objectives-train-v4` |
| `replay_prompt_lk` | Prompt-only generator with on-policy replay | 500 | `eagle3-generator-replay-train-v4` |
| `replay_prefix_lk` | Verified-prefix refresh with replay | 500 | `eagle3-generator-refresh-train-v4` |
| `replay_prefix_feedback_lk` | Verified-prefix refresh plus previous acceptance feedback | 500 | `eagle3-generator-refresh-train-v4` |

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
the final step, including the full training checkpoint and provenance. All runs
finish and upload at step 1,000; repository names use `4k-1000-v4`. Change `repo_id`
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
All other studies use the newly trained step-1000 local bundles for both
generators and the static-LoRA reference.

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

### Conditioning measurements and questions (2026-10-27)

**Evidence status.** The user supplied `summary.json`, `state.json`,
`failures.json`, `suite_request.json`, and the final 30 lines of server/client
logs for three policies. All six category tasks report `completed`, attempt 1,
50 successful requests, and `wandb_published: true`; each suite reports one server
attempt and an empty failure map. These are transcribed measurements from the
supplied outputs, not a rerun or an independent verification of the remote files.
No same-category results have been supplied, so the conditioning study is partial.

**Setup recorded in the requests.** All three policies use
`azeeer/eagle3-generator-all24k-10ep-v1`, revision
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`, subfolder `step-00004000`.
They use prompt-once conditioning, one conditioning example, fixed effective
scale 2, no feedback, three speculative tokens, concurrency 1,
`fixed_shared_adapter` batching, 50 test prompts per category, a 256-output-token
cap, three warmups and seed 42. This checkpoint is the existing remote model,
not one of the newly trained step-1000 v4 bundles. The requests reference
`$SPECULATORS_STUDY_ROOT/inputs/test/manifest.json`; fixed and cross-category
policies also reference `inputs/conditioning.safetensors`.

**Measurements.** Throughput is aggregate output tokens/s, including prefill and
serving overhead. Acceptance is accepted draft tokens / proposed draft tokens;
accepted prefix excludes the bonus token. Latency is median request latency.

| Policy | Category | Output tokens/s | Draft acceptance | Accepted prefix | Median latency (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Correct | math | 79.2963 | 65.0545% | 1.9516 | 3.1410 |
| Fixed donor | math | 77.1801 | 64.9183% | 1.9476 | 3.2382 |
| Cross-category | math | 70.1106 | 53.9614% | 1.6188 | 3.5765 |
| Correct | chat | 59.9882 | 40.8347% | 1.2250 | 4.1086 |
| Fixed donor | chat | 54.0325 | 35.4138% | 1.0624 | 4.5356 |
| Cross-category | chat | 55.2330 | 35.5605% | 1.0668 | 4.4738 |

The native counters supporting acceptance and accepted-prefix measurements are:

| Policy | Category | Accepted draft tokens | Proposed draft tokens | Verification rounds |
| --- | --- | ---: | ---: | ---: |
| Correct | math | 8474 | 13026 | 4342 |
| Fixed donor | math | 8466 | 13041 | 4347 |
| Cross-category | math | 7921 | 14679 | 4893 |
| Correct | chat | 6663 | 16317 | 5439 |
| Fixed donor | chat | 6230 | 17592 | 5864 |
| Cross-category | chat | 6243 | 17556 | 5852 |

**Questions and current answers.** Percentage throughput gains below use the
comparison policy as denominator; acceptance changes are percentage points.

1. **Does correct conditioning outperform one fixed training donor?** On chat,
   correct conditioning gives 11.0% higher throughput and 5.42 percentage points
   higher acceptance. On math, throughput is 2.7% higher but acceptance differs
   by only 0.14 percentage points. The evidence supports a chat benefit against
   this particular fixed donor; the small math timing difference needs repeats.
2. **Does conditioning from the wrong category hurt?** Relative to correct
   conditioning, the cross-category policy has lower acceptance and throughput
   in both categories. Correct conditioning gives 13.1% higher throughput and
   11.09 percentage points higher acceptance on math, and 8.6% higher throughput
   and 5.27 percentage points higher acceptance on chat. These measurements
   support sensitivity to cross-category donor substitution on this checkpoint.
3. **Is math insensitive to conditioning?** The fixed-donor comparison alone
   showed nearly unchanged acceptance, but the cross-category comparison does
   not. The narrower observation is that the particular fixed donor preserved
   math acceptance while the selected cross-category donors reduced it. No donor
   category or general donor-independence claim follows from the fixed result.
4. **Is a different prompt from the same category sufficient?** Open: no
   `same_category` measurement was supplied. Its implementation and command above
   are not measured evidence.
5. **Is the benefit category information or prompt-specific information?** Open:
   the same-category control is needed to separate these explanations. The
   available measurements establish neither explanation alone.
6. **Could generator execution cost explain the policy difference?** Reported
   generator time is about 1.95--2.00 ms/call across all six tasks, with exactly
   50 invocations and zero adapter refreshes per task. The observed acceptance
   changes are consistent with different draft quality rather than a large
   change in generator execution cost; these timings do not isolate every
   component of serving overhead.

**Evidence locations and identities.** The actual supplied cross-category output
folder is `benchmarks/cross-category`, differing from the suggested command's
`benchmarks/conditioning-cross_category`. All paths below are relative to
`/content/drive/MyDrive/dyn-spec-ablations-v4`.

| Policy | Output folder | Suite identity |
| --- | --- | --- |
| Correct | `benchmarks/conditioning-correct` | `860c8f59f81c51f5fce534dd9ba68394c0c23c998c00989e345cd820f9ceb04c` |
| Fixed donor | `benchmarks/conditioning-fixed` | `eb3cbb80955fb3ddbeba30f3ebb290d29b514347873a348e13dd1e2c82087646` |
| Cross-category | `benchmarks/cross-category` | `de3c3f45e592088bfa81fee4213b1d5fbc4aa71d5ca022a592e95a3053159b53` |

In each output folder, `summary.json` holds the metrics above and `state.json`
records completion and checksums for each task's `results.json`,
`benchmark_identity.json` and GuideLLM JSON. Native task results live under
`measurements/conditioning-<policy>/nemotron/nemo-<category>/attempt-0001/`;
server logs live under `servers/conditioning-<policy>/attempt-0001/server.log`,
using `cross_category` for that policy's name. All three policies report identical
per-category dataset hashes:

- Math: `d7e7f7f47881d09f1e1d1847691ea5a314a38cdb61400ae00b8782ec5170b89f`.
- Chat: `845fa6ea14a05c498e057b19cf8b43a26f5b7dbc9bbd70f262bd0bf0c7a939d6`.

**Limits and remaining evidence.** These are single executions on 50 prompts per
category, without repeat uncertainty estimates or supplied hardware metadata.
The same-category control remains missing. All policies record
`correctness_status: not_checked`; identical total output counts (12,800 math,
12,102 chat) do not establish identical text or correctness. The server tails
show force-kill/semaphore cleanup warnings after measurements completed; the
supplied task state and client logs report successful completion. Full resolved
plans, donor hashes/selections, command/version artifacts, checkpoint hashes and
patches were not supplied or independently inspected here. Retain the original
run directories and their provenance artifacts with these measurements; this
transcription does not replace those artifacts. No baseline speedup, general
model ranking, or final conclusion is established by this partial study.

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

## Replay compiler-cache failure (2026-10-26)

A live replay run stopped after its step-500 validation with
`FailOnRecompileLimitHit: Hard failure due to fullgraph=True`. The shared standalone
FlexAttention entry point exhausted Dynamo's default eight variants while replay
alternated gradient/no-gradient execution and changing query/KV lengths.

Streaming replay now routes through PyTorch SDPA, using implicit causal prefill,
unmasked singleton cached decode, and an explicit offset causal mask for
multi-token cached updates. This avoids materializing a quadratic mask for initial
prefill and avoids the standalone FlexAttention compiler cache for replay. Ordinary
training and validation retain their configured attention backend. Cache rollback
uses a negative crop count, compatible with the Transformers deprecation. The
causal KV policy, objective and experimental configuration identities are unchanged.

After updating code, rerun the same command with `--resume`; no config or workspace
change is required. The latest saved replay checkpoint is restored, or the step-500
parent is loaded when no replay checkpoint was saved. CPU regression checks cover
output/gradient parity, GQA, changing lengths and alternating gradient modes. The
CUDA stress test is included but cannot be executed in a CPU-only environment.
