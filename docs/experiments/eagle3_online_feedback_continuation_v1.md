# Online feedback continuation from generator step 4000

Status: implemented; CPU regression tests, no live A100 measurements for this stage.
This is a separate study from the v4 rejected-suffix replay experiments.

## Two independently launched arms

| Config | Starting generator | Trainable weights | Updates |
| --- | --- | --- | ---: |
| `regular.yaml` | Pinned step 4000 | Entire original generator | 1,000 additional |
| `projector_only.yaml` | Same pinned step 4000 plus zero feedback projection | Only the 3→64 bias-free feedback projection (192 parameters) | 1,000 additional |

Both arms use **exactly 4,000 training rows**: 1,000 each from math, code, STEM and
chat shard `00001`. Eight independent prompts per optimizer update give 8,000
presentations over 1,000 updates, or two passes over those 4K rows. No other training
shards are sampled. Held shard `00000` supplies 50 tuning examples per category;
the other 50 are reserved and never used by this training command.

The source is `azeeer/eagle3-generator-all24k-10ep-v1`, revision
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`, subfolder `step-00004000`.
Only the serving bundle and its provenance are downloaded, not the original Adam
state. **Both arms start a fresh optimizer** for this adaptation stage. Resuming
this new stage restores its own full optimizer and RNG checkpoint.

The local stage counter is 0→1,000; source-plus-stage lineage is 4000→5000.
Checkpoint directory `step-00001000` means 1,000 additional updates, not the original
model's step 1,000. Source step and parent weight hashes are recorded explicitly.

## Online protocol and what is isolated

There is no offline baseline-feedback dataset. Every training batch collects fresh
feedback using its **current** generator/projector parameters. All eight requests
in a batch use the same pre-update parameters; the optimizer steps only after their
gradients have accumulated. Rollouts are not reused after a parameter update.

Each request starts with zero feedback and the original prompt summary. Feedback
is `[last accepted/proposed ratio, acceptance EMA, observed flag]`; EMA decay is 0.9.
The prompt summary and prompt length remain fixed throughout the request. The
feedback arm regenerates the LoRA with updated feedback at verification-round
boundaries. It never changes conditioning to the latest prefix summary. The regular
arm ignores feedback; repeated generator calls yield the same adapter within a
request because its prompt inputs and weights are unchanged.

The frozen target supplies a greedy anchor token, the drafter proposes up to three
tokens, and full-target-vocabulary logits verify the longest accepted prefix.
Only the target anchor and accepted proposals enter the committed continuation.
Greedy correction is supplied by the target at the next round. EOS and context
limits terminate a request.

LK supervision uses accepted positions and the first correction prediction, whose
input prefix is still correct. A gradient-enabled pass teacher-forces only verified
previous tokens; **an incorrect proposed token never enters a supervised input**.
Positions after the first rejection are excluded. This differs from the old replay
objective that supervised invalidated drafted suffixes. Teacher loss logits remain
restricted to the pretrained draft vocabulary; verification uses the full target
vocabulary. There is no acceptance reward or policy-gradient objective.

Streaming draft attention uses SDPA. Collection and gradient passes keep dropout
disabled. Historical committed draft KV is detached and retains the adapters used
when it was created (`causal_stateful`); new slots use the current adapter. Gradients
do not propagate through discrete decisions or across prior verification rounds.
Loss is normalized by actual supervised positions per request and then averaged
over the eight requests.

`rollout_tokens: 64` bounds each online continuation to at most 64 new target-verified
tokens. It is explicit and identical in both configs, intended to limit online
collection cost. Saved response tokens are not used as feedback labels or training
targets here; the 4K source trajectories supply prompts and stable identities.
Increase this common setting before starting a new workspace if a longer online
horizon is required. Exact supervised-token exposure and verifier cost still vary
with acceptance/EOS and are logged; optimizer updates, prompt pool and rollout cap
are matched rather than token count or wall time.

This is a practical comparison of full-generator adaptation versus a small feedback
adapter; it does not by itself isolate information value from trainable capacity.
A later shuffled-feedback serving control would help distinguish informative
acceptance conditioning from a nearly constant latent steering effect. That control
is not part of these two training jobs.

## Setup

Run commands from the repository root. Reuse the pinned models and saved token
trajectories from [the online campaign setup](eagle3_generator_campaign_online.md).
No LoRA bank or activation-dump download is required.

```bash
export SPECULATORS_FEEDBACK_ROOT=/content/drive/MyDrive/dyn-spec-online-feedback-v1
CFG=examples/eagle3-lora/feedback_continuation_v1
python -m pip install -e '.[generator]'
wandb login
hf auth login
```

If the existing v4 extractor is still running with the same model, layer IDs and
shared directory, reuse it and copy its provenance instead of launching another:

```bash
mkdir -p "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor"
cp -a /content/drive/MyDrive/dyn-spec-ablations-v4/provenance/vllm-extractor/. \
  "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor/"
```

Otherwise launch the extraction server in a separate terminal/cell:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 18 33 --include-last-layer \
  --hidden-states-backend file --hidden-states-path /tmp/speculators-online \
  --provenance-dir "$SPECULATORS_FEEDBACK_ROOT/provenance/vllm-extractor" \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 16 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

This is the established A100 80 GB starting configuration; this new stage's live
memory fit and timings remain to be measured. The frozen full target LM head and
trainable generator add allocations outside vLLM's reservation. Both processes
must see the same absolute `/tmp/speculators-online` directory.

Training freezes the 4K membership automatically and resolves the parent bundle;
there is no donor-preparation command, no precomputed feedback-label step, and no
multi-run launcher. In each training terminal/cell export the same workspace root.

## Training commands

Run each command independently (sequentially on one GPU):

```bash
speculators generator train-feedback \
  --config "$CFG/regular.yaml" \
  --resume
```

```bash
speculators generator train-feedback \
  --config "$CFG/projector_only.yaml" \
  --resume
```

Add `--dry-run` to either command to inspect its resolved configuration without
model downloads, server calls or GPU work. `--resume` is valid on first launch and
continues the latest saved stage checkpoint after interruption. Changing experimental
settings in an existing run is refused; use a new workspace for altered settings.

## Outputs, W&B and publication

Both arms publish to the **new** W&B project `eagle3-generator-online-feedback-v1`,
group `step4000-4k-plus1000-online`, with distinct run names:
`regular-from4000-gseed42` and `projector-from4000-gseed42`.

- `training_losses`: LK on verified prefixes.
- `training_scores`: greedy draft acceptance and accepted-prefix length.
- `validation_scores`: current-policy online acceptance, overall and per category;
  this is not the earlier zero-feedback prompt probe.
- `validation_losses`: current-policy verified-prefix LK.
- `training_performance`, `validation_performance`, `cache_performance`: measured
  seconds, target requests/input tokens, actual supervised positions and prompt RAM.
- `feedback_diagnostics`: projector weight norm.

Validation runs at additional updates 0, 500 and 1,000, using actual causal feedback
for the projector arm. Prompt-only target activations can be cached in bounded RAM. The execution layer
caps this at half the detected system/container memory by default, also respecting
the configured upper bound; `--ram-cache-gib` sets a smaller explicit cap.
Policy-dependent continuations/feedback are freshly collected.
Transient extraction files are read and deleted. No full activation dataset is
retained on disk.

```text
$SPECULATORS_FEEDBACK_ROOT/
  inputs/membership.json
  sources/
  provenance/vllm-extractor/
  training/{regular-from4000,projector-from4000}/
    resolved_experiment.json, study.json, membership.json
    checkpoint.pt, metrics.jsonl, validation.json, result.json
    provenance/{parent/,attempt-*/}
    online_traces/update-*.json
    validation_traces/step-*.json
    checkpoints/step-00000500/{checkpoint.pt,bundle/,...}
    checkpoints/step-00001000/{checkpoint.pt,bundle/,hub.json,...}
```

At 500 additional updates a full local checkpoint and serving bundle are saved.
At 1,000 the final bundle, optimizer/RNG checkpoint and provenance are uploaded to:

- `azeeer/eagle3-online-regular-4k-plus1k-v1`
- `azeeer/eagle3-online-feedback-projector-4k-plus1k-v1`

Online and validation traces are published as W&B artifacts when tracking artifacts
are enabled. They include prompt/source identity, current policy step, proposed and
supervised token IDs, acceptance and the exact feedback inputs. Completed local
results survive a tracking failure and can be republished by rerunning with resume.

## Later serving

The feedback bundle must use original-prompt conditioning plus causal feedback:

```text
--adapter-experiment '{"refresh":"prompt","feedback":true,"scale":2}'
```

The runtime now supports this policy, refreshing factors from the stored original
prompt summary when feedback changes. It rejects prefix-summary refresh for these
continuation bundles to prevent a training/serving mismatch. Apply the repository's
pinned vLLM patch before serving and pass `--provenance-dir`; the existing benchmark
suite owns that provenance automatically. No serving benchmark is launched by
these training commands. Real GPU throughput and correctness remain unmeasured.

## Local verification

CPU checks cover the two 4K configs, source-weight loading, causal feedback from
the current policy, exclusion of invalidated suffixes, fixed prompt conditioning,
projector-only updates, full checkpoint recovery and export, and matching serving
inputs. The broader focused suite passed 130 tests with one CUDA-only test skipped.
The real A100 run remains required to measure fit and online-collection cost.

## Faster execution (2026-10-27)

Execution changes preserve the 4K prompt pool, 1,000 additional updates, 64-token
rollouts, 8-example optimizer batch and complete validation memberships:

- Independent requests retain separate KV caches and feedback. Proposal waves are
  produced before their verifier RPCs are released together, allowing vLLM to batch
  verification on the shared GPU. Gradient accumulation remains in input order and
  parameters change only after the complete optimizer batch.
- Validation uses the logits already computed during collection; it does not run
  a second generator/drafter pass for its loss diagnostic.
- Collection reuses factors when their conditioning input is unchanged. During
  training, adapter derivatives accumulate per distinct conditioning input and
  backpropagate through the generator once per such input. The regular arm therefore
  needs one generator backward per request rather than one per verification round.
  The feedback arm also reuses identical feedback inputs without rounding them.
- Extracted activations are memory-mapped and consumed before their temporary files
  are unlinked; this removes the full-prefix Python bytes/deserialization copy.
  The target still computes and emits full-prefix activations: incremental target
  KV verification is not implemented by this optimization.
- If interrupted before its first checkpoint, a resumed run can reuse its saved
  matching step-0 validation. It still restarts updates from the parent when no
  optimizer checkpoint was saved.

Resource settings are execution options, recorded in `execution.json` and each
attempt's provenance. They do not change the checkpoint's experiment identity, so
concurrency/RAM settings can be changed while resuming the same experiment. Automatic
concurrency is 8 with at least 60 GiB GPU memory and 2 otherwise; the default RAM
ceiling respects detected cgroup limits. Set the flags explicitly when desired.

On an 80 GB GPU, for each arm:

```bash
speculators generator train-feedback \
  --config examples/eagle3-lora/feedback_continuation_v1/regular.yaml \
  --rollout-concurrency 8 --ram-cache-gib 48 --resume
```

```bash
speculators generator train-feedback \
  --config examples/eagle3-lora/feedback_continuation_v1/projector_only.yaml \
  --rollout-concurrency 8 --ram-cache-gib 48 --resume
```

On the observed 40 GB GPU, use `--rollout-concurrency 2`, or omit the concurrency
flag for automatic selection. The extractor still needs its existing supported
configuration; concurrency cannot provide memory that the GPU does not have.

New performance panels separate collection wall seconds, backward wall seconds,
generator call count, verifier RPC seconds and activation-file mapping/validation
seconds. RPC seconds are summed across overlapping calls and must not be added to
wall seconds. Memory-map page faults and subsequent GPU copies can occur later and
are included in collection wall time rather than fully represented by the file
opening timer. `update_seconds` remains the total measured update wall time.

CPU checks establish serial/parallel output and feedback parity, actual overlapping
verification RPCs, gradient equivalence, validation-logit reuse and resume with
changed resource flags. The focused suite passed 136 tests with one CUDA-only test
skipped. These are correctness/work-reduction checks; real GPU speedup remains to
be measured from the new phase timings. No A100 speedup factor is claimed here.
