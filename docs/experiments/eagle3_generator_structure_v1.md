# Generator module and architecture studies v1

Status: implemented and locally tested with tiny models; GPU runs and new
measurements pending. These are eleven independent training jobs, launched
manually using the same CLI. There is no multi-job ablation command.

## Shared baseline and protocol

The historical reference for both families is
[`scale-1-direct-gseed42`, GEN-001](results/GEN-001-direct-data-scaling.md):
4K training rows and 1,500 optimizer updates, with `v_proj`/`o_proj` adapters.
Its default architecture is encoder_hidden=512, example_dim=128,
condition_dim=64, descriptor_dim=32, decoder_width=512, residual_blocks=2.
The new jobs start fresh generators on the same pretrained EAGLE3 drafter.
They do not continue step-4000 weights.

All jobs share:

- Exactly 1K training rows from each `math/code/stem/chat-00000` training split.
  Validation uses the separate 100-row validation split of each same subset:
  4K training rows and 400 validation rows. The loader verifies disjoint IDs.
- Original saved prompt-and-response trajectories, on-demand target activations,
  CPU RAM caching, `last` prompt conditioning, original three-step `lk_hybrid`
  teacher-forced loss. No generated-response rollouts, replay training, or feedback.
- Original `EpisodeSchedule`: four independently sampled cohorts per optimizer
  update, sizes 1–8, sampling with replacement across updates. Every config has
  the same schedule for seed 42. 1,500 updates are a fixed budget, not an epoch count.
- Packed microbatches up to 8,192 tokens. Rank 32, alpha 64 (coefficient 2),
  LoRA dropout 0.05; AdamW lr 1e-4, weight decay 0.01, clip 0.85,
  100 warmup updates; frozen base target and drafter parameters.
- Context-4 validation at updates **0, 250, 500, 750, 1000, 1250, 1500**.
  These metrics are offline agreement/loss, not measured serving acceptance or speed.
- Local and synchronous Hugging Face snapshots at **500, 1000, 1500**.
  Upload failure stops training after the local checkpoint is saved. Rerunning
  the same command restores optimizer/RNG/progress and retries the boundary upload.

GEN-001 has visual evidence and lacks an immutable historical configuration/run
export. This reconstructs its documented protocol; do not claim bitwise matched
historical execution or statistically significant gains from one seed.

## Config organization

All paths below are relative to `examples/eagle3-lora/structure_ablations_v1/`.
`common.yaml` contains shared settings, paths, and the baseline architecture.
Each run inherits it with `extends: ../common.yaml` and overrides one section.
Common-file and recipe references resolve relative to their YAML; runtime paths
resolve from the repository root. Edit shared hardware/storage paths there once.

| Family | Config | Change from shared baseline |
| --- | --- | --- |
| Modules | `modules/mlp.yaml` | up_proj, down_proj, gate_proj |
| Modules | `modules/attention.yaml` | q_proj, k_proj, v_proj, o_proj |
| Modules | `modules/all.yaml` | All seven projections |
| Architecture | `architecture/encoder_hidden-256.yaml` | encoder_hidden=256 |
| Architecture | `architecture/encoder_hidden-1024.yaml` | encoder_hidden=1024 |
| Architecture | `architecture/residual_blocks-1.yaml` | residual_blocks=1 |
| Architecture | `architecture/residual_blocks-4.yaml` | residual_blocks=4 |
| Architecture | `architecture/condition_dim-32.yaml` | condition_dim=32 |
| Architecture | `architecture/condition_dim-128.yaml` | condition_dim=128 |
| Architecture | `architecture/decoder_width-256.yaml` | decoder_width=256 |
| Architecture | `architecture/decoder_width-1024.yaml` | decoder_width=1024 |

Architecture runs change **one variable at a time**, keep `v_proj/o_proj`, and
share the historical baseline. This is not the 16-way architecture factorial.
No extra baseline launch is scheduled among the eleven. Module runs preserve
the baseline architecture, rank, alpha, and training budget; parameter counts
change, so this does not isolate module placement at equal parameter count.

## Setup and extractor

Use the [online campaign setup](eagle3_generator_campaign_online.md) for the
pinned local target/drafter downloads and saved trajectories if not already present.
The loader still needs all existing trajectory shards and metadata locally;
only subset 00000 contributes training/validation examples. It does not download
the 1-TB activation dataset. No `prepare-experiments` invocation is needed.

Run from the repository root. Authenticate with `hf auth login` and `wandb login`.
Default output is Drive: `/content/drive/MyDrive/dyn-spec-structure-v1`.
Ensure Drive is mounted, or change `output_root` in `common.yaml`.

Start one extractor in a separate terminal; keep it running while you launch
the training configs individually:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 18 33 --include-last-layer \
  --hidden-states-backend file \
  --hidden-states-path /tmp/speculators-online \
  --provenance-dir output/structure-ablations-v1/provenance/vllm \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 32 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

Extraction concurrency is 32, with a 64-example buffer. This is the next trial
for the user's RTX PRO 6000, not a measured fit or speed guarantee. The total
per-iteration token budget remains 8,193, so long sequences can limit actual
parallelism even with 32 outstanding requests. Keep GPU headroom for the trainer,
especially with the largest heads. The trainer defaults to a 96-GiB activation
RAM budget plus overhead; adjust this independently to available system RAM.

## Independent training commands

Set this shell variable once, then run whichever individual command you want.
It only abbreviates paths; it does not execute a group of experiments.

```bash
STRUCTURE_CONFIGS=examples/eagle3-lora/structure_ablations_v1
```

Module experiments:

```bash
speculators generator train-structure --config "$STRUCTURE_CONFIGS/modules/mlp.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/modules/attention.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/modules/all.yaml"
```

Architecture experiments:

```bash
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/encoder_hidden-256.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/encoder_hidden-1024.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/residual_blocks-1.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/residual_blocks-4.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/condition_dim-32.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/condition_dim-128.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/decoder_width-256.yaml"
speculators generator train-structure --config "$STRUCTURE_CONFIGS/architecture/decoder_width-1024.yaml"
```

Append `--dry-run` to any command to inspect its resolved configuration/job
without loading models, reading trajectories, creating a Hub repository, or training.
Rerun the exact same command to resume a local checkpoint; resume is automatic.
Use one training process at a time on the GPU alongside the extractor.

## W&B, Hub and recovery

Every run uses project **`eagle3-generator-ablations-v3`**. Groups are
`structure-v1/modules` and `structure-v1/architecture`. Run names begin with
`structure-v1-`, identify the changed setting and `4k-1500`, and end in `-gseed42`.
Stable run IDs are derived from configuration/data identity, so interrupted
launches return to the same W&B destination. Resolved settings, job, metrics,
and command/patch provenance are recorded with each run.

Each config has its own `azeeer/eagle3-generator-structure-v1-<name>` Hub repo.
Each repository receives `step-00000500/`, `step-00001000/`, `step-00001500/`:
portable generator bundle, full `checkpoint.pt` including optimizer/RNG/progress,
resolved experiment/data/model identities, metrics, commands, patches, checksums,
and the extractor provenance. Uploads are synchronous and acknowledged in
`hub_checkpoints.json`; only successfully acknowledged snapshots are durable on Hub.
There is no Hub upload at update zero.

Local run layout:

```text
<output_root>/pretraining/<name>/
  checkpoint.pt
  resolved_experiment.json
  configuration.json
  train_command.txt
  speculators.patch
  wandb.json
  metrics.json
  timing.json
  hub_checkpoints.json
  result.json                 # complete only after the final update/upload
  provenance/                 # retained previous launch attempts
```

After losing the Colab machine, restore the same code, data/models, extractor
paths and config, then download one **acknowledged** snapshot. Example for MLP
at update 1000:

```bash
hf download azeeer/eagle3-generator-structure-v1-modules-mlp-4k-1500 \
  --include 'step-00001000/*' --local-dir /tmp/structure-recovery
mkdir -p /content/drive/MyDrive/dyn-spec-structure-v1/pretraining/modules-mlp-4k-1500
cp -a /tmp/structure-recovery/step-00001000/. \
  /content/drive/MyDrive/dyn-spec-structure-v1/pretraining/modules-mlp-4k-1500/
speculators generator train-structure \
  --config examples/eagle3-lora/structure_ablations_v1/modules/mlp.yaml
```

Use an empty destination for restoration. Snapshot `result.json` is explicitly
marked `training_snapshot`, so it does not masquerade as a completed run.
Identity mismatches still fail; weights alone are not enough for exact resume.
The activation cache can be regenerated. Progress after the last acknowledged
Hub snapshot is lost if local storage is also lost.

## Memory and export limits

Calculated counts for the pinned Qwen3-8B EAGLE3 shape, rank 32, input_dim 12288.
The final column counts only FP32 parameters, gradients and two AdamW moments
(16 bytes/parameter), excluding frozen models, activations, temporaries and vLLM.

| Variant | Generator parameters | Optimizer training storage, GiB |
| --- | ---: | ---: |
| Historical v/o baseline | 293,288,224 | 4.37 |
| MLP | 814,397,760 | 12.14 |
| Attention | 646,297,952 | 9.63 |
| All modules | 1,453,177,280 | 21.65 |
| encoder_hidden=256 | 290,109,216 | 4.32 |
| encoder_hidden=1024 | 299,646,240 | 4.47 |
| residual_blocks=1 | 292,761,888 | 4.36 |
| residual_blocks=4 | 294,340,896 | 4.39 |
| condition_dim=32 | 293,267,712 | 4.37 |
| condition_dim=128 | 293,329,248 | 4.37 |
| decoder_width=256 | 149,860,384 | 2.23 |
| decoder_width=1024 | 581,716,768 | 8.67 |

Hub export and offline generator loading support all seven module names.
The existing vLLM merge/per-request serving hooks currently support only v/o;
they reject these new module sets explicitly. Extending serving hooks for
attention/MLP/all-module bundles is a separate follow-up. Architecture variants
retain the currently supported v/o set.

## Measurement record to add after execution

For each variant, retain W&B URL, uploaded commit/checkpoint hashes, resolved
config/code revision, full provenance, and per-domain losses/agreement at each
250-step validation. Compare against GEN-001 with its evidence limitations;
do not label these pending jobs as observed improvements. Track parameter count,
peak GPU/CPU memory and wall time because the decoder heads dominate model size.
