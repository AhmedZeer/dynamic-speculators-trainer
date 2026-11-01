# Attention generator: 100K examples, 2,000 updates

Train a **fresh Q/K/V/O generator** using the expanded Nemotron release
`r0002-100k`, with 25K training examples per category and separate validation.
The architecture, rank/alpha/dropout, learning rate, loss, context-size sampling,
BF16 execution, and four cohorts/update match the attention 24K run. The frozen
RedHat EAGLE3 drafter is retained; the 24K generator checkpoint is not loaded.

## Single-command launch

Finish the two generation workers and publish the complete trajectory dataset
first, once on either generation machine:

```bash
bash examples/eagle3-lora/nemotron_100k_v1/run.sh finalize
bash examples/eagle3-lora/nemotron_100k_v1/run.sh data
```

On the training machine, use the existing working trainer/vLLM environment
(`vllm==0.31.0` and the file hidden-state connector), `hf auth login`, and
`wandb login`. Stop the old response-generation vLLM server first: training owns
an extraction server and shares the GPU with it. Then, from the repository root:

```bash
git pull
CUDA_VISIBLE_DEVICES=0 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh
```

The launcher:

1. Resolves the completed trajectory release to a single immutable Hub commit;
   requires all four `data.json` records, 25 complete subsets/category, and matching
   model/source revisions. An unfinished release fails before GPU startup.
2. Downloads only the release metadata and trajectory Parquets, plus pinned target
   and drafter models. Cached downloads are reused; no activation archive or
   existing generator weights are downloaded.
3. Writes effective configs and a persistent dataset pin, and runs the existing
   full-data dry-run checks against the local dataset: **100K training rows,
   100 subsets, and exactly 2,000 planned optimizer updates**.
4. Starts its own file-backend vLLM extractor, waits for `/health`, then starts
   training with live training output in the terminal. Server startup/runtime
   output is saved to `vllm.log`.
5. Stops its owned trainer/server process groups on completion, Ctrl+C, SIGTERM,
   startup failure, or a training/server error. It does not kill unrelated servers.

Rerun the **same command and output root** to resume local checkpoints and retry
failed model uploads. The first resolved dataset commit is reused even if Hub
`main` changes. Replacing architecture/data/templates or the destination model
repo requires a new output root. An already-prepared run can adopt the 1K
validation cap while retaining its pinned data and cached downloads. If training
checkpoints already exist under the old validation protocol, use a new output
root rather than mixing validation histories. There is no automatic search for checkpoints
from a different machine or output folder.

The checked-in `training_recipe.yaml` contains an all-zero revision sentinel.
The launcher replaces it in the effective recipe; do not pass this template
straight to `train-full-data`.

## Settings and locations

- Eligible training pool: **100K**, from all `00000`–`00024` training splits in
  math, code, STEM and chat. Runtime validation: **1K rows, 250 per category**,
  selected deterministically from the existing held-out data. Each of the 25
  subsets contributes 10 held-out examples per category. The other 9K validation
  rows remain in the HF dataset but are not evaluated, cached as validation
  activations, or included in prompt-summary preparation for this run.
- Original epoch schedule (`--epochs 10`), capped at **2,000 optimizer updates**.
  This is a step budget, not a claim that every one of the 100K training rows is
  visited: it may stop before one complete pass over the expanded pool.
- Fresh attention generator: encoder 512, example 128, condition 64, descriptor
  32, decoder 512, two residual blocks. Rank 32, alpha 64, dropout 0.05.
- `lk_hybrid`, three drafting steps, last-prompt conditioning, cohort size 1–8,
  four cohorts/update, LR 1e-4, 100 warmup updates, 8,192-token microbatch budget.
- Validation initially, every **250** updates, and at the end.
- Local checkpoints and synchronous resumable HF uploads every **500** updates:
  `step-00000500`, `step-00001000`, `step-00001500`, `step-00002000`.
- HF model repo: **`azeeer/eagle3-generator-attention-100k-2k-v1`**.
- W&B project: **`eagle3-generator-ablations-v3`**; group
  **`attention-100k-2k-v1`**.
- Default output: `output/dyn-spec-attention-100k-2k-v1`.
  Checkpoints: `pretraining/all-data-generator-2000steps/` beneath that root.
- Dataset/input cache: `output/attention-100k-2k-v1/` on local disk; model weights
  use the normal Hugging Face cache.
- Effective configs, pinned revision, commands and each server attempt:
  `<output-root>/launcher/`. Available launcher configs/plan and server
  command/checksum/patch provenance are included in model snapshots.
- Extraction uses BF16, layers 2/18/33/36, sequence limit 8,193, token budget
  8,193, concurrency 128, eager execution, no prefix caching/chunked prefill,
  and **0.40 GPU memory utilization**. The remaining GPU memory is for training.
  This is different from the dedicated response server's 0.95 budget.
- Host activation cache: **96 GiB**, with an 8 GiB host reserve, a 256-example
  buffer and two updates of lookahead. First-use prompt-summary preparation
  covers the selected dataset and can take substantial time.

Optional persistent output location:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh \
  --output-root /content/drive/MyDrive/dyn-spec-attention-100k-2k-v1
```

Resolve/download/validate without starting vLLM or GPU training:

```bash
bash examples/eagle3-lora/attention_100k_2k_v1/run.sh --prepare-only
```

Other launcher options: `--port 8000`, `--gpu-memory-utilization 0.40`,
`--startup-timeout 1200`, `--hub-repo ...`, and `--trajectories-revision ...`.
An explicit trajectory ref is resolved once on a fresh run; resume must retain
its saved commit. `--help` lists all options. Keep the input/summary/compiler
caches on local SSD even when storing checkpoints on Drive.

Implementation validation uses synthetic release/model-download fixtures,
configuration comparison against the 24K reference, and subprocess lifecycle
checks. No 100K GPU training or performance measurement was executed locally.

## End-to-end pilot: 1,000 training samples, 10 updates

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh --pilot
```

This uses the same extractor, architecture, teacher-forced objective, cohort
batching, optimizer, validation, and HF publication path as the full run. It
selects exactly **1,000 eligible training examples**: 250/category, ten from each
of that category's 25 subsets, deterministically with seed 42. The 1K validation
selection stays unchanged and disjoint. Only these 2,000 examples need prompt
summaries; preparation does not extract summaries for the full 100K pool.
Trajectory Parquets and model downloads reuse the ordinary cache. A first run
still needs the complete release metadata/Parquets locally for membership checks.

The pilot performs **10 optimizer updates**, with a one-update warmup, initial
and final validation, and a local checkpoint/HF upload at step 10. It exercises
the real upload path; it requires HF and W&B authentication. Ten steps do not
constitute a full pass through all 1,000 eligible training samples.

Default isolated destinations:

- Output: `output/dyn-spec-attention-100k-2k-v1-pilot`.
- HF: `azeeer/eagle3-generator-attention-100k-2k-v1-pilot`, snapshot `step-00000010`.
- W&B group: `attention-100k-2k-v1-pilot` in `eagle3-generator-ablations-v3`.
- Pilot input metadata and summary-cache identity are separate from the full run.

Rerun `--pilot` to resume the pilot. Omit it for the fresh full 100K/2,000-step
run. Pilot and full modes reject reuse of each other's output directory; the full
run does not load pilot generator weights. Explicit `--output-root` / `--hub-repo`
overrides are supported. Add `--prepare-only` to inspect the 1,000-row/10-step
pilot plan without starting vLLM or training.
