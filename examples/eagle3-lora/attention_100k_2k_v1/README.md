# Attention generator: 100K examples, 2,000 updates

Train a **fresh Q/K/V/O generator** using the expanded Nemotron release
`r0002-100k`, with 25K training examples per category and separate validation.
The architecture, rank/alpha/dropout, learning rate, loss, context-size sampling,
BF16 execution, and four cohorts/update match the attention 24K run. The frozen
RedHat EAGLE3 drafter is retained; the 24K generator checkpoint is not loaded.

## Single-command launch

For a separate fresh 10,000-step run using the same 100K training pool and
training architecture, select the 10K configuration. This variant now uses
**seven TTT prediction steps** (`training_recipe_ttt7.yaml`); the 2K variant
continues to use three. It starts a fresh TTT7 experiment and requires a separate
output root from any TTT3 run:

```bash
CUDA_VISIBLE_DEVICES=0 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh \
  --variant 10k --output-root /path/to/attention-100k-10k-v1-ttt7
```

Omit `--output-root` to use `output/dyn-spec-attention-100k-10k-v1-ttt7`.
The default destination is `azeeer/eagle3-generator-attention-100k-10k-v1-ttt7`;
use `--hub-repo` to change it. This starts from a fresh generator, saves and
uploads every 1,000 updates, and retains the same validation and data caches as
the 2K run. It does not resume the 2K checkpoint. Rerun with the same output
root to resume this 10K run. Add `--prepare-only` to check the exact schedule
without starting vLLM; `--pilot` retains its isolated 10-step behavior.
HF snapshots include `step-00001000` and `step-00002000` for comparison with
the earlier TTT3 runs. Seven TTT steps increase training compute and memory;
this change does not automatically change the speculative length used by benchmarks.

### Two GPUs on one node

```bash
CUDA_VISIBLE_DEVICES=2,3 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh \
  --variant 10k --output-root /path/to/attention-100k-10k-v1-ttt7
```

The first listed GPU runs vLLM; the second runs the trainer. Each subprocess
sees only its assigned GPU as `cuda:0`. GPU UUIDs work too. With one listed GPU,
both processes share it. With no visibility list the original inherited-device
behavior is retained. Additional listed GPUs are not used by this launcher.
Assignments are printed and recorded in `launcher_commands.json` alongside
the existing extraction provenance. The processes exchange activation files
through the same node's `/tmp` directory; local NVMe is not required.

For more overlap on two dedicated 96 GB GPUs, an initial tuning experiment is:

```bash
CUDA_VISIBLE_DEVICES=2,3 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh \
  --variant 10k --output-root /path/to/attention-100k-10k-v1-ttt7 \
  --gpu-memory-utilization 0.70 --max-num-batched-tokens 16386 \
  --lookahead-updates 8
```

This lets the extractor batch more tokens and prepare more upcoming updates.
The default client concurrency remains 128 and the buffer remains 256; override
them with `--extraction-concurrency` and `--buffered-examples` if measurements
show extraction is still limiting throughput. More concurrency/buffering can
increase host RAM and temporary-file use; the RAM cache remains capped at
120 GiB. These settings do not change the training cohort, loss or token budget.
This tuning suggestion has not been measured on the cluster.

The same TTT7 output root can adopt a different GPU assignment, extraction queue
settings, and the 1K checkpoint/upload cadence on resume. Stop the existing
launcher cleanly first; resume uses the latest saved local checkpoint. The
10-step pilot still checkpoints/uploads at step 10; the 2K variant retains its
500-step cadence. Initial recovery checkpoints and final saves still apply.

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
  8,193, **max-num-seqs 128**, eager execution, no prefix caching/chunked prefill,
  and **0.40 GPU memory utilization**. Client request concurrency and its bounded
  buffer are configured separately under `online` in the training YAML. The remaining GPU memory is for training.
  This is different from the dedicated response server's 0.95 budget.
- Host activation cache: **120 GiB**, with an 8 GiB host reserve, a 256-example
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

Other launcher options: `--port 8000`, `--gpu-memory-utilization 0.40`, `--max-num-seqs 128`,
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

## Runtime restored to the 24K attention run

Runtime defaults match `attention_24k_6k_v1`:

- vLLM GPU memory utilization **0.40**, active sequences **128**.
- Extraction client concurrency **128**, buffer **256** examples.
- Host activation cache **120 GiB**, reserve **8 GiB**, lookahead **2** updates.
- BF16 generator/drafter, generator batching, **8,192-token** training budget.
- Eager extraction, 8,193 max context/batched tokens, no prefix caching or
  chunked prefill; layers 2/18/33/36.

The 100K training pool, 2,000-update budget, 1K runtime validation set, and
500-update checkpoint/upload cadence stay configured for this experiment.
Already-prepared runs can adopt the restored cache/queue settings using the same
output root and pinned data; completed summary files are reused. Architecture,
loss, learning-rate, and dataset changes still require a new experiment.

```bash
git pull
CUDA_VISIBLE_DEVICES=0 bash examples/eagle3-lora/attention_100k_2k_v1/run.sh
```

Stop any old server through its original session before restarting. This
launcher owns one extractor and one trainer; using a different port does not
isolate GPU memory from other jobs on the same card.

## Benchmark step 1,000 and step 2,000 across all categories

Both published checkpoints in `azeeer/eagle3-generator-attention-100k-2k-v1`
are pinned at revision `cf09002ae6fbca1fe0e9139ab8a0eee3b149a124`, subfolders
`step-00001000` and `step-00002000`. Both inference bundles were verified present.
The suite uses these exact folders; it does not substitute another training step.

Stop the training extractor first. The suite starts and cleans up its own servers:

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/attention_100k_2k_v1/benchmarks/all_categories_c1.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/attention-100k-2k-v1-all-categories-c1 \
  --resume
```

There are **14 measurements**: two checkpoints across Nemotron math, chat, STEM,
and code, plus Red Hat OOD HumanEval, math reasoning, and summarization.
Settings match the full historical comparison suites: concurrency **1**, 50
prompts/category, 256 output tokens, three draft tokens, three warmups, seed 42,
shared/merged adapters, one conditioning example, fixed shared-adapter batching.
The W&B project is `eagle3-generator-structure-bench-v1`, group
`attention-100k-2k-v1-all-categories-c1`.

Rerun the same command/output root with `--resume` after interruptions. Add
`--dry-run` to validate the suite without model downloads or GPU serving.
`summary.md` provides output tokens/s; `summary.json` also retains acceptance,
generator/merge/restore timings, workload hashes, provenance references, and
correctness status for extending the full seven-category table. Compare workload
hashes and preserve correctness/repeatability limitations when adding results.
Configuration/dry-run validation was performed locally; GPU benchmark results
have not been measured here.

## Supplied serving results

The two-checkpoint, seven-category measurements are archived in
[GEN-013](../../../docs/experiments/results/GEN-013-attention-100k-2k.md), including
the full historical table and acceptance comparisons. Step 2K improves on 1K in
all categories; comparisons with the earlier 24K suite remain observational,
with correctness and runtime/repeat equivalence unchecked.
