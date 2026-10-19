# Direct generator training with online activations

Run from the repository root on one A100 80 GB. The trainer replays saved prompt-and-response token IDs through
an external vLLM extractor. It never downloads activation dumps or restores a
PEFT directory tree. No speculative decoding benchmark is run.

For the measured cache results, memory sizing, eviction policy, and instructions
for reusing the cache in other workflows, see
[Reusing online activations in CPU RAM](eagle3_activation_ram_cache.md).

Install in the vLLM environment (and install Speculators in the trainer
environment too, if separate):

```bash
python -m pip install -e ./hs_connectors
python -m pip install -e '.[generator]' 'vllm==0.31.0'
wandb login
```

Download once outside training; omit commands for inputs already present:

```bash
hf download azeeer/eagle3-qwen3-8b-trajectories-v1 \
  releases/r0001/math/data.json releases/r0001/code/data.json \
  releases/r0001/stem/data.json releases/r0001/chat/data.json \
  --revision 058f01d1f0585e98327b2c190a65c796da04fdd4 \
  --local-dir ./trajectories --repo-type dataset
hf download azeeer/eagle3-qwen3-8b-trajectories-v1 \
  --revision 058f01d1f0585e98327b2c190a65c796da04fdd4 \
  --local-dir ./trajectories --repo-type dataset \
  --include 'releases/r0001/*/*/train.parquet' \
    'releases/r0001/*/*/validation.parquet'
hf download Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --local-dir ./models/qwen3-8b
hf download RedHatAI/Qwen3-8B-speculator.eagle3 \
  --revision 08610ffa01dd9f16731fe8f627b85905b6aa51c4 \
  --local-dir ./models/qwen3-8b-eagle3
```

The four `data.json` files are required metadata, alongside the subset folders.
If the trajectory directory already contains the Parquet files but training
reports a missing `data.json`, run only the first trajectories command above:
it downloads four small metadata files without downloading Parquet or activations.

Start extraction in another terminal on the same machine, GPU, and filesystem:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 18 33 --include-last-layer \
  --hidden-states-backend file \
  --hidden-states-path /tmp/speculators-online \
  --provenance-dir output/generator-direct-campaign-v3/provenance/vllm \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 16 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

The 40% vLLM budget leaves approximately 48 GB for training and other GPU
allocations. Actual fit and throughput require the pilot on the real GPU.
Both processes must see `/tmp/speculators-online` at the same absolute path.
The model alias and layer IDs must match the command above. Keep local model
directories at the downloaded pinned revisions.

After the server is ready, run the pilot in another terminal:

```bash
CUDA_VISIBLE_DEVICES=0 speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --trajectories-path ./trajectories --stage pilot
```

Then run the complete campaign:

```bash
CUDA_VISIBLE_DEVICES=0 speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --trajectories-path ./trajectories --stage all
```

Completed jobs, including the pilot, are reused; interrupted jobs resume from
checkpoints. Alternatively use `--stage pretraining`, then `--stage adaptation`.
`--stage prepare` checks local inputs without contacting vLLM or allocating GPU
memory. Missing files fail locally. `--vllm-endpoint` overrides the YAML endpoint;
model paths, shared directory, concurrency, and buffer size are in `online`.
Update budgets and the output path may change while reusing compatible checkpoints.
The configured update count is the total target, including already completed
updates; it is not an additional budget. The runner preserves saved checkpoint
identities and archives the effective configuration under `provenance/resume-*`.
Completed jobs remain completed; changing a budget does not restart them. A
checkpoint already beyond a reduced target is retained without rolling back
updates. An output-path change uses checkpoints already present at that path;
it does not search other folders or move checkpoints. Use a new `output_root`
for changes to data, architecture, loss, learning rate, or other training settings.

The pilot runs 100 updates each for a fresh generator using `lk_hybrid` and a
fresh ordinary LoRA, including context-4 offline validation. Its first update exercises long trajectories from every category.
It passes when both jobs complete; the old activation-download wait-fraction
threshold does not apply. Inspect `output/generator-direct-campaign-v3/pilot/report.json`
and per-run `timing.json` for extraction requests/seconds/tokens, data wait, compute
time, process RSS, GPU allocations, whole-device GPU memory, disk IO, temporary
file high-water mark, free disk, and checkpoint sizes. These also go to W&B.
Extraction seconds sum concurrent requests, so can exceed elapsed wall time.

Console validation progress prints completed contexts and measured elapsed seconds.
Each validation pass logs `validation_performance/pass_seconds`, cumulative
validation seconds, and examples/s to W&B. Per-subset metrics and measured seconds
are collected in one `validation_details/subsets` table. Training logs each
reported update's wall seconds, cumulative training seconds, and
`run_summary/session_elapsed_seconds`. `result.json`
records `training_seconds`, `validation_seconds`, and `data_wait_seconds`; W&B
summaries use the same names under `run_summary/`. These totals include completed updates
and passes restored from checkpoints; `elapsed_seconds` measures only the current
job invocation, including model loading, initial validation, tracker startup, and
checkpoint writes, but excluding final reporting and process cleanup.
`setup_seconds` separately records preparation/model loading before the training
routine. All durations are measured with a monotonic clock, with CUDA synchronized at phase
boundaries; no remaining-time estimate is substituted for a measurement.

Before initial validation/training, generator arms prepare prompt summaries for
all training and validation rows in the job's selected subsets. The pilot also
prepares whole selected subsets rather than only the examples sampled in 100
updates. Progress and `preparation_performance/seconds` show this first-use work.
`result.json` records `preparation_seconds` and counts of generated, imported,
loaded, and in-memory examples. W&B includes `run_summary/preparation_seconds`.
This cost is included in session wall time, separate from optimizer-update and
validation time. Ordinary-LoRA arms do not need this preparation.

Per-subset summary matrices are saved under `online.summary_cache_dir` (default
`output/generator-summary-cache`). Their identity includes source data, frozen
model revisions/IDs, layers, dtype, and conditioning mode. Complete shards
survive restarts and W&B project changes; old `/tmp` per-example summary files
are imported when available. Keep this directory alongside checkpoints when
moving machines. An interrupted shard is rebuilt; completed shards are reused.
Training gathers vectors directly from RAM without repeated prompt hashing or
Arrow-row reads.

Compiler artifacts use `online.compiler_cache_dir` (default
`output/generator-compiler-cache`), with Triton artifacts in its `triton` folder.
The configured path overrides PyTorch's import-time temporary default. Set
`compiler_cache_dir: null` to leave existing environment/default settings alone.
The effective path is printed and recorded in results. Retain the
directory to reuse compatible artifacts; the first invocation can still pay
compilation/autotuning costs.

The main campaign has seven runs, all using `lk_hybrid`:

- `scale-1-direct`, `scale-2-direct`, and `scale-4-direct` pretrain fresh generators
  on 1/2/4 subsets per category (4K/8K/16K training examples), respectively.
  The example configuration uses 1500 optimizer updates for each run.
- `fresh_generator`, `direct_1`, `direct_2`, and `fresh_lora` each train for 1000
  updates jointly on subsets 00002–00005 across math/code/stem/chat (16K examples).
  `direct_1` and `direct_2` initialize from completed 4K and 8K direct pretraining
  checkpoints; optimizer state starts fresh for adaptation. Their pretraining data
  does not overlap the adaptation data. The 16K pretraining checkpoint is used only
  for the scaling comparison because its data overlaps adaptation.

Training context sizes are 1–8; offline validation uses only context 4. Curves
report `full_acc_0` offline agreement, not speculative decoding acceptance.
Model, activation and adapter settings live in `generator_training_recipe.yaml`.
No bank download, catalog, saved LoRA snapshots, or reconstruction objective is used.
The generated and ordinary adapters retain rank 32, alpha 64, and `o_proj`/`v_proj`
targets. AdamW uses learning rate 1e-4, weight decay 0.01, clipping 0.85, and 100
learning-rate warmup updates; each optimizer update has four episodes and an 8192
microbatch token budget. The drafter and target remain frozen except for adapters.

Direct training and validation request full saved trajectories and reuse those
states for prompt conditioning. Defaults are sixteen
concurrent extraction requests and sixty-four buffered examples. At 8192 tokens,
4 layers, and 4096 BF16 features, a full example is approximately 256 MiB;
sixty-four examples contain about 16 GiB of tensor payloads, plus loading/collation
overhead. In addition, `online.ram_cache_gib: 96` retains up to 96 GiB of full
CPU activation tensors across sequential jobs in one campaign process. Validation
entries are prioritized; training entries use the remaining space with LRU eviction.
Prompt summaries are retained independently, outside this full-activation budget.
Shared files are still deleted after reading; cache misses regenerate full states.
Restarting the CLI clears full activations in RAM, while disk prompt summaries
remain reusable. Set `ram_cache_gib: 0` to disable the full-activation RAM cache.
Allow extra RAM for summaries, the sixty-four-example buffer, collation, models, and
checkpoint serialization. Models, local trajectories, summaries,
checkpoints, and optimizer states need additional persistent disk space.

Lookahead covers the current update and the next `online.lookahead_updates`
updates (default 2). Only sixty-four requests/results are queued at a time;
consuming a result refills the queue while the trainer computes. Episode
transitions retain useful future requests. Sampling and optimizer-update order
are unchanged. Validation/checkpoint boundaries drain or cancel pending requests;
completed activations remain available through the RAM cache.

The default W&B project is `eagle3-generator-ablations-v3`, with run groups
`pilot`, `experiment-1`, and `experiment-2`. Metric prefixes create separate
sections: `validation_scores`, `validation_losses`, `training_losses`,
`training_performance`, `validation_performance`, `preparation_performance`, `cache_performance`,
`resources`, `optimization`, and `run_summary`. Memory/storage charts use GiB;
durations use measured seconds. Conditional-accuracy diagnostics remain logged
but are hidden from automatic panels. Add them manually when needed.
The validation-detail table replaces per-subset scalar chart proliferation.
Agreement is plotted against optimizer updates and measured cumulative training
seconds. Aggregate comparison images use distinct colors by scale; adaptation arms have
four distinct colors.

This design uses a new output root, `output/generator-direct-campaign-v3`, so
previous bank campaigns/checkpoints remain untouched. It does not automatically
import old runs. Existing persistent prompt-summary and compiler caches remain
reusable because their identities do not depend on bank catalogs. Subsequent
invocations resume compatible direct-only runs normally.

Preserve `provenance/vllm` alongside the training provenance, configurations,
data identities, patches, and checkpoint checksums when sharing results.
