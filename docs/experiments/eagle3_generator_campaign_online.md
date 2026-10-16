# Local packed bank and online activations

Run from the repository root on one A100 80 GB. The trainer reads packed
Safetensors directly and replays the saved prompt-and-response token IDs through
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
hf download azeeer/eagle3-qwen3-8b-lora-bank-r32-packed-v1 \
  --local-dir ./data --repo-type dataset
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
  --provenance-dir output/generator-campaign-online/provenance/vllm \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 4 \
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
  --data-path ./data --trajectories-path ./trajectories --stage pilot
```

Then run the complete campaign:

```bash
CUDA_VISIBLE_DEVICES=0 speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --data-path ./data --trajectories-path ./trajectories --stage all
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

The pilot runs 100 updates for each of bank reconstruction, a fresh generator
using `lk_hybrid`, and a fresh ordinary LoRA, including context-4 offline
validation. Its first update exercises long trajectories from every category.
It passes when all three jobs complete; the old activation-download wait-fraction
threshold does not apply. Inspect `output/generator-campaign-online/pilot/report.json`
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

The existing design stays the same: six matched bank/direct pretraining jobs
with 1/2/4 subsets per category (4K/8K/16K examples), 10K updates, and identical
episode sampling within each pair. Bank targets are the ten first-collection-epoch
snapshots per subset. Four adaptation jobs compare a fresh generator, bank-2
generator, bank-1 generator, and fresh ordinary LoRA, each trained jointly for
1K updates on subsets 00002–00005 in all categories (16K examples). Training
context sizes are 1–8; offline validation uses only context 4. Curves report
`full_acc_0` offline agreement, not measured speculative decoding acceptance.

Bank reconstruction requests prompt states only, then caches small last-token
conditioning vectors. Direct training and validation request full saved
trajectories and reuse those states for prompt conditioning. Defaults are two
concurrent extraction requests and eight buffered examples. At 8192 tokens,
4 layers, and 4096 BF16 features, a full example is approximately 256 MiB;
eight examples contain about 2 GiB of tensor payloads, plus loading/collation
overhead. In addition, `online.ram_cache_gib: 64` retains up to 64 GiB of full
CPU activation tensors across sequential jobs in one campaign process. Validation
entries are prioritized; training entries use the remaining space with LRU eviction.
Prompt summaries are retained independently, outside this full-activation budget.
Shared files are still deleted after reading; cache misses regenerate full states.
Restarting the CLI clears full activations in RAM, while disk prompt summaries
remain reusable. Set `ram_cache_gib: 0` to disable the full-activation RAM cache.
Allow extra RAM for summaries, the eight-example buffer, collation, models, and
checkpoint serialization. Models, local bank/trajectories, summaries,
checkpoints, and optimizer states need additional persistent disk space.

The default W&B project is `eagle3-generator-ablations-v2`, with run groups
`pilot`, `experiment-1`, and `experiment-2`. Metric prefixes create separate
sections: `validation_scores`, `validation_losses`, `training_losses`,
`training_performance`, `validation_performance`, `cache_performance`,
`resources`, `optimization`, and `run_summary`. Memory/storage charts use GiB;
durations use measured seconds. Conditional-accuracy diagnostics remain logged
but are hidden from automatic panels. Add them manually when needed.
The validation-detail table replaces per-subset scalar chart proliferation.
Agreement is plotted against optimizer updates and measured cumulative training
seconds. Aggregate comparison images retain dark bank/light direct colors by
scale; adaptation arms have four distinct colors.

Run the same `--stage all` command after pulling this change. The RAM-cache
settings trigger a fresh pilot profile. Existing compatible completed training
jobs are reused and their saved validation history is logged to the new project;
interrupted jobs resume from checkpoints. These operational changes do not alter
the training identity or require a new output directory. Legacy configurations
retain their existing logging layout unless `wandb.layout: sections` is selected.

Preserve `provenance/vllm` alongside the training provenance, configurations,
data identities, patches, and checkpoint checksums when sharing results.
