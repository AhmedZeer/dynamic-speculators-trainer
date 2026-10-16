# Local packed bank and online activations

Run from the repository root on one A100 80 GB. The trainer reads packed
Safetensors directly and replays the saved prompt-and-response token IDs through
an external vLLM extractor. It never downloads activation dumps or restores a
PEFT directory tree. No speculative decoding benchmark is run.

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
Use a new `output_root` when changing experiment settings or local data identity.

The pilot runs 100 updates for each of bank reconstruction, a fresh generator
using `lk_hybrid`, and a fresh ordinary LoRA, including context-4 offline
validation. Its first update exercises long trajectories from every category.
It passes when all three jobs complete; the old activation-download wait-fraction
threshold does not apply. Inspect `output/generator-campaign-online/pilot/report.json`
and per-run `timing.json` for extraction requests/seconds/tokens, data wait, compute
time, process RSS, GPU allocations, whole-device GPU memory, disk IO, temporary
file high-water mark, free disk, and checkpoint sizes. These also go to W&B.
Extraction seconds sum concurrent requests, so can exceed elapsed wall time.

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
overhead. Shared activation files are deleted after buffered reading; full states
are regenerated on subsequent visits. Models, local bank/trajectories, summaries,
checkpoints, and optimizer states need additional persistent disk space.

Preserve `provenance/vllm` alongside the training provenance, configurations,
data identities, patches, and checkpoint checksums when sharing results.
