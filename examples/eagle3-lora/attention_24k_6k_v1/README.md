# Attention generator: 24K examples, 6,000 updates

Fresh Q/K/V/O generator training on the same pinned saved-trajectory release
used for the historical 24K V/O generators. Train once, then benchmark the
2,000-, 4,000- and 6,000-update snapshots independently. This is a prepared
experiment; no new training or serving measurements have been collected here.

## Protocol

- **24K training rows:** 1K rows from each subset `00000`–`00005` in each of
  math, code, STEM and chat (6K/domain). Pinned trajectory dataset revision:
  `058f01d1f0585e98327b2c190a65c796da04fdd4`, release `r0001`.
- Original `train-full-data` epoch schedule, seed 42: shuffle every training row
  once per epoch, same-subset cohorts of 1–8 examples, four cohorts/update.
  No fixed-eight/K1 conditioning-size study schedule is substituted. `--epochs 10`
  provides a sufficiently long plan; `--max-updates 6000` stops at exactly 6,000
  optimizer updates. It does not train ten completed epochs.
- Frozen pretrained target and EAGLE3 drafter; a **fresh generator**, not a
  continuation of the historical step-4000 generator. The original saved-response,
  teacher-forced three-step `lk_hybrid` objective and last-prompt conditioning
  are retained. There are no generated-response rollouts, replay training or feedback.
- Q/K/V/O adapters, rank 32, alpha 64, dropout 0.05. Architecture matches the
  earlier attention generator: encoder 512, example 128, condition 64,
  descriptor 32, **decoder 512**, two residual blocks.
- BF16 generator parameters/optimizer state and cohort batching retained from
  the latest training implementation. These numerical/execution settings differ
  from the historical FP32 generator, so this is not bitwise matched execution.
- Context-4 offline validation initially, every **250 updates**, and at the end.
  It uses the held-out `00000` validation split in each domain (400 rows total).
- Local checkpoints and synchronous resumable HF snapshots every **2,000**
  updates: `step-00002000`, `step-00004000`, `step-00006000`.
- HF repo: `azeeer/eagle3-generator-attention-24k-6k-v1`.
  Training W&B project: `eagle3-generator-ablations-v3`;
  group: `attention-24k-6k-v1`.

## Train

Run from the repository root with Drive mounted, the existing trainer/extractor
packages installed, and `hf auth login` / `wandb login` completed. Reuse the
pinned local models and trajectories; see the [download/setup commands](../../../docs/experiments/eagle3_generator_campaign_online.md).
No activation dump is downloaded. Hardware settings retain concurrency 128,
a 256-example buffer, a 120-GiB activation cache and an 8-GiB host reserve.
Concurrency here controls extraction requests; it is not the optimizer batch size.

Start the extraction server in a separate terminal (stop an old serving server
first). Its provenance directory must match the training command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 18 33 --include-last-layer \
  --hidden-states-backend file \
  --hidden-states-path /tmp/speculators-online \
  --provenance-dir output/attention-24k-6k-v1/provenance/vllm \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 128 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

Then train:

```bash
CUDA_VISIBLE_DEVICES=0 speculators generator train-full-data \
  --config examples/eagle3-lora/attention_24k_6k_v1/train.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-attention-24k-6k-v1 \
  --epochs 10 --max-updates 6000 --hub-upload-interval 2000 \
  --hub-repo azeeer/eagle3-generator-attention-24k-6k-v1 \
  --vllm-provenance-dir output/attention-24k-6k-v1/provenance/vllm
```

Append `--dry-run` to inspect local membership and the planned budget without
Hub access, GPU inference or training. Confirm `job.training_examples: 24000`,
24 subsets and `job.updates: 6000`. This requires the local trajectory files.
The checked-in config was schema-checked and its schedule checked using expected
release membership; actual local data/GPU execution has not been checked here.

Rerun the same training command to resume an interruption or retry a failed Hub
upload. Local run files and the upload ledger are under
`<output-root>/pretraining/all-data-generator-6000steps/`. Training pauses for each
upload; keep the local checkpoint and provenance until its commit is acknowledged.
Snapshots include the inference bundle, optimizer recovery checkpoint, checksums,
training/eval configuration and available extraction command/patch provenance.

## Benchmark the three checkpoints

After all three HF snapshots are uploaded, stop the training extractor. The
suite owns its servers and benchmarks **math and chat at concurrency 1**, matching
the latest comparison: 50 prompts/category, 256 output tokens, three draft tokens,
three warmups, seed 42, shared/merged adapters with one conditioning prompt.
There is one row per training snapshot, six measurements total.

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/attention_24k_6k_v1/benchmarks/math_chat_c1.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/attention-24k-6k-v1-math-chat-c1 \
  --resume
```

`--dry-run` validates the suite without downloading weights or starting servers.
The actual suite requires the existing patched vLLM/GuideLLM environment and
pins each model revision in its resolved-model artifacts. It requires the exact
step folders; no shorter checkpoint is substituted. Reuse the output directory
and `--resume` to retain completed measurements. `summary.md` gives throughput;
`summary.json` also records acceptance, generator/merge/restore timings, TTFT,
TPOT, workload identity and request counts. Runtime launch/eval provenance is
saved by the managed suite. Correctness is not checked by this timing suite.

Compare all three snapshots under one hardware/runtime setup. Match workload
hashes to GEN-010 before extending its table. Training data now matches the
historical 24K pool, but data matching does not establish equal serving runtime
or eliminate uncertainty from single measurements. Keep the 4K rows labeled
separately and retain the historical rows' evidence limits.
