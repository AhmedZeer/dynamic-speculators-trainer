# BATCH-002 — Matched 24K, 1K-update conditioning-size study

Status: configs and implementation prepared; full GPU runs have not been executed
by the agent. Seed 42, one training run per arm and one ID benchmark pass.

The owner selected **four separately trained generators**, rather than four
inference settings applied to one checkpoint. Each generator is trained and
served with K=1/2/4/8 conditioning summaries. One separately trained static LoRA
provides a matched-data/update baseline. No bank inputs or activation downloads
are used. The trainer replays existing token trajectories through an online
vLLM activation extractor.

## Controlled setup

| Arm | Config under `examples/eagle3-lora/` | Conditioning summaries K | Loss examples per episode | Optimizer updates |
| --- | --- | ---: | ---: | ---: |
| Generator K1 | `generator_conditioning_24k_1k_k1.yaml` | 1 | 8 | 1,000 |
| Generator K2 | `generator_conditioning_24k_1k_k2.yaml` | 2 | 8 | 1,000 |
| Generator K4 | `generator_conditioning_24k_1k_k4.yaml` | 4 | 8 | 1,000 |
| Generator K8 | `generator_conditioning_24k_1k_k8.yaml` | 8 | 8 | 1,000 |
| Static LoRA | `lora_conditioning_24k_1k.yaml` | not used | 8 | 1,000 |

All five arms share the 24K pool (`00000`–`00005`, math/code/STEM/chat), epoch
permutations, four loss episodes per update, seed, frozen models, rank-32/alpha-64
V/O adapters, `lk_hybrid`, AdamW LR 1e-4, weight decay 0.01, clipping 0.85,
100-update LR warmup, and 8,192-token microbatch budget. Generator architectures
and initial weights match across K. K examples are selected reproducibly from
each 8-example loss cohort; **all eight contribute to the loss**, so K does not
change the query/data/token budget.

On the current 24 × 1,000-example release, an epoch is 750 updates. Planning two
epochs and applying the 1,000-update cap gives each arm exactly 32,000 example
presentations: one complete pass over 24K and 8K presentations from the next
shuffle. `max_training_updates: 1000` in each config supplies the cap; explicit
`--max-updates 1000` below also makes it visible in the command. The epoch count
is a scheduling horizon, not a request to finish two epochs.

Local recovery checkpoints are saved every 250 updates. Context-4 offline
validation runs at 0/500/1,000. For that validation, conditioning uses min(K, 4)
summaries from the four-example evaluation cohort; it is a progress check, not
the final B=8 serving comparison. Hub backup occurs at update 1,000 and includes
weights, optimizer/progress/RNG state, inference bundle, and provenance.

## Start the activation extractor

Use the existing locally downloaded pinned model and trajectory inputs from
the [online runbook](eagle3_generator_campaign_online.md). Stop the previous
inference benchmark server before using its GPU/port for training. Run this
extractor in a separate terminal/process:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 18 33 --include-last-layer \
  --hidden-states-backend file \
  --hidden-states-path /tmp/speculators-online \
  --provenance-dir output/conditioning-24k-1k-extractor \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 16 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

If reusing an already running extractor, supply its actual provenance directory
in the training commands instead. The shared activation-file path must match
the config. Existing RAM limits remain 96 GiB with 16 extraction requests in
flight and a 64-example buffer, intended for A100 80 GB and the existing host RAM.

## Train four generators, sequentially

Run from the repository root with a write-capable Hugging Face login. The Drive
output paths preserve intermediate recovery checkpoints on Colab. This loop
stops if an arm fails; repeat it to reuse completed arms and resume interrupted
ones from their latest local checkpoint. Hub repo names match the benchmark YAML.

```bash
set -e
for k in 1 2 4 8; do
  speculators generator train-full-data \
    --config "examples/eagle3-lora/generator_conditioning_24k_1k_k${k}.yaml" \
    --arm generator --epochs 2 --max-updates 1000 \
    --output-root "/content/drive/MyDrive/dyn-spec-conditioning-24k-1k/generator-k${k}" \
    --hub-repo "azeeer/eagle3-generator-all24k-k${k}-1k-v1" \
    --vllm-provenance-dir output/conditioning-24k-1k-extractor
done
```

Use `--dry-run` on any command to print membership, schedule, cap and resolved
configuration without GPU training or Hub upload. Local trajectory files must
still be available. The complete budget is 4 × 1,000 generator updates, not
1,000 updates split between the generators.

## Train the matched 24K static LoRA

Keep the same extractor running:

```bash
speculators generator train-full-data \
  --config examples/eagle3-lora/lora_conditioning_24k_1k.yaml \
  --arm lora --epochs 2 --max-updates 1000 \
  --output-root /content/drive/MyDrive/dyn-spec-conditioning-24k-1k/static-lora \
  --hub-repo azeeer/eagle3-lora-all24k-1k-v1 \
  --vllm-provenance-dir output/conditioning-24k-1k-extractor
```

Each repo contains its final bundle under `step-00001000/`. There is no separate
manual export or publish step. Local runs are under `pretraining/`, labeled
`all-data-generator-1000steps-kK` or `all-data-lora-1000steps`.

## Benchmark ID only at fixed serving concurrency B=8

Stop the extraction server first. The suite owns inference-server startup and
cleanup and downloads only the inference bundle/provenance from each new repo,
excluding optimizer checkpoints.

```bash
python scripts/apply_vllm_generator_patch.py --apply
speculators benchmark suite \
  --config examples/eagle3-lora/benchmark_conditioning_24k_1k.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-conditioning-24k-1k/benchmarks
```

Append `--resume` for interrupted benchmark execution. Training recovery uses
the same training command, without an extra resume flag. A final remote snapshot
contains optimizer state for remote recovery too; local Drive checkpoints retain
progress before the first Hub backup.

The suite compares static LoRA and the four separately trained generators on
**Nemotron math/code/STEM/chat only**, 64 held-out prompts per category, 256-token
caps, seed 42 and three warmup requests. At B=8, 64 requests can form eight whole
cohorts without an arithmetic final partial group. Arrival timing or aborts can
still yield smaller groups. All variants reuse the same frozen category export.
No repeated benchmark sweep or Red Hat evaluation is included in this protocol.

The generator receives exactly min(K, actual cohort size) prompt summaries and
produces one shared LoRA for every cohort member. A seeded uniform subset avoids
systematically conditioning on the first/shortest prompt. Actual
`mean_conditioning_size` and `mean_batch_size` are recorded separately, alongside
acceptance, adapter timings and latency. All five models use the same closed
admission policy and B=8; this is not a concurrency sweep.

Read `summary.md` for throughput and `summary.json` for acceptance, timing and
conditioning counters. Generator labels encode training K and serving K. This
study measures their combined policy effect across separately optimized models;
it does not isolate inference K while holding one model's weights fixed. Native
continuous-batching baselines and verified output correctness remain separate
open questions.
