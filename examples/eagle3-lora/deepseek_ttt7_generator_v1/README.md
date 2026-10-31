# DeepSeek EAGLE3 ttt7: attention generator, 4K / 1,500 updates

Fresh generator on the frozen `deepseek-ai/eagle3_qwen3_8b_ttt7` drafter,
pinned revision `f6485ba8d21e11942958617dbe7e71b467f38f38`. The external drafter
baseline is 82.50 math / 54.10 chat output tok/s at C1; see
[GEN-010](../../../docs/experiments/results/GEN-010-structure-serving.md).
The subsequently supplied paired serving results are recorded in
[GEN-012](../../../docs/experiments/results/GEN-012-deepseek-generator-4k.md);
training was not executed locally here.

## Settings

- Original 4K saved-response pool: 1K training rows from each
  `math/code/stem/chat-00000`, plus the separate 100-row validation split/domain.
  Same pinned trajectory release and cohort schedule as the earlier attention
  generator: random same-subset cohorts of 1–8, four cohorts/update, seed 42.
- **1,500 optimizer updates**, original three-step `lk_hybrid` teacher-forced
  objective. No response rollouts, replay training or feedback. The source
  checkpoint's `ttt7` training does not change this comparison to seven steps.
- Generate LoRAs for **Q/K/V/O**, rank 32, alpha 64, dropout 0.05. Base target
  and DeepSeek drafter remain frozen; only the fresh generator is trained.
- Same generator architecture as the earlier attention arm: encoder 512,
  example 128, condition 64, descriptor 32, **decoder 512**, two residual blocks.
  BF16 generator training and cohort batching enabled.
- DeepSeek consumes five target decoder outputs: zero-based layers
  `[1,9,17,25,33]`. Extraction/native feature indices are `[2,10,18,26,34]`,
  followed by final verifier state 36 for supervision. Generator input width
  is **5 × 4096 = 20,480**, replacing the older drafter's 12,288-wide context.
  Prompt conditioning remains `last`; the loss/optimizer schedule is unchanged.
- Validate initially, every **250 updates**, and finally. Local recovery saves
  and synchronous HF backups every **500** updates: 500/1,000/1,500.
- HF: `azeeer/eagle3-generator-deepseek-ttt7-attention-4k-1500`.
  W&B project: `eagle3-generator-ablations-v3`; run prefix
  `deepseek-ttt7-generator-v1-`. Default output is Drive:
  `/content/drive/MyDrive/dyn-spec-deepseek-ttt7-generator-v1`.

[train.yaml](train.yaml) is the independent launch config.
[training_recipe.yaml](training_recipe.yaml) describes the actual frozen models
and extraction layers. `trajectory_recipe` separately points at the original
saved-response recipe. Original token membership/model provenance is checked
before constructing new runtime extraction metadata; the source lineage remains
in `trajectory_source` and the metadata hashes. No source files are relabeled or
modified. Reusing these responses is valid only for the same pinned target;
a target/model-revision or dataset mismatch is rejected.

## Prepare the native drafter once

Run from the repository root with the current code installed. Reuse the pinned
Qwen3-8B target and saved trajectories from the
[existing setup](../../../docs/experiments/eagle3_generator_campaign_online.md),
mount Drive, and authenticate with `hf auth login` and `wandb login`.
The previously benchmarked DeepSeek weights are reused from the HF cache.

```bash
python scripts/evaluate/prepare_deepspec_eagle3.py \
  --training --verifier models/qwen3-8b
```

The helper retains the vLLM serving directory and writes native training metadata
under `output/deepseek-eagle3-ttt7-training/<revision>/`. Every published tensor
name/shape is checked against the native Qwen3 EAGLE3 model on the meta device,
including Q/K normalization. Tensor bytes are unchanged and linked to the pinned
cache. Only verifier-owned head/norm tensors are absent from the source and
loaded from the local frozen target at runtime. Source and generated configs
and preparation checksums are retained and copied into training provenance.
Keep the HF cache available while using these linked directories.

## Start the five-feature extractor

Stop any existing extractor using port 8000, then run this in a separate terminal.
Old three-feature activation payloads are incompatible with this model.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py train ./models/qwen3-8b \
  --target-layer-ids 2 10 18 26 34 --include-last-layer \
  --hidden-states-backend file \
  --hidden-states-path /tmp/speculators-deepseek-ttt7-online \
  --provenance-dir output/deepseek-ttt7-generator-v1/provenance/vllm \
  -- --served-model-name Qwen/Qwen3-8B \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 \
  --max-model-len 8193 --max-num-batched-tokens 8193 --max-num-seqs 128 \
  --gpu-memory-utilization 0.40 --enforce-eager \
  --no-enable-prefix-caching --no-enable-chunked-prefill
```

Concurrency 128 controls extraction requests, not optimizer batch size. The
120-GiB activation cache retains the earlier host headroom limit; five features
plus the verifier state increase bytes/token relative to four-state extraction.
It remains bounded, so fewer examples may stay cached. The shared directory,
provenance path and configured layer list must match this command.

## Train

```bash
CUDA_VISIBLE_DEVICES=0 speculators generator train-structure \
  --config examples/eagle3-lora/deepseek_ttt7_generator_v1/train.yaml
```

Append `--dry-run` to inspect the 4K/1,500-update plan without GPU/Hub work.
Rerun the identical command to resume interrupted training or retry a failed
checkpoint upload. Preserve the output and provenance directories. Run files
are under `<output-root>/pretraining/deepseek-ttt7-attention-4k-1500/`.

## Benchmark the trained generator against its baseline

After HF upload of `step-00001500` succeeds, stop the training extractor and run
this independent suite from the repository root:

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/deepseek_ttt7_generator_v1/benchmarks/math_chat_c1.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/deepseek-ttt7-generator-vs-base-4k-1500-nemotron-c1 \
  --resume
```

The [benchmark config](benchmarks/math_chat_c1.yaml) now measures both the frozen
DeepSeek EAGLE3 ttt7 baseline and the trained step-1500 attention generator, on
**math, STEM, code and chat**, preserving the expanded category list. Concurrency
is 1, with 50 prompts/category, 256 output-token cap, three warmups,
**three draft tokens**, and seed 42: two models × four categories = eight cells.
The generator uses a prompt-conditioned shared/merged adapter with one
conditioning example; the baseline uses the same frozen drafter without it.
The baseline row reuses the pinned, metadata-prepared checkpoint created by the
preparation command above. Both models receive the same category workloads.
The filename retains `math_chat_c1` for compatibility, but covers all four domains.
W&B project remains `eagle3-generator-structure-bench-v1`, group
`deepseek-ttt7-generator-vs-base-4k-1500-nemotron-c1`.

The suite resolves and records an immutable Hub revision from `main`, requires
the exact `step-00001500` folder, and launches the frozen DeepSeek drafter with
the baseline's metadata translation. Earlier snapshots are not substituted.
Use the existing patched vLLM/GuideLLM environment; the suite manages its own
server and records extraction-independent serving/eval provenance. Keep the
new output directory separate from earlier math/chat-only suites: their workload
identity differs now that STEM/code are included. `--resume` preserves
successful cells and retries failures. Add `--dry-run` to inspect the config
without model downloads or serving; it does not verify snapshot availability.

`summary.md` reports throughput; `summary.json` includes acceptance, acceptance
length, generator/merge/restore timings, TTFT/TPOT and workload hashes. Compare
the two freshly measured rows within each category to assess whether generator
training helped. The historical 82.50 math / 54.10 chat baseline remains separate
evidence; check runtime conditions and workload hashes before historical
comparisons. Correctness remains unchecked by this timing suite. The supplied
paired results are now archived in GEN-012.

## Validation status

The actual published tensor header matches all native model keys/shapes; only
the two verifier-owned tensors are absent. CPU checks cover metadata conversion,
weight-byte preservation, original cohort settings, real saved-response loading
with separate source/runtime identities, target mismatch rejection, and reuse
of the measured baseline's vLLM metadata translation for generator serving.
These checks are not a full-size GPU training or generated-adapter throughput
measurement. The combined suite retains three draft tokens and shared C1
workloads for the baseline and generator across all four domains.


The user supplied all eight paired measurements: the generator improves measured
throughput by 6.56–16.08% and draft acceptance by 7.71–12.92 percentage points
across the four domains. Use the paired baseline (86.58 math / 56.68 chat tok/s)
when calculating gains; its earlier standalone timings are separate evidence.
Executed training/checkpoint/preparation provenance and numerical parity remain
unverified from the supplied aggregate summary.
