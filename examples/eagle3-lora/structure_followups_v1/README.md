# Matched static-LoRA baselines and compact all-module generator

Three independent training jobs following
[GEN-009](../../../docs/experiments/results/GEN-009-generator-structure.md).
Run each separately from the repository root, with the existing activation
extractor running. Shared data/model downloads, extractor command, provenance,
and checkpoint recovery are in the
[structure runbook](../../../docs/experiments/eagle3_generator_structure_v1.md).

## Protocol and comparisons

All three inherit `../structure_ablations_v1/common.yaml`: 4K training rows
(1K/category), 1,500 optimizer updates, validation every 250, and resumable
Hugging Face snapshots every 500. They use the same seed-42 episode schedule,
four cohorts/update, sizes 1–8, 8,192-token packing budget, rank 32, alpha 64,
dropout 0.05, LK objective, AdamW lr 1e-4, 100 warmup updates, weight decay 0.01,
and clip 0.85. Validation is the separate 100-row split/category at context 4.
The original saved-response training pipeline is retained.

| Config | What is trained | Comparison |
| --- | --- | --- |
| `static_lora/attention.yaml` | One fixed LoRA on q/k/v/o | Existing attention generator, same targets and data/update budget |
| `static_lora/all.yaml` | One fixed LoRA on all seven projections | Existing all-module generator, same targets and data/update budget |
| `generator/all_decoder256.yaml` | All-module generator, decoder_width=256 | Existing all-module generator with decoder_width=512 |

Both static baselines train only the LoRA A/B parameters, with BF16 parameters,
gradients, and AdamW moments. They have no hypernetwork or conditioning input;
the base drafter remains frozen. Their trainable parameter count is inherently
smaller than the generators'. The control matches target modules and the
data/update recipe, not parameter count or compute. Static updates accumulate
the same four cohorts before one optimizer step. The generator batching settings
in the common file are ignored for static arms.

The compact generator uses BF16 and independent-cohort batching. Only its decoder
width changes; encoder width, condition dimension, residual depth, adapter rank,
and the all-module target set retain their existing settings. It starts from
scratch and does not attempt to resize an old generator checkpoint.

All settings inherit the latest common YAML. Numerical execution changes during
older generator runs remain a comparison limitation: retain their provenance
and compare under matched execution settings when possible. No training or
serving results are claimed for these follow-ups yet.

## Commands

Static attention LoRA:

```bash
speculators generator train-structure \
  --config examples/eagle3-lora/structure_followups_v1/static_lora/attention.yaml
```

Static all-module LoRA:

```bash
speculators generator train-structure \
  --config examples/eagle3-lora/structure_followups_v1/static_lora/all.yaml
```

All-module generator with decoder width 256:

```bash
speculators generator train-structure \
  --config examples/eagle3-lora/structure_followups_v1/generator/all_decoder256.yaml
```

Append `--dry-run` to inspect a config without loading data/models or creating
a Hub repository. Rerunning an interrupted command resumes its own checkpoint.
Run one trainer at a time alongside the extractor.

## Output and publishing

W&B project remains **`eagle3-generator-ablations-v3`**. Static arms use group
`structure-v1/baseline`; the compact generator uses `structure-v1/architecture`.
Run names carry `static-lora-attention`, `static-lora-all`, or
`modules-all-decoder256`, plus the 4K/1,500 budget and seed. Each has a new local
directory under `<output_root>/pretraining/<name>/` and its own Hub repository:

- `azeeer/eagle3-static-lora-structure-v1-attention-4k-1500`
- `azeeer/eagle3-static-lora-structure-v1-all-4k-1500`
- `azeeer/eagle3-generator-structure-v1-modules-all-decoder256-4k-1500`

Snapshots are `step-00000500/`, `step-00001000/`, and `step-00001500/`, including
the full optimizer/RNG checkpoint, portable bundle, checksums, and training and
extraction provenance. Static runs publish `static_lora` bundles; the compact
run publishes a `generator` bundle. Existing experiments are not overwritten.

The present vLLM adapter hook still supports only v/o. These attention/all-module
bundles support training, offline validation, and export; their serving support
must be extended before applying the throughput suite to them.
