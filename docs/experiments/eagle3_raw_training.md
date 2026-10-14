# Train directly from published activations

For automatic scratch-only collection and publication of LoRA banks, use the
[cloud bank command](eagle3_cloud_banks.md). The manual staging commands below
remain useful when preparing selected inputs for generator training.

Compaction is optional. LoRA-bank training and generator training accept both
per-example raw safetensors and previously packaged shards. Downloaded files retain
their Hub repository paths; the trainer reads them directly without another
activation copy. A configured local activation folder is strict: missing files
raise an error rather than downloading during GPU training. Readers validate file
sizes; the staging command also verifies SHA-256 before training.

## Restore metadata and stage selected subsets

Run from the repository root. The cloud configs use scratch paths for
metadata, datasets and outputs. Restoring hidden metadata does not download every
activation. Staging downloads the exact references from the restored index,
including small summary files when an existing shard needs them. Downloads run
concurrently, defaulting to eight workers; no tensor repackaging or Hub upload runs.

```bash
export HF_HOME=/tmp/speculators-hf
export TMPDIR=/tmp/speculators-temp
mkdir -p "$HF_HOME" "$TMPDIR"

speculators lora-bank restore \
  --config examples/eagle3-lora/bank_stem_cloud.yaml \
  --revision 058f01d1f0585e98327b2c190a65c796da04fdd4 --stage hidden

speculators lora-bank stage-hidden \
  --config examples/eagle3-lora/bank_stem_cloud.yaml \
  --local-dir /tmp/speculators-generator-inputs \
  --subset stem-00000 --workers 8

speculators lora-bank train \
  --config /tmp/speculators-cloud/b0002/stem/raw_training.yaml
```

The generated `raw_training.yaml` selects only the staged bank subsets for LoRA
training. Repeat `--subset` to select several; `--subset all` stages all six subsets,
including the reserved adaptation subset. The command estimates the missing bytes
and checks free space plus the configured reserve before downloading. It retains
all staged files. Run categories sequentially if their combined files exceed disk
capacity; nothing automatically removes a user-owned download folder.

Existing `hf download --local-dir` folders are also supported: set
`execution.activation_local_dir` in a bank config to that folder and
`execution.training_subsets` to the staged subset IDs. The folder must contain
`releases/...` paths matching the restored index. The staging command can reuse and
verify files already downloaded into the same folder.

## Mix subsets in a shared generator

Use [generator_mixed.yaml](../../examples/eagle3-lora/generator_mixed.yaml) for the new
`generator pretrain` command. Its example mixes `math-00000` and `stem-00000`.
First restore and stage math alongside the STEM files above:

```bash
speculators lora-bank restore \
  --config examples/eagle3-lora/bank_math_cloud.yaml \
  --revision 058f01d1f0585e98327b2c190a65c796da04fdd4 --stage hidden

speculators lora-bank stage-hidden \
  --config examples/eagle3-lora/bank_math_cloud.yaml \
  --local-dir /tmp/speculators-generator-inputs \
  --subset math-00000 --workers 8

speculators lora-bank train \
  --config /tmp/speculators-cloud/b0002/math/raw_training.yaml

speculators generator pretrain \
  --config examples/eagle3-lora/generator_mixed.yaml
```

Both selected subsets need their collected LoRA snapshots under their bank roots
before generator pretraining. Existing collected snapshots can be reused there.
Raw target activations alone do not supply the LoRA reconstruction targets.

Each update accumulates four episodes. Each episode samples a subset uniformly,
samples 1–8 training examples from that subset, then samples a seed and checkpoint
from its own LoRA bank. The shared generator minimizes the existing LoRA-factor L1
reconstruction loss. It does not mix unrelated domains in a single context or use
a LoRA from a different subset as the target. The example runs 10,000 updates,
clips gradient norm at 0.85, and evaluates every 250 updates plus the final update.
Validation uses each selected subset's held-out examples at context sizes 1, 4, 8.
Saved metrics include individual subset results and their equal-weight mean;
W&B uses the separate `eagle3-generator-pretraining` project. Training provenance,
source identities and subset membership are recorded with the checkpoint.

To use other subsets, change `pretraining_subsets` to an explicit list and add
their bank YAMLs to `additional_bank_configs`. To use every bank subset in those
configs, set `pretraining_subsets: all`. Subsets marked for adaptation are excluded
and cannot be explicitly selected for bank pretraining. Model revisions, activation
layout and LoRA shapes must be compatible across banks. Dataset metadata is
combined with distinct row offsets; activation payloads are read on demand rather
than concatenated into RAM.

Staging all domains requires enough scratch for all selected activation files. Smaller combinations work with the same machinery. Setting
`activation_local_dir: null` retains the existing bounded Hub cache behavior,
including downloads on cache misses; the parallel staging path avoids those
training-time downloads. The original conditioning, heatmap and adaptation
commands retain their existing single-subset configuration. For later adaptation,
pass the mixed checkpoint explicitly with `--pretrained-checkpoint` and use a
normal adaptation config.
