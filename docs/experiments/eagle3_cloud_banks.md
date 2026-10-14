# Collect and publish LoRA banks on cloud scratch

The environment and Hugging Face write authentication must already be configured.
Cloud configs restore the existing public `r0001` trajectories and activation
indexes from pinned trajectories commit
`058f01d1f0585e98327b2c190a65c796da04fdd4`. They do not regenerate target responses,
extract activations, or compact raw files.

Run one domain:

```bash
speculators lora-bank train-cloud \
  --config examples/eagle3-lora/bank_stem_cloud.yaml \
  --download-workers 8
```

Or collect all four domains sequentially:

```bash
for domain in math code stem chat; do
  speculators lora-bank train-cloud \
    --config "examples/eagle3-lora/bank_${domain}_cloud.yaml" \
    --download-workers 8 || break
done
```

Configs put scratch artifacts in `/tmp/speculators-cloud/b0002/<domain>`. If the cloud
provider mounts scratch elsewhere, add `--scratch-root /your/scratch/eagle3` to
each command. This changes the bank output and restored selection folders;
model downloads still use the environment's Hugging Face cache location.

To collect just a chosen subset, add `--subset stem-00000`; repeat the flag for
multiple subsets. Without subset flags, all five bank-role subsets are selected.
Reserved adaptation subset `00005` is excluded. The catalog's assigned training
seed takes precedence over the fallback config seed list.

## Cloud versus regular configs

Regular `bank_<domain>.yaml` configs prepare new data in producer mode under
`out/data/eagle3`. Cloud configs consume the existing public data at a pinned
trajectories SHA and use `/tmp` scratch. They also name the public bank repo and
bank release explicitly, and reserve 12 GiB during staging instead of 8 GiB.
Model identities, layers, rank/alpha, loss and learning rates are the same.
The current STEM cloud override has one collection epoch; regular STEM and the
other cloud configs have four. Cloud mode does not inherently change the recipe.

Configure the adapter count with:

```yaml
training:
  loras_per_epoch: 10
```

Ten means exactly ten saved adapters in each collection epoch; there are no
warmup adapters. With 117 updates, snapshots are at updates
12, 24, 36, 47, 59, 71, 82, 94, 106 and 117. Counts reset each epoch, while
snapshot directory IDs retain the cumulative collection update number.
Requesting more adapters than optimizer updates raises an error.

`b0002` separates this policy from older interval-based banks in `b0001`.
Changed schedules require a new run output directory; the cloud configs now
include the bank release in their scratch path to avoid mixing checkpoints.
Existing `r0001` trajectories and activations remain the inputs.

The fused KL forward and backward kernels now own separate autotuning configs,
fixing TorchDynamo's shared-config mutation assertion. After pulling the fix,
retry with the cloud config. For a failed STEM run staged under the old scratch
path, reuse the downloaded inputs before retrying:

```bash
if [ -d /tmp/speculators-cloud/stem/.cloud-activations/stem-00000 ] &&
   [ ! -e /tmp/speculators-cloud/b0002/stem/.cloud-activations/stem-00000 ]; then
  mkdir -p /tmp/speculators-cloud/b0002/stem/.cloud-activations
  mv /tmp/speculators-cloud/stem/.cloud-activations/stem-00000 \
    /tmp/speculators-cloud/b0002/stem/.cloud-activations/stem-00000
fi
```

Only downloaded inputs move; previous run checkpoints remain in their original
profile. Staging still verifies local file hashes, even when it reports zero
bytes to download.

## What the command does

1. Restore the trajectory dataset, manifest and activation index.
2. Check for matching completed runs in the public LoRA bank repository.
3. Stage the next unpublished subset's activations with parallel, verified
   downloads. Training reads raw files or existing shards directly from scratch.
4. Train each pending seed, using the existing bank recipe: one warmup epoch at
   LR `1e-4`, four collection epochs at LR `1e-6`, rank 32 / alpha 64, KL loss,
   and `training.loras_per_epoch: 10` adapters per collection epoch.
   STEM cloud retains its customized one-collection-epoch setting; other
   cloud domains use four collection epochs.
5. Publish each completed seed in one atomic Hub commit, including adapter
   snapshots, adapter configs, snapshot entries, epoch validation, the final
   recovery states and optimizer/RNG files, training command, source patch,
   resolved settings and any available preparation/server provenance.
6. Once the subset's seeds are published, remove only that subset's downloaded
   activation folder before staging the next subset. LoRA outputs stay locally
   until scratch is discarded. Add `--keep-activations` to retain downloaded inputs.

The command currently uses one training worker and one subset at a time to keep
scratch usage bounded. It checks space before staging; the subset's full inputs,
models and training output must fit. Uploads happen after each completed seed,
not after every checkpoint or at the end of the whole category.

The public dataset repository is
`azeeer/eagle3-qwen3-8b-lora-bank-r32-v1`. Published paths have the form:

```text
releases/b0002/math/math-00000/seed-47/
  completion.json
  runs/math-00000/seed-47/
    snapshots/step-00000010/adapter/adapter_model.safetensors
    snapshots/step-00000010/adapter/adapter_config.json
    snapshots/step-00000010/entry.json
    epochs/...
    recovery/...
    train_command.txt
    speculators.patch
    bank_context.json
    resolved_train.json
  provenance/...                 # When source/server provenance is available
```

Artifacts have SHA-256 sidecars. `completion.json` includes source fingerprints,
subset membership, recipe identity and checksums for the portable file set. It
appears in the same commit as the files, so its presence marks a completed upload.
The final bank repository SHA is printed after publication. Existing paths are
immutable; change `hub.bank_release` from `b0002` for a different training recipe.
Each domain has five catalog seeds/runs, giving five upload commits per domain,
or twenty across all four, excluding repository initialization. Repeating a
successful command makes no upload commits for its already-published runs.

## Restart and consume

Rerun the same `train-cloud` command after a session ends. Published matching runs
are skipped without activation downloads. If scratch still exists, the ordinary
trainer can resume the current run from its local recovery checkpoint. If scratch
is gone, an unpublished run starts again; intermediate optimizer progress is not
uploaded during training. Upload failure leaves local LoRAs and downloaded inputs
in place. Existing training outputs can be published separately:

```bash
speculators lora-bank publish-bank \
  --config examples/eagle3-lora/bank_stem_cloud.yaml \
  --subset stem-00000
```

To restore completed bank runs for generator pretraining, restore the source
manifest first, then pass a pinned **bank repository** SHA to `restore-bank`:

```bash
speculators lora-bank restore \
  --config examples/eagle3-lora/bank_stem_cloud.yaml \
  --revision 058f01d1f0585e98327b2c190a65c796da04fdd4 --stage hidden

speculators lora-bank restore-bank \
  --config examples/eagle3-lora/bank_stem_cloud.yaml \
  --revision BANK_REPOSITORY_COMMIT_SHA
```

`restore-bank` reads without repository creation or write permission and verifies
file checksums. It restores the usual `runs/<subset>/seed-*/snapshots/...` layout,
which the generator already consumes. It downloads only runs completed at the
pinned bank revision; incomplete or absent runs are not fabricated.
