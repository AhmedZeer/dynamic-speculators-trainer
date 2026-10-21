# Full-data generator training with epoch checkpoints on Hugging Face

This trains a fresh generator on every training subset in the local pinned
trajectory release. The current release has six subsets per domain across
math/code/stem/chat: **24,000 training examples**. Validation rows are excluded
from optimization. Exact epoch scheduling visits every training row once per
epoch, shuffles between epochs, forms same-subset episodes of 1–8 examples, and
accumulates four episodes per optimizer update (possibly fewer at an epoch end).
This is different from the campaign's fixed-step sampling with replacement.

Stop inference serving and start the training activation server as described in
the [online runbook](eagle3_generator_campaign_online.md). Existing token
trajectories and frozen target/drafter files must already be downloaded. The
trainer generates full activations on demand with the existing bounded RAM cache;
it does not download the activation dump.

## Command

From the repository root:

```bash
hf auth login
speculators generator train-full-data \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --output-root output/generator-all24k-10epochs \
  --epochs 10 --save-every-epochs 2 \
  --hub-repo azeeer/eagle3-generator-all24k-10ep-v1 \
  --vllm-provenance-dir output/generator-direct-campaign-v3/provenance/vllm
```

Use the provenance directory of the extraction server actually running. Add
`--private` for a new private Hub repository. A write-capable Hub login is
checked before training. To inspect the data membership, actual number of
examples, optimizer updates and epoch boundaries without contacting the Hub or
running model inference, add `--dry-run`. It still reads local trajectory inputs.

The config supplies architecture, optimizer, learning-rate warmup, episode sizes,
activation service, cache settings, and W&B project. The total optimizer-update
count is derived from exact epoch membership rather than the config's old
1,500-update budget. Ten epochs on the current release total **240,000 training
example presentations**. Context size and final partial episodes make the number
of optimizer updates per epoch vary slightly. `train/epoch` records examples
seen divided by training pool size.

For seed 42, context sizes 1–8, four episodes/update, and 24 subsets of 1,000
training examples, the 10-epoch plan is **13,369 optimizer updates**. Hub saves
occur at updates 2,676, 5,349, 8,015, 10,681, and 13,369. The command computes
these boundaries from the actual local inputs; do not reuse these numbers for
a different dataset or schedule.

Context-4 validation runs initially and at epochs 2, 4, 6, 8, and 10, using the
existing held-out `00000` validation rows in each domain. It does not run a
vLLM inference benchmark. Local recovery checkpoints also follow the config's
optimizer-step checkpoint interval; these intermediate recovery saves are not
separate Hub snapshots.

## Remote checkpoints

At epochs 2, 4, 6, 8, and 10 the training loop saves a checkpoint, then waits for
a successful Hub commit before proceeding. Each snapshot is a separate folder:

```text
azeeer/eagle3-generator-all24k-10ep-v1/
  epoch-0002/
  epoch-0004/
  epoch-0006/
  epoch-0008/
  epoch-0010/
```

Each folder contains `bundle.json`, `model.safetensors`, full `checkpoint.pt`
(including optimizer, progress and RNG state), training configuration and
identity, metrics/timing, commands, available patches and extraction provenance,
and checksums. Its `result.json` explicitly describes an `epoch_snapshot` and
records completed epochs/updates; it is not the final whole-run completion marker.
The folder can be passed to `launch_vllm.py serve --adapter-bundle` later.

An upload failure stops training with the local checkpoint intact. Rerunning
the same training command resumes and retries a pending boundary upload first.
Successful commits are recorded locally in
`output/generator-all24k-10epochs/pretraining/all-data-10epochs/hub_checkpoints.json`.
The latest local checkpoint is retained; older requested snapshots persist in
their Hub folders. No local checkpoint is deleted before its upload is confirmed.

## Download for inference or recovery

Download only the desired snapshot rather than every multi-GB optimizer state:

```bash
hf download azeeer/eagle3-generator-all24k-10ep-v1 \
  --include 'epoch-0010/*' --local-dir ./bundles/generator-all24k
```

Add `--revision COMMIT_SHA` to pin the commit recorded by the upload. The serving
bundle directory is `./bundles/generator-all24k/epoch-0010`. For inference-only
downloads, request just `epoch-0010/bundle.json`, `epoch-0010/model.safetensors`,
and desired provenance files explicitly; the optimizer checkpoint is not needed
by the server.

For recovery after losing local storage, download the last uploaded epoch
(for example `epoch-0008`) and copy the raw training files into the original run
directory. **Do not copy the snapshot's `result.json` into the active run**:
the existing runner uses that filename as a whole-run completion marker.

```bash
hf download azeeer/eagle3-generator-all24k-10ep-v1 \
  --include 'epoch-0008/*' --local-dir ./recovered
mkdir -p output/generator-all24k-10epochs/pretraining/all-data-10epochs
cp -a ./recovered/epoch-0008/{checkpoint.pt,resolved_experiment.json,configuration.json,train_command.txt,speculators.patch,metrics.json,timing.json} \
  output/generator-all24k-10epochs/pretraining/all-data-10epochs/
```

Restart the training activation server with the same pinned models and compatible
settings, then rerun the original training command. Validation/data identities,
training settings, and epoch plan must match. Do not delete the local run until
the relevant remote commit is confirmed and its artifacts are downloadable.
