# Prepare and publish the scale-up dataset

This implementation covers selection, target response generation, portable
trajectory publication, hidden-state extraction/publication, and restoration on
another machine. It uses public dataset repositories under `azeeer`, with the
existing preparation commands. The expanded ten-epoch collection schedule,
LoRA-bank publication, append-selection migration, and joint generator training
are subsequent implementation phases; these commands do not launch them.

## Install on the GPU machine

Use local SSD storage for `out/data/eagle3`; edit `output_root` and
`hub.selection_root` in the YAMLs if that directory is unavailable. Model
downloads use the usual Hugging Face cache. A local activation cache is bounded
at 40 GiB with an 8 GiB free-disk reserve; pending extraction and model weights
need additional space. Do not point staging at a network-mounted Drive folder.

```bash
git clone https://github.com/AhmedZeer/dynamic-speculators-trainer.git
cd dynamic-speculators-trainer
python -m pip install -e '.[generator]' 'vllm==0.31.0'
hf auth login
mkdir -p out/data/eagle3
```

For an existing checkout, use `git pull` and reinstall the editable package.
Authenticate as `azeeer` using a token that can create/write dataset repositories.
Accept any source-dataset access conditions on Hugging Face before selection.
The repositories are created by the preparation command when needed. No token
is written to YAML. This preparation path does not need the generator serving
patch or a separately installed Torch version.

## Select math membership

```bash
speculators lora-bank prepare \
  --config examples/eagle3-lora/bank_math.yaml --stage select
```

The first selection streams chat, math, code, and stem on CPU and uniformly
samples source-row occurrences with a bounded reservoir. It does not deduplicate
text, merge aliases, or exclude matching prompts from other domains. It still
scans the eligible source rows to give each row equal selection probability,
and periodically logs progress. Later category selections reuse the frozen
catalog. Model/source
revisions are resolved once and saved with the selection. Source answers are
discarded; source UUID aliases and licenses are retained.

The full math configuration prepares six subsets, each 1,000 train + 100
validation examples: 6,600 math examples. Subset 00005 is marked reserved
adaptation. For a one-subset GPU pilot, use `bank_math_pilot.yaml` consistently
in every command instead; it uses separate selection/release/output paths and
prepares 1,100 math examples. Its first CPU selection still checks all domains.
Pilot selection is not a production append or migration.

The examples use the existing selection `s0001` and release `r0001` so that
published data can be resumed without regenerating it. Existing frozen catalogs
are reused with their exact membership, including catalogs created before the
sampler changed. New selections retain repeated source rows; `prompt_hash` is
metadata, not a filter. A change to the sampler does not change an existing
selection. To deliberately select new membership, choose a new selection ID,
release ID, and local roots.

If you already published responses and trajectories under `r0001`, pull the
updated checkout and rerun `prepare --stage hidden` with the same category config.
A missing local manifest is restored from the published data stage. Previously
sealed extraction batches are recovered from their completion records. You do
not need to rerun selection, responses, or trajectory preparation.

## Generate responses

Start the response server in terminal A:

```bash
speculators lora-bank serve \
  --config examples/eagle3-lora/bank_math.yaml --mode responses
```

Wait until vLLM is ready. In terminal B:

```bash
speculators lora-bank prepare \
  --config examples/eagle3-lora/bank_math.yaml --stage responses
```

`serve` downloads the selected target revision and launches `launch_vllm.py`
with the original model ID as its served name, context length 8,192, and a
provenance directory. Generation uses thinking disabled, temperature 0.7,
top-p 0.8, top-k 20, seed 0, and at most 4,096 output tokens. The full YAML sets
64 concurrent requests; change `execution.concurrency` for the machine.

Responses are generated in chunks of 1,000 selected prompts. Each completed
chunk is uploaded before the next chunk starts, and the final smaller chunk is
also published. An incomplete chunk fails visibly; rerunning reuses its locally
saved successes. Published chunks can be recovered on a replacement machine
without regenerating their responses. Do not run two publishers concurrently
for the same category/release.

## Prepare portable trajectories

```bash
speculators lora-bank prepare \
  --config examples/eagle3-lora/bank_math.yaml --stage data
```

This converts the exact response token sequences into local Arrow rows and
publishes per-subset train/validation Parquet files. The generated dataset card
defines category configurations and explicitly explains the reserved subset
role. Hub consumers reconstruct Arrow locally while preserving the original
row order and trajectory fingerprint.

## Extract hidden states

Stop the response server in terminal A with Ctrl+C. Start the extraction server:

```bash
speculators lora-bank serve \
  --config examples/eagle3-lora/bank_math.yaml --mode hidden
```

Wait until vLLM is ready, then run in terminal B:

```bash
speculators lora-bank prepare \
  --config examples/eagle3-lora/bank_math.yaml --stage hidden
```

The server extracts auxiliary layers `[2,18,33]` plus final layer `36` on the
saved token trajectories. Extraction works in bounded batches of 16 and
publishes approximately 1 GiB Safetensors shards; batches and subset/split
boundaries may make smaller shards. Each shard has stable prompt-derived tensor
keys, a compact raw last-prompt-token summary, and a completion record containing
its exact repository commits and checksums. A final category index is published
only after every expected row is durable.

Completed payloads are removed from local staging after their upload group is published.
An upload failure retains pending extraction files; rerunning retries them.
After machine loss, extraction recovers previously sealed shard records and
generates only missing rows. Unsaved/in-flight work can still be lost; upload
completion is the durability boundary.

Extraction batches remain at 16 examples, but publication groups accumulate
until `hub.upload_batch_examples` (256) or `hub.upload_batch_bytes` (2 GiB) is
reached. The byte threshold is checked after each extraction batch, so a group
can exceed it by one batch. The final smaller group is always published.
Each group uploads all its shards, conditioning summaries, completion records,
and checksum sidecars in a single atomic commit. Provenance files are also
published together in one commit. This reduces commit creation and lets vLLM
process several batches between uploads; uploads still block extraction while
that group is being published.

Local pending files survive a failed upload and are reused on retry. On another
machine, only successful publication groups are recoverable; uncommitted groups
must be extracted again. Old individually published shards/completion records
remain readable. Increasing the thresholds reduces commit frequency but
increases local staging space and the work exposed to machine loss. Existing
server-side rate limits must expire before publishing again; batching does not
reset them.

Nested Hugging Face upload/download progress bars are disabled by default to
keep notebook output responsive. Large uploads print their size; slow uploads
emit a heartbeat every 30 seconds. Small metadata uploads are quiet, while
extraction counters and errors remain visible. Set `hub.progress_bars: true`
to retain the library's progress-bar setting, or adjust
`hub.upload_log_interval` to change the heartbeat interval.

Every successful stage prints `Hub trajectories revision: <COMMIT_SHA>`.
Save the final hidden-stage SHA for consumers. Stage manifests and checksum
sidecars are immutable. Different bytes under a previously published path fail
explicitly rather than overwriting data. `publication.json` records successful
uploads locally, and provenance is published under content-derived paths.

## Restore on a training machine

Install the same checkout, authenticate if needed, and set the YAML's
`output_root` to an available local directory. Then:

```bash
speculators lora-bank restore \
  --config examples/eagle3-lora/bank_math.yaml \
  --revision <FINAL_TRAJECTORIES_COMMIT_SHA> --stage hidden
```

This restores frozen membership, responses, trajectory rows, and the activation
index. Full activation shards download on demand through a bounded shared cache.
Restore never runs a target server or repartitions source prompts. Consumers
must provide a complete commit SHA, not `main`. A missing requested stage fails.
The existing bank reader and generator behavioral reader can consume this index;
fixed `last` conditioning can use the compact summaries directly.

`dataset.source` continues to identify the original prompt dataset. The separate
`hub` section controls publication/restore; changing the source to the prepared
dataset repository would mean something different.

## Prepare the other categories

Use `bank_code.yaml`, `bank_stem.yaml`, and `bank_chat.yaml` with the same four
stages. They share the frozen selection and repositories but have independent
local roots and category paths. Run category publication serially. You can
reuse an already-running server while its pinned target and context settings
match the next category; give each server invocation its own provenance as
shown by the `serve` command.

All four production configs use `out/data/eagle3/<category>-s0001` and the
shared `out/data/eagle3/selection-s0001` registry. Together they prepare 24
subsets (26,400 examples): 20 bank subsets and four reserved adaptation subsets.
Each subset has 1,000 training examples and 100 validation examples. The frozen
catalog assigns one seed per bank subset: chat 42–46, math 47–51, code 52–56,
and STEM 57–61. Subset 00005 of each category is reserved for adaptation and
excluded from bank collection.

With the response server running, select every category and generate its
responses and trajectories in terminal B:

```bash
for domain in math code stem chat; do
  for stage in select responses data; do
    speculators lora-bank prepare \
      --config "examples/eagle3-lora/bank_${domain}.yaml" --stage "$stage" || break 2
  done
done
```

Switch terminal A to the hidden-state server, then extract every category:

```bash
for domain in math code stem chat; do
  speculators lora-bank prepare \
    --config "examples/eagle3-lora/bank_${domain}.yaml" --stage hidden || break
done
```

After extraction finishes, collect LoRAs for all four categories:

```bash
for domain in math code stem chat; do
  speculators lora-bank train \
    --config "examples/eagle3-lora/bank_${domain}.yaml" || break
done
```

This uses the currently implemented schedule: one warmup epoch, four collection
epochs at the lower learning rate, checkpoints every 10 updates, and epoch-end
validation. These are four bank collection commands, not joint generator
training. The existing `generator_math.yaml` still describes the original math
ablation; the all-domain generator training interface remains a subsequent
phase of the scale-up plan.

The old math experiment config is preserved as `bank_math_legacy.yaml`.
The existing generator and vLLM ablation YAMLs reference that legacy config so
they continue to address the original experiment.

## Storage and validation

Public Hub storage remains subject to account capacity. The one-subset pilot
measures actual bytes and upload behavior before committing to the larger
activation corpus. Keep an independent durable copy if account storage cannot
accept all planned shards. Splitting repositories does not increase an account's
quota.

CPU tests exercise public immutable upload, interrupted response/shard upload,
retained pending files, recovery after deleting producer storage, portable
Parquet restoration, and lazy shard reads with a fake Hub and extractor.
These tests do not establish real-model extraction correctness or production
Hub capacity; the GPU pilot establishes those practical limits.

See [the scale-up data contract](eagle3_generator_scaleup.md) for the full
experimental plan and future implementation phases.
