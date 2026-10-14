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
saved trajectories. Extraction requests one output token, so the hidden-state
server reserves one extra position: 8,193 for saved inputs up to 8,192 tokens.
Keep `responses.context_length` and `hidden_states.sequence_length` at 8,192;
changing those fields changes the prepared-data contract. Response serving
continues to use the configured generation context limit.

The bank extracts states on the
saved token trajectories. The bank calls the original extractor once per
subset. Each subset uses a persistent queue, with concurrency
controlled by `execution.hidden_state_concurrency`. The production config has
six subsets of 1,000 train + 100 validation examples; each queue handles 1,100
examples. There are no per-batch Arrow copies, client restarts, or uploads during
extraction. `hub.extraction_batch_size` is accepted for older YAMLs but ignored.

Raw files are saved under `hidden_states/.raw-states/hs_<global-row>.safetensors`.
Completed local files are trusted for resume without a payload scan. Newly
saved files are not scanned for finite values. Deferred compaction checks token
alignment, shape, and dtype as it reads them. Previous per-batch extraction files are
migrated by their saved primary IDs and token sequences, avoiding regeneration.

After extraction finishes, `hub.activation_format: raw` uploads the original
Safetensors files unchanged. There is no tensor repacking or conditioning-summary
generation on the extraction machine. Checksums and small completion records
still identify durable examples. `hub.upload_batch_examples` (512),
`hub.upload_batch_bytes` (8 GiB), and `hub.upload_max_files` (128) bound commit
groups. Each commit includes the raw files, checksums, and its completion record.
A successful group is recorded locally before its original files are removed.
Upload failures retain those files and metadata for retry. Previously sealed
packages are uploaded as-is rather than unpacked or recomputed.
The next subset starts only after publication and cleanup finish.
The final activation index is published once all rows are durable, followed by
the hidden-stage record and provenance.

**Local storage must fit one subset's remaining activations before publication.** RAM is
small during raw preparation, but the 40 GiB reader-cache limit does not cap raw
extraction storage. `hub.extraction_scope: subset` is the default. Set it to
`all` only when local storage can fit the whole category. On machine loss,
unpublished raw files and packages are lost;
published completion records remain recoverable. Existing individually published
shards also remain usable. A server-side rate limit must expire before another
commit can succeed.

If an older category-wide run filled the disk, stop preparation and the
extraction server, keep the local output directory, and publish its completed
files before requesting more activations:

```bash
speculators lora-bank publish-hidden \
  --config examples/eagle3-lora/bank_code.yaml
```

This command does not contact vLLM. It uploads completed raw files directly,
reuses previously sealed packages, and releases successfully published storage.
It needs working space for small metadata records and the upload library,
but does not create a second activation copy. Missing rows are left for the next
extraction run. Partial completion
records are published, but the immutable final index and hidden-stage record are
written only when every row is durable. Restart the server and rerun preparation
after this command frees space. The same command works for math, STEM, and chat.

## Build the training-ready release later

Once the source category is fully published, run compaction on a separate CPU
machine with sufficient local SSD space. It reads both old shards and new raw
files, builds approximately 1 GiB training shards and compact conditioning
summaries, and publishes to a different release. The raw/source release remains
unchanged:

```bash
speculators lora-bank compact-hidden \
  --config examples/eagle3-lora/bank_code.yaml \
  --release r0001-training \
  --output-root out/data/eagle3/code-r0001-training
```

No vLLM server or GPU is used. Inputs are restored at a pinned source revision.
Trajectories reference their original immutable Hub artifacts instead of being
generated again. Compaction processes and publishes one subset at a time, using
a bounded source-download cache; allow space for that cache, the current subset,
and a shard's working space. Do not reuse the source output directory or source
release ID. The resulting pinned consumer config is written to:

```bash
speculators lora-bank train \
  --config out/data/eagle3/code-r0001-training/bank_training.yaml
```

The reader supports mixed source indexes containing old shards and raw files.
Raw indexes compute last-token summaries on demand; the compacted release
provides separate summary files for fast generator preprocessing. Setting
`hub.activation_format: sharded` retains the old preparation behavior explicitly.

### Local compaction on the extra SSD

The four `bank_<domain>_local.yaml` configs pin the complete source release at
`058f01d1f0585e98327b2c190a65c796da04fdd4`. They use a shared 20 GiB activation
cache at `/media/azeer/extra-segment/eagle3/activation-cache`, an 8 GiB disk
reserve, and 2 GiB upload groups. Compaction applies these local cache settings
instead of the cache budget embedded in the source index. Run domains serially.

```bash
mkdir -p /media/azeer/extra-segment/eagle3/{hf-cache,tmp}
export HF_HOME=/media/azeer/extra-segment/eagle3/hf-cache
export TMPDIR=/media/azeer/extra-segment/eagle3/tmp
hf auth login

for domain in stem chat code; do
  speculators lora-bank compact-hidden \
    --config "examples/eagle3-lora/bank_${domain}_local.yaml" \
    --release r0001-training \
    --output-root "/media/azeer/extra-segment/eagle3/${domain}-training" || break
done
```

All target categories share the same training release ID but have separate Hub
paths and output directories. Math is already entirely sharded, so it does not
require compaction. Its local config is provided if a uniform training release
is desired; add `math` to the loop only for that purpose. Existing responses and
activations are reused; no server, extraction run, or GPU is needed.

To resume code on a new machine, with its saved trajectories already on Hub,
start the extraction server in terminal A:

```bash
speculators lora-bank serve \
  --config examples/eagle3-lora/bank_code.yaml \
  --mode hidden --resume-from-hf
```

The server flag restores only selection metadata and prompts, then downloads
and launches the frozen target revision. It does not perform selection,
response generation, activation downloads, or publication. When the server is
ready, start extraction in terminal B:

```bash
speculators lora-bank prepare \
  --config examples/eagle3-lora/bank_code.yaml \
  --stage hidden --resume-from-hf
```

The flag restores frozen membership, saved responses, and prepared trajectories
before extraction. If the YAML points at `main`, the command resolves it to one
commit SHA for input restoration. It does not regenerate responses or reselect
prompts. Existing activation completion records are recovered separately, so
already published examples are skipped without downloading their full activation
shards. The same flag works with math, STEM, and chat configs. Rerunning with
local files present reuses those as well.

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
