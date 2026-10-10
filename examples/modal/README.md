# One-L40S data preparation

This runner starts fresh on the current branch. It submits no training jobs and
has no dependency on the BSC branch, Conda archives, or Slurm.

## Setup and execution

From the repository root:

```bash
python -m pip install -r examples/modal/requirements.txt
modal setup
```

Set a **$25 workspace spend limit** in the Modal dashboard before paid execution:
https://modal.com/docs/guide/budgets. This caps out-of-pocket workspace spending,
not just this app. Credits, image builds, CPU preparation, storage, and egress
have separate billing implications. The eight-hour GPU guard is cumulative per
run across resumptions and is a secondary safeguard, not a dollar cap.

```bash
# Twelve examples, all four domains, separate nemotron-v1-smoke artifacts.
modal run scripts/modal/app.py --command prepare --smoke

# Production: select all 24 subsets, complete responses + hidden states serially.
modal run scripts/modal/app.py --command prepare

# Optional: process only subset zero across all four domains; membership stays fixed.
modal run scripts/modal/app.py --command prepare --subsets 0

# CPU-only status; use a config with run_id: nemotron-v1-smoke for smoke status.
modal run scripts/modal/app.py --command status

# Rerun prepare to resume missing records after interruption.
# Download includes activations and may be large; no GPU is allocated.
modal run scripts/modal/app.py --command download --destination output/modal
```

Only launch one instance of this app at a time. `max_containers=1` limits the
single GPU function within an app; separate simultaneous `modal run` apps can
allocate separate GPUs. The runner uses one input at a time and does not fan out.
The default `status` command does not download models or start a GPU.

## Artifacts and compatibility

`speculators-models` holds the pinned target model, HF cache, and compilation
cache. `speculators-data` holds `/artifacts/<run_id>/`:

- `selection.json`: resolved revisions, immutable selection identity, and all
  subset-bank paths, ordered by subset then chat/math/code/stem.
- `banks/<domain>/<domain>-0000N/`: ordinary `manifest.json`, `bank.yaml`,
  `prompts.jsonl`, `responses.jsonl`, Arrow `data/`, and `hidden_states/hs_i.safetensors`.
- `manifest.json.global_indices` and prompt `global_index` map local row `i` to
  the original full selection. Both train and validation memberships are fixed.
- `complete.json`: validated row coverage, token count, and activation bytes.
- `source.json`, `speculators.patch`, per-stage provenance, and server logs.
- `metrics.json` per bank: response/extraction times, saved rows, and throughput.
- `gpu_runtime.json` and `last_attempt.json`: cumulative runtime and attempt metrics.
- Per-attempt source/configuration snapshots and installed dependency versions.

Each subset is independently compatible with the existing bank reader. Subset
five is reserved for later adaptation; this runner prepares its data but does
not train it. Execution changes such as batch size/concurrency do not repartition
examples. Scientific settings require a new run ID. Models and dataset revisions
are resolved once and subsequently pinned to the saved selection.

Download preserves artifacts verbatim, including provenance. Absolute paths in
bank YAML/manifests still refer to Modal mounts: a future local training/import
step must explicitly remap those paths rather than mutate the original identity.
No second persistent activation copy is created during preparation.

## Resource and recovery behavior

The GPU worker requests one L40S, four physical CPU cores, and 64 GiB RAM. It uses
bf16, tensor parallelism one, and automatic attention backend selection. Serving
runs in a separate process group with CUDA multiprocessing set to `spawn`.
Responses and extraction servers run sequentially and are fully stopped on error.
Extraction reserves one additional context token (8193) without shortening data.
The CUDA development image includes nvcc; Ninja is installed for FlashInfer JIT.
The pinned vLLM extraction path is used; the generator inference patch is not
needed for data preparation and is not applied.

Responses are appended directly to the Volume in 100-row request batches. Torn
final JSON records are repaired on resume; duplicate IDs and interior corruption
fail visibly. Transient requests retry with backoff; invalid requests fail
immediately. Hidden states are atomically published and token/layer-shape/dtype checked
before completion. Volume commits occur after batches and during long extraction;
workers reload before reading. Uncommitted work may need repeating after preemption.
Completed runs and exhausted runtime guards are checked on CPU before requesting
a GPU. After a hard kill, the ledger conservatively retains a one-minute runtime
reserve; clean exits record actual execution time.

The dataset selection uses disk-backed deduplication and the existing sorted-ID
seeded shuffle, avoiding retention of every source prompt in RAM. Only selected
prompts are excluded from subsequent domains, in chat/math/code/stem order.

The $25 budget is a stopping limit, not a promise that all 26,400 examples fit.
Inspect throughput and artifact bytes from smoke/production before extending the
budget. Long-context hidden states can occupy hundreds of GiB; all are retained
because training is deferred. Cold starts, interrupted attempts, and retries also
consume time. Modal bills CPU/RAM and storage in addition to the GPU.
