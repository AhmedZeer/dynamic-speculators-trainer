# EAGLE3 LoRA banks for weight-generator supervision

## Agreed behavior

The bank supplies trained adaptation weights for a generator that will condition
on target hidden states. First the generator learns from bank weights; later it
is optimized through the drafter against the target's behavior. This change
implements bank preparation, training, and inspection. It does not implement the
generator or claim that weight reconstruction alone maximizes acceptance.

For the initial math experiment, use `Qwen/Qwen3-8B` as target and
`RedHatAI/Qwen3-8B-speculator.eagle3` as drafter. Train only rank-32 `o_proj` and
`v_proj` LoRAs, with **fixed alpha 64**. The generator will produce factors, not
alpha. Freeze the pretrained drafter's remaining parameters and preserve its
architecture, normalization, and vocabulary mapping.

Each seed starts independently from the pretrained drafter. Train for **one full
warmup epoch**, then lower the LR and collect for **four full epochs**. Save every
**10 completed collection optimizer updates**, continuously across epoch
boundaries. Evaluate **once at each epoch end**, including warmup. Evaluation is
reporting, not a plateau gate. There is no intermediate validation, best-only
bank pruning, or return to warmup.

Subset size controls training exposure and adaptation scope; rank 32 does not
establish a fixed number of examples that can be encoded. Existing ablations
motivate useful adapters, while quality and variation of the collected bank
remain assumptions to measure.

## Hierarchy and artifact ownership

```text
Experiment (resolved models, configuration, provenance)
  Category (math; later code, stem, chat)
    Source dataset (identity, configuration, split, revision)
      Shared target trajectories and hidden-state cache
        Subset (train and held-out validation membership)
          Training seed
            Warmup epoch
            Four collection epochs
              Immutable adapter snapshots
  Future generator episodes
    1–32 examples from one subset + random adapter from its bank
```

The initial preset selects four groups of 1,100 distinct prompts: 1,000 training
examples and 100 extra held-out validation examples per group. Identical prompts
are deduplicated **before** partitioning; source IDs are retained as aliases.
Partitioning uses a deterministic shuffle of sorted prompt identities. Validation
membership is shared across the seeds of a subset. Incomplete groups are reported
and excluded. V1 accepts text-only, single-user prompts; original source answers
are discarded in favor of target-generated answers.

```text
output/lora-bank-math/
  manifest.json                 # pinned revisions, cache identity, memberships
  experiment.json               # complete resolved experiment controls
  prompts.jsonl                 # stable prompt IDs and source aliases
  responses.jsonl               # exact target token IDs and boundary masks
  data/                         # shared Arrow rows; stable cache indices
  hidden_states/hs_<i>.safetensors
  provenance/                   # target server and response-generation artifacts
  runs/math-00000/seed-42/
    bank_train.yaml             # runnable trainer configuration
    run.yaml                    # existing trainer provenance configuration
    resolved_train.json         # all resolved fields, including inherited values
    bank_context.json
    train_command.txt
    speculators.patch
    recovery/<generation>/      # last two full-precision durable states
    epochs/<epoch>.json         # one validation result per completed epoch
    snapshots/step-00000010/
      adapter/                  # PEFT factors/config, without merged base model
      entry.json                # step, identity, alpha, rank, checksums
      train_command.txt
      speculators.patch
      run.yaml
      resolved_train.json
      bank_context.json
  bank.json                     # aggregate entries and inspection diagnostics
```

Epoch numbers in artifacts are zero-based. A snapshot's `epoch_validation`
refers to the enclosing epoch's eventual validation, not an individual evaluation
of that checkpoint. Adapter entries are immutable; epoch results are separate.

For a run with collection-epoch step counts `S_e`, the candidate count is
`floor(sum(S_e) / save_interval)`. Sum this over subsets and seeds for the bank.
**1,000 examples is not 1,000 steps:** the loader packs variable-length examples
into a token budget, and distributed world size also changes epoch length.
There is no off-cadence final bank entry. A run shorter than the save interval
can complete with zero candidates; inspection makes this visible.

## External parameters

The example configuration is
[`examples/eagle3-lora/bank_math.yaml`](../../examples/eagle3-lora/bank_math.yaml).
All groups reject unknown keys. Paths are relative to the working directory.
Values below are initial experimental defaults, not established optimal values.

| Group | Controls | Effect |
| --- | --- | --- |
| `models` | Target/drafter IDs and revisions | Defines representations, frozen weights, and adapter compatibility; revisions resolve to immutable Hub SHAs |
| `dataset` | Source, revision, configuration, split, category, sample limit | Defines available prompts; sample limit applies before deduplication |
| `partition` | Training/validation sizes, seed, subset IDs | Controls adaptation scope, independent subsets, and membership |
| `responses` | Thinking, temperature, top-p/top-k, seed, output/context limits | Defines the target trajectory distribution |
| `hidden_states` | Ordered layer IDs, dtype, sequence length | Defines conditioning features and target loss features |
| `lora` | Rank, fixed alpha, dropout, target modules | Controls update capacity, scaling, and regularization |
| `training` | Seeds, warmup/collection epochs and LRs, save interval | Controls optimization trajectories and bank density |
| `training` | Weight decay, noise, token budget, TTT count/decay, loss | Changes exposure and training objective; token budget also changes updates/epoch |
| `execution` | Endpoints, request timeout/retries, response concurrency | Controls server interaction and throughput, not generator batch size |
| `execution` | Hidden-state request/write concurrency | Separates GPU requests from bounded CPU validation and Drive copies |
| `execution` | Prompt RAM chunk size | Controls bounded input prefetch without changing request concurrency |
| `execution` | Response staging directory, sync interval | Controls local durability and destination-write frequency; `null` disables staging |
| `execution` | Processes, attention implementation | Controls distributed execution and numerical behavior |
| `conditioning` | Maximum examples, prompt-state policy | Records the future generator contract; does not affect bank training |

The preset uses warmup LR `1e-4`, collection LR `1e-6`, three seeds `[42,43,44]`,
alpha/rank `64/32`, dropout `0.05`, AdamW, and token budget 8192. Default response
sampling is thinking disabled, temperature 0.7, top-p 0.8, top-k 20, seed 0,
maximum output 4096 and context 8192. Long prompts are left-truncated by the
server to `context_length - max_tokens`; the returned token IDs define the actual
prompt states and loss boundary. Complete returned sequences must fit the
configured extraction length; preprocessing does not silently truncate them.

Store layers `[2,18,33,36]`: the first three are concatenated EAGLE3 input
features, while the final target state supplies token-distribution supervision.
The recipe assumes Qwen3-8B's width of 4096 and the released drafter's one
`o_proj` and one `v_proj`; supporting another model shape needs a compatible
recipe.

Bank v1 uses the file backend and zero dataloader workers so augmentation RNG
can be restored without losing prefetched worker state. The dataloader has its
own seeded generator; training seeds also seed the pack sampler. Single GPU and
replicated DDP are supported; FSDP LoRA remains unsupported. Bank mode rejects
conflicting max-step caps, LR schedulers, merged-per-snapshot saves, and
best-only checkpointing. Resolved trainer settings are saved in
`resolved_train.json` rather than inferred from future package defaults.

Bank training reads upcoming packed batches in a background thread and retains
raw hidden states in a bounded RAM cache. The example configuration uses:

```yaml
execution:
  training_prefetch_batches: 2
  training_cache_gib: 8
```

The lookahead counts future packed batches. The cache budget is per training
process; retained tensors are reused across batches and epochs while they fit.
It excludes the current batch, an in-flight read, and filesystem page cache.
Noise augmentation and packing still run on the training thread, preserving
seed and resume behavior. No hidden-state validation scan is added. Set
`training_prefetch_batches: 0` to disable it; these two I/O settings can change
when resuming an existing run. Each epoch logs cache bytes, sample count, file
reads, cache hits, and consumer waits to help tune the budget.

## Running the pipeline

Install the LoRA extra (`pip install -e '.[lora]'`). Preparation requires a target
server; bank training uses cached states and does not need a running target.
Start with selection, which pins model/dataset revisions and writes membership:

```bash
speculators lora-bank prepare --config examples/eagle3-lora/bank_math.yaml --stage select
```

Read `manifest.json` for `revisions.target_sha`. First launch ordinary target
serving for response generation, using that SHA and context limit. This mode
adds no extraction model, hidden-state connector, or scale-out endpoint defaults.
**Always pass `--provenance-dir`**:

```bash
python scripts/launch_vllm.py responses Qwen/Qwen3-8B \
  --provenance-dir output/lora-bank-math/provenance/response-server \
  -- --revision TARGET_SHA --max-model-len 8192

speculators lora-bank prepare --config examples/eagle3-lora/bank_math.yaml --stage responses
speculators lora-bank prepare --config examples/eagle3-lora/bank_math.yaml --stage data
```

Keep the served model name `Qwen/Qwen3-8B`. The response endpoint must support
exact prompt and completion token IDs. Stop the response server, then launch
extraction against the same pinned target. Its layer order and dtype must match
the recipe:

```bash
python scripts/launch_vllm.py train Qwen/Qwen3-8B \
  --target-layer-ids 2 18 33 --include-last-layer \
  --provenance-dir output/lora-bank-math/provenance/extraction-server \
  -- --revision TARGET_SHA --max-model-len 8192

speculators lora-bank prepare --config examples/eagle3-lora/bank_math.yaml --stage hidden
```

Extraction processes the exact saved prompt-plus-response tokens. Enabling it
during response generation captures states that the response stage does not use,
so splitting the servers avoids redundant activation storage and transfer.
Separate generation/extraction endpoints are also supported. `--stage all`
requires both capabilities to be available already; use the separate stages
above to switch one GPU between serving modes.

Responses resume by stable prompt identity. Local JSONL input is opened once
with buffered reads. `execution.prompt_chunk_size` (default 1000) controls the
RAM chunk size; the next chunk is prefetched off-thread while the current one
feeds the request queue. At most two prompt chunks plus the bounded request
queue and in-flight requests are retained. Chunk size, concurrent requests, and
Drive sync interval are independent controls. Input order and IDs are preserved,
and unfinished source iterators close on completion or interruption.

Arrow preparation reorders asynchronous output by prompt identity. Bank extraction
trusts saved final `hs_<row_index>.safetensors` files and resumes missing rows by
filename alone, without opening existing headers or payloads. New activations
are validated on local server storage before their first publication, copied to
a hidden `.pending` destination, then renamed to the final name after the copy
succeeds. Resume ignores pending filenames. There is no post-extraction or
pre-training activation scan.

The preset uses 16 hidden-state requests and 2 concurrent validation/publication
workers, independently of 256 response-generation requests. File-lock waiting
uses the configured request timeout (600 seconds by default), rather than a
separate fixed 10-second timeout. On a fail-fast error, extraction stops scheduling
new rows, finishes in-flight requests/saves, and reports the failing row and phase.
It no longer calls `os._exit`, which could kill other Drive writes halfway through.
Files complete out of index order; successful files are retained on failure.
These publication guarantees rely on filesystem rename semantics, including those
provided by the Drive mount.

A changed cache recipe or prepared content is rejected; use another
output root for another experiment.

Response generation stages both JSONL outputs on local disk by default:

```yaml
execution:
  prompt_chunk_size: 1000
  response_staging_dir: /tmp/speculators-responses
  response_sync_interval: 1000
```

Each completed example is flushed locally. Every 1,000 completed examples
(including failed examples), the client copies a flushed snapshot of the response
and error files to `output_root` in a background thread. Other workers continue
while that copy runs. A final sync runs on completion and ordinary Ctrl+C; forced
termination or runtime loss can lose responses not yet synced to Drive. Local
staging is retained, so restarting in the same runtime resumes those responses.
A new runtime resumes from the last Drive snapshot. Existing Drive responses are
copied locally on first use; divergent local/Drive files are rejected. Run only
one response-generation process per output root. Set `response_staging_dir: null`
to use the original direct append behavior. Staging and sync controls do not alter
the cache identity and can be changed for an existing experiment.

The launcher uses `os.execvp` and inherits stdout/stderr; it does not redirect
statistics to another file or suppress them. Look in the server process's output,
including any redirection used by the Colab cell that started it. If logs are
filtered, launch with `VLLM_LOGGING_LEVEL=INFO` and omit `--disable-log-stats`.
Keep passing `--provenance-dir`. Statistics visibility also depends on the vLLM
version and its logging configuration.

Wait for `prepare --stage hidden` to finish successfully before training. The
bank requires one cached file for every prepared row (4,400 for the full pilot
preset). Extraction is concurrent, so higher-numbered files can appear while
lower indices are still missing. If interrupted, rerun the same hidden stage
against the extraction-mode server; existing files are reused. Training reports
present/required counts and missing indices for an incomplete cache. It does not
silently drop missing examples or generate states during training.

Stop the target server to free GPU memory, then run:

```bash
speculators lora-bank train --config examples/eagle3-lora/bank_math.yaml
speculators lora-bank inspect --config examples/eagle3-lora/bank_math.yaml
```

Training resolves pinned model snapshots and runs subset/seed combinations
sequentially. `execution.processes` enables local `torchrun` DDP. Rerunning
resumes existing compatible runs and skips completed epochs. Change output roots
for changed optimization settings; expansion should use a separately identified
cache/experiment rather than mutating existing memberships.

Recovery commits model factors, full-precision optimizer state, phase/counters,
rank-specific Python/NumPy/Torch/CUDA RNG states, and data position atomically.
A signal may interrupt an update, so recovery uses the **last committed boundary**
rather than saving partially updated weights. Warmup may replay from its start;
collection replays at most the updates since the last interval/epoch checkpoint.
An interruption during validation can repeat that incomplete validation on resume.
Ordinary uninterrupted runs evaluate exactly five times. Resume requires the same
configuration and distributed world size.

Inspection writes `bank.json`, reports count/completion, exact factor duplicates,
effective scaled-update norms, and adjacent-update distances. It uses factor
Gram matrices rather than allocating dense updates. These diagnostics do not
establish behavioral diversity or certify acceptance gains; they do not silently
filter otherwise valid entries. Selected adapters can be merged for serving via
PEFT and evaluated using the existing evaluation tools. Include train/eval/vLLM
commands, model hashes, and patches when publishing artifacts or results.

## Future generator and experiments

An episode selects a subset, chooses `b` between 1 and 32, samples `b` distinct
training examples, and chooses a random adapter from that subset's bank. Inputs
are only prompt-prefix states from layers `[2,18,33]`, with token masks and
example boundaries. The output is **one** shared rank-32 adapter with fixed
alpha. This is an acceptable supervision target, not necessarily the best adapter
for that particular episode. Later training through the drafter supplies the
performance objective. Generator architecture and reconstruction loss are deferred.

Reserve entire subsets for generator generalization evaluation, not only unseen
checkpoints from training trajectories. Compare conditioned generation against a
fixed math adapter and shuffled conditioning. Sweep subset size, seeds,
collection epochs, save spacing, and conditioning sizes including 1, 10, and 32.
Measure quality and useful variation separately from checkpoint count. Plateau
and reversible warmup/collect scheduling are future experiments, requiring epoch
criteria, hysteresis, and cycle limits; they are not part of v1.

Relevant precedent:

- [Drag-and-Drop LLMs](https://arxiv.org/pdf/2506.16406): dataset-paired prompt/checkpoint supervision and dense low-LR collection; its math recipe uses 10K examples and 100 collection updates.
- [Text-to-LoRA](https://arxiv.org/pdf/2506.06105): adapter reconstruction and task-loss training; its objective ablation motivates evaluating downstream behavior rather than reconstruction alone.
- [Doc-to-LoRA](https://arxiv.org/pdf/2602.15902): conditioning on frozen-model activations and distillation objectives. Its learned scale is not adopted here.
- [SHINE](https://arxiv.org/pdf/2602.06358): mapping model-derived representations to adapters and training through the adapted model.

These results motivate the pipeline; they do not set optimal EAGLE3 subset size,
checkpoint count, or conditioning budget.

Bank hidden-state extraction resumes by row ID, filling missing files even
when higher-numbered files already exist. The progress bar includes reused files
and advances only after a new file has been published successfully. Console
summaries show completed/total, reused, saved, failed, remaining, and files/second
every `execution.hidden_state_log_interval` seconds (default 10), including while
requests are pending. Failed rows remain in the remaining count. Summaries use
in-memory counters and add no Drive reads or writes. The standalone extraction
CLI exposes the same setting as `--progress-log-interval`.

Startup also logs dataset loading and the resume scan immediately, with periodic
`still running` messages at the same interval. These filesystem stages run in a
background thread so slow Drive access does not block the logging heartbeat.

The bank prepare entry point emits plain stderr diagnostics before loading the
configuration or manifest, including the code path, resolved output root, PID,
and client process RSS. Stage heartbeats also cover manifest/source loading and
post-extraction presence checking, independently of the root training logger. RSS is
this process's resident memory; Colab's overall RAM includes the vLLM server and
filesystem cache too. `--stage hidden` requires an existing manifest and fails
immediately if it is missing, instead of selecting the full source corpus. Check
the Drive mount and `output_root` when this happens.

Extraction uses a plain console handler independently of Rich and root logging
filters. It prints an event-loop startup marker, dataset/cache stages, and the
server model/endpoint once connected. If startup has not completed within 30
seconds, a one-time faulthandler thread stack dump is printed to stderr. This
also works when Python's event loop or logging is blocked. The watchdog is
cancelled when the cache is already complete, when the server is ready for
requests, or when the command exits. A stack dump is diagnostic, not an abort.

Bank training logs manifest loading, cache presence checking, snapshot resolution,
and each subset/seed subprocess launch. Saved artifacts are assumed valid. Startup
checks only the experiment configuration and whether all final filenames listed
by the manifest's row count exist. It does not reopen the prepared dataset, hash
its token contents, stat each cache file, or scan activation payloads. Existing
`hidden_states_validation.json` receipts are ignored and left untouched.

Bank extraction enables `trust_existing_outputs` automatically. The standalone
extraction CLI exposes this as `--trust-existing-outputs`; its default remains
header validation when `--validate-outputs` is used. Normal training loads the
selected samples as needed by the dataloader.

Training console output identifies the subset and seed throughout each run. It
reports model/vocabulary loading, LoRA installation, packed dataloader creation,
optimizer/device/recovery initialization, and restored epoch/step counters.
Warmup and collection epochs have explicit start/end messages and timed
heartbeats every 10 seconds. Epoch-end evaluation prints its metrics, recovery
saves print their stage, and published LoRA snapshots print their destination.
The parent command reports the run number out of the subset/seed total and
completion of the final bank index.

`training.log_freq` controls optimizer-step progress messages (default 10 in the
math recipe), alongside existing training metrics. The first and final steps of
each epoch are always reported. Progress includes full-epoch step totals, global
updates, collection updates, LR, steps/second, approximate epoch ETA, and updates
until the next snapshot. Mid-epoch resume uses the original epoch denominator.
Only rank zero emits these stage/progress messages under DDP. Logging adds no
activation scans, extra evaluations, or extra checkpoint writes.

Bank training uses a plain root console handler. Full hyperparameter records
are persisted in `run.yaml` and delivered to configured metric backends without
rendering the entire configuration in the terminal. Training metrics still
appear in the console. Startup stages cover metric-configuration publishing,
rotary alignment, run-identity checks, and provenance writes before vocabulary
and model loading. If initialization has not reached the training loop within
60 seconds, a one-time thread stack dump appears on stderr. It is cancelled
when training becomes ready or initialization exits.
