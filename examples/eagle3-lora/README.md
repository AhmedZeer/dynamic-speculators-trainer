# Qwen3-8B EAGLE-3 LoRA training on Nemotron

This note describes the Colab workflow in
[`eagle3_qwen3_8b_nemotronchat_1k_lora.sh`](eagle3_qwen3_8b_nemotronchat_1k_lora.sh).
It is intended for repeated training runs and hyperparameter sweeps that reuse
the same regenerated responses, prepared data, and verifier hidden states.

## Storage layout

The script separates shared data from each training run:

```text
/content/drive/MyDrive/dynamic-speculators-dump-v2/
├── eagle3_qwen3_8b_nemotron_lora_ordered_v1/ # DATA_ROOT, shared cache
│   ├── regenerated/qwen3_8b.jsonl
│   ├── data/                           # prepared Arrow rows
│   ├── hidden_states/                  # hs_<row-index>.safetensors
│   └── vLLM provenance files
└── runs/<RUN_ID>/                      # checkpoints and per-run logs
    ├── checkpoints/
    └── logs/
```

The default `DATA_ROOT` uses append-stable ordering. The older `v2` cache used
size-dependent shuffling and must not be mixed with this cache. Override
`DATA_ROOT` to select another shared cache. Keep one cache for runs that use the
same dataset, verifier model, sequence length, and target layers. Use a separate
cache if any of those data-generation inputs change.

`RESPONSE_ROOT` may point at an existing regenerated-response directory. This
allows a new ordered Arrow/hidden-state cache to reuse the much smaller JSONL
from an older cache without copying it. Arrow data and hidden states must stay
together under the new `DATA_ROOT`.

`RUN_ID` defaults to `v1`; set a distinct value for each independent sweep run.
Reuse the same `RUN_ID` when resuming a run. `RUN_DIR` can also be set directly.

## Cache size versus training size

`MAX_SAMPLES` is the number of rows to regenerate, prepare, and cache.
`TRAIN_SAMPLES` caps the number of examples taken from the training split for a
particular run. The training CLI option `--max-train-samples` applies this cap
without copying or rewriting the prepared dataset. The default
`TRAIN_DATA_RATIO=0.9` reserves the final 10% of the prepared rows for validation;
the cap applies to the training split and leaves validation data unchanged.

For example, the first run can build a 10k cache and train on 3k examples:

```bash
MAX_SAMPLES=10000 TRAIN_SAMPLES=3000 RUN_ID=sweep-3k \
  bash examples/eagle3-lora/eagle3_qwen3_8b_nemotronchat_1k_lora.sh
```

After generation and hidden-state extraction complete, a second run can reuse
the same cache and train on 2k examples:

```bash
MAX_SAMPLES=10000 TRAIN_SAMPLES=2000 RUN_ID=sweep-2k \
  bash examples/eagle3-lora/eagle3_qwen3_8b_nemotronchat_1k_lora.sh \
    --skip-regenerate --skip-missing
```

`--skip-regenerate` disables response generation. `--skip-missing` disables
hidden-state generation and sets training to skip rows without hidden states.
The script reports how many selected training rows have hidden states and how
many will be skipped. Without those flags, it detects completed artifacts and
generates only missing responses or hidden states. If the requested cache size
is larger than the prepared cache, the script expands the prepared data from
the regenerated JSONL using `--preserve-order`. It builds the expanded Arrow
dataset in `data.next`, verifies that every existing row has identical token
IDs, and only then swaps it into place. This validation reads the compact Arrow
data and does not open existing hidden-state tensors. Existing hidden-state
files therefore remain valid when growing a cache from 10k to 20k. With
`--skip-regenerate`, expansion requires the regenerated JSONL to already have
the requested number of prompts.

## W&B and training metrics

The script enables W&B with `--logger wandb`. `WANDB_PROJECT` selects the
project; `WANDB_RUN_NAME` is passed through `--run-name`. Defaults are
`dynamic-speculators-v2` and `qwen3-8b-nemotronmath-1k-lora-{time}`. Configure
authentication in the Colab environment before training.

The trainer logs the global gradient norm before clipping as `train/grad_norm`
and logs `train/gradient_clipped` as 1 when that norm exceeds the clipping
threshold (0 otherwise). Gradient clipping is currently fixed at a maximum
norm of 1.0. These metrics are available in runs using code that includes the
logging change.

The example currently passes `--scheduler-type none`, so its learning rate is
constant at `1e-4`. Replace that with `cosine` or `linear` to enable a scheduler;
the current learning rate is logged as `lr`.

Validation runs at the end of each epoch. `--train-data-ratio` controls the
split size, not validation frequency, and there is not currently a CLI option
to disable validation. `--log-freq` controls training metric logging frequency;
increasing it (for example, to 10) reduces logging and timing-profiler overhead.

Always pass `--provenance-dir` to `scripts/launch_vllm.py`; this example stores
data-generation vLLM provenance under `DATA_ROOT` beside the shared artifacts.

## Dataset-specific banks

See [the bank design and workflow](../../docs/developer/lora_bank.md) and
[`bank_math.yaml`](bank_math.yaml) for the rank-32 math bank: one warmup epoch,
four collection epochs, snapshots every 10 updates, and epoch-end validation.
