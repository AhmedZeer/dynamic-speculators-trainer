# Offline EAGLE3 LoRA generator experiments

The generator produces one rank-32 LoRA for EAGLE3's `o_proj` and `v_proj`,
conditioned on **1–8 examples' target hidden states**. Alpha remains fixed at 64.
The target is `Qwen/Qwen3-8B`; the drafter is
`RedHatAI/Qwen3-8B-speculator.eagle3`. These experiments use the existing math
bank and require neither a vLLM server nor new response/activation generation.

## Run order

Install plotting and W&B support from the checkout, then run from the repository root:

```bash
pip install -e '.[generator]'
speculators generator prepare --config examples/eagle3-lora/generator_math.yaml
speculators generator conditioning --config examples/eagle3-lora/generator_math.yaml
speculators generator heatmap --config examples/eagle3-lora/generator_math.yaml
speculators generator adaptation --config examples/eagle3-lora/generator_math.yaml
```

The generator configuration references `bank_math.yaml` relative to its own
location. Its other paths follow the bank's working-directory convention.
Existing bank configuration, manifests, checkpoints, and LoRAs remain intact.
Both preparation and training announce their stage, report stalled reads and
periodic progress, and write separate experiment artifacts.

## 1. Choose conditioning on math-00001

Train three fresh generators through the **frozen drafter's prediction loss**,
without bank reconstruction pretraining. Each sees every training example once
per epoch, for five epochs. Compare:

| Name | Prompt representation per example |
| --- | --- |
| `last` | Last prompt-token states at target layers 2, 18, 33 |
| `last_mean` | Last prompt-token and masked mean prompt states at those layers |
| `projected` | Last and mean states after EAGLE3's frozen input normalization and FC projection |

Evaluate before training and after each epoch. Select using the final epoch,
not the best intermediate checkpoint. A tie selects the first variant in the
table. Decoder architecture and training schedule are shared; encoder input
widths, and therefore parameter counts, differ and are reported.

## 2. Bank heatmap on math-00000

Fix the winning conditioning and initialize a new generator for every cell:

- Columns: seed 42; seed 43; seed 44; seeds 42 & 44; seeds 43 & 42;
  seeds 42 & 43 & 44. The missing 43 & 44 pair is intentionally excluded.
- Rows: collection strides 10, 20, 30, yielding **18 generators**.
- Keep the same collection-step interval across all cells: the overlap of the
  three available seed runs. Keep only steps divisible by the requested stride;
  never include warmup or off-stride epoch-end checkpoints.
- Train each generator for **1,000 optimizer updates**, four reconstruction
  episodes per update. Uniformly choose a bank seed, then a retained checkpoint
  within that seed. Uniformly choose context size 1–8 and sample training prompts
  without replacement within each episode. Context and adapter draws are
  independent within the same subset.
- Use saved A/B factors directly as targets. Average the mean L1 loss of the
  four factors equally. There is no factor alignment, dense-update conversion,
  factor standardization, learned alpha, or latent sampling.

Each cell's evaluation applies **the generator's output** to the frozen drafter;
it does not average the saved adapters' scores. Training budgets are identical
even when checkpoint counts differ. The highest final score selects the
pretrained generator for experiment 3. Ties follow row/column order.

Exports include CSV/JSON values, checkpoint counts, and PNG/PDF heatmaps for
the primary score and each context size.

## 3. One-epoch adaptation on math-00001

Compare the winning bank-pretrained generator, an identically shaped fresh
generator, a fresh ordinary LoRA, and a transferred ordinary LoRA. The transfer
control uses seed 42's latest collected checkpoint from math-00000, independently
of the heatmap winner. Experiment 1's trained generator weights are not reused.

Train each arm for exactly one epoch through the same drafter loss. Use matching
shuffled example groups, token budgets, hidden-state noise, dropout random seeds,
and optimizer-update boundaries. Ordinary LoRA arms maintain one adapter across
groups; generator arms produce one adapter per group. Report epoch-zero scores,
epoch-one scores, and their differences.

## Architecture and data flow

This is a [Text-to-LoRA](https://arxiv.org/abs/2506.06105)-inspired large
full-factor decoder with an activation-set encoder replacing the task encoder.
Each example's summary and log prompt length pass through a shared
`input → 512 → 128` MLP with GELU and output LayerNorm. Aggregate embeddings
using mean, population standard deviation, and log example count. A
`257 → 128 → 64` MLP produces the condition. Pooling is invariant to example
order; singleton standard deviation is zero.

Concatenate the condition with 32-dimensional module and layer descriptors;
decode through a 512-wide network with two residual MLP blocks. Separate full
A/B heads produce the four factors. At the example's default Qwen3 dimensions,
the output-head weights alone contain 285,212,672 parameters; the encoder and
decoder add more. This configuration prioritizes the agreed architecture over
deployment latency, which is outside this experiment.

Output heads start with zero weights, random ordinary LoRA A biases, and zero B
biases, so a fresh generator initially preserves the base drafter. Functional
parameter substitution preserves gradients into the generator. Alpha/rank
scaling is applied inside the drafter, and only the generator or ordinary LoRA
parameters are optimized. EAGLE3 uses the configured attention backend.
The model forward runs outside compilation for functional factor substitution;
the CUDA FlexAttention kernel is compiled separately to preserve fusion.

Only prompt states before `prompt_length` condition the generator. Cached final
target states and the frozen target norm/head supply response training signals.
The full target decoder is not instantiated. A conditioning group shares one
adapter across token-budget microbatches, with loss weighted by supervised
response-token counts before one optimizer update. FlexAttention microbatches
are padded to multiples of 128 tokens; padding has document ID -1 and a false
loss mask. This aligns cached draft KV segments with their sparse attention
blocks. With this backend, `token_budget` must also be a multiple of 128.

During training, generated factors are exposed to the drafter as differentiable
leaf tensors. Each microbatch backpropagates independently and releases its
attention graph. The accumulated factor gradients then backpropagate through
the generator once per group. This preserves the shared adapter and weighted
gradient while supporting compiled backward kernels that donate saved buffers;
it requires neither `retain_graph=True` nor disabling donated buffers.

## Evaluation definition

Evaluate on the existing 100 validation examples at context sizes **1, 4, 8**,
using the same deterministic grouping across all runs. Include every example;
the final short group uses its actual size. Aggregate the existing metric raw
counts across microbatches, then compute the normalized `full_acc_0`. The primary
score is the arithmetic mean of the three context-size scores. Preserve all
other existing draft-step metrics.

`full_acc_0` is offline first-step draft/target argmax agreement under the training
setup. It is not a measured decoding acceptance rate or inference speed.
Reusing math-00001 validation for conditioning selection and adaptation makes
the final comparison exploratory, rather than an independent held-out result.

## External parameters, I/O, and recovery

| Controls | Effect |
| --- | --- |
| `bank_config`, subset IDs, `transfer_seed` | Existing data and adapter sources |
| `context.*` | Number of examples sharing one generated adapter and evaluation sizes |
| `architecture.*` | Encoder/decoder capacity and full output-head size |
| `optimization.lr`, `weight_decay` | Shared AdamW optimization |
| `optimization.max_grad_norm` | Global L2 gradient clipping threshold; defaults to 0.85 |
| `conditioning_epochs`, `pretraining_updates`, `episodes_per_update` | Comparison and reconstruction budgets |
| `adaptation_epochs` | Fixed to one |
| `token_budget` | Activation microbatch size, independent of context size |
| `heatmap.*` | Bank checkpoint selections and grid shape |
| `seed` | Generator initialization, grouping, and bank sampling |
| `n_workers` | Independent concurrent runs on the visible device; no GPU assignment |
| `dtype`, `device` | Frozen-model compute precision and device; generator and ordinary LoRA parameters remain float32 |
| `staging_dir`, `factor_cache_mib` | Local summary storage and bounded bank-factor LRU cache |
| `checkpoint_interval`, `log_interval` | Reconstruction recovery and progress frequency |

Prepared summaries are small and loaded into RAM. Preparation reads each needed
hidden-state payload once and releases it after summarizing; summaries are cached
locally and copied to the experiment output once. Training reads response
activations with the bank's buffered file reader, avoiding Drive-backed mmap.
There is no full-activation prefetch cache or exhaustive payload/hash scan.

Every run saves resolved configuration, selected source metadata, package/git
provenance (`train_command.txt`, `speculators.patch`), recovery state, and results.
Resuming refreshes the command/code provenance and archives previous attempts
under `provenance/`; include that history when publishing a resumed run.
Checkpoints are serialized locally and streamed to one replaceable recovery file
in the run directory, rather than accumulating full generator snapshots.
Reconstruction resumes at checkpointed update boundaries; drafter-loss training
resumes at epoch boundaries. Repeated commands skip completed matching runs.
Changed experiment settings require a new output root. Changing worker count,
staging location, factor-cache capacity, or W&B settings does not invalidate completed runs.

### W&B monitoring

Install the optional integration with `pip install -e '.[generator]'`. The example
configuration enables **offline** logging, with separate projects:

- `eagle3-generator-conditioning`: drafter loss and per-epoch acceptance for each conditioning representation.
- `eagle3-generator-heatmap`: reconstruction L1 loss for each seed/stride cell, final acceptance, and a summary run with the 18-cell table and heatmap images.
- `eagle3-generator-adaptation`: drafter loss, initial/final acceptance, and score change for each of the four arms.

To monitor live cloud dashboards, run `wandb login` and set `wandb.mode: online`.
Set `wandb.entity` to select an account/team; each project name is configurable,
but the three names must differ. `wandb.group` defaults to the output directory
name, and `wandb.name_prefix` can distinguish experiment campaigns.
For example, `heatmap-math-00000-last-stride20-bankseeds42+44-gseed42-ctx1-8`
identifies its data subset, representation, bank selection, generator seed, and
conditioning range. Adaptation names also identify the arm.

`wandb.log_interval` controls optimizer-update logging independently of console
logging. Plot `train/drafter_loss` or `train/reconstruction_l1` against
`optimizer_step`. Every experiment clips the accumulated optimizer gradients
to a global L2 norm of `optimization.max_grad_norm` (default 0.85), immediately
before its optimizer step. W&B logs `train/grad_norm` **before clipping** and
`train/gradient_clipped` (1 when clipped, otherwise 0). Non-finite gradient
norms stop the run before applying an update.
Clipping is part of the training identity; runs created before clipping was
introduced require a new `output_root` to keep the experiment comparisons valid.
Validation logs include `validation/score_mean_full_acc_0`
and every reported metric under `validation/context_1/*`, `context_4/*`, and
`context_8/*`. Run summaries include final/best scores and parameter counts;
heatmap cells also record the number of supervising checkpoints.

Online restarts reuse a deterministic W&B run ID with `resume: allow`; the
optimizer counter is saved in recovery checkpoints. Drafter training still
recovers at epoch boundaries, so updates after the last checkpoint can repeat.
Previously completed runs backfill their saved validation history without
loading models or retraining; historical step losses cannot be recovered.
Repeated commands skip results already logged to the same destination.
Offline attempts produce local W&B bundles under `staging_dir/wandb`; they
do not automatically merge into one cloud history. Use `wandb sync` to upload
offline bundles when desired. Set `wandb.enabled: false` to disable tracking.

`wandb.upload_artifacts` defaults to false. Enabling it attaches result/config
files, command/git/package provenance, source patches, and archived resume
attempts; heatmap exports include PNG/PDF/CSV/JSON. Full model checkpoints and
hidden-state payloads are excluded. W&B working files use local staging storage
to avoid additional Drive traffic during optimizer updates.

Use one worker initially: every worker owns the large generator, optimizer,
and frozen drafter. Publish provenance and resolved configuration with results.
A real-model GPU smoke run is required before spending the full experimental
budget; CPU checks exercise synthetic data and gradient paths.
