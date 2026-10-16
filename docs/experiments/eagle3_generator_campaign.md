# Paired bank warmup and direct pretraining

For the local packed-bank workflow with online vLLM activations on an A100
80 GB, see [the online campaign runbook](eagle3_generator_campaign_online.md).

This campaign runs six pretraining jobs and four joint adaptation jobs, serially
on one A100 40 GB. It reads existing Hugging Face activations on demand. It does
not launch vLLM, generate new trajectories, or measure decoding acceptance.
Training curves show **offline agreement (`full_acc_0`) at context size 4**.

## Configure and run

Install the repository with the generator extra (`pip install -e '.[generator]'`).
Configure W&B authentication for online logging. Start from
[`generator_campaign.yaml`](../../examples/eagle3-lora/generator_campaign.yaml).
Set `input_root`, `output_root`, and `staging_dir` to local SSD paths. Check that
`/tmp` is disk-backed rather than a RAM filesystem before using the default.
The intended machine has 64 GB CPU RAM and 500 GB local SSD. Model downloads use
the normal Hugging Face cache; put that cache on the same adequately sized SSD.
Keep the output directory, including checkpoints and provenance, across restarts.

```bash
# No downloads, filesystem writes, model loading, or GPU work.
speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign.yaml --dry-run

# Read-only Hub restoration: metadata and first-epoch bank adapters, not activations.
speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign.yaml --stage prepare

# Training-only cold/warm cache pilot, including context-4 offline validation.
speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign.yaml --stage pilot

# Each stage resumes matching checkpoints and skips completed jobs.
speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign.yaml --stage pretraining
speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign.yaml --stage adaptation
```

`--stage all` performs preparation, pilot, pretraining, and adaptation in order.
Full training requires a matching successful pilot report. The pilot chooses
between four and eight download workers and checks a 20% warm-cache data-wait
threshold. If it fails, inspect its report and tune IO settings before retrying.
After changing pilot budgets or training settings, use a new output root.
Operational download/prefetch/cache settings can change without invalidating
training recovery; rerunning the pilot records the current conditions.

## The ten training jobs

Each category is math, code, STEM, or chat. Prepared subsets contain 1,000 training
examples and 100 separate validation examples.

| Pretraining scale | Subsets in each category | Total training examples | Jobs |
| --- | --- | --- | --- |
| 1 | 00000 | 4,000 | Bank and direct |
| 2 | 00000–00001 | 8,000 | Bank and direct |
| 4 | 00000–00003 | 16,000 | Bank and direct |

The bank arm reconstructs saved LoRA factors with L1 loss for 10,000 updates.
There is no subsequent behavioral training in this arm. The direct arm optimizes
the same generator through the frozen drafter using `lk_hybrid`, also for 10,000
updates. Each pair starts from identical generator weights and uses the same
index-addressable episode schedule. Four episodes accumulate into one optimizer
update. Each episode samples a category uniformly, then a subset uniformly, and
then 1–8 training examples. Snapshot selection has a separate random stream and
always targets the sampled subset.

Only ten snapshots from the first collection epoch of each bank run are used.
The pinned published math bank has four collection epochs; the other domains
have one. Preparation reads published completion identities to identify the
correct first epoch instead of assuming current cloud YAMLs match past runs.
The bank commit is pinned to `2305403f0f3b560fafb6c6d261c370ab3a95759d`;
the source trajectory commit is inherited from the pinned cloud configs.
Only snapshots needed by scales 1/2/4 are restored, together with their original
snapshot provenance. Bank creation compute is historical and is not included
in campaign timing.

Adaptation uses four subsets per category: **00002–00005**, totalling **16,000**
training examples per job. Four initializations train jointly for 1,000 updates:
fresh generator, bank-warmed scale-2 generator, bank-warmed scale-1 generator,
and fresh ordinary LoRA. All use `lk_hybrid` and the same context schedule.
Only generator weights transfer; optimizers and LR ramps restart. Neither
transferred generator has seen the adaptation subsets. The ordinary LoRA is
one shared adapter across all adaptation categories.

Defaults retain `last` conditioning, the existing generator architecture, rank 32,
alpha 64, and `o_proj`/`v_proj` adapters. The remaining drafter parameters stay
frozen. AdamW uses LR 1e-4, weight decay 0.01, 100 LR-ramp updates, and gradient
clipping at 0.85. Frozen compute is BF16; trainable parameters and optimizer state
are FP32. The token budget is 8192, with three TTT steps and the bank's masks and
noise settings. Episodes contribute equally; behavioral microbatches within an
episode are weighted by supervised response-token count. Oversized rows fail
explicitly; they are never silently truncated.

Validation occurs at update 0, every 250 updates, and completion, at context size
4 only. Experiment 1 uses the same 100 validation examples from subset 00000 in
each category for every job. Experiment 2 uses the 100 validation examples from
each of its 16 subsets. Token-level counts aggregate within each subset, then
subset/category scores receive equal weight. Final checkpoints are compared;
validation does not select an adaptation winner.

W&B groups separate pretraining, adaptation, and pilot jobs. Pretraining uses
one hue per scale, dark for bank and light for direct. The saved comparison
PNG/PDF and W&B image preserve those exact colors; W&B's default per-run line
chart colors are not modified automatically. Every job also logs per-category
and per-subset scores. `comparison.json` preserves the underlying curves.

## Lazy activations and the training-only pilot

A single application cache is shared across categories and production jobs.
Eight updates are prefetched ahead, with four initial download workers and two
checksum workers. Only required immutable references are fetched. Files are
verified once, atomically published, and deduplicated by repository, revision,
path, and checksum. Verification receipts detect local modifications. Per-file
locks protect downloads/reads; network transfers do not hold a global cache lock.
An LRU policy evicts files outside the active prefetch window. Prompt summaries
are created only when requested, persisted separately, and retained after full
activation eviction. A bounded in-memory summary cache avoids repeated small
reads. No eager whole-corpus summary preparation runs.

The activation cache cap is 350 GiB with a 50 GiB free-disk floor. In-flight
reservations and temporary checkpoint writes participate in quota checks.
If protected files cannot fit, reduce `downloads.lookahead_updates` or increase
capacity. Published shards must be downloaded as whole files; this is not remote
row-level tensor streaming. Uniform sampling will eventually touch much of the
dataset, so on-demand access cannot remove the underlying bandwidth requirement.

The pilot runs 100 updates per job, testing bank reconstruction, direct generator
training, and ordinary LoRA with cold and warm application caches at four and
eight download workers: twelve short jobs. The first update includes the longest
training trajectories from every category. Warm jobs reset weights and replay
the same schedule. Dedicated pilot caches are cleaned after each completed pair;
reports, checkpoints, and provenance remain. HF/Xet and OS caches are not cleared,
so a cold application cache is not a guaranteed cold network/storage stack.

Inspect `pilot/report.json` and each job's `timing.json` for:

- Downloaded bytes, worker time, retries, checksum time, file reuse and evictions.
- Exposed data-wait/preparation time, update wall time, and CUDA compute time.
- GPU peak allocated/reserved memory; sampled GPU utilization when NVML is
  available; CPU RSS and process disk IO.
- Peak reserved cache bytes, actual cache occupancy, free disk, checkpoint sizes,
  and exact metadata-based full activation working sets for the ten jobs.
- Cold/warm linear training-time projections, excluding model startup,
  validation and checkpoint overhead. Larger datasets can exceed cache capacity
  and perform worse than the pilot; the report explicitly marks those cases.

Pilot and campaign training states are separate. Successful CPU tests establish
correctness of the machinery, not A100 fit or throughput. Run this pilot on the
actual GPU before spending the full budget.

## Recovery and reproducibility

Each run saves its resolved configuration, immutable data identity, selected
snapshots or source checkpoint hash, `train_command.txt`, `speculators.patch`,
metrics, and timing. Recovery includes weights, optimizer state, update position,
validation history, and Torch/CUDA RNG state. It is saved every 500 updates and
at completion, using an atomic replacement. Resume archives earlier command/code
provenance; completed matching runs are skipped. Changed training identities
require a new output directory. Include provenance and resume history whenever
publishing results.
