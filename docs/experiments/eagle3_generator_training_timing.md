# Generator and drafter training timings

The original campaign/structure training loop records separate component times
on every new optimizer update. The existing `train-structure` commands enable
this automatically; there is no additional profiling config or launcher.
The model, optimizer, cohort schedule, and loss are unchanged.

## What is measured

| W&B metric under `training_performance/` | Scope per optimizer update |
| --- | --- |
| `generator_forward_seconds` | All generator forward calls: set encoder/pooling, conditioning, module decoders and A/B heads |
| `generator_backward_seconds` | Accumulated LoRA-factor gradients propagated back through the generator |
| `drafter_forward_seconds` | All packed training microbatches: frozen verifier norm/head target logits, three EAGLE3 training steps, losses and metrics |
| `drafter_backward_seconds` | Weighted loss backward through the drafter into the factor leaves; into the fixed adapter parameters for static LoRA |
| `optimizer_seconds` | Gradient clipping plus AdamW update |
| `data_wait_seconds` | Existing host-clock measurement of data preparation, activation waits, collation and transfers; separate from component intervals |

These are training timings, not vLLM serving latency, standalone kernel times,
or verifier acceptance measurements. Frozen drafter weights still participate
in backward computation to obtain the adapter gradients.

On CUDA, each component uses CUDA events on the current stream. Events are
resolved after the loop's existing update-end synchronization; no per-component
synchronization is introduced. Stream intervals can include CPU launch gaps or
GPU contention with the extraction process, so they are not sums of individual
kernel execution times. On CPU the measurements use a host clock.

`gpu_compute_seconds` is the sum of the five recorded component intervals.
It does not include every GPU bookkeeping operation. Scalar finite checks and
host logging are outside those intervals. GPU times, host data times, and update
wall time are different measurements, not an exact additive wall-time breakdown.
Startup, compilation, validation, checkpoint saving and Hub uploads are outside
the per-update component totals.

## Normalization and cohort batching

Additional W&B metrics:

- `generator_forward_ms_per_call`: total generator forward milliseconds divided
  by actual calls. With batching enabled, one call handles four independent
  cohorts and produces four distinct LoRAs. With serial execution, there are
  normally four calls.
- `generator_forward_ms_per_cohort`: amortized cost per cohort. This is not the
  measured latency of a standalone singleton invocation.
- `drafter_forward_ms_per_microbatch`: total drafter forward milliseconds divided
  by packed microbatches. One microbatch can contain multiple sequences and all
  three training prediction steps; its latency depends on token count.
- `generator_forward_calls`, `generator_backward_calls`, `drafter_microbatches`:
  counts defining the denominators. Static-LoRA runs report zero generator time
  and zero generator calls.

Example console format (illustrative fields, not measurements):

```text
... timings/update; generator_forward=...ms; generator_backward=...ms; drafter_forward=...ms; drafter_backward=...ms; optimizer=...ms; generator_calls=1; drafter_microbatches=...
```

This line follows the existing wall/data/cache line at update 1 and then every
configured logging interval (currently 10). W&B receives the component metrics
through the existing tracking configuration and project. Each timing row is
also stored with `train/` prefixes in
`<output_root>/pretraining/<name>/timing.json`, persisted at checkpoint saves
and run completion. No new Hub repository is required for instrumentation.

## Running and comparing

Pull the latest code and rerun the same command for an ongoing experiment:

```bash
speculators generator train-structure \
  --config examples/eagle3-lora/structure_followups_v1/generator/attention_decoder256.yaml
```

Resume uses the last saved checkpoint. Timing fields are added only for future
updates; earlier aggregate measurements cannot be split retroactively.
Completed runs remain completed and are skipped rather than modified.

Compare a window of warm updates, rather than the first compiled update or one
randomly sampled cohort. Keep hardware, precision, generator batching, extraction
settings/cache state, and model targets fixed. Inspect microbatch and supervised
token counts alongside timings. Use generator forward time to study generation
cost, backward time to study training overhead, and clipping/AdamW time to locate
large-parameter optimizer costs. A serving-latency comparison still needs the
separate vLLM benchmark/profiling path.
