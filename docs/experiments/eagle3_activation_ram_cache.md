# Reusing online activations in CPU RAM

The online generator campaign uses a shared CPU cache to avoid repeatedly asking
vLLM for frozen-target activations. This is useful when multiple training arms
reuse saved token trajectories and validation membership. It needs no activation
download and leaves GPU memory available for the target server and trainer.

## Observed results

The A100 80 GB pilot reported the following measured wall times on 400 validation
examples, evaluated in groups of four:

| Validation set | First pass, seconds | Warm pass, seconds |
| --- | ---: | ---: |
| Math | 35.18 | 6.25 |
| Code | 24.49 | 5.47 |
| STEM | 15.53 | 3.41 |
| Chat | 21.70 | 4.93 |
| **All categories** | **96.90** | **20.06** |

The warm pass was about **4.83 times faster**, a **79.3% reduction in wall time**. The first
pass includes extraction and startup effects, so this is an observed end-to-end
improvement rather than an isolated cache benchmark. The warm pass still runs
generator/drafter evaluation; caching does not reuse validation predictions.

The 400 validation payloads occupied **16.81 GiB**. The following direct-generator
arm began with a **19.99-second** validation pass, demonstrating reuse across
training jobs in the same process. Its final validation took **18.74 seconds**.
The ordinary-LoRA arm subsequently validated in **16.86 seconds** at initialization.

During direct training, the full-activation cache reached **63.97 GiB** and
reported **365 evictions**. Trainer process RSS was **69.67 GiB** at completion:
the configured cache budget is smaller than total process memory. These figures
come from the pilot logs supplied on 2026-10-16:

- [Bank pilot](https://wandb.ai/ahmadzeer2005-y-ld-z-technical-university/eagle3-generator-ablations-v2/runs/1b9c55983dccdbbf)
- [Direct-generator pilot](https://wandb.ai/ahmadzeer2005-y-ld-z-technical-university/eagle3-generator-ablations-v2/runs/1c244b27909aed60)
- [Ordinary-LoRA pilot](https://wandb.ai/ahmadzeer2005-y-ld-z-technical-university/eagle3-generator-ablations-v2/runs/6867c93b5df2afa4)

The shared RAM-cache implementation was introduced in
[`fde90cf`](https://github.com/AhmedZeer/dynamic-speculators-trainer/commit/fde90cf).
Keep each run's resolved configuration, `train_command.txt`, patches, checkpoint
checksums, and the server's `provenance/vllm` directory when reproducing or sharing
measurements. The figures above are transcribed from the supplied console logs.

## What is cached

There are three distinct stores:

| Store | Contents | Retention |
| --- | --- | --- |
| Full activations | CPU `token_ids` and `hidden_states` tensors for complete trajectories | Byte-bounded, validation preferred, LRU within each priority |
| Prompt summaries | Last prompt-token features used to condition the generator | All encountered summaries retained in RAM for the campaign |
| Disk summaries | Small conditioning vectors under `staging_dir/online-summaries` | Reusable after process restart |

On a full-activation miss, vLLM replays the original saved token IDs, including
the response. The trainer waits for the shared-file lock, loads the Safetensors
payload, validates token IDs, finiteness, layer shape, hidden width, and dtype,
then caches the CPU tensors and deletes the temporary file. The same payload
also supplies its prompt summary. A cache hit skips the extraction and file IO.

Bank reconstruction needs only prompt summaries. When a summary is missing,
only prompt tokens are sent to vLLM; the full prompt tensor is discarded after
the last-token vector is saved. Training noise is applied during batch creation,
after reading cached frozen-target states, so the cache does not freeze noise.

## Configuration and memory sizing

The settings in
[`generator_campaign_online.yaml`](../../examples/eagle3-lora/generator_campaign_online.yaml)
are:

```yaml
online:
  ram_cache_gib: 64
  concurrency: 2
  buffered_examples: 8
  shared_dir: /tmp/speculators-online
```

The cache grows on demand; it does not preallocate 64 GiB. Set `ram_cache_gib: 0`
to disable full-activation retention while keeping prompt summaries and bounded
prefetching. Concurrency limits extraction workers; buffering limits pending
results independently of the retained cache.

Full hidden-state storage is approximately:

```text
sequence tokens × extracted layers × target hidden width × bytes per element
8192 × 4 × 4096 × 2 bytes (BF16) = 256 MiB per maximum-length example
```

Token IDs add a small amount. A 64 GiB cache holds about 256 maximum-length
examples; actual capacity depends on sequence lengths. The pilot's validation
set averaged about 43 MiB per example.

The limit covers retained full-payload tensor bytes only. Leave RAM for prompt
summaries, in-flight requests, consumed batches, deserialization, token datasets,
model loading, and checkpoint serialization. Evicted payloads can remain alive
briefly if an active batch or future still references them. Process RSS can
therefore exceed the cache limit, as observed in the pilot.

One prompt summary has `(layers - 1) × hidden_width` FP32 values. With this model,
26,400 summaries would occupy about 1.21 GiB of raw tensor data, plus container
and any additional summary-storage overhead.

## Retention and lifetime

`ActivationRAM` preferentially evicts training payloads before validation
payloads. If validation alone exceeds capacity, older validation entries can
also be evicted. A training payload is not retained when making room would
require evicting only validation entries. Oversized payloads remain usable for
the current request but are not retained.

`LocalInputs` owns the cache; each sequential job's `OnlineActivations` client
borrows it. Closing a job shuts down its workers and client without clearing the
shared tensors. Use one campaign invocation to exploit this sharing:

```bash
speculators generator campaign \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --data-path ./data --trajectories-path ./trajectories --stage all
```

Separate CLI invocations do not share RAM. Checkpoint resume restores training
state, not full activations; a restarted process repopulates those on demand.
Disk prompt summaries survive while the staging directory remains available.

## Reusing the implementation

The implementation lives in
[`online_activations.py`](../../src/speculators/generator/online_activations.py).
For a new sequential workflow, create one owner and pass it to each client:

```python
from speculators.generator.online_activations import ActivationRAM, OnlineActivations

ram = ActivationRAM(capacity_gib=64, validation_indices=validation_indices)
store = OnlineActivations(cfg, rows, bank, ram=ram)
try:
    payload = store.get_full(row_index)
finally:
    store.close()
# Keep `ram` alive and pass it to the next job's client.
```

The caller supplies the prepared rows, bank configuration, and validation
indices. Full-payload keys are integer row indices, not content hashes: reuse
the owner only with the same immutable row ordering, target weights, extraction
layers, and dtype. Create a fresh owner when any of those change. The campaign
enforces source identities before constructing its shared inputs; `ActivationRAM`
itself does not validate identities for arbitrary callers. Treat returned
payloads as read-only; perform augmentation on separate batch tensors.

Disk summary keys include pinned model revisions, layer IDs, prompt tokens,
prompt boundary, and conditioning mode. They do not depend on the current
generator weights, which change during training.

## Reading the measurements

The sectioned W&B layout exposes these useful checks:

- `cache_performance/cache_gib`: retained full-payload bytes converted to GiB.
- `cache_performance/hit_percent`: full-activation cache hits; prompt summaries
  have their own `conditioning_hit_percent` and hit/miss counters.
- `cache_performance/requests`, `full_requests`, and `prompt_requests`: successful
  extraction counts under the same prefix; warm hits do not issue requests.
- `cache_performance/extraction_seconds`: summed extraction/transfer/validation
  durations across successful concurrent requests, not elapsed wall time.
- `cache_performance/evictions`: cumulative across jobs sharing the owner;
  request and hit/miss counters restart with each job's client.
- `validation_performance/pass_seconds`: actual full-pass wall time.
- `training_performance/update_seconds` and `data_wait_percent`: whether remaining
  data preparation still dominates updates.
- `resources/host_rss_gib`: trainer-process RSS, including memory beyond the cache.

Conditioning hits include reused disk summaries as well as RAM summaries; a 100%
conditioning hit rate does not imply a 100% full-activation hit rate. Prefetch
buffer hits are also distinct from retained-cache hits. Local `timing.json` and
`result.json` retain the original raw counters and durations.

When comparing training arms, report cold cache population and warm validation
separately. Sequential jobs inherit different cache states, so their total wall
times depend on run order. Use measured training seconds and validation seconds
to explain that difference. The cache removes repeated target work; it does not
guarantee that every training trajectory fits or eliminate first-use extraction.

See the [online campaign runbook](eagle3_generator_campaign_online.md) for pinned
downloads, vLLM launch arguments, provenance, and experiment commands.
Cache eviction and cross-job reuse are covered by
[`test_online_campaign.py`](../../tests/unit/generator/test_online_campaign.py).
