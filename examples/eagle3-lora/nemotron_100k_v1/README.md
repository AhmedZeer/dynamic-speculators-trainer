# Nemotron expansion: 100K train + 10K validation

Each of math, code, STEM, and chat has 25 subsets of 1,000 training examples
and 100 separate validation examples: **25,000 train + 2,500 validation per
category**, or **110,000 responses total**. This is a final size, not 100K
additional requests.

The configs append to frozen `s0001`/`r0001` at trajectory commit
`1caca05caace9f2eb774141bdd8b2d3382a919f5`. That catalog contains 6,600 selected
prompts per category (six 1,000/100 subsets). Existing completed responses are
restored from that pinned release; selected prompts without responses are generated.
The existing subset order, IDs, train/validation membership, and adaptation role
of subset 00005 remain intact. Additional rows exclude all inherited source UUIDs
and aliases. As in the existing source-row sampler, repeated text and overlap
between different source domains are not deduplicated. This is not a new
train/evaluation decontamination claim.

New artifacts go to the existing public dataset
`azeeer/eagle3-qwen3-8b-trajectories-v1`, under `selections/s0002-100k` and
`releases/r0002-100k`; the original release stays intact. Source/model commits,
sampling parameters, context limit, and thinking-off behavior match the parent.
Do not change the frozen selection or parent revision midway through this run.

## Run

From the repository root, with the existing environment and Hugging Face write
authentication (`hf auth login`):

```bash
git pull
python -m pip install -e '.[generator]'
bash examples/eagle3-lora/nemotron_100k_v1/run.sh select
```

Selection is CPU/network work. It scans the pinned source once per category;
subsequent selections restore the shared frozen catalog. Do this before starting
the GPU server so initialization does not consume paid GPU time unnecessarily.

Terminal A (leave running):

```bash
bash examples/eagle3-lora/nemotron_100k_v1/run.sh serve
```

After the server is ready, terminal B:

```bash
bash examples/eagle3-lora/nemotron_100k_v1/run.sh responses
```

Rerun the same command after interruption. It restores already published chunks
and retains locally completed responses. There is no need for `--resume` or
`--resume-from-hf` with these producer configs. After generation, optionally
publish the train/validation trajectory Parquets:

```bash
bash examples/eagle3-lora/nemotron_100k_v1/run.sh data
```

These commands generate responses and publish trajectories; hidden-state
extraction remains a separate stage, needed before generator training.
Each category can also run independently with the same CLI:

```bash
speculators lora-bank prepare \
  --config examples/eagle3-lora/nemotron_100k_v1/math.yaml --stage responses
```

## Publication cadence and performance

`hub.response_chunk_size: 1000` publishes a response JSONL chunk to Hugging Face
before advancing to the next chunk. The final chunk has 500 responses per
category. When a chunk contains inherited responses, it needs fewer than 1,000
new generations. Publication waits for the whole chunk to succeed; failures stop
advancement and the same command resumes it. `execution.response_sync_interval`
only controls local disk sync and is not the Hub upload setting. Command/patch
provenance is published, including the shared server launch artifacts.

The runner uses response-only serving, BF16, client concurrency 128, vLLM
`max-num-seqs=128`, `max-num-batched-tokens=16384`, and GPU memory utilization
0.90. It keeps the original 4,096 output-token cap and 8,192 context limit.
Use local SSD for `out/data`, HF caches, and response staging.

Costs of the current implementation:

- Every 1K chunk drains before upload; long tail requests and synchronous Hub
  commits temporarily leave the GPU idle. Small immutable chunks provide recovery
  checkpoints at the cost of this throughput overhead.
- Each chunk reparses the growing local response file, and restarts regeneration
  setup. At 27.5K responses per category this adds CPU, disk I/O, and transient
  host memory costs; there is no full hidden-state cache in this response stage.
- Initial source selection scans all eligible rows for uniform sampling. Resume
  downloads/restores existing data and republishes inherited responses into the
  new release, which costs network time but avoids target-model generation.

For a tuning experiment, increase both `execution.concurrency` in the YAMLs and
`--max-num-seqs` in the runner to 256. Measure completed output tokens/second and
watch KV-cache preemptions; more concurrency is not guaranteed to be faster.
Batch-token budget can also be tuned against KV headroom. The dedicated generation
server can use more GPU memory than the earlier combined training/vLLM setup.
An asynchronous uploader, a continuously filled request queue, and incremental
response indexing would remove avoidable boundary/I/O costs while retaining 1K
Hub checkpoints; those optimizations are not implemented in this change.

Validation performed locally: parent recipe checks against the four published
selection records; synthetic tests for split preservation, disjoint appended
source IDs, reuse without extra generation calls, interrupted-upload recovery,
and unchanged existing Hub workflow tests. No GPU throughput measurement or
100K generation was run here.
