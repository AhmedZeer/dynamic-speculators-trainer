# Runtime profiling: merged and per-request EAGLE3 adapters

Implemented 2026-10-25. The command runs the ordinary unprofiled measurement
first, then starts fresh servers for bounded CPU/CUDA traces. These are inference
runs; there is no training or extra `check-endpoint` pass. Native CUDA execution
must be checked in the user's vLLM/A100 environment; development checks include
real CPU Torch traces and a real GuideLLM client against a mock profile endpoint.

## Questions and controls

The question is why improved draft acceptance does not translate into comparable
throughput gains. Trace CPU routing, synchronization, low-rank kernels, normal
model kernels, transfers and GPU activity gaps separately.

| Config | Comparison | Concurrency | ID categories |
| --- | --- | ---: | --- |
| `profile_generator_serial.yaml` | Original EAGLE3; generator merged into weights per request; generator through native multi-LoRA | 1 | Math, chat |
| `profile_generator_continuous.yaml` | Original EAGLE3; per-request generator through native multi-LoRA | 4 | Math, chat |

All generator entries use the same pinned step-4000 bundle, last-token prompt
conditioning, K=1, rank 32 and alpha 64. Both phases use 16 prompts/category,
128-token caps, seed 42, three unrecorded warmup requests/category, BF16 and eager
V1 execution. The serial merged arm has a model-specific batching-policy
override; at C1 there is no multi-request admission barrier. All C4 arms use the
native continuous scheduler. Original-source hashes are checked for the baseline;
no profiler-specific model/runner patch is installed on it.

The merged/unmerged comparison includes numerical differences from BF16
rounding, and any resulting acceptance differences must be retained. Matching
the generator checkpoint and input prompts does not guarantee identical output
trajectories or exact proposal equality. Both profiler and benchmark results
retain `correctness_status=not_checked`.

## Commands

Use the existing vLLM **0.31.0** environment, with this repo installed editable
and the existing GuideLLM dependencies. Stop old servers, then run from the repo:

```bash
git pull
python scripts/apply_vllm_generator_patch.py --apply

speculators benchmark profile \
  --config examples/eagle3-lora/profile_generator_serial.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-profiles/serial-v1

speculators benchmark profile \
  --config examples/eagle3-lora/profile_generator_continuous.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-profiles/continuous-v1
```

Append `--resume` to either profiling command after interruption. Completed
unprofiled and profiled measurements are checked and reused independently. Add
`--dry-run` to inspect the recipe without model downloads or server startup.
Both phases use the exact checkpoint revisions and frozen workloads resolved
by the first phase. Changing profile/workload settings requires a new root.
The suite owns server/client startup and process-group cleanup.

## Capture lifecycle

The second phase passes vLLM's native `--profiler-config` through the launcher:
CPU/CUDA activities, gzip traces, `ignore_frontend=true`, and a 30-iteration
limit. Stack, shape and memory capture default to false to keep captures small;
the YAML exposes all three if a deeper trace is needed.

After tokenization checks and warmups, the client snapshots metrics and posts
`/start_profile`. It runs GuideLLM once, then posts `/stop_profile` in a `finally`
block. The stop call waits for trace flushing, including on client failure.
Native vLLM can stop automatically at the iteration limit; the final stop call
is idempotent. Traces are copied per measurement after flushing, without
including previous categories' traces. A missing trace, missing native worker
iteration annotations, or missing CUDA activity fails the measurement.

The window covers at most the configured worker iterations, not all requests.
At 128 output tokens a 30-iteration capture may finish before request completion
or batch refill. Read phase/occupancy counts to determine coverage; a later
larger-window run can examine refill or preemption specifically. Do not infer
untested scheduling behavior from an initial fully occupied decode window.

The existing controller `profiling()` method still serves its original purpose:
memory sizing during engine startup. Runtime tracing is a separate mechanism.
The server records the profiler config and normal launch/model/source provenance.

## Artifacts and interpretation

Under each output root:

```text
benchmark/summary.md                 # unprofiled aggregate output tokens/s
benchmark/summary.json               # acceptance, latency and adapter counters
profiled/profile_summary.md          # measurement index with links to details
profiled/summary.json                # diagnostic index; no throughput table
profiled/measurements/<model>/<dataset>/<category>/attempt-NNNN/
  profile_capture.json              # config, trace checksums and scope
  profile_summary.md                # CPU and CUDA operation tables
  profile_summary.json              # structured accounting and coverage
  cpu_operations.csv
  kernel_summary.csv
  traces/                           # original native JSON/gzip traces and tables
  results.json                      # full-workload metrics, marked diagnostic
```

The profiled index links the corresponding **unprofiled** throughput and
aggregate seconds per output token. That wall-time/token ratio is not per-request
TPOT. Native request-level GuideLLM results and Prometheus before/after snapshots
remain available in each measurement. Include launch provenance when sharing
results; the client copies verified baseline source files, not the entire
installed package or earlier trace payloads.

Open native `.pt.trace.json.gz` files directly in Perfetto. Tables summarize:

- CPU inclusive and exclusive time, with recorded child intervals subtracted.
- CUDA kernel, copy and memset durations, call counts and mean operation time.
- CPU synchronization calls and CUDA attribution to native/custom ranges where
  launch correlation/external IDs are available.
- Native worker iteration counts, prefill/decode/mixed phase counts and mean
  scheduled requests per captured iteration.
- CUDA activity union time and gaps between the first and last recorded CUDA
  activity. This is **not SM utilization**, and excludes startup/export idle time
  outside that span.

Custom ranges cover `generator`, `adapter_to_cpu`, `adapter_prepare`,
`adapter_load`, `adapter_pack`, `adapter_upload`, `token_mapping_build`,
`token_mapping`, `adapter_cleanup`, `adapter_merge`, `adapter_restore`,
`draft_forward`, and the native QKV/O projection wrappers. Projection ranges
include their base GEMMs; use individual shrink/expand kernel rows for the
low-rank branch. These ranges are enabled only for profiling launches. Native
baseline traces have no custom adapter ranges.

CPU totals are nested; CUDA operations can overlap across streams. Do not add
those totals as an end-to-end latency breakdown. Missing CUDA attribution stays
unknown; it is not automatically classified as target verification. Native
worker iteration annotations anchor `vllm.execute_model.prefill/decode/mixed`
attribution. Those scopes include target execution and its input preparation;
they are not pure verification-kernel timers. Operator names, optional
shapes/stacks and the timeline support further inspection of target kernels.
This pass does not collect hardware DRAM-bandwidth or SM-occupancy counters.

Acceptance and request counters cover the **whole client workload**, while
trace summaries cover the bounded window. Exact traced CUDA time per accepted
or output token is unavailable from these aggregates. The trace identifies
candidate costs; confirm any improvement with an unprofiled run.

## Diagnosis to make next

- Large mapping CPU time/synchronization: inspect mapping rebuild frequency and
  caching, CPU/GPU metadata round trips and small metadata kernels.
- Large low-rank kernel time or many short launches: inspect the native V/O
  adapter execution path and opportunities to reduce launches.
- Similar draft cost but high idle gaps: inspect worker iteration boundaries,
  prefill/decode transitions and actual batch occupancy in the timeline.
- Large merge/unmerged difference at C1: quantify recurring adapter execution
  cost before attributing the C4 result to batching policy.
- Similar execution costs: inspect acceptance differences and target verification
  work per emitted token using the unprofiled counters and per-request records.

No performance optimization or new speedup conclusion is implied by adding
these diagnostics. The ordinary benchmark command never enables profiling.
