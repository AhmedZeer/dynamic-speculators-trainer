# Generation benchmark archive

The per-request continuous-batching comparison is recorded in
[BATCH-003](../BATCH-003-per-request-continuous.md), with
[24 supplied measurements](2026-10-25-per-request-4000-v1/summary.json) and
[derived throughput/acceptance/latency comparisons](2026-10-25-per-request-4000-v1/analysis.json).
The upload contains C1/C2/C4 only; full runtime provenance, request-level records,
correctness verification and C8 results are unavailable.

The completed concurrency 1/2/4/8 comparison is recorded in
[BATCH-001](../BATCH-001-shared-adapter-concurrency.md), with its
[full supplied table](2026-10-24-comparison-v2-concurrency/summary.md),
[112 measurements](2026-10-24-comparison-v2-concurrency/metrics.json), and
[derived matched-concurrency comparisons](2026-10-24-comparison-v2-concurrency/derived_comparisons.json).
Serial values are reused from the earlier comparison. The subsequently supplied
[detailed suite summary](2026-10-24-comparison-v2-concurrency/suite_summary_detailed.json)
and [derived acceptance/timing analysis](2026-10-24-comparison-v2-concurrency/detail_analysis.json)
include aggregate acceptance and adapter counters. Native request-level GuideLLM
records and full server/checkpoint provenance remain unavailable.

The current four-model `comparison-v2` tables are recorded in
[GEN-003](../GEN-003-generator-suite-throughput.md), with separate Nemotron
in-distribution and Red Hat out-of-distribution tables and
[structured metrics](2026-10-22-comparison-v2/metrics.json).

The [complete nemoall50 comparison](2026-10-21-nemoall50/README.md) contains
the EAGLE3/LoRA/generator table across math, code, stem, and chat, with full native
measurement files and available provenance archived alongside it. These are new
measurements; they do not replace the earlier math-only records below.

Each run has its own directory containing:

- `console.txt`: supplied console output, retained as evidence.
- `metrics.json`: normalized metrics with explicit units, source path, evidence
  quality, missing provenance, and any derived values identified separately.

Archive subsequent submissions in new directories; do not replace earlier runs.
When supplied, add native `guidellm.json`, benchmark identity, correctness reports,
server configuration, command files, patches, and checkpoint checksums to the
same run directory. Until then, console-only records remain preliminary.
Unknown settings stay null rather than being inferred from current defaults.
The receipt date is not assumed to be the execution date or timezone.

| Run | Domain | Output tokens/s (server mean) | Request latency median / p95 | Evidence |
| --- | --- | --- | --- | --- |
| [scale-1-direct-v2](2026-10-21-nemomath50-scale-1-direct-v2/metrics.json) | nemo-math | 74.9 | 3.3 / 4.1 s | Console only; exact-output correctness not established |
| [eagle3-base-4](2026-10-21-nemomath50-eagle3-base-4/metrics.json) | nemo-math | 63.7 | 4.0 / 4.8 s | Console only; exact-output correctness not established |
| [scale-1-lora-v2](2026-10-21-nemomath50-scale-1-lora-v2/metrics.json) | nemo-math | 73.6 | 3.4 / 4.1 s | Console only; exact-output correctness not established |

Throughput and latency are different measurements. Keep GuideLLM's per-request
token-rate statistics separate from aggregate server token throughput. Calculate
speedups only after comparable baseline measurements are available.

## Preliminary generator versus EAGLE3 comparison

The two console logs report equal input/output token totals (4,930 / 12,800)
and concurrency 1. Generator `scale-1-direct-v2` reports 74.9 output tokens/s
versus 63.7 for `eagle3-base-4`: approximately **1.18x throughput**. Total
duration falls from 201.1 to 170.9 seconds, approximately **15.0% less time**.
Median request latency falls from 4.0 to 3.3 seconds (approximately 1.21x ratio).
Median TTFT is higher for the generator: 48.1 versus 43.4 ms.

See the [structured comparison](2026-10-21-nemomath50-generator-vs-eagle3.json)
for formulas and source metric hashes. This is a single-run, console-only
observation: prompt identity and hardware/runtime equivalence are unverified,
and exact-output correctness remains unresolved under GEN-002. Streaming tokens
per iteration do not establish speculative acceptance. No target-only timing
baseline has been supplied, so these ratios measure the observed additional
benefit over EAGLE3, not end-to-end speedup over target-only decoding.

## Preliminary trained-LoRA comparison

Trained LoRA `scale-1-lora-v2` reports **73.6 output tokens/s**, taking 174.1 s
for 12,800 output tokens. Its median/p95 request latency is 3.4/4.1 s; median
TTFT is 43.4 ms. Reported input/output token totals and concurrency match the
other two runs; workload hashes, checkpoints, training configurations, hardware,
and runtime equivalence have not been supplied for this measurement.

- [LoRA versus EAGLE3 base](2026-10-21-nemomath50-lora-vs-eagle3.json):
  approximately **1.16x throughput** (73.6 / 63.7).
- [Generator versus trained LoRA](2026-10-21-nemomath50-generator-vs-lora.json):
  approximately **1.018x throughput**, or **1.8% higher throughput**
  (74.9 / 73.6). Total measured duration is 170.9 versus 174.1 s, about 1.8%
  less time; median TTFT is higher for the generator (48.1 versus 43.4 ms).

The generator and trained LoRA perform closely in these individual runs. The
small difference does not establish a reliable generator advantage without
repeated measurements. The ordinary LoRA captures most of the observed timing
improvement over EAGLE3 base on this workload. These results do not establish
relative performance on external distributions or resolve GEN-002 correctness.
