# Generation benchmark archive

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
