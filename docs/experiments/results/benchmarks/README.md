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

Throughput and latency are different measurements. Keep GuideLLM's per-request
token-rate statistics separate from aggregate server token throughput. Calculate
speedups only after comparable baseline measurements are available.
