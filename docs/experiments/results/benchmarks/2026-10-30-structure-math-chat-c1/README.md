# Structure models: supplied math/chat C1 measurements

Received 2026-10-30 in two parts:

- [summary.md](summary.md) preserves the initial rounded throughput table.
  [metrics.json](metrics.json) transcribes those values; its missing-evidence
  statements apply to that initial receipt.
- [analysis.json](analysis.json) preserves the initial rounded-value comparisons
  and separately calculated configured parameter counts.
- [suite_summary_detailed.json](suite_summary_detailed.json) archives the later
  user-supplied suite JSON, preserving numerical precision and fields. It adds
  aggregate acceptance, adapter timings, workload hashes and request counts.
  It was not retrieved directly from the execution environment; original
  request-level files and runtime provenance were not supplied.
- [detailed_analysis.json](detailed_analysis.json) derives current comparisons
  from the detailed summary and records its SHA-256. Regenerate and validate
  with `python3 docs/experiments/results/benchmarks/2026-10-30-structure-math-chat-c1/analyze.py`
  from the repository root (standard library only).

Configured intent: five step-1500 models trained on 4K rows, shared/merged
adapters at concurrency 1, 50 math and 50 chat prompts, output cap 256, seed 42.
All ten detailed measurements report 50 successful and completed requests;
workload hashes and output-token totals agree across models within each
category. All acceptance and throughput identities pass validation.

Correctness was not checked. Request-level records, executed checkpoint hashes,
hardware/code/package provenance and repeated measurements remain unavailable.
CUDA-event adapter durations do not isolate drafter/verifier time and are not
a complete wall-time partition.

See [GEN-010](../../GEN-010-structure-serving.md) for the extended historical
table, updated acceptance and timing interpretation, and evidence limits.
