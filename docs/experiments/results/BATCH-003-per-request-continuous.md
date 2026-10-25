# BATCH-003 — Per-request adapters against native EAGLE3

## Record and setup

- Received 2026-10-25; execution date and hardware provenance not supplied.
- Preliminary: one supplied summary, no repeat uncertainty, correctness `not_checked`.
- [Original summary](benchmarks/2026-10-25-per-request-4000-v1/summary.json),
  [throughput table](benchmarks/2026-10-25-per-request-4000-v1/summary.md), and
  [derived comparisons](benchmarks/2026-10-25-per-request-4000-v1/analysis.json).
- 24 completed measurements: two model labels × C1/C2/C4 × four Nemotron ID
  categories. C8 is absent; its execution/failure status cannot be inferred.
- Each pair uses the same workload hash and completes 50 requests. Generator
  invocation, preparation and completion counters all equal 50; conditioning
  and requests per adapter both equal 1. Mean draft batch sizes are 1.95–1.96
  at C2 and 3.77–3.84 at C4.

The intended [implementation/runbook](../eagle3_per_request_lora.md) uses one
singleton-conditioned adapter per request with native continuous admission.
The intended checkpoint is `azeeer/eagle3-generator-all24k-10ep-v1/step-00004000`;
the baseline is original EAGLE3 using original vLLM sources and the native
scheduler. The supplied summaries contain labels and counters, but not imported
source hashes, model hashes, launch settings or runtime provenance. Those exact
identities cannot be independently checked from this upload.

## Throughput relative to the baseline

Each cell is `100 × (generator aggregate output tok/s / baseline tok/s − 1)`.
Throughput includes prefill and serving overhead.

| Concurrency | Math | Code | STEM | Chat |
| --- | ---: | ---: | ---: | ---: |
| 1 | +6.42% | +1.26% | +0.59% | −7.25% |
| 2 | +5.45% | −2.09% | −1.22% | −9.19% |
| 4 | +9.33% | −0.70% | −0.82% | −4.76% |

At C4 the absolute baseline → generator rates are 234.25 → 256.09 (math),
222.77 → 221.21 (code), 222.86 → 221.03 (STEM), and 194.59 → 185.33 (chat).
The small code/STEM gaps do not establish a reproducible difference.

Summing the four category measurement durations gives 846.09 → 845.98 s at C1,
430.05 → 439.19 s at C2, and 225.81 → 224.99 s at C4. These sums exclude setup
and warmups and are not simultaneous mixed-domain measurements. They describe
an approximately unchanged total workload duration at C1/C4, with C2 about
2.12% longer. Output-token totals differ slightly for some pairs, so these are
observed workload durations rather than exact equal-token latency comparisons.

## Acceptance and latency

Acceptance length is the suite metric `1 + accepted draft tokens / draft cycles`;
it is not a percent acceptance rate. At C4:

| Category | Baseline acceptance length | Generator acceptance length | Change |
| --- | ---: | ---: | ---: |
| Math | 2.416 | 2.954 | +22.24% |
| Code | 2.299 | 2.624 | +14.14% |
| STEM | 2.230 | 2.522 | +13.09% |
| Chat | 2.058 | 2.180 | +5.93% |

The acceptance advantage is present at all three reported concurrencies. It
does not translate into a throughput win in every category.

Median TTFT is worse for the generator in all 12 matched pairs: +10.01–11.66 ms
at C1, +15.93–18.03 ms at C2, and +13.16–15.55 ms at C4. At C4 baseline medians
are 77.28–89.72 ms and generator medians are 92.38–105.26 ms.

Generator execution costs 3.12–3.30 ms/request, with adapter loading another
1.07–1.13 ms/request. Their timed total is only 0.098–0.419% of the measured
category wall time. Those one-time operations cannot alone explain the large
steady-state throughput gap relative to the acceptance improvement. Repeated
token mapping and unmerged multi-LoRA projection work are plausible costs to
profile; neither is separately measured here. Timed GPU counters also do not
account for every CPU-side operation.

## Relation to the earlier shared-adapter experiment

All four category hashes match the archived BATCH-001 workload. At C4 the old
generator label's median TTFT values were 218.41/350.18/362.20/514.96 ms for
math/code/STEM/chat; the new values are 92.46/105.26/92.38/93.05 ms. This is a
large descriptive latency improvement over those closed-cohort measurements.

This is not a controlled isolation of the scheduling change: adapter execution,
conditioning policy and baseline scheduling changed, and exact runtime/hardware
provenance is unavailable. At C1 the new generator is also slower than the old
merged generator measurement, despite no multi-request cohort being present.
That makes recurring adapter execution/routing a useful investigation target,
but does not establish its cost from cross-run differences.

## Interpretation and next check

The counters support successful execution of the new per-request path and
concurrent draft forwards. Full request-level semantic isolation and exact
output correctness remain unverified by these aggregate summaries.

The proposal-quality hypothesis holds across these ID categories. Broad
throughput acceleration over native EAGLE3 does not: math has a consistent
gain, code/STEM are close, and chat consistently regresses. Continuous admission
removes the closed-cohort deployment constraint; it does not by itself make
the current multi-LoRA implementation faster.

The smallest useful follow-up is a same-session C1 comparison of the same
checkpoint and singleton conditioning using merged versus native multi-LoRA
execution, followed by profiling token-mapping updates and V/O LoRA kernels.
Collect full server/source provenance and repeats before treating small gaps
as stable. Keep these measurements separate from verified-correctness claims.
