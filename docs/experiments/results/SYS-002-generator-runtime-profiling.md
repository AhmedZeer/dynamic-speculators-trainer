# SYS-002 — Generated-adapter runtime profiling

## Record and evidence

Received 2026-10-26. Preliminary, single supplied run per arm/category, output
correctness unchecked. The archive includes unprofiled benchmark summaries and
ten completed CPU/CUDA profile summaries, each with 30 worker iterations, plus
CPU/kernel CSVs. Raw timelines, complete server/checkpoint provenance and hardware
identity were not supplied. The configured workload is 16 prompts per category
with a 128-token cap; all unprofiled measurements complete 16 requests.

- [Supplied ZIP, unchanged](profiling/2026-10-26-generator-runtime/profiling-inspection.zip)
- [Derived metrics and source checksums](profiling/2026-10-26-generator-runtime/analysis.json)
- [Profiling protocol](../eagle3_runtime_profiling.md)
- [Previous per-request results](BATCH-003-per-request-continuous.md)

The labels identify the step-4000 checkpoint, singleton conditioning, and BF16
merged or unmerged execution. Actual checkpoint/source hashes cannot be verified
from these summaries. The current exports differ from the earlier pasted console
rates; the tables below use the supplied JSON exports as evidence.

## Unprofiled throughput and acceptance

Acceptance rate is accepted draft tokens divided by proposed draft tokens.
Throughput is aggregate output tokens/s, including prefill and serving overhead.
All arms within each category have matching workload hashes. At C1, output token
totals also match: 2,048 math and 1,569 chat. At C4, math totals match at 2,048;
chat has 1,573 baseline versus 1,569 generator output tokens.

| Concurrency | Arm | Math tok/s | Chat tok/s | Math acceptance | Chat acceptance |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Vanilla EAGLE3 | 60.91 | 52.71 | 47.74% | 37.30% |
| 1 | Generator, merged | 78.32 | 54.05 | 70.05% | 39.20% |
| 1 | Generator, multi-LoRA | 65.20 | 46.14 | 69.73% | 38.80% |
| 4 | Vanilla EAGLE3 | 225.40 | 173.61 | 47.62% | 36.64% |
| 4 | Generator, per-request multi-LoRA | 241.62 | 138.50 | 68.92% | 39.35% |

At C1, multi-LoRA is **16.75% slower on math and 14.64% slower on chat** than
merged generator execution. Acceptance differs by only −0.31 and −0.40 percentage
points. This control strengthens the evidence for an execution-path cost, though
small numerical/acceptance differences remain and no repetitions are available.

The merged generator improves C1 math throughput by 28.57% over vanilla; the
multi-LoRA path retains only 7.04%. Chat changes are +2.56% merged and −12.46%
multi-LoRA. At C4, the per-request generator is +7.19% on math and −20.23% on chat
relative to vanilla. These are descriptive single-run changes.

## What the bounded profiles show

The following are **instrumented trace-window costs**, not unprofiled overhead
estimates. CPU mapping and draft ranges are inclusive. They are distinct scopes,
but their constituent operation times must not be added again.

| Arm/category | Mapping CPU ms | Draft-forward CPU ms | Stream sync calls | CUDA launch calls | LoRA kernel CUDA ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 merged, math | — | 143.78 | 35 | 16,192 | — |
| C1 multi-LoRA, math | 142.18 | 258.37 | 401 | 18,660 | 4.29 |
| C1 merged, chat | — | 143.83 | 35 | 16,199 | — |
| C1 multi-LoRA, chat | 143.87 | 259.71 | 401 | 18,667 | 3.64 |
| C4 vanilla, math | — | unavailable | 29 | 16,304 | — |
| C4 per-request, math | 138.26 | 250.56 | 437 | 19,242 | 6.76 |
| C4 vanilla, chat | — | unavailable | 29 | 16,304 | — |
| C4 per-request, chat | 141.92 | 263.41 | 437 | 19,251 | 6.07 |

The generator makes 90 draft forwards and 90 mapping calls in each window:
three draft forwards per target iteration. Mapping costs approximately
1.54–1.60 ms CPU-inclusive per call under profiling, about 4.6–4.8 ms per target
iteration. The isolated shrink/expand kernels execute 360 times per window and
consume only 3.64–6.76 ms of summed CUDA time in total.

C1 math additionally shows 120 `aten::sort` and 120 `aten::_unique2` calls in the
multi-LoRA trace, versus none in the merged trace. Scalar extraction calls rise
from 62 to 1,032. CUDA launch calls increase by 2,468; only 360 are the identified
LoRA shrink/expand kernels. This identifies substantial additional metadata,
launch and host-side work, rather than simply expensive low-rank GEMMs.

The native manager caches only its last mapping. Our proposer alternates packed
verification-input queries and one-token-per-request draft queries, so repeated
layout transitions are a concrete cache-miss candidate. The pinned native
metadata path includes sorting, unique-ID processing and CPU/GPU synchronization.
The summaries establish repeated work; they do not directly count manager cache
hits or attribute every scalar extraction to a particular source line.

## GPU activity and scheduling evidence

In the C1 math control, the recorded CUDA activity span grows from 1,715.18 to
2,035.96 ms (+320.79 ms), while the CUDA busy union grows only from 493.93 to
509.32 ms (+15.39 ms). Gaps within the observed activity span grow by 305.39 ms.
Target decode-attributed CUDA activity is essentially unchanged: 407.007 versus
407.019 ms across 29 decode iterations.

At C4 math, the activity span increases by 279.50 ms, busy union by 17.84 ms, and
gaps by 261.66 ms relative to vanilla. The corresponding chat increases are
329.61, 17.48 and 312.13 ms. These are first-to-last recorded CUDA spans, not SM
utilization, complete request wall times or deployment idle fractions.

All C4 windows show the same mean scheduled request count, 3.9: one prefill
iteration, one mixed iteration and 28 decode iterations. C1 windows show one
prefill and 29 decode iterations. The generator ran once in each C1 capture and
four times in each C4 capture. No completion/refill or preemption coverage is
established from these windows. There is no occupancy disadvantage visible in
this initial captured C4 window.

Taken together, unchanged target decode work, small additional CUDA computation,
increased host work and growing activity gaps point toward routing/launch
control overhead as the leading optimization target. Profiling amplifies costs,
especially for many tiny operations. It cannot establish how many milliseconds
of unprofiled latency a particular optimization will recover. Synchronization
call counts rise sharply, but synchronization wait time alone is not the whole
explanation: C4 stream synchronization totals are only about 4 ms per window.

## Proposed next change, not implemented by this inspection

Keep native continuous admission. Optimize the per-request execution path first:

1. Reuse mapping metadata for the recurring packed-query and single-query
   layouts, with explicit invalidation on request order, query lengths,
   rejection/layout changes, adapter IDs and GPU slot assignment. A last-value
   cache already exists; useful caching must handle the alternating layouts.
2. Reduce repeated scalar GPU-to-CPU reads, metadata allocations and tiny
   launches in that path. Retain lifecycle and token-routing checks.
3. Re-run the same unprofiled C1 merged/multi-LoRA and C4 controls. Confirm
   acceptance and adapter lifecycle remain stable, then use another short trace
   to check that sort/unique, sync and mapping counts actually decreased.
4. Examine remaining low-rank kernel overhead after measuring the mapping fix.

The evidence does not justify abandoning continuous batching or concluding that
LoRA matrix multiplication is the primary bottleneck. It does justify treating
current multi-LoRA orchestration as a practical cost despite retained acceptance.
