# GEN-008 — Fixed LoRA coefficient, step-4000 generator

## Record

- Receipt date: 2026-10-28; execution date and hardware not supplied.
- Status: preliminary, six supplied aggregate measurements, no repeats.
- [Supplied summary JSON](benchmarks/2026-10-28-coefficient-4000-c1/summary.json),
  [measurement table](benchmarks/2026-10-28-coefficient-4000-c1/summary.md), and
  [derived comparisons](benchmarks/2026-10-28-coefficient-4000-c1/analysis.json).
- Suite identity: `658b0f75ab30477a6e0881812f93a3e39d4d2e2ade17f503c6efecdb3782cfa3`.
- Correctness: `not_checked`. Output mismatch override: false. These do not
  establish token equality or semantic correctness.

## Question and setup

Vary only the effective coefficient in `W_effective = W_base + s * B @ A`, using
the intended pinned step-4000 generator, shared/merged execution, singleton prompt
conditioning and concurrency 1. Requested coefficients: 0.5, 4, 8. There is no
new training or adaptive coefficient controller in the configured experiment.

See [configs and commands](../../../examples/eagle3-lora/coefficient_sweep_4000/README.md).
The configured source is `azeeer/eagle3-generator-all24k-10ep-v1`, revision
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`, subfolder `step-00004000`.
Executed server/model hashes and the resolved plan were not provided, so those
identities are configured intent rather than independently verified provenance.

All three coefficients have identical category prompt hashes, 50 completed
requests/category, and identical output totals: math 12800, chat 11032. Every
measurement reports mean batch/conditioning size 1, 50 generator calls and zero
adapter refreshes. Effective scale means and scale sum/count identities confirm
the requested constant coefficient in each measurement.

The math/chat prompt hashes also match the archived shared-adapter coefficient-2
run in [BATCH-001's detailed summary](benchmarks/2026-10-24-comparison-v2-concurrency/suite_summary_detailed.json).
Its historical acceptance counters imply 65.34% draft acceptance and 2.960
acceptance length for math, and 39.65% / 2.189 for chat. These are separate-run
references, not additional results from the current sweep.

Counters named `feedback_*` record verifier outcomes even when feedback conditioning
is disabled; their presence does not imply feedback-trained weights or a controller.
The zero refresh count and one generator call/request are consistent with a fixed
prompt-generated adapter. Output totals/hashes do not prove output-token equality.

## Measurements

| Coefficient | Math output tok/s | Chat output tok/s | Math draft acceptance | Chat draft acceptance | Math acceptance length | Chat acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 66.97 | 53.66 | 55.38% | 37.94% | 2.662 | 2.138 |
| 2 — historical | 79.11 | 56.49 | 65.34% | 39.65% | 2.960 | 2.189 |
| 4 | 64.08 | 51.33 | 51.14% | 34.73% | 2.534 | 2.042 |
| 8 | 37.10 | 36.94 | 15.36% | 15.70% | 1.461 | 1.471 |

Throughput is the aggregate server output token rate, including prefill and
serving overhead. Draft acceptance is accepted tokens / proposed draft tokens.
Acceptance length is `1 + accepted tokens / draft cycles`, including the verifier
bonus token; it is not a percentage. Full precision and per-position rates are
preserved in the supplied JSON.

Relative to coefficient 0.5, coefficient 4 reduces throughput by 4.31% in math and
4.34% in chat. Coefficient 8 reduces it by **44.60% in math and 31.16% in chat**.
Acceptance and acceptance length fall in both categories at each increase.

Generator + merge + restore timings stay near 0.176 seconds per math measurement
and 0.179–0.180 seconds per chat measurement across all scales. Median TTFT is
also similar (math about 49.7–49.8 ms; chat about 50.3–50.9 ms). At coefficient 8,
median TPOT rises to about 26.7 ms, from 14.7 ms math / 17.8 ms chat at 0.5.
The large slowdown accompanies lower proposal acceptance and more draft cycles,
not a measured increase in one-time generator execution cost.

## Extended historical throughput table

| Model | Math output tok/s | Chat output tok/s |
| --- | ---: | ---: |
| EAGLE3 base — historical | 64.34 | 54.97 |
| Trained LoRA — historical | 72.50 | 56.35 |
| Generator, 2000 steps — historical | 77.58 | 57.58 |
| Generator, 4000 steps, coefficient 2 — historical | 79.11 | 56.49 |
| Generator, 4000 steps, coefficient 0.5 — new | 66.97 | 53.66 |
| Generator, 4000 steps, coefficient 4 — new | 64.08 | 51.33 |
| Generator, 4000 steps, coefficient 8 — new | 37.10 | 36.94 |

Historical rows are from [GEN-003](GEN-003-generator-suite-throughput.md), not
reruns in this sweep. In particular, coefficient 2 has no new acceptance
measurement here. Historical acceptance is from the archived detailed summary
linked above. Its historical throughput and acceptance are higher, but runtime/hardware
provenance and repeat uncertainty are missing. This combined table is an archive
extension, not a controlled seven-arm same-session benchmark.

## Interpretation and next check

Among the three newly measured coefficients, **0.5 is best in both categories**
for throughput and acceptance. Coefficient 8 strongly harms proposal quality in
this setup. Scaling the adapter more aggressively does not monotonically improve
the drafter. No universal optimum or statistically reliable 0.5-versus-4 gap is
established from single measurements.

The essential missing control is coefficient 2 under the same current runtime,
hardware and workload. It is needed before selecting 0.5 over the existing default
or treating the historical 79.11/56.49 rates as a direct coefficient comparison.
No additional experiment is launched or added to the requested sweep here.

## Updates

- 2026-10-28: Archived six supplied results and extended the historical table.
