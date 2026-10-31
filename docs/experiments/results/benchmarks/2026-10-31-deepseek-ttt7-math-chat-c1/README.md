# DeepSeek EAGLE3 ttt7: supplied math/chat C1 results

Received 2026-10-31. [summary.md](summary.md) preserves the displayed throughput
table; [summary.json](summary.json) archives the user-supplied detailed JSON with
full numerical precision. It was not retrieved from the execution environment.
[analysis.json](analysis.json) records source/reference SHA-256 values, validation,
formulas and full-precision comparisons to the five structure models.

Both cells report 50 successful requests, with no output-mismatch override.
Acceptance counters satisfy three proposals/cycle, accepted/proposed rate,
accepted-prefix/cycle and acceptance-length identities. Output-token totals
and workload hashes match the original five models in each category. Aggregate
throughput equals output tokens / measured seconds.

Math: 82.50 output tok/s, 74.04% draft acceptance, 3.221 acceptance length.
Chat: 54.10 output tok/s, 37.13% draft acceptance, 2.114 acceptance length.
This is a completed supplied measurement, not evidence of token-level correctness.
Adapter metrics are empty, so no isolated drafter or verifier latency is available.

The [configured suite and metadata preparation](../../../../../examples/eagle3-lora/structure_benchmarks_v1/deepseek_ttt7.md)
use the pinned external checkpoint, one concurrency point and the existing
math/chat workload. Actual prepared-config/weight hashes, server/client logs,
request-level records and hardware/package/code provenance were not supplied.
Training data/budget and architecture differ from the generator/static-LoRA
experiments. Single-run differences do not establish statistical significance
or attribute gains to a specific architecture/training choice.

See the [extended table and interpretation](../../GEN-010-structure-serving.md).
