# Structure models: supplied math/chat C1 throughput

Received 2026-10-30. [summary.md](summary.md) preserves the supplied table.
[metrics.json](metrics.json) transcribes its ten rounded throughput values and
records missing evidence; it is not the unavailable native `summary.json`.
[analysis.json](analysis.json) contains explicitly derived pairwise percentage
changes and separately calculated parameter counts for the two attention generators.

Configured intent: five step-1500 models trained on 4K rows, shared/merged
adapters at concurrency 1, 50 math and 50 chat prompts, output cap 256, seed 42.
Executed workload hashes, native request/acceptance/latency records, checkpoint
hashes, and hardware/code provenance were not supplied. Correctness was not checked.

See [GEN-010](../../GEN-010-structure-serving.md) for the extended historical
table, interpretation and limits. Do not infer acceptance or generator latency
from aggregate output tokens/s.
