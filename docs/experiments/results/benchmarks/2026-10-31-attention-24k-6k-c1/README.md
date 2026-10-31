# Attention generator on 24K: 2K/4K/6K checkpoint results

Received 2026-10-31. [summary.json](summary.json) archives the supplied full-precision
suite JSON, [summary.md](summary.md) tabulates throughput, and
[analysis.json](analysis.json) gives validated identities, formulas, source and
reference SHA-256 values, per-checkpoint changes, adapter timing fractions and
descriptive comparisons to earlier measurements. These were supplied by the
user, not fetched from the execution environment.

Three checkpoints × math/code/STEM/chat, concurrency 1; all twelve cells report
50 successful/completed requests, 50 generator invocations and conditioning size 1.
Three-token acceptance identities, adapter means and aggregate throughput
identities pass. Hashes, subsets and output-token totals match within category
across all three checkpoints. Math/chat also match the five earlier structure
models and the separately measured DeepSeek baseline.

These are the generator trained on the original Red Hat EAGLE3 drafter, not the
new DeepSeek-based generator. Step 2,000 has the highest measured throughput in
all four categories; later checkpoints improve acceptance relative to step 2,000.
STEM acceptance peaks at 4,000 with a small decline at 6,000. The discrepancy
between better acceptance and lower throughput is not diagnosed by these data.
Timed adapter costs are approximately 6.53–6.60 ms/request, under 0.22% of wall.

One result per checkpoint/category; correctness unchecked. Actual training and
serving checkpoint/config/weight hashes, hardware/package/code provenance,
request-level logs and repeated trials are not supplied. Historical V/O rows
have matching reported data/step budgets but different precision/execution and
unverified hardware/runtime equivalence. Comparisons with 4K / 1,500 training
also change both data pool and update count. Preserve those evidence limits.

See [GEN-011](../../GEN-011-attention-24k-6k.md) for acceptance tables and
interpretation, and [GEN-010](../../GEN-010-structure-serving.md) for the extended
math/chat throughput table.


[Extended historical all-category table](extended_historical_table.md) combines
the original four-model, seven-category comparison with the three new checkpoints.
New Red Hat out-of-distribution cells are explicitly missing; all new measured
cells are Nemotron math/code/STEM/chat.
