# Paired DeepSeek baseline and 4K/1,500 attention generator

Received 2026-10-31. [summary.json](summary.json) archives the supplied full-precision
suite JSON; [summary.md](summary.md) gives throughput; [analysis.json](analysis.json)
records source/reference SHA-256 values, validation, formulas, paired gains,
adapter costs and the earlier-versus-current baseline comparison. None was
retrieved directly from the execution environment.

Eight cells (two models × math/code/STEM/chat), concurrency 1, three proposals
per draft cycle. All cells report 50 successful requests with no mismatch override.
Each generator cell reports 50 prepared/completed requests/batches and 50 calls,
mean conditioning size 1. Acceptance identities, adapter means and throughput
identities pass. Per-category hashes, subset lists and output-token totals match
both models, earlier attention-generator workloads, and the earlier DeepSeek
math/chat baseline where available.

Use the baseline in this paired suite when computing gains. Its math/chat
acceptance dictionaries are identical to the earlier baseline, but throughput
is 4.94%/4.77% higher. Preserve the earlier run as separate evidence; runtime
conditions and variance are not diagnosed by two summaries.

The generator improves acceptance and throughput in all four categories in this
suite. Recorded generate/merge/restore cost is 6.54–6.57 ms/request, under 0.24%
of its measured wall time. Baseline adapter metrics are empty; there are no
isolated drafter/verifier timings. Correctness and repeated-trial uncertainty
remain unchecked. Executed checkpoint/config/preparation hashes, hardware/code
provenance, training membership and request-level files were not supplied.
There is no matched DeepSeek static-LoRA arm or conditioning-control experiment.

See [GEN-012](../../GEN-012-deepseek-generator-4k.md) for the detailed working
findings and [GEN-010](../../GEN-010-structure-serving.md) for the extended table.
These are in-distribution Nemotron results; new OOD cells remain unmeasured.
