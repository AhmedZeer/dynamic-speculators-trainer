# Experiment results

On explicit request, these working records are consolidated into the
[evolving LaTeX experimental report](../../../reporting/README.md).
The report keeps measurements, implementation status, and open questions
separate; it is not generated on a daily schedule.

This directory records observations separately from implementation plans and runbooks.
Each experiment gets a stable ID and a dated record. Add subsequent measurements
to the record with their own date and evidence; preserve earlier conclusions and
explain when new evidence changes them.

| ID | Experiment | Status | Finding |
| --- | --- | --- | --- |
| GEN-001 | [Direct generator training: data pool size](GEN-001-direct-data-scaling.md) | Preliminary; one seed, visual evidence | Similar context-4 validation agreement and losses for 4K, 8K, and 16K training pools at 1,500 updates. |
| GEN-002 | [HTTP token equality and repeatability](GEN-002-http-token-equality.md) | Failed exact equality; investigation deferred | Target-only repeat differs on 27/50 prompts; EAGLE3 repeat matches 50/50. Timing experiments proceed with an explicit correctness override. |
| GEN-003 | [Generator suite throughput: in-/out-of-distribution](GEN-003-generator-suite-throughput.md) | Preliminary; supplied suite summary | Four-model comparison: generator gains are larger on Nemotron math/code/STEM than on the selected Red Hat categories. |
| GEN-004 | [Bank warmup and direct-training decision](GEN-004-bank-warmup-adaptation.md) | Preliminary adaptation evidence; bank workflow retired | Fresh generator catches up to bank-pretrained generator; use original data directly with drafting loss. |
| SYS-001 | [Activation RAM reuse](../eagle3_activation_ram_cache.md) | Measured console pilot; no repeated trials | Recorded validation wall times: 96.90 s on the first pass and 20.06 s on a warm pass; extraction/startup effects are included. |
| BATCH-001 | [Shared-adapter concurrency sweep](BATCH-001-shared-adapter-concurrency.md) | Supplied GPU throughput summary; acceptance and repeated-run uncertainty unavailable | Generators consistently beat EAGLE3 on Nemotron; G4's mean ID gain over static LoRA narrows to 1.08% at C8, with mixed OOD results. |

Use [the results template](TEMPLATE.md) for further experiments. Report the
comparison budget, validation membership, evidence quality, limitations, and
decision explicitly. Distinguish offline agreement from measured speculative
acceptance and serving throughput. Record unavailable measurements as unavailable,
rather than estimating precise values from plots.

Generation logs and normalized timing measurements are kept in the
[generation benchmark archive](benchmarks/README.md), with one directory per run.

For reproducibility, link run IDs, exact configuration snapshots, code revision,
checkpoint hashes, and training/evaluation/serving provenance when available.
If these are unavailable, mark the record preliminary and list what is missing.
When publishing artifacts, include the associated command files, patches, and
checksums described in the repository agent guide.
