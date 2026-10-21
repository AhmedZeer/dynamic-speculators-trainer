# Nemotron: 50 prompts per category

Aggregate output tokens/s from the native measurement files. All arms use the same per-category workload hashes, concurrency 1, seed 42, and 50 successful requests. Output token totals also match per category. Correctness was not checked.

| Model | nemo-math | nemo-code | nemo-stem | nemo-chat |
| --- | ---: | ---: | ---: | ---: |
| eagle3 | 64.18 | 60.87 | 59.04 | 55.32 |
| lora | 72.56 | 65.62 | 64.26 | 56.52 |
| generator | 77.81 | 68.45 | 66.58 | 57.68 |

Rates include prefill and serving overhead; they are not isolated decode-only kernel throughput. These are individual measurements, not estimates across repeated runs.

The [structured summary](summary.json) contains exact measurements and source hashes. [Full original artifacts](artifacts.tar.gz) preserve native GuideLLM outputs, exact workloads, commands, configuration, checkpoint checksums, and available patches/provenance. Extract with `tar -xzf artifacts.tar.gz` into a separate directory.
