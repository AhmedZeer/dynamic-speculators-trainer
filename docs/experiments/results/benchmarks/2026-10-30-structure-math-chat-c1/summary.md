# Benchmark suite: output tokens/s

Aggregate output token throughput includes prefill and serving overhead. Correctness was not checked.

| Model | nemotron/nemo-math | nemotron/nemo-chat |
| --- | ---: | ---: |
| static-lora-attention-4k-1500 | 75.72 | 56.48 |
| static-lora-all-4k-1500 | 77.90 | 57.26 |
| generator-attention-4k-1500 | 78.18 | 55.41 |
| generator-all-4k-1500 | 76.72 | 55.05 |
| generator-attention-decoder256-4k-1500 | 77.19 | 55.57 |
