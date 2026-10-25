# Benchmark suite: output tokens/s

Aggregate output token throughput includes prefill and serving overhead. Correctness was not checked.

-cN denotes concurrency N; native continuous admission is enabled.

| Model | nemotron/nemo-math | nemotron/nemo-code | nemotron/nemo-stem | nemotron/nemo-chat |
| --- | ---: | ---: | ---: | ---: |
| eagle3-vanilla-c1 | 62.98 | 59.49 | 57.57 | 53.43 |
| eagle3-vanilla-c2 | 123.50 | 117.16 | 113.06 | 105.33 |
| eagle3-vanilla-c4 | 234.25 | 222.77 | 222.86 | 194.59 |
| generator-4000-per-request-c1 | 67.02 | 60.24 | 57.92 | 49.56 |
| generator-4000-per-request-c2 | 130.23 | 114.72 | 111.68 | 95.65 |
| generator-4000-per-request-c4 | 256.09 | 221.21 | 221.03 | 185.33 |
