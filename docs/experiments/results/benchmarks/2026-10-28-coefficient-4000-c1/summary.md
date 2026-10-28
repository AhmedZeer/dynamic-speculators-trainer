# Coefficient sweep: shared/merged, concurrency 1

Output throughput includes prefill and serving overhead. Correctness was not checked.

| Coefficient | Math tok/s | Chat tok/s | Math draft acceptance | Chat draft acceptance | Math acceptance length | Chat acceptance length |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 66.97 | 53.66 | 55.38% | 37.94% | 2.662 | 2.138 |
| 4 | 64.08 | 51.33 | 51.14% | 34.73% | 2.534 | 2.042 |
| 8 | 37.10 | 36.94 | 15.36% | 15.70% | 1.461 | 1.471 |
