# Supplied completed suite summary

Aggregate output tokens/s, including prefill and serving overhead. Correctness was not checked.

| Model | nemo-math | nemo-code | nemo-stem | nemo-chat | specbench-humaneval | specbench-math-reasoning | specbench-summarization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eagle3-base-c1 | 64.34 | 60.93 | 59.28 | 54.97 | 68.32 | 66.55 | 58.29 |
| eagle3-base-c2 | 116.85 | 110.79 | 108.42 | 85.08 | 121.51 | 124.04 | 108.60 |
| eagle3-base-c4 | 213.73 | 201.16 | 202.13 | 157.48 | 222.33 | 236.27 | 198.76 |
| eagle3-base-c8 | 370.40 | 344.12 | 358.96 | 269.56 | 386.17 | 419.55 | 327.71 |
| trained-lora-c1 | 72.50 | 64.79 | 64.23 | 56.35 | 67.24 | 67.19 | 58.30 |
| trained-lora-c2 | 136.08 | 122.63 | 118.62 | 92.40 | 116.13 | 122.64 | 107.49 |
| trained-lora-c4 | 248.65 | 222.99 | 220.87 | 163.45 | 215.16 | 235.21 | 197.72 |
| trained-lora-c8 | 446.95 | 389.09 | 397.33 | 278.03 | 366.84 | 433.09 | 348.00 |
| generator-24k-2000step-c1 | 77.58 | 68.11 | 66.50 | 57.58 | 69.24 | 67.99 | 58.81 |
| generator-24k-2000step-c2 | 139.21 | 124.74 | 120.36 | 92.44 | 118.98 | 124.27 | 107.19 |
| generator-24k-2000step-c4 | 256.04 | 228.38 | 219.65 | 165.02 | 217.22 | 239.08 | 194.58 |
| generator-24k-2000step-c8 | 448.07 | 397.04 | 397.62 | 277.83 | 362.63 | 422.07 | 342.32 |
| generator-4000step-c1 | 79.11 | 69.83 | 67.92 | 56.49 | 69.44 | 68.32 | 58.69 |
| generator-4000step-c2 | 142.55 | 128.85 | 121.48 | 92.11 | 122.91 | 128.52 | 109.25 |
| generator-4000step-c4 | 263.18 | 237.29 | 224.17 | 169.35 | 220.45 | 234.68 | 196.91 |
| generator-4000step-c8 | 451.59 | 393.68 | 397.94 | 283.40 | 368.11 | 427.44 | 338.82 |
