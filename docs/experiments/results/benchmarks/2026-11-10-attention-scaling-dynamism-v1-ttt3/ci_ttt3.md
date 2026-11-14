# Bootstrap confidence intervals, TTT3

Bootstrap over the 50 prompts of each cell (5,000 resamples, seed 42); tok/s = total output tokens / total request time (C1). Acceptance proxy = (output tokens / streamed token iterations − 1) / 3 per cell; across all 69 cells it tracks vLLM's native acceptance with correlation 1.00 and sits 0.4–1.4 pp lower (mean −0.8 pp). Paired differences resample matched prompts. CIs reflect prompt sampling only, not run-to-run/server timing variance (one run per cell).

## Per cell

| category   | arm                        |   steps | tok/s [95% CI]    | acceptance proxy % [95% CI]   |
|:-----------|:---------------------------|--------:|:------------------|:------------------------------|
| chat       | Generator / correct        |    1000 | 59.0 [55.3, 62.5] | 40.1 [35.6, 44.5]             |
| chat       | Generator / correct        |    2000 | 59.4 [55.7, 63.0] | 41.0 [36.3, 45.5]             |
| chat       | Generator / correct        |    3000 | 60.4 [56.5, 64.1] | 41.1 [36.4, 45.8]             |
| chat       | Generator / correct        |    4000 | 60.5 [56.7, 64.1] | 41.7 [37.0, 46.2]             |
| chat       | Generator / correct        |    5000 | 59.6 [55.9, 63.2] | 41.9 [37.1, 46.6]             |
| chat       | Generator / correct        |    6000 | 57.9 [54.3, 61.4] | 41.7 [36.8, 46.2]             |
| chat       | Generator / cross_category |    1000 | 56.8 [53.5, 59.9] | 37.1 [32.9, 41.1]             |
| chat       | Generator / cross_category |    2000 | 57.6 [54.2, 60.8] | 37.4 [33.3, 41.4]             |
| chat       | Generator / cross_category |    3000 | 56.8 [53.6, 59.8] | 36.5 [32.4, 40.4]             |
| chat       | Generator / cross_category |    4000 | 55.0 [51.8, 58.0] | 36.2 [32.2, 40.1]             |
| chat       | Generator / cross_category |    5000 | 54.8 [51.6, 57.8] | 36.2 [32.1, 40.0]             |
| chat       | Generator / cross_category |    6000 | 55.0 [51.9, 58.0] | 35.6 [31.7, 39.4]             |
| chat       | Generator / fixed          |    1000 | 54.2 [50.9, 57.3] | 37.1 [32.8, 41.1]             |
| chat       | Generator / fixed          |    2000 | 53.9 [50.7, 56.9] | 37.2 [33.0, 41.3]             |
| chat       | Generator / fixed          |    3000 | 52.9 [50.0, 55.8] | 36.3 [32.3, 40.2]             |
| chat       | Generator / fixed          |    4000 | 54.0 [51.0, 57.0] | 36.1 [32.2, 40.0]             |
| chat       | Generator / fixed          |    5000 | 52.5 [49.5, 55.5] | 36.1 [32.1, 40.0]             |
| chat       | Generator / fixed          |    6000 | 52.6 [49.7, 55.4] | 35.4 [31.5, 39.2]             |
| chat       | Static LoRA / static       |    1000 | 56.8 [53.3, 60.1] | 39.3 [34.8, 43.6]             |
| chat       | Static LoRA / static       |    2000 | 57.2 [53.7, 60.6] | 39.6 [35.0, 44.2]             |
| chat       | Static LoRA / static       |    3000 | 58.7 [55.0, 62.3] | 40.0 [35.4, 44.5]             |
| chat       | Static LoRA / static       |    4000 | 56.1 [52.8, 59.3] | 40.1 [35.6, 44.3]             |
| chat       | Static LoRA / static       |    5000 | 57.4 [53.8, 60.8] | 40.6 [35.9, 44.9]             |
| math       | Generator / correct        |    1000 | 79.3 [77.4, 81.3] | 64.8 [62.4, 67.3]             |
| math       | Generator / correct        |    2000 | 78.1 [75.9, 80.2] | 65.4 [62.6, 68.0]             |
| math       | Generator / correct        |    3000 | 81.5 [79.5, 83.5] | 66.3 [63.8, 68.7]             |
| math       | Generator / correct        |    4000 | 78.6 [76.5, 80.7] | 66.5 [63.8, 69.1]             |
| math       | Generator / correct        |    5000 | 78.4 [76.3, 80.5] | 66.8 [64.1, 69.4]             |
| math       | Generator / correct        |    6000 | 78.6 [76.5, 80.5] | 67.4 [64.8, 69.9]             |
| math       | Generator / cross_category |    1000 | 72.2 [70.6, 73.7] | 55.8 [53.8, 57.7]             |
| math       | Generator / cross_category |    2000 | 71.0 [69.4, 72.7] | 54.4 [52.3, 56.5]             |
| math       | Generator / cross_category |    3000 | 71.9 [70.4, 73.4] | 55.1 [53.2, 57.0]             |
| math       | Generator / cross_category |    4000 | 69.9 [68.4, 71.4] | 54.8 [52.9, 56.7]             |
| math       | Generator / cross_category |    5000 | 69.3 [67.8, 70.8] | 54.8 [52.9, 56.7]             |
| math       | Generator / cross_category |    6000 | 70.1 [68.6, 71.5] | 54.3 [52.4, 56.1]             |
| math       | Generator / fixed          |    1000 | 75.5 [73.7, 77.5] | 64.8 [62.5, 67.3]             |
| math       | Generator / fixed          |    2000 | 75.6 [73.5, 77.6] | 65.5 [62.8, 68.1]             |
| math       | Generator / fixed          |    3000 | 76.3 [74.3, 78.3] | 66.2 [63.7, 68.8]             |
| math       | Generator / fixed          |    4000 | 77.7 [75.5, 79.7] | 66.7 [64.0, 69.2]             |
| math       | Generator / fixed          |    5000 | 76.0 [73.8, 78.1] | 66.9 [64.1, 69.5]             |
| math       | Generator / fixed          |    6000 | 81.0 [78.9, 83.1] | 67.5 [64.8, 70.0]             |
| math       | Static LoRA / static       |    1000 | 72.5 [70.8, 74.2] | 59.4 [57.2, 61.5]             |
| math       | Static LoRA / static       |    2000 | 74.0 [72.1, 75.8] | 61.3 [58.9, 63.6]             |
| math       | Static LoRA / static       |    3000 | 74.0 [72.2, 75.8] | 62.6 [60.1, 64.9]             |
| math       | Static LoRA / static       |    4000 | 73.7 [71.8, 75.5] | 63.0 [60.5, 65.4]             |
| math       | Static LoRA / static       |    5000 | 74.6 [73.0, 76.4] | 63.2 [61.1, 65.5]             |
| stem       | Generator / correct        |    1000 | 67.1 [65.6, 68.6] | 49.9 [48.1, 51.8]             |
| stem       | Generator / correct        |    2000 | 67.0 [65.3, 68.7] | 50.7 [48.6, 52.8]             |
| stem       | Generator / correct        |    3000 | 68.4 [66.8, 70.0] | 50.7 [48.7, 52.7]             |
| stem       | Generator / correct        |    4000 | 66.2 [64.7, 67.7] | 51.0 [49.1, 52.9]             |
| stem       | Generator / correct        |    5000 | 65.9 [64.3, 67.5] | 51.2 [49.2, 53.2]             |
| stem       | Generator / correct        |    6000 | 66.8 [65.2, 68.4] | 51.3 [49.2, 53.4]             |
| stem       | Generator / cross_category |    1000 | 64.0 [62.5, 65.5] | 45.9 [44.1, 47.7]             |
| stem       | Generator / cross_category |    2000 | 63.6 [62.1, 65.1] | 45.6 [43.8, 47.4]             |
| stem       | Generator / cross_category |    3000 | 64.1 [62.7, 65.5] | 44.7 [43.0, 46.4]             |
| stem       | Generator / cross_category |    4000 | 61.4 [60.0, 62.8] | 44.1 [42.4, 45.8]             |
| stem       | Generator / cross_category |    5000 | 61.4 [60.1, 62.8] | 44.3 [42.6, 46.0]             |
| stem       | Generator / cross_category |    6000 | 60.8 [59.5, 62.1] | 43.5 [41.9, 45.2]             |
| stem       | Generator / fixed          |    1000 | 60.3 [58.9, 61.7] | 45.5 [43.8, 47.3]             |
| stem       | Generator / fixed          |    2000 | 60.8 [59.3, 62.3] | 45.3 [43.4, 47.1]             |
| stem       | Generator / fixed          |    3000 | 59.2 [57.9, 60.5] | 44.1 [42.4, 45.8]             |
| stem       | Generator / fixed          |    4000 | 60.3 [59.0, 61.6] | 43.9 [42.3, 45.6]             |
| stem       | Generator / fixed          |    5000 | 58.1 [56.8, 59.4] | 43.5 [41.9, 45.1]             |
| stem       | Generator / fixed          |    6000 | 61.9 [60.5, 63.2] | 43.2 [41.5, 44.8]             |
| stem       | Static LoRA / static       |    1000 | 63.5 [62.0, 65.0] | 47.4 [45.5, 49.3]             |
| stem       | Static LoRA / static       |    2000 | 63.8 [62.3, 65.3] | 48.3 [46.4, 50.2]             |
| stem       | Static LoRA / static       |    3000 | 63.8 [62.2, 65.3] | 48.6 [46.6, 50.7]             |
| stem       | Static LoRA / static       |    4000 | 63.1 [61.6, 64.6] | 49.1 [47.1, 51.0]             |
| stem       | Static LoRA / static       |    5000 | 63.8 [62.2, 65.3] | 49.4 [47.4, 51.3]             |

## Paired differences (a − b on identical prompts)

| comparison       | category   |   steps |   n_paired | Δ tok/s [95% CI]    | Δ acceptance pp [95% CI]   |
|:-----------------|:-----------|--------:|-----------:|:--------------------|:---------------------------|
| correct-cross    | chat       |    1000 |         50 | 2.2 [1.6, 3.0]      | 3.0 [2.3, 3.8]             |
| correct-cross    | chat       |    2000 |         50 | 1.8 [1.1, 2.5]      | 3.6 [2.8, 4.5]             |
| correct-cross    | chat       |    3000 |         50 | 3.6 [2.6, 4.7]      | 4.6 [3.5, 5.8]             |
| correct-cross    | chat       |    4000 |         50 | 5.5 [4.6, 6.6]      | 5.5 [4.5, 6.6]             |
| correct-cross    | chat       |    5000 |         50 | 4.8 [3.8, 5.9]      | 5.7 [4.5, 7.0]             |
| correct-cross    | chat       |    6000 |         50 | 2.9 [1.9, 4.0]      | 6.0 [4.7, 7.4]             |
| correct-cross    | math       |    1000 |         50 | 7.1 [6.0, 8.2]      | 9.1 [7.8, 10.4]            |
| correct-cross    | math       |    2000 |         50 | 7.1 [5.8, 8.4]      | 11.0 [9.5, 12.5]           |
| correct-cross    | math       |    3000 |         50 | 9.6 [8.2, 11.0]     | 11.2 [9.5, 12.9]           |
| correct-cross    | math       |    4000 |         50 | 8.6 [7.2, 10.0]     | 11.6 [9.9, 13.3]           |
| correct-cross    | math       |    5000 |         50 | 9.2 [7.6, 10.6]     | 12.0 [10.3, 13.7]          |
| correct-cross    | math       |    6000 |         50 | 8.5 [7.2, 9.8]      | 13.1 [11.4, 14.7]          |
| correct-cross    | stem       |    1000 |         50 | 3.1 [2.3, 3.9]      | 4.0 [3.2, 4.8]             |
| correct-cross    | stem       |    2000 |         50 | 3.4 [2.6, 4.2]      | 5.1 [4.2, 6.0]             |
| correct-cross    | stem       |    3000 |         50 | 4.3 [3.5, 5.1]      | 6.0 [5.0, 7.1]             |
| correct-cross    | stem       |    4000 |         50 | 4.8 [3.9, 5.7]      | 6.9 [5.9, 8.0]             |
| correct-cross    | stem       |    5000 |         50 | 4.5 [3.6, 5.4]      | 6.9 [5.9, 8.1]             |
| correct-cross    | stem       |    6000 |         50 | 6.0 [5.1, 7.0]      | 7.8 [6.7, 9.0]             |
| correct-fixed    | chat       |    1000 |         50 | 4.8 [4.1, 5.5]      | 3.0 [2.3, 3.9]             |
| correct-fixed    | chat       |    2000 |         50 | 5.5 [4.8, 6.3]      | 3.7 [2.9, 4.6]             |
| correct-fixed    | chat       |    3000 |         50 | 7.5 [6.4, 8.6]      | 4.8 [3.8, 6.0]             |
| correct-fixed    | chat       |    4000 |         50 | 6.5 [5.5, 7.5]      | 5.6 [4.5, 6.8]             |
| correct-fixed    | chat       |    5000 |         50 | 7.0 [6.0, 8.2]      | 5.8 [4.6, 7.2]             |
| correct-fixed    | chat       |    6000 |         50 | 5.3 [4.3, 6.4]      | 6.2 [5.0, 7.7]             |
| correct-fixed    | math       |    1000 |         50 | 3.8 [3.4, 4.2]      | 0.0 [-0.3, 0.3]            |
| correct-fixed    | math       |    2000 |         50 | 2.5 [2.1, 3.0]      | -0.1 [-0.4, 0.2]           |
| correct-fixed    | math       |    3000 |         50 | 5.2 [4.8, 5.6]      | 0.1 [-0.3, 0.4]            |
| correct-fixed    | math       |    4000 |         50 | 0.9 [0.5, 1.3]      | -0.3 [-0.6, 0.1]           |
| correct-fixed    | math       |    5000 |         50 | 2.5 [1.9, 3.0]      | -0.1 [-0.4, 0.3]           |
| correct-fixed    | math       |    6000 |         50 | -2.4 [-2.9, -1.9]   | -0.1 [-0.4, 0.3]           |
| correct-fixed    | stem       |    1000 |         50 | 6.8 [6.2, 7.5]      | 4.4 [3.6, 5.2]             |
| correct-fixed    | stem       |    2000 |         50 | 6.2 [5.3, 7.0]      | 5.5 [4.5, 6.4]             |
| correct-fixed    | stem       |    3000 |         50 | 9.2 [8.4, 10.0]     | 6.6 [5.6, 7.6]             |
| correct-fixed    | stem       |    4000 |         50 | 5.9 [5.1, 6.8]      | 7.2 [6.2, 8.2]             |
| correct-fixed    | stem       |    5000 |         50 | 7.8 [7.0, 8.7]      | 7.7 [6.7, 8.8]             |
| correct-fixed    | stem       |    6000 |         50 | 4.9 [4.0, 5.8]      | 8.2 [7.1, 9.3]             |
| cross-fixed      | chat       |    1000 |         50 | 2.5 [2.4, 2.8]      | 0.0 [-0.2, 0.2]            |
| cross-fixed      | chat       |    2000 |         50 | 3.7 [3.4, 4.0]      | 0.1 [-0.1, 0.3]            |
| cross-fixed      | chat       |    3000 |         50 | 3.9 [3.5, 4.3]      | 0.2 [-0.0, 0.5]            |
| cross-fixed      | chat       |    4000 |         50 | 0.9 [0.7, 1.1]      | 0.1 [-0.1, 0.4]            |
| cross-fixed      | chat       |    5000 |         50 | 2.3 [2.0, 2.6]      | 0.1 [-0.1, 0.3]            |
| cross-fixed      | chat       |    6000 |         50 | 2.4 [2.1, 2.7]      | 0.2 [-0.0, 0.5]            |
| cross-fixed      | math       |    1000 |         50 | -3.3 [-4.3, -2.3]   | -9.1 [-10.4, -7.7]         |
| cross-fixed      | math       |    2000 |         50 | -4.5 [-5.7, -3.4]   | -11.1 [-12.5, -9.6]        |
| cross-fixed      | math       |    3000 |         50 | -4.4 [-5.8, -3.0]   | -11.2 [-12.9, -9.4]        |
| cross-fixed      | math       |    4000 |         50 | -7.7 [-9.2, -6.3]   | -11.9 [-13.5, -10.3]       |
| cross-fixed      | math       |    5000 |         50 | -6.7 [-8.2, -5.2]   | -12.1 [-13.8, -10.3]       |
| cross-fixed      | math       |    6000 |         50 | -10.9 [-12.2, -9.6] | -13.2 [-14.8, -11.5]       |
| cross-fixed      | stem       |    1000 |         50 | 3.7 [3.4, 4.1]      | 0.3 [0.1, 0.6]             |
| cross-fixed      | stem       |    2000 |         50 | 2.8 [2.4, 3.2]      | 0.4 [0.0, 0.7]             |
| cross-fixed      | stem       |    3000 |         50 | 4.9 [4.6, 5.3]      | 0.6 [0.4, 0.9]             |
| cross-fixed      | stem       |    4000 |         50 | 1.1 [0.8, 1.4]      | 0.3 [0.0, 0.5]             |
| cross-fixed      | stem       |    5000 |         50 | 3.3 [3.0, 3.7]      | 0.8 [0.5, 1.2]             |
| cross-fixed      | stem       |    6000 |         50 | -1.1 [-1.5, -0.8]   | 0.4 [0.1, 0.7]             |
| generator-static | chat       |    1000 |         50 | 2.2 [1.7, 2.8]      | 0.9 [0.3, 1.5]             |
| generator-static | chat       |    2000 |         50 | 2.1 [1.6, 2.7]      | 1.3 [0.7, 1.9]             |
| generator-static | chat       |    3000 |         50 | 1.7 [1.1, 2.3]      | 1.1 [0.6, 1.7]             |
| generator-static | chat       |    4000 |         50 | 4.4 [3.8, 5.1]      | 1.6 [1.1, 2.3]             |
| generator-static | chat       |    5000 |         50 | 2.2 [1.7, 2.8]      | 1.4 [0.7, 2.0]             |
| generator-static | math       |    1000 |         50 | 6.8 [5.9, 7.7]      | 5.5 [4.4, 6.5]             |
| generator-static | math       |    2000 |         50 | 4.1 [3.2, 5.1]      | 4.1 [3.0, 5.2]             |
| generator-static | math       |    3000 |         50 | 7.5 [6.6, 8.3]      | 3.6 [2.7, 4.7]             |
| generator-static | math       |    4000 |         50 | 4.9 [3.9, 5.8]      | 3.5 [2.4, 4.5]             |
| generator-static | math       |    5000 |         50 | 3.8 [2.8, 4.8]      | 3.6 [2.6, 4.6]             |
| generator-static | stem       |    1000 |         50 | 3.6 [3.0, 4.2]      | 2.5 [1.8, 3.2]             |
| generator-static | stem       |    2000 |         50 | 3.2 [2.6, 3.7]      | 2.4 [1.8, 3.1]             |
| generator-static | stem       |    3000 |         50 | 4.6 [4.0, 5.2]      | 2.0 [1.4, 2.7]             |
| generator-static | stem       |    4000 |         50 | 3.1 [2.3, 3.9]      | 2.0 [1.3, 2.7]             |
| generator-static | stem       |    5000 |         50 | 2.1 [1.5, 2.8]      | 1.9 [1.1, 2.6]             |
