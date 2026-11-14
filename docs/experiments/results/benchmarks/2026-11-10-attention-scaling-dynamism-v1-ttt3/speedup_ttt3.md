# Speedup vs target-only baseline (no drafter), TTT3

Speedup = output tok/s ÷ target-only Qwen3-8B tok/s on the same 50 prompts per category (C1, max 256 tokens, greedy, seed 42, identical vLLM flags; target-only via `launch_vllm.py responses`, original drafter via `launch_vllm.py serve --drafter`). 95% CI: paired bootstrap over prompts (5,000 resamples). Baseline and original-drafter runs used a different A100 VM (2026-11-10 13:50–15:03 UTC) than the 69 suite cells (04:57–11:55 UTC); VM-to-VM timing variation is not in the CIs. Acceptance column: per-request proxy (tracks native within ~1 pp; original drafter native: math 47.0, STEM 40.1, chat 36.1).

## Math — baseline 29.18 tok/s

| arm                        | steps   |   tok/s | speedup [95% CI]   |   acceptance proxy % |
|:---------------------------|:--------|--------:|:-------------------|---------------------:|
| Original Eagle3 drafter    | —       |   64.31 | 2.20× [2.14, 2.27] |                46.19 |
| Static LoRA                | 1K      |   72.49 | 2.48× [2.42, 2.54] |                59.36 |
| Static LoRA                | 2K      |   73.99 | 2.54× [2.47, 2.60] |                61.29 |
| Static LoRA                | 3K      |   74.03 | 2.54× [2.47, 2.60] |                62.63 |
| Static LoRA                | 4K      |   73.7  | 2.52× [2.46, 2.59] |                63    |
| Static LoRA                | 5K      |   74.64 | 2.56× [2.50, 2.62] |                63.22 |
| Generator / correct        | 1K      |   79.28 | 2.72× [2.65, 2.79] |                64.82 |
| Generator / correct        | 2K      |   78.12 | 2.68× [2.60, 2.75] |                65.36 |
| Generator / correct        | 3K      |   81.5  | 2.79× [2.72, 2.86] |                66.28 |
| Generator / correct        | 4K      |   78.57 | 2.69× [2.62, 2.76] |                66.45 |
| Generator / correct        | 5K      |   78.43 | 2.69× [2.61, 2.76] |                66.82 |
| Generator / correct        | 6K      |   78.58 | 2.69× [2.62, 2.77] |                67.41 |
| Generator / fixed          | 1K      |   75.5  | 2.59× [2.52, 2.65] |                64.82 |
| Generator / fixed          | 2K      |   75.57 | 2.59× [2.52, 2.66] |                65.46 |
| Generator / fixed          | 3K      |   76.31 | 2.62× [2.55, 2.68] |                66.24 |
| Generator / fixed          | 4K      |   77.66 | 2.66× [2.58, 2.73] |                66.71 |
| Generator / fixed          | 5K      |   75.98 | 2.60× [2.53, 2.68] |                66.87 |
| Generator / fixed          | 6K      |   81.01 | 2.78× [2.70, 2.85] |                67.49 |
| Generator / cross-category | 1K      |   72.18 | 2.47× [2.42, 2.53] |                55.76 |
| Generator / cross-category | 2K      |   71.05 | 2.44× [2.38, 2.49] |                54.39 |
| Generator / cross-category | 3K      |   71.91 | 2.46× [2.41, 2.52] |                55.08 |
| Generator / cross-category | 4K      |   69.93 | 2.40× [2.34, 2.45] |                54.84 |
| Generator / cross-category | 5K      |   69.27 | 2.37× [2.32, 2.43] |                54.78 |
| Generator / cross-category | 6K      |   70.09 | 2.40× [2.35, 2.46] |                54.3  |

## STEM — baseline 29.16 tok/s

| arm                        | steps   |   tok/s | speedup [95% CI]   |   acceptance proxy % |
|:---------------------------|:--------|--------:|:-------------------|---------------------:|
| Original Eagle3 drafter    | —       |   59.28 | 2.03× [1.99, 2.08] |                39.54 |
| Static LoRA                | 1K      |   63.47 | 2.18× [2.12, 2.23] |                47.38 |
| Static LoRA                | 2K      |   63.83 | 2.19× [2.14, 2.24] |                48.29 |
| Static LoRA                | 3K      |   63.76 | 2.19× [2.13, 2.24] |                48.66 |
| Static LoRA                | 4K      |   63.11 | 2.16× [2.11, 2.21] |                49.08 |
| Static LoRA                | 5K      |   63.75 | 2.19× [2.13, 2.24] |                49.35 |
| Generator / correct        | 1K      |   67.09 | 2.30× [2.25, 2.35] |                49.9  |
| Generator / correct        | 2K      |   66.98 | 2.30× [2.24, 2.36] |                50.72 |
| Generator / correct        | 3K      |   68.4  | 2.35× [2.29, 2.40] |                50.71 |
| Generator / correct        | 4K      |   66.2  | 2.27× [2.22, 2.32] |                51.04 |
| Generator / correct        | 5K      |   65.9  | 2.26× [2.20, 2.32] |                51.2  |
| Generator / correct        | 6K      |   66.81 | 2.29× [2.23, 2.35] |                51.32 |
| Generator / fixed          | 1K      |   60.27 | 2.07× [2.02, 2.11] |                45.53 |
| Generator / fixed          | 2K      |   60.79 | 2.08× [2.03, 2.14] |                45.26 |
| Generator / fixed          | 3K      |   59.18 | 2.03× [1.99, 2.08] |                44.1  |
| Generator / fixed          | 4K      |   60.27 | 2.07× [2.02, 2.11] |                43.86 |
| Generator / fixed          | 5K      |   58.1  | 1.99× [1.95, 2.04] |                43.49 |
| Generator / fixed          | 6K      |   61.89 | 2.12× [2.08, 2.17] |                43.17 |
| Generator / cross-category | 1K      |   64    | 2.19× [2.14, 2.25] |                45.87 |
| Generator / cross-category | 2K      |   63.6  | 2.18× [2.13, 2.23] |                45.62 |
| Generator / cross-category | 3K      |   64.12 | 2.20× [2.15, 2.25] |                44.74 |
| Generator / cross-category | 4K      |   61.39 | 2.10× [2.06, 2.16] |                44.14 |
| Generator / cross-category | 5K      |   61.43 | 2.11× [2.06, 2.16] |                44.27 |
| Generator / cross-category | 6K      |   60.77 | 2.08× [2.04, 2.13] |                43.54 |

## Chat — baseline 29.31 tok/s

| arm                        | steps   |   tok/s | speedup [95% CI]   |   acceptance proxy % |
|:---------------------------|:--------|--------:|:-------------------|---------------------:|
| Original Eagle3 drafter    | —       |   56    | 1.91× [1.80, 2.01] |                35.66 |
| Static LoRA                | 1K      |   56.8  | 1.94× [1.82, 2.05] |                39.26 |
| Static LoRA                | 2K      |   57.22 | 1.95× [1.83, 2.06] |                39.64 |
| Static LoRA                | 3K      |   58.7  | 2.00× [1.88, 2.12] |                40.04 |
| Static LoRA                | 4K      |   56.11 | 1.92× [1.80, 2.03] |                40.08 |
| Static LoRA                | 5K      |   57.4  | 1.96× [1.83, 2.08] |                40.56 |
| Generator / correct        | 1K      |   59    | 2.01× [1.89, 2.13] |                40.12 |
| Generator / correct        | 2K      |   59.37 | 2.03× [1.90, 2.15] |                40.96 |
| Generator / correct        | 3K      |   60.4  | 2.06× [1.92, 2.19] |                41.15 |
| Generator / correct        | 4K      |   60.49 | 2.06× [1.93, 2.19] |                41.72 |
| Generator / correct        | 5K      |   59.6  | 2.03× [1.91, 2.16] |                41.91 |
| Generator / correct        | 6K      |   57.93 | 1.98× [1.85, 2.10] |                41.68 |
| Generator / fixed          | 1K      |   54.24 | 1.85× [1.74, 1.96] |                37.08 |
| Generator / fixed          | 2K      |   53.88 | 1.84× [1.73, 1.94] |                37.22 |
| Generator / fixed          | 3K      |   52.91 | 1.80× [1.70, 1.90] |                36.34 |
| Generator / fixed          | 4K      |   54.02 | 1.84× [1.74, 1.95] |                36.11 |
| Generator / fixed          | 5K      |   52.55 | 1.79× [1.69, 1.90] |                36.1  |
| Generator / fixed          | 6K      |   52.61 | 1.79× [1.69, 1.89] |                35.44 |
| Generator / cross-category | 1K      |   56.79 | 1.94× [1.82, 2.05] |                37.1  |
| Generator / cross-category | 2K      |   57.59 | 1.97× [1.85, 2.08] |                37.35 |
| Generator / cross-category | 3K      |   56.78 | 1.94× [1.83, 2.04] |                36.54 |
| Generator / cross-category | 4K      |   54.95 | 1.88× [1.77, 1.98] |                36.25 |
| Generator / cross-category | 5K      |   54.83 | 1.87× [1.76, 1.98] |                36.24 |
| Generator / cross-category | 6K      |   55.03 | 1.88× [1.77, 1.98] |                35.65 |

