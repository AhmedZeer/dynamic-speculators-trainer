# GEN-014: Training TTT3/TTT7 crossed with serving lengths 3/7

Measured evidence: user supplied two detailed 18-cell C1 summaries. Each cell reports 50 successful requests; matching workload hashes, subset lists and output totals are present across models and serving lengths. Correctness is unchecked; no repeat distributions or executed hardware/checkpoint checksums were supplied.

Implementation/configuration: six checkpoint rows, math/STEM/chat, 50 prompts per category, max 256 output tokens, singleton conditioning, fixed shared adapter, three warmups, seed 42. Configured repository pins are in [the suite](../../../examples/eagle3-lora/attention_100k_2k_v1/benchmarks/ttt_comparison_c1.yaml); those pins are not executed model-checksum evidence.

[Selected metrics and native counts](benchmarks/2026-11-05-attention-ttt-comparison-c1/selected_metrics.json) preserve throughput, acceptance and accepted-prefix length. The original full summaries and runtime provenance were supplied in the conversation rather than archived here. All 36 transcribed cells satisfy proposed_tokens = serving_length * verification_rounds. Acceptance and prefix length are recomputed from native counts.

## Serving draft length 3

Cells show output tokens/s / draft acceptance (%). Row names describe training TTT and optimizer updates.

| Model | Math | STEM | Chat |
| --- | ---: | ---: | ---: |
| attn_gen_ttt3_1k | 76.58 / 65.15% | 66.94 / 51.68% | 56.69 / 39.73% |
| attn_gen_ttt7_1k | 77.94 / 64.02% | 66.77 / 50.77% | 58.30 / 39.53% |
| attn_gen_ttt3_2k | 79.37 / 67.43% | 67.83 / 51.99% | 57.85 / 40.26% |
| attn_gen_ttt7_2k | 77.21 / 65.46% | 67.07 / 51.29% | 56.85 / 39.73% |
| attn_gen_ttt7_3k | 78.92 / 66.04% | 67.85 / 51.94% | 57.33 / 39.81% |
| attn_gen_ttt7_4k | 81.05 / 67.06% | 68.34 / 51.69% | 58.78 / 39.82% |

## Serving draft length 7

Cells show output tokens/s / draft acceptance (%). Row names describe training TTT and optimizer updates.

| Model | Math | STEM | Chat |
| --- | ---: | ---: | ---: |
| attn_gen_ttt3_1k | 87.58 / 40.89% | 69.67 / 29.30% | 55.36 / 20.57% |
| attn_gen_ttt7_1k | 88.71 / 40.66% | 70.18 / 29.06% | 56.23 / 20.48% |
| attn_gen_ttt3_2k | 89.56 / 42.90% | 69.48 / 29.60% | 54.67 / 20.77% |
| attn_gen_ttt7_2k | 91.37 / 42.41% | 71.33 / 29.77% | 55.94 / 20.43% |
| attn_gen_ttt7_3k | 91.92 / 43.08% | 71.51 / 30.00% | 56.19 / 20.72% |
| attn_gen_ttt7_4k | 93.31 / 44.35% | 71.09 / 30.23% | 55.28 / 20.76% |

## Comparisons and open questions

At 2K optimizer updates, switching training TTT3 to TTT7 changes throughput as follows (TTT3 denominator):

| Serving length | Math | STEM | Chat |
| --- | ---: | ---: | ---: |
| 3 | -2.72% | -1.13% | -1.73% |
| 7 | +2.02% | +2.66% | +2.31% |

At the TTT7 2K checkpoint, extending serving length 3 to 7 changes math/STEM/chat throughput by +18.34%, +6.35%, -1.60%. The same direction holds for all six checkpoints: math and STEM improve, chat declines. This supports a workload-dependent serving-length choice in these supplied runs; it does not establish a hardware-independent optimum.

The matched-update training effect remains small and mixed. At 2K with serving length 7, math acceptance is 42.41% for TTT7 versus 42.90% for TTT3 despite higher measured throughput. Therefore the timing gain alone is not evidence of improved overall acceptance. Equal optimizer updates also do not match training compute: TTT7 supervises a longer rollout. Comparisons against the TTT7 4K checkpoint have a larger update budget.

Draft acceptance rates across serving lengths have different proposal denominators. At TTT7/2K on math, accepted-prefix length rises from 1.964 to 2.969 while acceptance falls from 65.46% to 42.41%; the longer block still increases throughput.

Decision/open work: retain both serving-length tables, compare matched 1K/2K training budgets first, and repeat measurements before attributing small timing differences to training horizon. No final conclusion about TTT7 superiority or output correctness follows.
