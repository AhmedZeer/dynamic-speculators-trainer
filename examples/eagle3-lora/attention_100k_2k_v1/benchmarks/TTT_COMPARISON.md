# Training TTT3/TTT7 crossed with serving draft length 3/7

The row name describes the training objective and optimizer step. The `--ttt`
argument sets the number of speculative tokens proposed during serving; it does
not change or retrain the checkpoint. Run all six checkpoints with both lengths
to separate training-horizon effects from serving-length effects.

| Row | Training TTT steps | Optimizer updates |
| --- | ---: | ---: |
| attn_gen_ttt3_1k | 3 | 1,000 |
| attn_gen_ttt7_1k | 7 | 1,000 |
| attn_gen_ttt3_2k | 3 | 2,000 |
| attn_gen_ttt7_2k | 7 | 2,000 |
| attn_gen_ttt7_3k | 7 | 3,000 |
| attn_gen_ttt7_4k | 7 | 4,000 |

The models are pinned to immutable HF commits, verified on 2026-11-05:

- TTT3: `azeeer/eagle3-generator-attention-100k-2k-v1`,
  `cf09002ae6fbca1fe0e9139ab8a0eee3b149a124`.
- TTT7: `azeeer/eagle3-generator-attention-100k-10k-v1-ttt7`,
  `e2d9fb419311aec5134bd96c221037e504bf2596`.

From the repository root, with the existing supported benchmark environment:

```bash
for ttt in 3 7; do
  speculators benchmark suite \
    --config examples/eagle3-lora/attention_100k_2k_v1/benchmarks/ttt_comparison_c1.yaml \
    --ttt "$ttt" \
    --output-root "output/benchmarks/attention-100k-ttt-comparison/ttt${ttt}" \
    --resume || break
done
```

Each suite uses math, STEM and chat, 50 prompts per category, concurrency 1,
256 maximum output tokens, seed 42, three warmups, and singleton prompt
conditioning. It retains the earlier suite's fixed shared adapter policy.
This is 18 measurements per serving length, 36 total. The CLI override enters
the effective suite configuration, provenance, and resume identity. A different
serving length requires a different output directory. The W&B groups receive
`-ttt3` or `-ttt7` suffixes.

Tables are written to:

- `output/benchmarks/attention-100k-ttt-comparison/ttt3/summary.md`
- `output/benchmarks/attention-100k-ttt-comparison/ttt7/summary.md`

Each directory also contains `summary.json`, native acceptance counters, and
the existing reproducibility artifacts. Repeat the loop to resume incomplete
measurements. Add `--dry-run` to inspect configuration without downloading models
or starting vLLM. These suites have not been measured locally.

Compare matched training checkpoints at a fixed serving length first. Equal
optimizer updates do not imply equal training compute across TTT3 and TTT7.
For serving lengths 3 versus 7, compare throughput and accepted-prefix length
alongside draft acceptance: acceptance percentages have different denominators.
