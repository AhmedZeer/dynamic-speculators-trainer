# Matched static/generator scaling and conditioning controls

The launcher prepares one frozen workload and training-only donor bank, then
runs pinned checkpoints and collects the tables needed for the planned figures.
It does not launch training or generate the figures/report.

```bash
bash examples/eagle3-lora/attention_100k_2k_v1/benchmarks/run_scaling_dynamism.sh \
  --output-root output/attention-scaling-dynamism-v1
```

Use the existing supported trainer/vLLM benchmark environment with the generator
patch applied, and authenticate to HF and W&B. The launcher owns the extractor
and benchmark services; stop other services on its port first. `--gpu 2` chooses
physical GPU 2 for this sequential benchmark. `--port 8100` chooses another
port. The donor-preparation extractor runs alone on the GPU, so it uses
`--extractor-gpu-memory-utilization 0.85` (the serving value of earlier Colab A100
runs) rather than the co-located training default of 0.40, which leaves no KV
cache for Qwen3-8B on a 40 GB A100. No NVMe is required. On Colab, a persistent `--output-root` preserves the
workload, donors, completed tasks and collected tables across runtime resets.

Serving draft length defaults to 3. To repeat the full matrix at both lengths:

```bash
bash examples/eagle3-lora/attention_100k_2k_v1/benchmarks/run_scaling_dynamism.sh \
  --output-root output/attention-scaling-dynamism-v1 --ttt 3 7
```

Other selections:

- `--phase scaling`: generator correct-prompt checkpoints 1K–6K and static LoRA
  1K–5K; the paired scaling table uses 1K–5K. The extra generator 6K result is
  reusable by the dynamism phase.
- `--phase dynamism`: generator correct, fixed and cross-category at 1K–6K.
- `--phase all`: both phases, the default. Correct-prompt measurements are
  shared, never rerun for the second phase when already completed.
- `--dry-run`: writes/validates the requested declarative configurations and
  prints counts without model/data downloads or GPU startup.
- `--collect-only`: rebuilds tables from completed summaries without GPU work.

Rerun the same command/root to resume. Each underlying suite uses `--resume`.
Incomplete or failed suites stop the launch; absent cells are not replaced by
zeros. Configuration changes require a new root. Both serving lengths have
separate suite directories and W&B groups.

## Checkpoint coverage and workload

HF revisions checked on 2026-11-08:

| Arm | Repository | Pinned commit | Checkpoints |
| --- | --- | --- | --- |
| Generator, trained TTT3 | azeeer/eagle3-generator-attention-100k-2k-v1 | 3edcce5e4d2fbfc56dbfa71ae0cd96fb8dcc1cba | 1K–6K |
| Static attention LoRA, trained TTT3 | azeeer/eagle3-static-lora-attention-100k-10k-v1 | 8b5ef5792fcd5e3ca7e49a4959edc0b347b496e6 | 1K–5K |

All suites use math, STEM and chat, concurrency 1, 50 measured prompts/category,
max 256 output tokens, three warmups, seed 42, shared merged adapters, scale 2
(rank 32/alpha 64), and singleton generator conditioning. Generator control
policies use the same experimental controller with prefix refresh/feedback
disabled and per-round JSON recording disabled. Static LoRA is prompt independent;
it needs no fixed/cross-category control.

All configurations point to `study/inputs/test/manifest.json`. The existing
`prepare-experiments` procedure shuffles the 100 held-out rows of shard 00000
with seed 42 into 50 tuning and 50 test rows per category. Scaling and dynamism
use the exact same 50 test prompts. These are a new shared frozen workload,
different from historical suites that sampled several validation shards; do not
splice their results into the same curve without rerunning the matched controls.
No benchmark outputs are used to select checkpoints or donors.

For donor preparation, the launcher calls the existing managed data preparation
for the exact 100K training release, then starts its provenance-enabled vLLM
extractor and runs `generator prepare-experiments`. Target/drafter weights and
trajectory Parquets are downloaded if missing; cached downloads are reused.
Only 64 training prompts/category from shard 00001 supply donor summaries.
The extractor stops before speculative-serving benchmarks begin. Future launches
reuse the frozen workload and checksum-validated donor bank.

The fixed policy uses the first math training donor for every query, exactly as
the previous conditioning test. Cross-category selection is deterministic from
the prompt hash: math → chat, chat → math, STEM → math, excluding identical
prompt tokens. Thus STEM cross-vs-fixed measures different choices from the math
donor pool; the fixed donor is not category neutral. Donor IDs, prompt hashes,
categories and checksums are saved with the bank. Per-suite model/donor hashes,
native acceptance counters and server/eval provenance remain in the suite outputs.

Counts per serving length: 18 correct + 15 static + 18 fixed + 18 cross = **69
measurements**, or 3,450 measured requests plus warmups. Both lengths total 138
measurements. The reused correct suite accounts for all six generator steps.

## Tables and figure inputs

Under the chosen output root:

- `configs/`: frozen suite YAMLs and donor-preparation configuration.
- `suites/ttt3/{correct,static,fixed,cross_category}/`: resumable raw benchmarks,
  `summary.json`, throughput tables, native metrics and provenance.
- `tables/scaling_ttt3.csv` and `.md`: 30 rows = five training steps × two
  methods × three categories, with `tokens_per_second` and `acceptance_percent`.
- `tables/dynamism_ttt3.csv` and `.md`: 18 rows = six steps × three categories,
  with all three acceptance percentages and the two requested signed deltas.
- Corresponding `ttt7` files if that serving length was requested.

Scaling figure inputs: x = `steps`; left y = `tokens_per_second`; right y =
`acceptance_percent`; model distinguishes generator/static. Preserve category
curves or panels rather than silently pooling tasks.

Dynamism figure inputs: x = `steps`; y = **percentage-point differences**:
`correct_minus_fixed_pp = 100 * (rate_correct - rate_fixed)` and
`cross_category_minus_fixed_pp = 100 * (rate_cross_category - rate_fixed)`.
The latter is intentionally cross-category minus fixed, not correct minus cross.
The collector verifies workload hashes before comparing/subtracting measurements.

Implementation status: schema, coverage and collection tests plus dry-run plans;
GPU benchmarks are pending. A single 50-prompt run per cell does not quantify
timing uncertainty. Correctness remains whatever the native suite reports;
throughput/acceptance measurements alone do not establish output correctness.
