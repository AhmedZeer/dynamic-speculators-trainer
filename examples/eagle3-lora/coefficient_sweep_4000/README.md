# Step-4000 LoRA coefficient sweep

Inference only, using the pinned original step-4000 generator. Fixed effective
coefficients are **0.5, 4 and 8** in `W_effective = W_base + coefficient * B @ A`.
They replace alpha/r (normally 2); they do not multiply that coefficient again.
No training, feedback controller, prefix refresh or per-request multi-LoRA path
is used. Execution is shared/merged, concurrency 1, singleton prompt conditioning.

The existing benchmark suite owns server/client launch, cleanup and provenance.
The workload uses the same Nemotron dataset selection as `benchmark_suite.yaml`,
restricted to math/chat: 50 prompts/category, seed 42, 256 output-token cap,
three speculative tokens, three warmups. It deliberately does not use the newer
v3 training validation split or v4 ablation test manifest. Workload hashes must
be checked against old artifacts before claiming exact historical prompt reuse.

Run the three-model suite for one combined throughput table:

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/coefficient_sweep_4000/suite.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/coefficients-4000-v1 \
  --resume
```

Or run each config manually with its own output directory:

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/coefficient_sweep_4000/s05.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/coefficient-4000-s05-v1 --resume
```

For the other independent commands replace `s05.yaml` with `s4.yaml` / `s8.yaml`
and the directory suffix with `s4-v1` / `s8-v1`. This uses the existing suite
command; no new multi-command launcher is introduced. Add `--dry-run` to inspect.
Use new output directories rather than changing the identity of the old suite.

W&B project: **`eagle3-generator-coefficient-bench-v1`**; group
`step4000-merged-math-chat-c1`. Per-round file recording is disabled consistently
for the sweep; aggregate acceptance counters remain available. Scale bounds are
explicitly raised to 8 in all three configs so scale 8 validates.

The combined suite writes `summary.md`, `summary.csv`, `summary.json`, per-category
results and owned server/client logs. Output tok/s includes prefill and serving
overhead. Preserve aggregate acceptance counters/length alongside throughput.
Suite-generated launch provenance includes the required `--provenance-dir`.
Apply the repository's supported vLLM generator patch before adapter serving.

Extended shared/merged C1 table (historical reference plus supplied new sweep):

| Model | Math output tok/s | Chat output tok/s |
| --- | ---: | ---: |
| EAGLE3 base (historical) | 64.34 | 54.97 |
| Trained LoRA (historical) | 72.50 | 56.35 |
| Generator, 2000 steps (historical) | 77.58 | 57.58 |
| Generator, 4000 steps, coefficient 2 (historical) | 79.11 | 56.49 |
| Generator, 4000 steps, coefficient 0.5 | 66.97 | 53.66 |
| Generator, 4000 steps, coefficient 4 | 64.08 | 51.33 |
| Generator, 4000 steps, coefficient 8 | 37.10 | 36.94 |

Historical values are from [GEN-003](../../../docs/experiments/results/GEN-003-generator-suite-throughput.md).
They are not rerun or silently inserted into the new suite's measured summary.
Runtime/hardware provenance for the old table is incomplete; cross-run differences
need that qualification. New measurements and acceptance metrics are archived in
[GEN-008](../../../docs/experiments/results/GEN-008-coefficient-sweep.md).
Coefficient 0.5 leads the three new settings; 8 sharply reduces acceptance and
throughput. Coefficient 2 still needs a same-runtime measurement for a controlled
comparison. Exact-output correctness is not established by timing.
