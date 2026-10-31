# DeepSeek Qwen3-8B EAGLE3 ttt7: math/chat C1

Use [deepseek_ttt7_math_chat_c1.yaml](deepseek_ttt7_math_chat_c1.yaml) to benchmark
only `deepseek-ai/eagle3_qwen3_8b_ttt7`, pinned at revision
`f6485ba8d21e11942958617dbe7e71b467f38f38` (metadata checked 2026-10-31).
This is an external pretrained drafter, without our generated or static adapters.

The suite uses exactly the same benchmark/settings sections as
[math_chat_c1.yaml](math_chat_c1.yaml): Qwen3-8B target, math/chat only,
concurrency 1, 50 prompts/category, 256 output-token cap, three warmups,
three draft tokens, seed 42 and the same managed serving runtime.
`ttt_length: 7` is checkpoint training metadata; this comparison does not change
the draft-token count to seven. W&B remains
`eagle3-generator-structure-bench-v1`, group `deepseek-eagle3-ttt7-math-chat-c1`.

## Prepare and benchmark

From the repository root, in the existing installed Speculators + patched
vLLM 0.31.0 + GuideLLM environment:

```bash
python scripts/evaluate/prepare_deepspec_eagle3.py
```

This downloads the pinned config and model weights into the HF cache, then
creates a serving directory under
`output/deepseek-eagle3-ttt7-vllm/f6485ba8d21e11942958617dbe7e71b467f38f38`.
Keep the HF cache available: the prepared weights are linked to that cache.
The original config, serving config, and a preparation record with checksums
are preserved. No model tensors are converted or retrained.

Stop any training/extraction/inference process using GPU 0 or port 8000, then:

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/structure_benchmarks_v1/deepseek_ttt7_math_chat_c1.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/deepseek-ttt7-math-chat-c1 \
  --resume
```

Use this separate output root; the existing multi-model suite is unchanged.
The managed suite stages and hashes the prepared files, pins model identity,
passes `--provenance-dir`, and collects launch/eval provenance, logs, acceptance,
adapter metrics and throughput in `summary.json` and `summary.md`. Rerun with
`--resume` to reuse successful cells and retry failures. Add `--dry-run` to
inspect the suite without weight downloads or GPU launch; that does not prove
checkpoint readiness or GPU compatibility.

After completion, provide `summary.json` to extend
[GEN-010's table](../../../docs/experiments/results/GEN-010-structure-serving.md)
with a measured `DeepSeek EAGLE3 ttt7` row. Do not populate it with inferred
numbers. Check the two workload hashes against GEN-010 and preserve the
preparation record and actual runtime/hardware provenance. Correctness is not
checked by the timing suite.

## Why metadata preparation is required

The [pinned checkpoint config](https://huggingface.co/deepseek-ai/eagle3_qwen3_8b_ttt7/blob/f6485ba8d21e11942958617dbe7e71b467f38f38/config.json)
declares `Qwen3Eagle3Model` and zero-based target decoder layers
`[1, 9, 17, 25, 33]`. vLLM 0.31.0's
[EAGLE wrapper](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/transformers_utils/configs/eagle.py)
does not translate that architecture name to the registered loader. Pointing
the existing suite directly at the untouched repository would therefore be
insufficient for this pinned runtime.

The helper selects the registered
[`Eagle3Qwen3ForCausalLM` implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/model_executor/models/qwen3_eagle3.py)
and explicitly sets five auxiliary states, the existing raw-hidden residual
convention, and raw recurrent output. DeepSpec
[extracts `hidden_states[layer_id + 1]`](https://github.com/deepseek-ai/DeepSpec/blob/main/deepspec/modeling/eagle3/common.py);
vLLM captures states before the indexed decoder layer, so the serving auxiliary
indices are `[2, 10, 18, 26, 34]`. Without this translation, the runtime could
select three default features despite the checkpoint's five-feature FC matrix.
The small safetensors header was inspected: `fc.weight` is `[4096, 20480]`, and
the attention projections include Q/K normalization and concatenated
embedding/hidden inputs, consistent with vLLM's Qwen3 EAGLE3 implementation.

Four local unit checks cover config translation, unexpected metadata rejection,
source/weight preservation and repeat preparation, and exact workload/settings
agreement with the existing structure suite. The actual CLI dry run passes.
These checks do not establish GPU forward parity or model correctness; the first
real serving launch remains untested here. Keep this a prepared external-model
benchmark until measurements are collected.
