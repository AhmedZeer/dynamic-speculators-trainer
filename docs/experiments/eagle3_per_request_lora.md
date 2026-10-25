# Per-request generated adapters with native continuous admission

Implementation date: 2026-10-25. Status: implemented with CPU routing/lifecycle
checks; the native CUDA kernels and full A100 serving run have not been executed
in the development environment. This is a new serving policy, separate from
historical closed-cohort measurements. See the
[source audit](eagle3_per_request_lora_feasibility.md).

Each new request completes target prefill, then the generator receives that
request's last prompt-token auxiliary states and prompt length (K=1). Its rank-32
V/O factors are registered directly in a **draft-only** native vLLM LoRA manager.
The target remains frozen and has no request adapters. There is no per-request
factor file or Hub transfer.

Native vLLM multi-LoRA kernels execute tokens with their request's adapter.
The EAGLE proposer refreshes token mappings after constructing each draft
forward's query layout, including mixed prefill/decode and subsequent speculative
steps. Base projection weights are never overwritten. Factors remain fixed
throughout the request; CPU originals survive preemption and KV recomputation,
while only scheduled requests occupy GPU adapter slots. Completion/cancellation
releases only that request's adapter. HTTP and engine cleanup are idempotent.

The ordinary vLLM scheduler admits new requests while others are decoding;
`FixedBatchScheduler` is not selected. This implementation initially retains
BF16, eager execution, V1, synchronous scheduling, unchunked prefill, disabled
prefix caching, TP/PP/DP=1 and concurrency up to 8. These restrictions are explicit
comparison settings, not a claim to reproduce all default vLLM launch options.
Static-LoRA continuous serving is outside this initial implementation.

## Baseline contract

The baseline is original EAGLE3 on **original vLLM sources with its native
scheduler**. The launcher creates an isolated source view using checksum-verified
original files from the patch installer's backup; other package files and
compiled extensions are shared through symlinks. It does not revert or modify
site-packages while another process could be using them. The worker reports the
hashes of the imported speculative config, EAGLE proposer and GPU model runner;
readiness rejects a baseline with patched hashes.

Both arms use the same BF16/eager/V1 comparison settings and speculative token
count. The baseline has no adapter path and no generator/controller. The HTTP
status extension and middleware provide measurement diagnostics and request IDs;
they do not change its model, decoding, KV allocation or scheduling. The
baseline should not be described as default-tuned vLLM: CUDA graphs, chunking,
prefix caching and async scheduling remain disabled in this matched study.

Baseline provenance contains `vanilla_source.json`, original source copies in
`vanilla-vllm/`, and an empty `vllm.patch`. Generator provenance includes the
applied patch. Standard model hashes, launch commands and source provenance are
retained for both arms.

## Benchmark the step-4000 checkpoint

Use the existing vLLM **0.31.0** environment and install this repository's
benchmark dependencies as in the suite runbook. Stop old servers first, pull
these changes, and upgrade the recognized patch. No retraining is needed.

```bash
python scripts/apply_vllm_generator_patch.py --apply
speculators benchmark suite \
  --config examples/eagle3-lora/benchmark_per_request_4000.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/per-request-4000-v1
```

Append `--resume` when restarting this same suite. Use a new output root: the old
fixed-cohort measurements have different serving semantics and cannot be reused
as continuous-batching results.

The config downloads only the inference bundle/provenance under
`azeeer/eagle3-generator-all24k-10ep-v1/step-00004000`, excluding optimizer
checkpoints. The generator repository is pinned to
`bf3fb2818dc1c09f3324b7193d00ceb292a8a854`. The exported
bundle identifies target Qwen3-8B at `b968826d9c46dd6066d109eabc6255188de91218`
and original EAGLE3 at `08610ffa01dd9f16731fe8f627b85905b6aa51c4`.

The suite runs **two models × four concurrencies (1/2/4/8) × four ID categories**
(math/code/STEM/chat), using 50 held-out prompts per category, a 256-token output
cap, three warmups and seed 42. The suite owns launching/killing every server
and GuideLLM client. It stops on a failed measurement. Output correctness is
marked `not_checked`, consistent with the existing timing workflow.

`summary.md` contains throughput; `summary.json` retains latency, speculative
acceptance and adapter counters. `servers/<model>/attempt-*/server.log` and
`measurements/<model>/<dataset>/<category>/attempt-*/client.log` hold diagnostics.
Adapter counters require one invocation per prepared request and completed
cleanup for all prepared adapters. `mean_draft_batch_size` comes from actual
proposer forward layouts. `mean_requests_per_adapter` and
`mean_conditioning_size` are both 1; neither is a GPU batch-size measurement.

For a smaller first GPU check, copy the YAML, use `concurrencies: [1, 2]`, one
category and a small `max_samples`, and give that pilot a different output root.
This checks the full native backend before committing to all 32 measurements.
The regular suite command itself does not run a separate correctness pass.

## Standalone generator server

If the bundle has already been downloaded locally:

```bash
python scripts/launch_vllm.py serve \
  --adapter-bundle ./bundles/generator-4000/step-00004000 \
  --adapter-policy per_request --conditioning-examples 1 \
  --batching-policy continuous --concurrency 8 \
  --provenance-dir output/http-servers/generator-4000-per-request-c8
```

A native baseline uses normal target/drafter arguments plus
`--batching-policy continuous --vanilla-vllm --concurrency 8` and its own
`--provenance-dir`. The suite supplies the correct pinned model arguments.

## Evidence limits

CPU tests cover distinct per-request routing against explicit low-rank matrix
products, Q/K preservation, unchanged base weights, newcomer admission,
reordering, singleton generation counts, pause/recompute, slot reuse, cleanup,
profiling and original-source launch isolation. They also execute the installed
proposer patch against a test double to check every-forward routing. Native
adapter registration tests cover destructive packing/scaling isolation.
These are not native CUDA kernel or full-model equivalence tests. BF16 unmerged
LoRA branches can round differently from historical merged BF16 weights. No
new speedup or output-correctness conclusion is established by implementation.
