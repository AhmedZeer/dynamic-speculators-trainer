# Per-request generated LoRAs and continuous batching

Date: 2026-10-25. Status at audit time: source feasibility audit; no new serving implementation
or GPU benchmark. Audited vLLM 0.31.0, release commit
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, matching the repository patch manifest.
Repository baseline: `ab8ffc1`. Subsequent implementation is recorded in the
[per-request serving runbook](eagle3_per_request_lora.md); the findings below
preserve the source-audit decision and GPU validation requirements.

## Finding

The design is feasible at the component level: generate one adapter from each
request's prompt, register its tensors in a draft-only multi-LoRA manager, and
route that request's draft tokens through its adapter. Requests can then enter
and leave ordinary continuous batches independently. Their adapters do not need
to be merged into the same shared draft weight tensor.

**This is not an existing supported serving mode in our pinned vLLM integration.**
Generating separate factors alone, or adding `--enable-lora`, does not implement
it. The audit supports a prototype, not a claim that the new path already works
or will improve throughput. The fixed-cohort serving policy remains available for historical experiments.
A separate opt-in implementation now exists; full-model GPU checks remain pending.

## Evidence from the pinned source

Links below identify the exact audited revision, rather than current upstream.

| Check | Finding | Implication |
| --- | --- | --- |
| In-memory adapter registration | [`LoRAModel`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/lora/lora_model.py) accepts a dictionary of layer weights; [`LoRAModelManager`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/lora/model_manager.py) exposes `add_adapter`, `activate_adapter`, `set_adapter_mapping`, and `remove_adapter`. | Generated factors can be registered without a Hub upload, PEFT directory, or per-request disk write. |
| Packed V-only adaptation | [`MergedQKVParallelLinearWithLoRA`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/lora/layers/column_parallel_linear.py) supports separate Q/K/V slices; unset slices are reset and left without updates. | The intended V-only adapter has a plausible native kernel path while preserving Q/K. Actual shape and scaling compatibility still need execution checks. |
| Output projection adaptation | Native [`ReplicatedLinearWithLoRA`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/lora/layers/replicated_linear.py) exists, alongside row-parallel wrappers. | Draft output projection adaptation does not inherently require globally merged weights. |
| EAGLE3 model structure | [`Eagle3LlamaForCausalLM`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/llama_eagle3.py) inherits `LlamaForCausalLM` at runtime, whose class metadata includes LoRA support and packed QKV mappings. The first draft layer uses a QKV input width of **2 × hidden size**. | Reuse looks possible, but target-model adapter dimensions must not be assumed for the drafter. Inherited metadata alone is not an end-to-end support guarantee. |
| Existing CLI LoRA routing | [`GPUModelRunner.load_model`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/worker/gpu_model_runner.py) wraps `self.model` with the LoRA manager. Its input preparation sets active LoRAs from the target input batch. | Standard LoRA flags operate on the target-side path; they do not supply independent generated draft adapters. The verifier must remain unchanged. |
| EAGLE proposer routing | [`LLMBaseProposer`](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/v1/spec_decode/llm_base_proposer.py) loads the draft without installing a LoRA manager or setting per-token LoRA mappings. | A draft-only manager and routing hooks are missing and must be implemented. |
| Multiple draft forwards | The proposer first forwards packed prefill/verification tokens, then changes to one query token per request for subsequent draft steps. | Routing must be updated for **every draft forward**, using its actual token layout. A mapping set only once at the start of `propose()` is insufficient. |

The repository's current `AdapterMerger` copies `W_base + scale * B @ A` into
one shared draft projection. `prepare_proposer` rejects newcomers while that
adapter is active. Those implementation choices require closed cohorts; the
generator's ability to produce a singleton-conditioned adapter does not.

## Proposed implementation contract

1. Keep the target model and its request LoRA IDs unchanged. Install a separate
   native LoRA manager on the EAGLE3 draft, targeting only packed V and O.
2. After each new request's complete target prefill, generate its adapter from
   the last prompt-token auxiliary states and prompt length, before its first
   draft prefill. Condition on that request alone (K=1). Keep its factors fixed
   until completion or cancellation.
3. Allocate an adapter ID per internal request ID. Register factors directly
   in memory, with explicit rank, alpha scaling, dtype, and shape checks.
   Retain factors across pauses/preemption; never evict an active adapter or
   silently replace it with base weights. Bound GPU residency by scheduled
   concurrency, with an explicit policy for retained paused requests.
4. Rebuild the token-to-adapter mapping after the proposer constructs each
   forward's inputs. Cover mixed decode/prefill, changing request order,
   speculative rejection, and all later draft steps. Clear it for profiling
   and dummy forwards.
5. Release only the finished request's factors and slot. Preserve the existing
   named worker RPC and engine lifecycle hooks, including idempotent HTTP
   cleanup. New requests may join while unrelated requests continue decoding.
6. Use the native scheduler rather than `FixedBatchScheduler` for this mode.
   Initially retain eager BF16, synchronous scheduling, TP/PP/DP=1, unchunked
   prefill and disabled prefix caching. Restoring continuous admission does
   not require enabling every other vLLM optimization at once.
7. Record a distinct serving policy and adapter counters in provenance and
   benchmark identity. Count one generator invocation per prepared request,
   not one per shared cohort. Use a new benchmark output root; historical
   fixed-cohort measurements must retain their original policy labels.

Calling `LoRAGenerator` on K summaries produces **one shared adapter**. Producing
K independent adapters requires K singleton calls or an explicitly vectorized
independent-request forward; passing all summaries into the existing pooling
forward would recreate the current shared-adapter behavior. Existing K=1
checkpoints are the clearest initial candidate. Serving a larger-K-trained
checkpoint at K=1 is mechanically accepted by the model, but changes its
conditioning distribution and needs measurement.

## Required functional check before switching

Start with synthetic distinct adapters and a tiny pinned draft to establish
token routing independently of the generator. Compare mixed-request packed
projection/forward results against separate executions with the same factors.
Check V-only QKV slicing, O, alpha/rank scaling, and base-weight immutability.

Then exercise the actual generator bundle at C1/C2/C4/C8 with staggered arrivals,
different prompt/response lengths, request reordering, completion, cancellation,
slot reuse, speculative rejection, and forced KV preemption/resumption. Confirm
that newcomers are admitted before all older requests finish, every request
keeps its original adapter, adapter/slot counts remain bounded, and cleanup
does not affect another request.

Separate numerical checks from performance: merged BF16 weights and unmerged
LoRA branches have different rounding, so exact equality with the historical
merged runtime is not guaranteed. Establish the new multi-request path against
its own single-request execution first. The previously unresolved target-only
output-equality investigation is not resolved by this source audit.

Only after the functional checks should a matched GuideLLM sweep compare
native-continuous base EAGLE3, static LoRA merged once, and per-request generator
adapters on the same ID manifests. Record throughput, TTFT/TPOT, acceptance,
adapter generation/loading time, and memory. Faster scheduling may be offset by
per-token multi-LoRA work; no speedup is established yet.

## Audit limits

The local test environment has Torch `2.14.1+cpu`, no installed vLLM, and no
CUDA-enabled Torch runtime. The available local GPU is an 8 GB RTX 5060, not the
target A100 80 GB setup. This check inspected the pinned upstream source and
repository hooks; it did not execute native multi-LoRA kernels or a full model.
