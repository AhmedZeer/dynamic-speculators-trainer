# Structure models: math/chat, shared adapters, concurrency 1

This suite measures the attention/all-module generators, the attention-only
decoder-width-256 generator, and two matched static LoRAs: five models, all at
**step 1,500**. It uses the existing managed vLLM + GuideLLM
pipeline: one server/model, two category measurements, cleanup, next model.
No training, response replay, adaptive scale or per-request adapters are enabled.

## Model readiness

Checked on 2026-10-30 using the Hub API:

| Model | Repository | Step-1500 readiness |
| --- | --- | --- |
| Attention generator | `azeeer/eagle3-generator-structure-v1-modules-attention-4k-1500` | Present; revision `a8acae6ac086408a2cbce584c5ceaf1ead57f3c9` |
| All-module generator | `azeeer/eagle3-generator-structure-v1-modules-all-4k-1500` | Present; revision `b077b468207123d82a8ae57d9e249d6c5508abc6` |
| Attention static LoRA | `azeeer/eagle3-static-lora-structure-v1-attention-4k-1500` | Present; latest verified revision `dfb78b270a39f625bc7a39ef75cfaaab5868eb54` |
| All-module static LoRA | `azeeer/eagle3-static-lora-structure-v1-all-4k-1500` | Present; revision `8ded0674b4e9f9a29c4938596bc2aacea58d1d4f` |

The attention decoder-256 generator is added as
`azeeer/eagle3-generator-structure-v1-modules-attention-decoder256-4k-1500`.
At the latest check, revision `afd0df1af02dfb6f379eb4f0d529029a1abdc075`
contains step 500 and step 1000; step 1500 is pending. Its config uses `main`
and **only** `subfolder: step-00001500`. No shorter snapshot is substituted.
Launch after that upload is acknowledged. Attention LoRA also retains `main`
so already-started suite requests remain compatible. Model resolution pins
exact SHAs in `resolved_models.json` and `suite.json`; resume keeps those pins.
Missing snapshots fail before GPU server launch. The suite does not poll
indefinitely or start training a missing model.

## Settings

- Nemotron **math and chat only**, 50 validation prompts/category, seed 42.
- 256 output-token cap, three warmup requests, three speculative tokens.
- Concurrency **1**; generator conditioning size 1, prompt-generated shared
  adapter, default fixed coefficient alpha/rank=2. No per-verifier refresh.
- Static LoRAs merge once at startup; generators generate/merge per request
  and restore after completion, matching the earlier shared C1 execution.
- W&B project: `eagle3-generator-structure-bench-v1`.
- Correctness is not checked; the existing suite collects throughput and
  acceptance counters without requiring a separate `check-endpoint` report.

The workload-export settings match the historical math/chat study. Before
treating rows as matched prompts, check the emitted dataset hashes against the
historical math `d2990b90cdc231a932afe725035f9b403a82f483d390062888beed4358d93b28`
and chat `4462e1e17606a424d791e4d34ad18558f8d327aa646597dd3362e336afd0f91f`.
Different hardware/runtime versions or generator precision remain limitations
of comparison with historical measurements, even if prompt hashes match.

## Command

Pull the code and use the existing serving/GuideLLM environment with pinned
vLLM 0.31.0 and its Speculators patch installed. Stop the training/extraction
server first: the suite owns GPU 0 and port 8000 and will not kill unrelated
processes to take that port.

Once decoder-256 step 1500 is uploaded:

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/structure_benchmarks_v1/math_chat_c1.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/structure-math-chat-c1 \
  --resume
```

`--resume` also works for a new directory. Subsequent invocations retain valid
completed measurements and retry failed/incomplete tasks. Adding the fifth
model to the original four-model suite is supported: keep the same output
directory and `--resume`; only missing measurements run. Existing model definitions,
workloads and settings must stay identical. Use a new directory for unrelated suites. Append `--dry-run` to
inspect the config without downloads or GPU launches; it does not establish
Hub snapshot readiness.

The suite automatically launches/stops its owned servers and clients, includes
`--provenance-dir`, verifies model identity, and writes:

```text
suite.json
resolved_models.json
state.json
summary.md                 # five new rows, math/chat output tokens/s
summary.csv
summary.json               # throughput, acceptance, adapter timing, workload hashes
failures.json
servers/<model>/attempt-*/server.log
measurements/<model>/nemotron/<category>/attempt-*/client.log
```

Send `summary.json` and `summary.md` after completion to extend the archived
table. [The prepared extension](../../../docs/experiments/results/GEN-010-structure-serving.md)
retains historical rows and marks new values as pending, not measured zeroes.
Acceptance rate is accepted/proposed draft tokens; acceptance length includes
the verifier bonus token. Both are available in detailed JSON when reported
by the server; neither should be inferred from offline validation agreement.

## Projection support and validation limits

Shared merged serving now maps q/k/v to packed QKV rows, gate/up to the first/
second packed MLP halves, and o/down to their separate matrices. It checks
exported dimensions, retains original weights, restores after generated-adapter
requests, and rolls back failed merges. Static adapters remain merged once.
The target model is never adapted. Existing v/o behavior is retained.

Packing was checked against the pinned vLLM sources:
[EAGLE3 layer](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/llama_eagle3.py),
[Llama attention and MLP](https://github.com/vllm-project/vllm/blob/db9527a46873454610df6dbedf79a36d6bf1a7f6/vllm/model_executor/models/llama.py).
CPU tests cover explicit LoRA merge parity, untouched packed slices, exact
restoration, idempotence, failed-merge rollback and controller lifetimes.
The new attention/all-module GPU serving run is pending; these tests do not
establish GPU benchmark speed or token equality. Per-request native multi-LoRA
remains v/o-only and rejects these wider target sets explicitly.
