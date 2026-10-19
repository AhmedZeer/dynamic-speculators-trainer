# Independent HTTP serving and GuideLLM evaluation

Serving and evaluation are separate processes. `launch_vllm.py serve` owns one
vLLM engine; the evaluator only sends HTTP requests. The same evaluator can
measure target-only, original EAGLE3, a generated adapter, or a trained ordinary
LoRA applied to the EAGLE3 drafter; the target stays unchanged. It never starts/stops a server or loads GPU weights. The older
`speculators generator benchmark` serial harness remains available for historical
experiments; it is not required for this workflow.

The initial matched configuration is BF16, eager execution, V1 runner, one active
sequence, synchronous scheduling, no prefix caching, and no chunked prefill.
Generator conditioning uses the current prompt alone (context 1). Training still
uses contexts 1–8 and its existing context-4 validation. No training changes or
activation downloads are needed.

## Environment and validation workload

The serving launcher registers `AdapterWorkerExtension` with vLLM. Status and
request cleanup use named worker RPC methods, so insecure pickle serialization
is unnecessary. If an older server reports `Object of type ... function ... is
not serializable` on `/speculators/status`, update the checkout and restart the
server using the updated launcher. Use a new `--provenance-dir` (for example,
`output/http-servers/target-rpc-fix`) because the code and launch configuration
changed, then rerun `check-endpoint`. Updating only the client cannot fix an
already running server.

Run commands from the repository root. Use the existing vLLM 0.31.0 environment:

```bash
python -m pip install -e '.[generator]'
python -m pip install 'guidellm==0.7.1'
python scripts/apply_vllm_generator_patch.py --check
python scripts/apply_vllm_generator_patch.py --apply
```

The patch is needed for generator and ordinary-LoRA bundles. Apply it before
starting servers; existing processes must be restarted after patch changes.
The launcher checks the pinned vLLM version and patch state rather than modifying
an installed library automatically.

Export the existing Nemotron validation split once:

```bash
speculators generator export-validation \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --output output/generator-http-validation \
  --tokenizer-path ./models/qwen3-8b \
  --max-tokens 256
```

For a smaller workload, choose domains and cap examples **per domain**:

```bash
speculators generator export-validation \
  --config examples/eagle3-lora/generator_campaign_online.yaml \
  --output output/generator-http-validation-50 \
  --tokenizer-path ./models/qwen3-8b \
  --categories math,code,stem,chat --max-samples 50 --seed 42
```

This exports 200 examples total. New manifest groups and result labels are
`nemo-math`, `nemo-code`, `nemo-stem`, and `nemo-chat`; each row also identifies
the dataset as `nemotron`. `--categories` accepts these prefixed names, short
domain names, or `all` (the default). `--max-samples 0` uses all available rows
(the default). Samples are seeded and interleaved across validation subsets so
a small cap does not select only the first subset. The cap applies to each
domain, not each subset; it uses all rows when fewer than the cap are available.
Selection settings are recorded in the checksummed manifest.

Use the new manifest path for both correctness checks and GuideLLM, and create
new correctness reports for it. Existing manifests remain readable. GuideLLM's
`--subsets math,code` selector also accepts `nemo-math,nemo-code`, including when
reading older manifests with short group names. Exporting never overwrites an
existing directory.

The uncapped export selects the union of the retained runs' validation subsets: `00000` and
`00002`–`00005` for math/code/stem/chat. With the example config this is 2,000
prompts (500 per category). Only validation Parquet files and metadata are read;
training Parquet files, banks, and hidden states are not read. The export records
source checksums, row/subset IDs, frozen-model revisions, prompt-token IDs, text,
and per-row output caps. Text must reproduce the original tokens exactly with
special-token insertion disabled; otherwise export fails. No response text is
included. Output caps are clipped to the remaining context space.

`manifest.json` and the four JSONL files form one immutable workload. Keep this
folder and use it for every checkpoint. These are development-validation results,
since this split already monitors training; they are not untouched-test results.

## Start one server

Stop the training activation server and trainer before measuring inference.
Run one server at a time on the A100 and leave it running in its terminal.
The following commands use port 8000 and the same served-model alias.

Target only:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --provenance-dir output/http-servers/target
```

Original EAGLE3:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py serve Qwen/Qwen3-8B \
  --revision b968826d9c46dd6066d109eabc6255188de91218 \
  --drafter RedHatAI/Qwen3-8B-speculator.eagle3 \
  --drafter-revision 08610ffa01dd9f16731fe8f627b85905b6aa51c4 \
  --provenance-dir output/http-servers/eagle3-base
```

For a published generator, download its portable bundle at the commit printed
by `push-to-hub` (replace `COMMIT_SHA` with that actual commit):

```bash
hf download azeeer/eagle3-generator-direct-4k-v3 \
  --revision COMMIT_SHA --local-dir ./bundles/scale-1-direct

CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py serve \
  --adapter-bundle ./bundles/scale-1-direct \
  --provenance-dir output/http-servers/scale-1-direct
```

For a trained ordinary LoRA, download its published bundle, then use the same
launcher; `bundle.json` determines which runtime to instantiate:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/launch_vllm.py serve \
  --adapter-bundle ./bundles/fresh-lora \
  --provenance-dir output/http-servers/fresh-lora
```

Repeat with the `scale-2-direct` and `scale-4-direct` bundles as needed. Bundle
model IDs/revisions are authoritative. This path does not depend on legacy arm
names or bank files. `--adapter-bundle` accepts an exported Speculators bundle,
not a raw recovery `checkpoint.pt` or an arbitrary PEFT directory. Use
`push-to-hub` to produce portable bundles first.

Models resolve to pinned Hugging Face cache snapshots, downloading missing model
weights once. Local positional model/drafter paths are accepted when their directory
names are their complete revision SHAs. `--dry-run` resolves cached snapshots
only, records provenance and prints the command; it does not start vLLM.
`--port`, `--host`, `--spec-tokens`, `--max-model-len`, and
`--gpu-memory-utilization` are explicit launcher options. Scheduling/precision
flags stay matched. Default speculative length is 3 and default context is 8192.

Each server directory contains model checksums, commands, required environment,
server configuration, patches, and source-bundle provenance. Changing server
configuration requires a new provenance directory. The generator remains on the
GPU, runs once per prepared request, and merges before draft prefill. The adapter
is fixed during decoding. Generator cleanup restores immutable base weights before HTTP
completion is returned, and on cancellation/errors. Ordinary LoRA bundles are
merged once at server startup and stay applied, avoiding artificial per-request
merge/restoration costs; their startup merge time is reported separately. Native request-finish hooks
remain as a second cleanup path. Fresh HTTP request IDs isolate cleanup from queued
requests. `/speculators/status` reports the actual runner/controller, current
adapter state, and cumulative generation/merge/restoration counters and seconds.

The server handles cleanup for both completions and chat completions.
The matched validation workflow uses `/v1/completions`. It does not reapply chat templates
or use training-time file-based activation extraction. Higher concurrent adapter
execution requires a different adapter-isolation design; increasing client
concurrency alone would introduce queuing.

## Establish exact-output correctness

With the target-only server running, create a reference:

```bash
speculators generator check-endpoint \
  --target http://127.0.0.1:8000/v1 \
  --validation-manifest output/generator-http-validation/manifest.json \
  --output output/http-checks/target.json
```

After stopping that server and starting the desired generator/LoRA/base-EAGLE3
server, check it against the reference:

```bash
speculators generator check-endpoint \
  --target http://127.0.0.1:8000/v1 \
  --validation-manifest output/generator-http-validation/manifest.json \
  --reference output/http-checks/target.json \
  --output output/http-checks/scale-1-direct.json
```

The checker verifies server-side tokenization, passes the original token IDs to
completions, and compares returned output-token IDs under greedy decoding and
seed 42. It also verifies the connected frozen-model revisions, V1 runner,
controller presence, idle adapter state, and execution/restoration counters.
It saves diagnostics and fails on token disagreement. A reference is required for
all adapter/base-EAGLE3 checks. It does not launch a server.

For a short integration check, add `--limit 3` to both correctness commands.
This selects three prompts per category; the performance command must then use
`--max-requests 3`. Use the full reference/check for full-workload measurements.
Correctness artifacts are bound to the server/checkpoint, workload and generation
settings; a report for another checkpoint cannot authorize a measurement.

## Measure the running endpoint using the existing evaluator

Leave the checked server running. In another terminal:

```bash
python scripts/evaluate/evaluate.py throughput \
  --target http://127.0.0.1:8000/v1 \
  --validation-manifest output/generator-http-validation/manifest.json \
  --correctness-report output/http-checks/scale-1-direct.json \
  --server-provenance output/http-servers/scale-1-direct \
  --tokenizer-path ./models/qwen3-8b \
  --max-concurrency 1 --max-requests 0 --warmups 3 \
  --output-dir output/http-benchmarks/scale-1-direct
```

Use the corresponding correctness/server/output paths for each other checkpoint,
including target-only and base EAGLE3. `--max-requests 0` means every frozen prompt;
a positive limit selects that many per category. `--subsets math,code` restricts
categories; default is all categories. `--validation-grouping subset` measures
individual subsets separately. Limits apply per category before regrouping.

The evaluator invokes GuideLLM 0.7.1's streaming OpenAI HTTP backend, completions
format, deterministic data order, exact per-row output caps, temperature 0 and
EOS stopping. Special tokens are not inserted. An explicit tokenizer path avoids
resolving the served alias as a Hugging Face model. Without that path, only the
pinned target tokenizer/config files are fetched, not model weights.

Three explicit warmup requests per group precede both counter snapshots and
GuideLLM measurement. Native vLLM speculative acceptance metrics are collected
from `/metrics`; adapter timings come from `/speculators/status`. The client
checks tokenization again and requires a successful correctness artifact covering
the selected workload. Failed/incomplete GuideLLM request counts, server changes,
missing adapter execution, or missing restoration fail the measurement.

Results contain:

- `perf_results.csv`: median latency, TTFT, ITL, median token-rate distribution,
  successful requests, total output tokens, measured seconds and aggregate
  output tokens/second, using the existing evaluation CSV column names.
- `acceptance.csv`: accepted/proposed draft counters and accepted length.
  Target-only acceptance is undefined and omitted.
- `adapter_performance.csv`: generator, merge, restoration seconds and counts.
- Per-group GuideLLM JSON, exact input JSONL, actual GuideLLM command, before/after
  Prometheus snapshots, server snapshots, correctness reference and provenance.
- `validation/manifest.json` and all original frozen workload files.

Use a new output directory for each measurement/repetition. Server construction
and warmups are excluded; request timing includes generation and HTTP adapter
cleanup. A performance result is not a training context-4 agreement measurement.

`sweep` is also accepted, but matched servers remain limited to one active
sequence. Acceptance and adapter counters are aggregate across the entire
GuideLLM invocation, not attributed to individual sweep points. Native GuideLLM
JSON retains point-specific timing distributions. For concurrent-capacity studies
of other servers/datasets, the evaluator's existing generic GuideLLM modes remain
available. Target-only performance runs no longer fail simply because speculative
acceptance counters are absent.

No real-model GPU performance is established by the CPU tests. They verify bundle
handling, cancellation/restoration, source/token identities, and the actual
GuideLLM client against a mock HTTP server. Run the short checks before investing
in full GPU measurements. Preserve all provenance when publishing benchmark results.
