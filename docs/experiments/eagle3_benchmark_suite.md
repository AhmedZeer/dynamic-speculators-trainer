# Automated vLLM and GuideLLM benchmark suites

`speculators benchmark suite` wraps the existing independent HTTP server and
GuideLLM evaluator. Declare the models and datasets in one YAML file; the runner
exports the selected prompts once, starts one model server, measures each
category, stops that server, and proceeds to the next model.

Start with [the example configuration](../../examples/eagle3-lora/benchmark_suite.yaml).
Replace its model repositories and checkpoint subfolders with your exports.

```bash
speculators benchmark suite \
  --config examples/eagle3-lora/benchmark_suite.yaml \
  --output-root /content/drive/MyDrive/dyn-spec-bench-v1/suites/comparison-v1
```

Use persistent storage for `--output-root` on Colab. After an interruption, run
the same command with `--resume`. Completed categories with matching artifact
checksums are skipped; incomplete, failed, or damaged measurements get a new
attempt directory. A completed suite resumes without launching any servers.

`--dry-run` validates and prints the configuration without downloads, exports,
or processes. It does not check Hub availability, model compatibility, or GPU
capacity. Changing the suite configuration requires a new output root.

## Model entries

Each model needs `name`, `type`, and `model`. Names must be unique. `model` is
either a Hub model repository or a local directory. Optional `revision` selects
a Hub branch, tag, or commit; the first execution resolves and records the full
commit SHA, and resume keeps that SHA even if the repository moves.

| Type | Required model contents | Serving behavior |
| --- | --- | --- |
| `drafter` | An EAGLE3 drafter checkpoint | Original drafter, with no adapter |
| `lora` | Exported `static_lora` inference bundle | LoRA merged into the EAGLE3 drafter |
| `generator` | Exported generator inference bundle | Prompt-conditioned adapter generation |

For generator and LoRA repositories, optional `subfolder` selects a checkpoint
folder containing `bundle.json` and `model.safetensors`, such as `step-002000`.
Point local model entries at the inference bundle or its parent plus `subfolder`.
Raw training checkpoints and generic PEFT folders must first be exported using
the existing generator/LoRA export commands. Bundle downloads include training
provenance and exclude optimizer `checkpoint.pt` files.

Local drafter directories require a full `revision` SHA or Hub local-download
metadata. The runner stages their files into a directory named by the SHA and
records file checksums. Relative local paths, `settings.tokenizer_path`, and
`settings.campaign_config` resolve from the suite YAML's directory.

All entries must use the same pinned target model. Bundles supply their target
and drafter references. A suite containing only bare drafters uses the recipe's
Qwen3-8B target, revision `b968826d9c46dd6066d109eabc6255188de91218`.
The shared workload checks the target/tokenizer; each entry's own drafter
revision is checked against its running server. The original export manifest
with its training drafter reference is preserved as `source_manifest.json`.

## Datasets and settings

`benchmarks` accepts one or more dataset selections:

| Dataset | Categories |
| --- | --- |
| `nemotron` | `math`, `code`, `stem`, `chat` |
| `RedHatAI/speculator_benchmarks` | `HumanEval`, `math_reasoning`, `qa`, `question`, `rag`, `summarization`, `tool_call`, `translation` |

Use `[all]` to select all categories. `writing` is an alias of the byte-identical
`question` workload. Red Hat export details and source revisions are recorded in
the [dataset inventory](speculator_benchmarks_inventory.md). Red Hat entries also
accept `prompt_format: auto` (default), `raw`, or `chat`. An optional benchmark
`name` distinguishes repeated selections of the same dataset.

Defaults are 50 samples **per category**, a 256-token output budget, seed 42,
three warmup requests per category, one concurrent request, GPU 0, port 8000,
8,192-token model context, three speculative tokens, and GPU memory utilization
0.85. Override these through `settings` in the example. Concurrency is currently
fixed at one to match the adapter runtime. Throughput uses the full exported
category; it does not apply a second sampling limit.

Nemotron uses the existing campaign's validation membership. Only the trajectory
metadata and selected validation Parquet files are downloaded if missing.
Red Hat uses its pinned JSONL sources and the recipe's target metadata. Neither
path downloads training activation dumps or generates training activations.
The tokenizer and model weights use the normal Hugging Face cache.

## Environment, lifecycle, and failures

Run from an editable checkout in the environment already used for independent
serving: vLLM 0.31.0 and GuideLLM 0.7.1, with the generator patch applied for
adapter models. See the [HTTP runbook](eagle3_generator_http.md) for installation
and patch instructions. The runner uses the current Python interpreter for both
server and client; it does not install dependencies or patch vLLM automatically.

The wrapper retains the existing BF16, eager, V1, single-sequence matched server
configuration. It waits for health and verified model status before measuring.
Stop any manually launched server on the chosen port first; the runner refuses
to reuse or terminate an unrelated service.

Server and client each run in an owned process group. Normal completion,
timeouts, Ctrl-C, and SIGTERM trigger cleanup of those groups. Notebook/runtime
loss relies on persisted output for resume. An output lock and a local port lock
prevent overlapping suite launches.

`startup_timeout_seconds` defaults to 900, including initial model downloads;
`client_timeout_seconds` defaults to 3,600 per category, and
`shutdown_timeout_seconds` to 30. With `continue_on_error: true` (default), a
failed category is recorded and its server is recycled before continuing. A
server startup failure marks that model's remaining categories failed and moves
to the next model. Set `continue_on_error: false` to stop at the first failure.
The command exits unsuccessfully if any measurements remain failed.

## Results and provenance

```text
<output-root>/
  suite_request.json          # requested configuration
  resolved_models.json       # immutable revisions and bundle checksums
  suite.json                 # resolved models, workloads, suite identity
  campaign_configuration.json
  state.json                 # per-category attempts and completion checksums
  summary.md                 # models × dataset/categories, output tok/s
  summary.csv
  summary.json               # table plus detailed performance/adapter metrics
  failures.json
  workloads/<dataset>/       # frozen manifest and category JSONLs
  suite_attempts/<attempt>/  # invocation, package/git provenance, source patch
  servers/<model>/attempt-*/ # server log, argv, launch provenance and hashes
  measurements/<model>/<dataset>/<category>/attempt-*/
                            # client log/argv, native GuideLLM results,
                            # status/metrics snapshots, copied provenance
```

The table reports aggregate output tokens per measured wall second, including
prefill and serving overhead. Full latency and adapter metrics remain available
in the native artifacts. Every actual server launch passes `--provenance-dir`;
client and suite invocations save command, package, git, and patch provenance.
Include the complete output directory when publishing results.

The suite runs throughput directly, with **no correctness-generation pass**.
Results explicitly record `correctness_status=not_checked`; these measurements
support exploratory speed comparisons. Server identity, tokenizer agreement,
successful request counts, and adapter execution/restoration are still checked
by the underlying evaluator. Token-exact correctness remains an independent
command in the HTTP runbook.
