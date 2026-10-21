# Speculator benchmarks: inventory before integration

Inspected 2026-10-21 at dataset revision
`2ae86affa2cb97a972b7fc681dd51c04fbff083e`:
[RedHatAI/speculator_benchmarks](https://huggingface.co/datasets/RedHatAI/speculator_benchmarks/tree/2ae86affa2cb97a972b7fc681dd51c04fbff083e).
All nine JSONL files were read to inspect row counts and schemas. This is an
inventory, not a new integration or completed benchmark.

| File selector | Task | Rows |
| --- | --- | ---: |
| `HumanEval` | Python code completion | 164 |
| `math_reasoning` | Math reasoning | 80 |
| `qa` | Question answering | 80 |
| `question` | MT-Bench mixed tasks | 80 |
| `rag` | Retrieval-augmented generation | 80 |
| `summarization` | Summarization | 80 |
| `tool_call` | Tool-use prompts | 200 |
| `translation` | German-to-English translation | 80 |
| `writing` | Same content as `question` at this revision | 80 |

There are 924 rows across files, or 844 after dropping the duplicate file
`writing`. Other cross-file or training-set duplicates have not been audited.

## Details affecting evaluation

- `question.jsonl` and `writing.jsonl` are **byte-identical**, SHA256
  `9db8ea4678e058c26f21a9cc0bf79f95a9750e370cafb4f90a8e3b058c2a2aa4`.
  Each includes 10 prompts for each of writing, roleplay, reasoning, math,
  coding, extraction, stem, and humanities. Do not treat them as two independent
  categories in aggregate scores.
- The card lists eight tasks, but omits `tool_call.jsonl`. That file includes
  20 source categories spanning simple, parallel, multiple-function, relevance,
  memory, web-search, format-sensitivity, and multi-turn prompts. These are
  serialized text prompts, not evidence that a benchmark executes actual tools.
- Every file has a string `prompt`. HumanEval has code-specific columns;
  other files use `question_id`, source categories, and sometimes references.
  Load each file separately to avoid incompatible cross-file schemas. The Hub
  viewer currently reports a schema-casting error when combining files.
- The generic evaluator already defaults to this repository and lists all nine
  selectors. That existing route is distinct from our pinned, checksummed
  Nemotron manifest workflow; it does not by itself provide an audited external
  workload or protect against the duplicated category.

## Proposed external-workload protocol

Use eight unique file groups, with labels such as `specbench-humaneval`,
`specbench-math-reasoning`, `specbench-mt-bench`, and `specbench-tool-call`.
Keep `nemo-*` labels for Nemotron validation. Start with 50 deterministic prompts
per unique group, preserve source category metadata, and balance MT-Bench
sampling across its eight categories. Fix the revision, tokenizer, prompt
formatting policy, output cap, concurrency, and seed for every server mode.

HumanEval is code-completion data; do not silently apply a conversational
template to every task. Decide prompt handling before freezing the workload.
Log realized prompt lengths and generated lengths; RAG and summarization may
stress different context lengths from the short Nemotron math workload.

This is an external-distribution evaluation candidate, not established
contamination-free OOD data. Audit exact and normalized prompt overlap against
the Nemotron training selections before making stronger OOD claims. Reference
answers and HumanEval tests are unnecessary for speed measurements; answer
quality requires a separate evaluation protocol.
