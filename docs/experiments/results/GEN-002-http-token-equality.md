# GEN-002 — HTTP token equality and repeatability

## Record

- Observation: 2026-10-21.
- Status: exact-output correctness failed; cause unresolved; investigation deferred
  by the experiment owner to prioritize performance iteration.
- Evidence: user-provided reports in local `tmp/`; these files are not published
  in the repository. No latency or throughput measurement is contained in them.

## Setup

50 Nemotron math validation prompts, manifest identity
`09006e3d4dafd6638a9a707ceb749ee23b496756a4c1c8b5f6affe8a77030ea8`.
Each output contains 256 tokens. Prompt/settings fingerprints match across runs.
Target Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`;
drafter RedHatAI/Qwen3-8B-speculator.eagle3 revision
`08610ffa01dd9f16731fe8f627b85905b6aa51c4`.
vLLM 0.31.0, V1 runner, seed 42, one active sequence, three speculative tokens.
See the [HTTP runbook](../eagle3_generator_http.md) for launch settings.

## Evidence

| Report | SHA256 |
| --- | --- |
| `target.json` | `69bf9491e2181d62a0d6e1e322ee8f28b817facf8fad76fc825d940c4f95d837` |
| `base2.json` (target-only repeat) | `3c2dbf23b79843f66d9d95734cb5494f8716f2b79aae336a03125ea428a6d0a3` |
| `eagle3-base.json` | `7ad7433b414063eedfdbd3a30dec5330413f5da31453fdaabfee8fefdfb61f2d` |
| `eagle3-base-2.json` | `6dbd1a83e4d8e5986baec97b3fa71919274a1e0ce175a6c6b84d653ed0cbb12b` |
| `scale-1-direct.json` | `4e4c33fa7a7b9cf1de8a487d27c07213d01751539049f71a8bf1069aefc50a4b` |

| Comparison | Exact sequence matches |
| --- | --- |
| Original target vs repeated target | 23/50 |
| Original EAGLE3 vs repeated EAGLE3 | 50/50 |
| Repeated target vs repeated EAGLE3 | 17/50 |
| Original generator vs repeated EAGLE3 | 50/50 |

The generator report records 50 invocations and 50 completed/restored requests.
Original target code revision was `c9384b5`, original speculative runs `6976d0b`,
and repeated runs `c08d0f5`. Serving/launch code and generation request semantics
did not change between those revisions; recorded patch identities match.
Full hardware, kernel, and environment provenance is not available in these
reports, so they cannot establish identical runtime environments.

## Interpretation and decision

The target-only reference is not reproducible across the supplied runs.
Numerical behavior, environment changes, and shared decoding bugs remain possible;
none is established as the cause. Identical generator/EAGLE3 outputs do not prove
an acceptance or speed benefit. Sequence mismatch counts are not token error rates.

Defer correctness diagnosis and allow **exploratory timing** with
`--allow-output-mismatch` in the GuideLLM evaluator. Preserve failed reports and
mark performance outputs correctness-unverified; do not rewrite them as matched.
Model, validation, seed, workload coverage, and connected-server checks remain
mandatory. Compare generator performance against both target-only and EAGLE3
base. Such speedups are empirical timing comparisons, not verified lossless
acceleration. No speedup has yet been calculated from these correctness reports.

## Updates

- 2026-10-21: Repeat comparisons recorded; failure explicitly deferred.
