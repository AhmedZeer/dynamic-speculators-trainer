# EAGLE3 scale-up with Hugging Face artifacts

Status: data contract and implementation plan. Public repositories under
`azeeer` are confirmed. This document replaces the BSC deployment assumptions
in `eagle3_generator_bsc.md`; the experimental design remains the starting point.
The Hub integration and proposed configuration fields below are not implemented
by the current bank CLI. No repositories have been created or uploaded yet.

## Initial experiment

Retain Qwen/Qwen3-8B and RedHatAI/Qwen3-8B-speculator.eagle3, with pinned
revisions, rank 32, alpha 64, and o_proj/v_proj adapters. Use math, code, stem,
and chat from the pinned Nemotron decontaminated source. Domain labels describe
the experiment category, independently of the original source repository.

Each subset has exactly 1,000 training examples and 100 validation examples.
Select six subsets per domain initially:

| Subsets in each domain | Role | Training | Validation |
| --- | --- | --- | --- |
| 00000–00004 | Bank training / generator pretraining | 5,000 | 500 |
| 00005 | Reserved joint adaptation / evaluation | 1,000 | 100 |
| Total | Six subsets | 6,000 | 600 |

This is 24 subsets and 26,400 prepared examples overall; math alone is 6,600.
Reserved validation is never used to select pretraining checkpoints. If tuning
on reserved validation is introduced, allocate another independent test subset.

Keep ten bank epochs: one warmup followed by nine collection epochs, ten
checkpoints per collection epoch, one persistent optimization trajectory per
bank subset. The previous plan proposes constant LR 1e-4 and lk_hybrid loss;
these are experiment settings, not existing bank defaults. Twenty bank runs
produce 1,800 adapters. No bank run is launched for reserved subset 00005.
Keep generator context sizes 1–8, last conditioning, one shared reconstruction
pretraining run, and the four joint adaptation arms from the earlier plan.

## Public repositories

All three are Hugging Face dataset repositories. The bank is a collection of
supervision targets; individual deployable models can be exported separately.

| Repository | Content | Typical reader |
| --- | --- | --- |
| `azeeer/eagle3-qwen3-8b-trajectories-v1` | Membership, prompts, target responses, token trajectories, releases | All stages |
| `azeeer/eagle3-qwen3-8b-activations-v1` | Full target-state shards, indexes, compact conditioning summaries | Bank training / behavioral adaptation |
| `azeeer/eagle3-qwen3-8b-lora-bank-r32-v1` | Adapter snapshots, run manifests, metrics and provenance | Generator pretraining |

The trajectories repository owns the release catalog. Other repositories refer
to its exact selection/trajectory fingerprints. A release pins the commit SHA
of each required repository. Consumers use those SHAs, not moving `main`.
Namespace, repository names, and public visibility belong in a shared storage
configuration. Access tokens come from the environment or HF login, never YAML.

Separate model identity, extraction profile, and LoRA training profile from
membership. A different generation or extraction policy creates a new profile;
a different rank creates a different bank family. Existing published artifacts
are never silently regenerated in place.

## Logical and physical structure

Stable identities are canonical prompt hashes, subset IDs such as `math-00000`,
explicit `train`/`validation` membership, and an experiment role of `bank` or
`adaptation`. Local row numbers and hs_123 filenames are implementation details;
they cannot serve as globally unique Hub identifiers.

An illustrative layout is:

```text
trajectories repository/
  README.md
  selections/s0001/catalog.json
  selections/s0001/prompt_registry.parquet
  selections/s0001/math/math-00000/membership.json
  selections/s0001/math/math-00000/prompts.parquet
  profiles/<generation-profile>/profile.json
  profiles/<generation-profile>/math/math-00000/responses/<chunk-hash>.jsonl
  profiles/<generation-profile>/math/math-00000/train.parquet
  profiles/<generation-profile>/math/math-00000/validation.parquet
  releases/r0001/catalog.json
  provenance/<stage-attempt>/...

activations repository/
  profiles/<extraction-profile>/profile.json
  profiles/<extraction-profile>/math/math-00000/index.parquet
  profiles/<extraction-profile>/math/math-00000/shard-00000.safetensors
  profiles/<extraction-profile>/math/math-00000/conditioning.safetensors
  profiles/<extraction-profile>/math/math-00000/complete.json
  provenance/<stage-attempt>/...

bank repository/
  profiles/<training-profile>/profile.json
  profiles/<training-profile>/math/math-00000/seed-47/run.json
  profiles/<training-profile>/math/math-00000/seed-47/epoch-02/snapshot-01/adapter.safetensors
  profiles/<training-profile>/math/math-00000/seed-47/epoch-02/snapshot-01/metadata.json
  profiles/<training-profile>/math/math-00000/seed-47/complete.json
  releases/r0001/bank.json
```

Use Parquet for portable tables; rebuild machine-local Arrow data on download.
Expose category configurations (`math`, `code`, `stem`, `chat`) and ordinary
train/validation splits in the trajectories dataset card. Keep `subset_id` and
`role` as columns; a dataset split is not the same thing as an experiment role.
Consumers always use explicit subset allowlists, so a domain-wide train split
does not accidentally include reserved adaptation data.

Each trajectory retains prompt ID, subset ID, role, membership, canonical
messages, source repository/revision/split/UUID aliases, per-source license
metadata, target model/tokenizer revision, input_ids, loss_mask, prompt_length,
sequence length, generation settings, truncation/termination metadata, and
response identity. Preserve upstream attribution and review retained source
licenses when preparing the public dataset card.

Full states use approximately 1 GiB Safetensors shards, bounded by bytes rather
than a fixed example count. Avoid shards crossing subsets or membership splits.
Use prompt-derived tensor keys and an index mapping prompt ID to shard path,
tensor key, dtype, layer order, sequence shape, and trajectory fingerprint.
Keep actual token lengths, not padded 8,192-token payloads. The full-state layer
order is [2,18,33,36], with BF16 dtype and explicit target revision.

The local reader downloads a needed shard once and reads selected examples.
It must integrate with a bounded shared disk cache; training never issues Hub
requests directly from the GPU forward pass. Existing per-example caches need
a shard-aware adapter before this format is usable.

Publish compact last-token conditioning summaries while extracting states.
Store the exact feature transform and model/profile identity used to produce
them; reuse the existing prompt-summary function to preserve training semantics.
Full states remain necessary for drafter-loss training and validation. Raw-factor
reconstruction can use summaries and LoRA targets without downloading all full
states; its behavioral validation still requires activation shards.

## Membership and future additions

Freeze source revisions, canonicalization, ordering seed, category ownership,
and membership before response generation. Build a shared prompt registry across
all four domains, preserving aliases. Assign cross-domain duplicates to one
owner using the fixed category order chat, math, code, stem, before partitioning.
Record unsupported rows and exclusion reasons rather than silently changing
membership. A math-first GPU run can consume this globally prepared selection.

Preserve the earlier initial seed mapping: chat bank seeds 42–46, math 47–51,
code 52–56, stem 57–61. Persist individual subset-to-seed entries in the catalog.
Expansion allocates new seeds explicitly; it never recomputes this mapping from
the new number of subsets. Partition, response, and generator seeds are separate.

Add new math subsets as math-00006, math-00007, etc. Never move 00005 from its
reserved role. Freeze old prompt identities, split memberships, and artifact
paths. If the source revision grows, exclude every previously registered prompt
and append new complete groups of 1,100 using a separately frozen ordering.
Keep incomplete remainders in an unassigned pool until enough new prompts exist.

Distinguish versions: v1 is the data/schema family; s0001 is an immutable
selection; r0001 is an immutable artifact release. A later selection extends
the prior registry, while a later release catalogs the available stages and
new subsets without rewriting old files. Explicit subset roles, counts, seeds,
profile IDs, checksums, and repository commits belong in each release.

Do not automatically import earlier local experiments merely because their
subset IDs match: verify prompt membership and profile identity first. Initial
IDs in this new Hub family refer to this new frozen selection.

## Publication and recovery

The user's sequence remains select → responses → data → hidden, running target
response generation and extraction on their own GPU machine. Both vLLM server
modes retain `--provenance-dir`. Local work directories are staging areas; Hub
commits contain the durable published artifacts.

| Stage | Durable publication boundary |
| --- | --- |
| select | Frozen membership and prompt tables |
| responses | Sealed response chunks, nominally every 1,000 successes; flush the final partial chunk |
| data | Completed per-subset train/validation Parquet and trajectory manifest |
| hidden | Each completed activation shard and progress record; completion marker last |
| bank training | Each immutable adapter snapshot; run completion marker after all 90 exist |

Files are uploaded before a release references them. A subset completion marker
is committed only after its required files are present. Cross-repository uploads
are not one transaction: publish payloads, resolve their commit SHAs, then commit
the authoritative catalog referencing those SHAs. Publish from one coordinator;
workers own disjoint paths and never race to overwrite the catalog.

Retries are idempotent by identity and checksum. Matching files are skipped;
different bytes under an existing immutable path are an error. Chunk/shard
publication has a durable local journal, bounded upload queue, and visible upload
progress. Upload failures remain pending and are retried on resume. A GPU stage
must not claim durable completion while required uploads are pending.

Restore only the requested subset/stage artifacts from a pinned release. Rewrite
local root paths when rebuilding the local manifest, preserving its logical
fingerprints. Missing responses/states are regenerated only in producer mode;
consumer mode fails if a selected release does not contain them. No whole-corpus
activation scan is added at training startup; verification happens at first
publication/download and is reused from the local cache index.

## Configuration and code changes

Keep `dataset.source` as the upstream prompt source. Add a separate Hub artifact
reference: prepared trajectories are not another source dataset to repartition.
Use producer and consumer configurations derived from the same experiment
definition. Producer mode accepts the four prepare stages and publishes their
outputs. Consumer mode pins a release and hydrates artifacts before training.

The intended external controls are repository IDs/public visibility, producer
or consumer mode, selection/release IDs and commit revisions, local staging root,
generation/extraction/training profiles, explicit bank/adaptation subset lists,
per-subset seeds, upload flush interval/workers/queue limits, activation shard byte
target, and bounded local cache capacity. Availability state is derived from
published manifests, not a user-set ready flag.

Implementation order:

1. Define the portable release and membership schema, cross-domain selection
   registry, subset roles, provenance, and immutable object identities.
2. Add idempotent Hub publication/download with progress journals, completed-stage
   markers, relative paths, and producer/consumer behavior. Cover interrupted
   uploads and restoration on a new machine before bulk generation.
3. Add activation sharding and compact summaries at extraction time, plus a local
   shard reader shared by bank training and generator behavioral training.
4. Extend bank config/trainer for constant LR, lk_hybrid, exact ten-snapshot
   epoch schedules, explicit per-subset seeds, and exclusion of reserved subsets.
5. Add multi-bank generator reconstruction and joint adaptation with the balanced
   sampling, matched exposure, and evaluation rules in the earlier design.
6. Provide math/code/stem/chat producer configs and pinned consumer configs, then
   run a one-subset publication/restore pilot before preparing all 26,400 examples.

Current limitations: the bank config accepts only kl_div/ce, requires a lower
collection LR, uses one shared seed list across all subsets, and trains every
subset in its local manifest. Changing its subset list invalidates that manifest.
The generator currently consumes a local single-bank corpus. These must be
implemented explicitly; editing YAML alone cannot deliver this experiment.

## Storage budget and references

Four BF16 layers of width 4,096 cost 32 KiB per token. For 26,400 trajectories
averaging 2,000 tokens, full states are approximately 1.57 TiB; at 8,192 tokens,
approximately 6.45 TiB. These are estimates, not measured sizes. Start with one
subset, measure actual lengths and publication size, and budget public Hub
capacity before bulk extraction. An upload backlog is not a durable copy.
Public storage on a free account is best-effort, not an unlimited entitlement;
the Hub currently directs substantial storage needs to paid plans or, in some
cases, community-impact grants. Public visibility alone does not establish that
this estimated activation corpus can be stored without cost. Repository
separation does not bypass account-wide limits. The one-subset pilot must check
the actual account capacity before committing to full-state publication at scale.

- [Original experimental design](eagle3_generator_bsc.md)
- [Source dataset card](https://huggingface.co/datasets/openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated/blob/main/README.md)
- [Hub dataset configurations](https://huggingface.co/docs/hub/en/datasets-manual-configuration)
- [Hub upload and commit APIs](https://huggingface.co/docs/huggingface_hub/en/guides/upload)
- [Hub storage guidance](https://huggingface.co/docs/hub/en/storage-limits)
