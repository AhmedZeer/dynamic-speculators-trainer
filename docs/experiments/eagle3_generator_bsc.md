# BSC EAGLE3 generator experiment plan

Status: experimental design, before implementation or submission. The user
confirmed four domains, ten total bank epochs, and joint adaptation. Defaults
marked proposed below are recommendations rather than confirmed measurements.

## Objective and fixed scope

Test whether a generator pretrained on a larger, diverse LoRA bank helps learn
useful adapters on unseen subsets across chat, math, code, and stem. Preserve
Qwen/Qwen3-8B, RedHatAI/Qwen3-8B-speculator.eagle3, rank 32, alpha 64, and
o_proj/v_proj adapters. Pin model and dataset revisions before transfer.
The serving integration being developed separately can be merged later.

The dataset is openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated,
configuration default. Use five bank subsets plus one reserved adaptation subset
per domain. Each subset contains 1,000 training prompts and 100 disjoint
validation prompts, preserving the earlier experiment convention.

| Quantity | Per domain | Total |
| --- | --- | --- |
| Bank subsets | 5 | 20 |
| Bank training prompts | 5,000 | 20,000 |
| Bank validation prompts | 500 | 2,000 |
| Reserved adaptation subsets | 1 | 4 |
| Adaptation training prompts | 1,000 | 4,000 |
| Adaptation validation prompts | 100 | 400 |
| All prepared prompts | 6,600 | 26,400 |
| Collected LoRAs | 450 | 1,800 |

## Membership, seeds, and extension

Deduplicate canonical prompts, deterministically shuffle, then select consecutive
blocks of 1,100: subset IDs 00000 through 00004 form the bank; 00005 is reserved
for adaptation. Consecutive refers to the shuffled sequence, following the
existing selector. Validate prompt identity disjointness across domains as well
as within domains; resolve cross-domain duplicates deterministically before
finalizing memberships. Preserve source IDs, licenses, aliases, and exclusions.

Assign exactly one stable training seed to each bank subset, distinct across all
20 subsets. Proposed mapping: base 42 + 5 * domain_index + subset_index, with
domain order chat, math, code, stem. Persist the mapping explicitly so expansion
never changes old seeds. Continue the same optimization trajectory across epochs;
do not restart adapters or optimizers between collection epochs. Partition,
response, and experiment RNG seeds remain separate controls.

Memberships and artifacts are immutable. Expansion appends new identified
subsets through a versioned manifest without repartitioning old prompts; training
uses an explicit bank-subset allowlist so reserved subsets never enter the bank.
The existing implementation requires new roots for changed memberships; safe
extension therefore needs explicit manifest/version support.

## Preparation stages

1. `select`: resolve revisions, validate/deduplicate source prompts, freeze
   membership, and write prompts and manifest. It does not generate responses.
2. `responses`: ask the pinned target to generate new responses, saving exact
   prompt/completion token IDs and boundaries. Source answers are discarded.
3. `data`: convert saved trajectories into indexed Arrow training rows and map
   subset memberships to those rows. It does not run the target again.
4. `hidden`: extract target states on those exact saved trajectories. Save
   layers [2,18,33,36] in bfloat16; the first three supply EAGLE3 inputs and
   prompt conditioning, the last supplies target-distribution supervision.

Run selection on the connected preparation machine. Generate responses and
extract states at BSC. Switch between response-mode and extraction-mode servers
as in the existing pipeline. Both server modes must receive `--provenance-dir`.
Preserve the original response/context settings initially: thinking disabled,
temperature 0.7, top-p 0.8, top-k 20, 4,096 output tokens, 8,192 context tokens.
Freeze these settings across domains. Account for any left truncation in the
saved token sequence and conditioning boundary.

## Bank optimization and exact collection

Each of 20 runs starts from the same frozen base drafter. Train ten total epochs:
epoch 1 is warmup with no collected entries; epochs 2 through 10 collect ten
entries each. Use the same constant LR in both phases, proposed 1e-4 from the
previous warmup. No LR decay or phase-boundary reduction. Preserve AdamW,
weight decay 0.01, dropout 0.05, noise 0.05, token budget 8,192, and three TTT
steps initially. Proposed loss: lk_hybrid for bank training and adaptation, to
match the latest generator experiment; this requires extending the bank schema.

For each collection epoch, set the sampler epoch and determine its actual full
optimizer-update count S_e, before resume skips. Save after updates
ceil(k * S_e / 10), k = 1,...,10. Fail explicitly if S_e < 10. This yields ten
distinct update boundaries, includes the epoch end once, and tolerates changing
packed batch counts. Do not approximate with a fixed global save interval.
Persist the resolved schedule and publication state in recovery metadata.
Evaluate at every epoch end; retain all 90 snapshots rather than pruning by
validation score. A completion check must verify exactly 90 immutable entries
per run and 1,800 overall. Inspection reports factor duplicates, update norms,
adjacent distances, and epoch metrics; selected snapshots can receive additional
behavioral evaluation to check whether weight variation is useful diversity.

## One shared pretraining run

Run one bank reconstruction pretraining job across all 20 subsets. There is no
conditioning sweep, stride/seed heatmap, or winner selection. Proposed fixed
conditioning: last prompt-token states, with the existing architecture and
context sizes 1 through 8.

Sample domain uniformly, subset uniformly within domain, then checkpoint
uniformly within that subset's 90 entries. Draw distinct conditioning examples
only from that subset's training membership. Keep raw-factor L1 reconstruction
to preserve comparability with the previous experiment. Validation uses bank
validation prompts only, balanced across domains; the reserved sixth subsets
must not enter pretraining or pretraining checkpoint selection.

Proposed initial budget: 10,000 optimizer updates, four episodes per update,
giving 40,000 episodes, or approximately 22 draws per bank snapshot in expectation.
Keep generator LR 1e-5 and the existing 100-update LR ramp. This generator ramp
is separate from the constant bank-training LR. Report checkpoint coverage,
reconstruction curves, per-domain drafter scores, and elapsed time. Validate
every 1,000 updates and use the final checkpoint for adaptation. Freeze budget
and hyperparameters before evaluating reserved validation prompts.

One seed per subset removes multiple independent initialization targets, but
does not make weight targets single-valued: identical conditioning can still
be paired with multiple trajectory snapshots. Raw LoRA factors also have basis
ambiguities. Deterministic L1 reconstruction may learn a compromise; behavioral
adaptation must establish whether this initialization is useful.

## Joint one-epoch adaptation

Use all four reserved 00005 subsets together. Compare the existing four arms:
bank-pretrained generator, fresh generator, fresh ordinary LoRA, and transferred
ordinary LoRA. Both ordinary LoRA arms maintain one shared adapter across domains.
For transfer, use a predeclared bank-validation selection rule: best macro-domain
validation score among final bank snapshots, with a deterministic tie-break.
No reserved validation results may influence that choice.

For every arm, process each of the 4,000 training prompts once in one epoch,
with balanced domain interleaving. Form conditioning groups within a domain;
match group order, contexts, token budgets, augmentation/dropout seeds, and
optimizer boundaries across arms. Every arm starts independently; training an
earlier arm must not alter the initialization of a later arm.

Evaluate before and after adaptation on the 400 held-out reserved prompts at
context sizes 1, 4, and 8. Report each domain separately and their equal-weight
macro average, with loss, full_acc_0, and other existing draft-step metrics.
Include frozen-base scores. Compare pretrained versus fresh generator to isolate
the pretraining benefit, and generator versus ordinary LoRA for adaptation
behavior. Preserve the capacity caveat: the generator is much larger than LoRA.
Record intermediate training progress and timings for learning-speed analysis.
Reserved validation is final evaluation only; if tuning is introduced, create
a separate final test set before claiming independent generalization.

Offline full_acc_0 is argmax agreement, not measured acceptance or speedup.
After merging compatible serving support, evaluate final artifacts on matched
held-out requests with real decoding acceptance, throughput, and generator
overhead, including conditioning and adapter-refresh policy.

## BSC resources and scheduling

The official MN5 ACC overview specifies four H100 GPUs with 64 GB each, 80 CPU
cores, 512 GB RAM, and 480 GB local NVMe per node. Budget against 64 GB per GPU.
The GPU request must be paired with 20 CPU cores per GPU. Use account etur22,
partition acc, and QoS acc_ehpc as provided by the user; actual account limits
and software/driver versions must be checked at BSC.

Prefer four independent single-GPU bank runs per node, explicitly bound one per
GPU, with 20 CPUs each. This preserves single-GPU epoch semantics and avoids
DDP overhead for 1K runs. The existing n_workers option does not assign devices;
GPU-aware dispatch must be implemented. Use arrays/bundled tasks with isolated
outputs, bounded concurrency, and stage dependencies. Never have multiple jobs
rewrite the same aggregate bank index; aggregate after all runs finish.

Benchmark response generation and extraction with one replica per GPU first.
For Qwen3-8B, test this layout against tensor-parallel serving and choose by
measured throughput and memory. Extraction concurrency must be tuned separately
because hidden-state traffic is large. For adaptation, four GPUs can run the
four matched arms concurrently; one shared bank pretraining run starts on one
GPU, with distributed support added only if profiling warrants it. There is no
automatic promise of four-GPU speedup from the current generator implementation.

Keep models, manifests, responses, activations, checkpoints, logs, and provenance
durably in GPFS. Use the job's TMPDIR for local staging, bounded activation LRU,
and Torch/Triton caches. BSC forbids user temporary data in /tmp, and deletes
job-local storage after termination. Measure quotas with bsc_quota and free local
space before scheduling. Flush offline W&B logs to GPFS for later synchronization.

Four bfloat16 layers of width 4,096 require approximately 32 KiB per sequence
token, excluding metadata and auxiliary tensors. At 2,000 tokens per trajectory,
26,400 examples require about 1.57 TiB; at the full 8,192-token limit, about
6.45 TiB. These are calculated estimates, not measured dataset sizes. Stage
active subsets rather than the entire activation corpus on the 480 GB SSD.
Float32 rank-32 adapter factors are approximately 2.125 MiB per snapshot at
the current model dimensions, or 3.74 GiB for 1,800 snapshots before metadata
and recovery state. Activations dominate storage. A pilot must measure actual
lengths, disk traffic, VRAM peaks, and per-stage runtime before bulk extraction.

## Offline delivery

Prepare a relocatable conda-pack archive on compatible Linux x86_64. Freeze the
tested Python/PyTorch/CUDA/vLLM/Transformers/PEFT stack rather than resolving broad
project ranges at BSC. Include hs_connectors, generator/LoRA dependencies,
compiler/JIT requirements, and local package wheels. Install project packages
from wheels for packing; editable installs retain external checkout references.
Separate serving and training archives if dependency compatibility requires it.
Finalize the serving archive after the other agent's integration is reconciled.

Deliver the code revision, wheels, archive hashes, explicit conda/pip inventories,
pinned target/drafter snapshots and tokenizers, selected prompt manifests, and
Slurm/config files. Dataset selection runs online, so unused dataset splits need
not be transferred. Preserve all manifest references when relocating: current
manifests contain absolute roots and need an explicit validated relocation path.

At BSC, extract at a stable prefix and run conda-unpack. Enable HF_HUB_OFFLINE,
HF_DATASETS_OFFLINE, and WANDB_MODE=offline. Offline readiness includes model
conversion, FlexAttention forward/backward, vLLM token-ID response generation,
hidden extraction, generator reconstruction/adaptation, and interrupted resume.
Test the transferred archives with networking unavailable before full submission.
Capture driver, module list, Slurm resources, environment hashes, commands,
git revision/patches, and checkpoint hashes. Always use launch_vllm.py's
provenance-dir, and retain provenance histories when publishing resumed runs.

## Implementation order and acceptance

1. Add constant bank LR support, one-seed-per-subset mapping, explicit training
   subset selection, and exact per-epoch snapshot schedules. Verify varying
   packing lengths and interrupted publication/resume without duplicates.
2. Add multi-bank corpus and balanced reconstruction, plus joint adaptation and
   cross-domain reporting. Verify membership isolation and matched arm exposure.
3. Add offline selection/relocation and pinned local-model resolution, environment
   build/pack instructions, GPU-aware dispatch, and Slurm stage dependencies.
4. Run a small BSC end-to-end pilot, including interruption/resume, then finalize
   concurrency, cache sizes, quotas, wall time, and tested environment locks.
5. Execute preparation, 20 bank runs, bank inspection, one pretraining run,
   four joint adaptation arms, and reporting. Merge serving work and run actual
   speculative decoding evaluation when ready.

The existing worktree contains serving integration changes. Do not switch its
branch or include those edits in BSC commits inadvertently; use an isolated
checkout for implementation and integrate deliberately later.

## Sources

- [Current bank pipeline](../developer/lora_bank.md)
- [Previous generator experiment](eagle3_generator.md)
- [Dataset metadata](https://huggingface.co/datasets/openeurollm/Nemotron-Post-Training-Dataset-v2-decontaminated/blob/main/README.md)
- [MN5 ACC hardware](https://www.bsc.es/supportkc/docs/MareNostrum5/overview)
- [MN5 GPU/Slurm resources](https://www.bsc.es/supportkc/docs/MareNostrum5/slurm/)
- [MN5 storage and temporary-directory rules](https://www.bsc.es/supportkc/docs/MareNostrum5/storage/)
- [conda-pack relocation](https://conda.github.io/conda-pack/index.html)
- [conda-pack editable-package handling](https://conda.github.io/conda-pack/api.html)
