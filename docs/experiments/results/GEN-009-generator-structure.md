# GEN-009 — Generator architecture and adapter target modules

## Record and evidence

- Receipt date: 2026-10-30. Status: preliminary visual evidence, one visible seed.
- Ten original user-supplied W&B screenshots are archived in
  [the evidence directory](evidence/2026-10-30-structure-v1/README.md), with
  [file hashes and image numbering](evidence/2026-10-30-structure-v1/manifest.json).
- [Configured protocol and commands](../eagle3_generator_structure_v1.md);
  [historical baseline](GEN-001-direct-data-scaling.md).
- Raw W&B histories, run URLs/IDs, checkpoint hashes, resolved executed configs,
  and per-run code/hardware provenance were not supplied with these images.
  Exact metric differences, uncertainty, completion status, and timing speedups
  are therefore unavailable. No numerical metric table is reconstructed from pixels.
- Several runs are visibly partial. Compare curves at shared optimizer steps,
  rather than comparing their last points as if all had completed 1,500 updates.

## Intended comparison

Each new config trains a fresh conditional generator for 1,500 updates on 4K
training rows: 1K each from Nemotron math, code, STEM, chat. The configured
validation is the separate 100-row split/category, grouped into context-4
cohorts, every 250 updates. Seed 42, rank 32, alpha 64, four training cohorts
of 1–8 examples/update, and the original saved-response three-step LK objective
are shared. These settings are configured intent, not independently verified
execution provenance for each screenshot.

The baseline `scale-1-direct-gseed42` uses v_proj/o_proj adapters and
encoder_hidden=512, condition_dim=64, residual_blocks=2, decoder_width=512.
Architecture variants change one of these settings at a time; example_dim=128
and descriptor_dim=32 remain fixed. Module variants preserve the baseline
generator architecture while changing its output heads and adapter targets.

The workflow changed during this study: generator BF16 parameters/moments,
batched independent cohort generation, activation concurrency/cache settings,
and checkpoint export were updated. Some changes can take effect on resume.
The user reported an RTX PRO 6000 for the current workflow; individual hardware
and execution settings are not verified from these screenshots. Historical and
new cumulative-update-time curves are not controlled architecture-speed tests.
That x-axis includes data waits and update work, and excludes separate validation
and upload pauses. It does not measure inference generation latency or end-to-end
experiment duration.

## Architecture observations

| Comparison | Evidence | What is visible | Supported interpretation |
| --- | --- | --- | --- |
| decoder_width 256 / 512 baseline / 1024 | Images 1–2 | Agreement and all three losses are close at shared steps. The 256 curve appears to stop around 500; 1024 around 1,250; baseline continues to 1,500. | No clear validation benefit from a wider factor decoder in the visible overlap. Width 256 is a compact candidate, not an established equal-quality final checkpoint. |
| condition_dim 32 / 64 baseline / 128 | Images 3–4 | Curves closely overlap; both variants are partial, roughly within the first 500–750 updates. | No clear benefit from increasing condition-vector width within this range and budget. Final convergence equivalence is untested. |
| residual_blocks 1 / 2 baseline / 4 | Images 5–6 | The two variants are close; some agreement and loss curves look slightly better than the historical baseline. One block reaches roughly 1,500 visible updates; four blocks roughly 1,250. | Extra residual depth has no clear advantage over one block here. This supports testing the simpler trunk, not claiming depth never helps. |
| encoder_hidden 256 / 512 baseline / 1024 | Images 7–8 | The 1024 encoder shows a small, reasonably consistent advantage in several agreement/loss panels; 256 remains near baseline and appears to stop around 750. | A wider encoder is a weak positive signal. It needs raw histories and repeats before choosing it over smaller encoders. |

The architecture sweep is largely insensitive over the tested range, with a
small encoder-width signal. The plotted losses support the same broad reading
as agreement. This does not establish exact equality or statistical significance.
The screenshot grid does not show the code/chat agreement panels for these
comparisons, so domain-specific claims are limited to displayed math/STEM and
the reported aggregate.

The larger encoder and the reduced decoder have not been tested together.
Neither has the combination of reduced decoder/condition width/depth. A combined
small model is a new experiment, not a result implied by separate ablations.

## Module observations

Images 9–10 compare:

| Target set | Projections | Calculated generator parameters |
| --- | --- | ---: |
| Historical baseline | v_proj, o_proj | 293,288,224 |
| MLP | up_proj, down_proj, gate_proj | 814,397,760 |
| Attention | q_proj, k_proj, v_proj, o_proj | 646,297,952 |
| All | All seven projections | 1,453,177,280 |

Counts are calculations for the configured pinned drafter/rank/architecture,
not measurements extracted from the screenshots.

At common visible checkpoints, the overall picture is **all modules > attention
> MLP > historical v/o baseline**. The all-module and MLP curves appear to stop
around 500, and attention around 750; the baseline continues to 1,500. This is
an early-checkpoint ordering, not a completed 1,500-update ranking. The ordering
is especially clear in overall/math agreement and the three validation losses;
it is not a claim of strict ordering in every displayed metric at every step.

Attention-only is stronger than MLP-only despite having fewer generator
parameters. This is evidence that **where adaptation is applied matters**, beyond
simply increasing hypernetwork size. Adding q/k to the baseline v/o set is a
promising change, but this study does not isolate q from k or identify which
individual projection contributes the gain.

All modules improve on attention alone in the visible overlap, suggesting
complementary value from MLP adaptation when attention is already adapted.
The all-module generator also has substantially more parameters. The evidence
does not separate target coverage, parameter count, initialization, and their
interactions. Attention-only and MLP-only are different non-nested target sets;
this is not proof that module count alone monotonically determines quality.

## Agreement, acceptance, and the architecture hypothesis

These screenshots report **offline validation agreement**, including later
training-step agreement, and LK losses. They do not supply serving acceptance
rates, acceptance lengths, output throughput, or generator/merge latency.
Improved later-step agreement is encouraging for speculative proposal quality,
but verifier behavior on generated prefixes and serving cost still need measurement.

Lack of sensitivity to trunk width/depth does **not** show that this architecture
is inadequate. It is compatible with sufficient trunk capacity, an output-space
or target-placement bottleneck, limitations of the fixed last-state/mean/std
conditioning representation, or insufficient signal/budget to distinguish variants.
These are alternatives to investigate, not findings established by the plots.
All variants retain the same basic pooling and full-factor generation design;
they test capacity within that family, not fundamentally different architectures.

The parameter distribution also matters. The baseline has 285,769,728 of
293,288,224 parameters in its factor heads. Changing condition width or residual
depth changes relatively little of the whole model. Decoder width does change
head size substantially (about 150M total parameters at width 256 versus 582M
at 1024), yet shows no clear advantage for the larger model at shared visible
steps. That makes a smaller decoder worth evaluating; it does not prove all
forms of architecture improvement are exhausted.

## Working direction and next measurements

The useful objective is **generate effective adapters for useful target modules
at low cost**, rather than maximize the number of generated weights. Wider target
coverage is promising here; weight count itself is not the success metric.
Dense full-factor heads grow quickly with coverage, and the acceptance benefit
must outweigh generation/merge/application overhead in the selected serving path.
Merged adapters incur generation/merge cost at preparation; per-request factor
application can add work during drafting. More target modules do not necessarily
add base-model forward FLOPs when adapters are merged.

Priorities:

1. Finish module runs to the same budget and export raw per-step histories.
   Repeat the strongest comparison across seeds. Prefer comparisons under the
   same BF16/batching/hardware settings over attributing historical timing gaps
   to architecture.
2. Compare attention and all-module generators against **matched static LoRAs
   for each same module set**, using the same 4K/1,500-update budget, rank, alpha,
   and validation. This distinguishes conditional generation from the benefit
   of simply adapting more projections.
3. Test conditioning usefulness: correct prompt context versus a fixed context
   and shuffled/mismatched contexts, with the same trained generator. Keep
   inference precision and adapter scale fixed. If appropriate conditioning
   does not outperform these controls, prompt-specific adaptation is not yet
   established, even if generator training beats the pretrained drafter.
4. Evaluate the strongest coverage set with a width-256 decoder, first as a
   single change. Measure losses/agreement, generator parameters and latency,
   then real acceptance length and output tokens/s at concurrency 1. The current
   vLLM hook supports only v/o: attention/all-module serving support is still
   required before these new sets can use the existing throughput suite.
5. Consider a different conditioning/output-generation architecture only after
   these controls identify a limitation: better context use if conditioning
   proves ineffective, or more efficient output generation if broad coverage
   helps quality but dense heads dominate cost. No new architecture is implemented
   or validated by this record.

Working decision: prioritize useful target coverage and efficient generation;
retain compact-trunk candidates. Do not record architecture failure, actual
acceptance improvement, or a throughput win from these screenshots alone.

Implementation follow-up: [three independent training configs and commands](../../../examples/eagle3-lora/structure_followups_v1/README.md)
now cover static attention LoRA, static all-module LoRA, and an all-module
generator with decoder width 256. Execution/results are pending.
