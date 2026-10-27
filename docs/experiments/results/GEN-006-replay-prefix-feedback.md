# GEN-006 — Replay, verified-prefix refresh and verifier feedback

## Record

- Observation date: 2026-10-27; screenshots supplied by the user.
- Status: preliminary, one displayed seed (42), visual evidence only.
- [Runbook](../eagle3_generator_ablations_v4.md).
- [Structured observations and image hashes](benchmarks/2026-10-27-replay-prompt-probe/metrics.json).
- Execution revisions, complete run IDs, checkpoint hashes and raw metric exports
  are unavailable for this record. Setup below describes repository configuration;
  exact correspondence with every executed run is not independently verified.

## Question and setup

Does replacing ordinary saved-response distillation with immediate replay of the
current drafter's proposals improve a prompt-conditioned LoRA generator? Does
refreshing its conditioning from the verified prefix, or adding previous verifier
acceptance, improve that continuation?

| Arm | Conditioning and objective | Budget |
| --- | --- | --- |
| `static_lora_lk` | One shared static LoRA, ordinary LK | 1,000 updates from initialization |
| `prompt_lk` | Prompt-conditioned generator, ordinary LK | 1,000 updates from initialization |
| `prompt_ce` | Prompt-conditioned generator, ordinary teacher-argmax CE within draft vocabulary | 1,000 updates from initialization |
| `replay_prompt_lk` | Prompt-conditioned generator, current-policy proposal replay, LK | Parent `prompt_lk` at 500 + 500 replay updates |
| `replay_prefix_lk` | Replay with verified-prefix conditioning refreshed each round | Same parent at 500 + 500 replay updates |
| `replay_prefix_feedback_lk` | Prefix replay plus previous acceptance vector | Same parent at 500 + 500 replay updates |

All configured arms use the same 4K training pool: 1K per category from shard
`00001`, eight independent requests per optimizer update, and rank-32 V/O LoRAs.
The target and drafter backbone are frozen. The intended common learning rate is
1e-4 with the existing warmup schedule. Replay restores parent optimizer moments
and RNG. Its additional feedback projection is zero-initialized.

The prompt probe uses 50 tuning examples per category from held shard `00000`,
separate from training and the reserved test rows. It uses singleton prompt
conditioning, saved-response teacher forcing and **zero feedback**, including for
prefix/feedback-trained models. Overall agreement is the equal-weight mean of the
four category `full_acc_0` values. It is not measured online speculative acceptance.

Optimizer updates are matched; tokens, target queries and wall time are not.
Ordinary updates supervise saved response trajectories. Replay supervises at most
4 verification rounds × 3 draft positions = 12 positions per request, or at most
96 positions per eight-request update. Thus the last 500 replay updates supervise
at most 48K positions. EOS/context limits can reduce that number. Exact ordinary
and replay token counts must be obtained from the run artifacts.

## Evidence and measurements

Approximate readings from screenshots, not raw numerical exports:

| Arm | Overall prompt agreement at 500 | At 1,000 | Observation |
| --- | ---: | ---: | --- |
| `prompt_lk` | ~78.8% | ~79.3% | Ordinary continuation improves the probe |
| `prompt_ce` | ~78.6% | ~79.2% | Close to ordinary LK; significance unknown |
| `static_lora_lk` | ~77.1% | ~77.7% | Below both prompt generators |
| `replay_prompt_lk` | ~78.8% | ~76.2% | ~2.6 percentage-point decline |
| `replay_prefix_lk` | ~78.8% | ~76.2% | Similar overall decline |
| `replay_prefix_feedback_lk` | ~78.8% | ~76.2% | Similar overall decline |

Small differences between the replay arms are deliberately not transcribed at
false precision. All three lose endpoint agreement in math, code, STEM and chat.
The replay plots join only two evaluations (500 and 1,000); the connecting lines
do not establish a smooth or monotonic degradation between those evaluations.

Figures:

- [Ordinary generator/objective/static-LoRA prompt probes](assets/GEN-006-objectives-prompt-probe.png).
- [Prompt-only replay prompt probe](assets/GEN-006-prompt-replay-prompt-probe.png).
- [Prefix and prefix-feedback replay prompt probes](assets/GEN-006-prefix-feedback-prompt-probe.png).
- [Prompt-only replay training loss](assets/GEN-006-prompt-replay-training-loss.png).
- [Prefix and prefix-feedback training losses](assets/GEN-006-prefix-feedback-training-loss.png).

The loss plots are noisy and show no clear visual separation between the prefix
and prefix-feedback arms. Their evolving proposal distributions prevent treating
these losses as measurements on a fixed evaluation distribution.

## What feedback actually enters the generator

For a completed round with J accepted draft tokens out of three proposals:

```text
acceptance = J / 3
EMA = acceptance                          on the first observed round
EMA = 0.9 * previous_EMA + 0.1 * acceptance   on subsequent rounds
feedback = [acceptance, EMA, has_previous_verification]
```

The first proposal receives `[0, 0, 0]`. Feedback is reset at each new request and
used only in the *next* round, never before verification. During training this
acceptance is computed locally from full-target greedy logits, rather than read
from vLLM serving acceptance counters. For example, accepting
2/3 then 0/3 gives successive next-round vectors `[2/3, 2/3, 1]` and `[0, 0.6, 1]`.

A learned bias-free linear map projects these three scalars into the generator's
conditioning dimension and **adds** them to its normal context embedding before
the factor decoder. The projection starts at zero. The entire generator, including
this projection, receives gradients from LK distillation. Acceptance is an input;
there is no acceptance reward, policy-gradient estimator, or differentiable
verifier decision. It does not provide token identities, the full rejection trace,
or target logits directly to the conditioning encoder.

Prefix conditioning is the concatenation of target layers 2, 18 and 33 at the
**last committed token**, plus prefix length through the existing length encoder.
It is not a pooled encoding of all response tokens. Prompt replay keeps the
original prompt summary and length. The configured effective LoRA coefficient is
fixed at alpha/r = 64/32 = 2 during these training arms; acceptance-based coefficient
control is a separate inference experiment.

Implementation: [feedback](../../../src/speculators/generator/feedback.py),
[conditioning injection](../../../src/speculators/generator/model.py).

## What replay actually does

For each selected training prompt, using the current generator parameters:

1. Extract frozen target states for the prompt. A fresh draft KV cache and fresh
   feedback state are created for this request.
2. Choose the target's greedy next token as an anchor/bonus token. Generate LoRA
   factors from the prompt or latest verified-prefix summary and optional previous
   feedback. There is no optimizer update inside a request.
3. With gradients disabled and the drafter in evaluation mode, recursively propose
   three tokens using detached LoRA factors and the current draft KV cache.
4. Run the frozen target on the prefix, anchor and all proposals. Full-vocabulary
   greedy target predictions determine the longest accepted draft prefix.
5. Re-run the **same drafted token prefixes** with gradients enabled and the
   drafter in training mode. LK matches the target distributions at all three
   proposal positions, including rejected positions and positions following a
   rejection. Targets there are conditioned on the drafted prefix, which may
   contain earlier errors. The target logits used by the loss are restricted to
   the fixed draft vocabulary; acceptance verification itself is full-vocabulary.
6. Backpropagate into the generator. Discard tentative draft KV and retain
   historical committed KV detached from the graph. Append the anchor and only the
   accepted draft prefix; obtain next-round feedback from the completed round.
   Any correction token is obtained from the target at the start of the next round.
7. Repeat up to four rounds, unless EOS or the context limit stops the episode.
   Accumulate gradients from eight requests, clip them and take **one** optimizer
   update. There is no long-lived replay buffer or repeated optimization of an old
   stored rollout dataset.

The rollout uses evaluation mode; gradient replay uses training mode, including
configured LoRA dropout (0.05 in the recipe). Forced token prefixes are identical;
this does not imply identical rollout and gradient-pass logits under dropout.
Loss gradients are normalized by eight requests and the configured maximum of four
rounds; early-ending requests therefore contribute fewer round terms. Historical
KV and discrete proposals/acceptance are detached, so this is not backpropagation
through the entire multi-round decoding process. The loss provides no explicit
prefix-survival or measured-throughput weighting.

Replay uses the documented `causal_stateful` KV policy: committed historical states
retain their past adapters; newly committed slots use the current adapter. The
current implementation uses SDPA for streaming replay after the compiler-cache fix;
ordinary validation/training keep the configured backend. Exact executed backend
revision must be confirmed from run provenance before attributing numerical effects.

Implementation: [rollout and replay](../../../src/speculators/generator/study_replay.py),
[streaming draft](../../../src/speculators/generator/streaming.py),
[optimizer accumulation and probe](../../../src/speculators/generator/study.py).

## Interpretation and limits

The evidence argues against **this replay recipe** for preserving or improving the
held-out prompt probe: replacing the second half of ordinary LK training with any
of the three replay continuations reduces endpoint agreement. Ordinary prompt LK
continues improving instead. Prefix refresh and acceptance inputs do not rescue
the probe. Since prompt-only replay also declines, the shared degradation cannot
be attributed solely to prefix refresh or the feedback encoder.

This does not establish that all replay or all verifier feedback is ineffective.
Supervision changes from long saved responses to short drafted blocks, and includes
positions after a first rejection; data distribution, supervised-token exposure,
dropout, KV history and target-compute costs differ. Forgetting, local objective
mismatch or implementation effects are hypotheses, not diagnosed causes. The
feedback-trained model is evaluated with zero feedback, so the probe does not test
whether informative online feedback improves its proposals.

There are no serving acceptance, accepted-prefix-length or throughput measurements
for these trained arms in the supplied evidence, and no repeated-seed uncertainty.
Do not label probe agreement as decoding acceptance or conclude a speed regression
from these charts.

## Decision and next check

Deprioritize the present replay training recipe and retain ordinary prompt LK as
the working reference. This is a scoped engineering decision, not a general claim
that replay is useless. A matched C1 math/chat inference comparison, if run, can
check whether there is any acceptance/throughput benefit despite the probe decline.
Raw validation and replay JSON traces should be archived before stronger causal
claims or new training experiments. No additional training or benchmarking is
launched as part of this record.

## Updates

- 2026-10-27: Archived five supplied figures, approximate endpoint comparisons,
  implementation-level replay/feedback semantics, evidence limits and the decision
  to deprioritize the current recipe.
