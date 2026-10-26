# Speculative decoding literature: implications for the conditional LoRA generator

## Record and scope

- Review date: 2026-10-26; repository inspected at `3d7b2c6`.
- Status: literature review, implementation audit, and proposed experiments. No new training or serving measurements.
- Discovery starting point: [mlsanity speculative decoding index](https://mlsanity.com/inference/speculative-decoding/), whose displayed coverage ends on 2026-10-21. This is a targeted review of relevant papers and their primary sources, not an exhaustive review of its entire catalog.
- Recent October 2026 preprints are preliminary evidence. Published percentage improvements refer to each paper's own models, baselines, budgets, and engines; they are not forecasts for this repository.
- Focus: whether useful input-dependent adaptation improves draft acceptance and ultimately decoding speed. Concurrency engineering remains a separate, deferred workstream.
- Related measurements: [runtime profiling](results/SYS-002-generator-runtime-profiling.md), [per-request adapters](results/BATCH-003-per-request-continuous.md), [rank study](results/GEN-005-rank-dynamism-and-serving.md), and [earlier scaling comparison](results/GEN-001-direct-data-scaling.md).

## Recommendation

Keep EAGLE3 with a conditional adapter as the near-term research platform. First establish the contribution of the conditioning signal, measure vocabulary and rejection-position limits, and improve training/deployment alignment. Evaluate smaller forms of conditional adaptation before expanding the generator or replacing the draft architecture.

The mechanism to establish is: supplying the appropriate context changes the adapter in a way that improves accepted prefixes over a matched fixed adapter. Adapter variation, parameter count, category separability, and ordinary language-model loss do not establish that mechanism individually.

## What is already implemented

| Component | Observed implementation | Consequence |
| --- | --- | --- |
| Loss | `losses/eager.py::lk_hybrid_loss` and its fused equivalent mix forward KL and TV using a detached overlap-dependent coefficient. | Acceptance-oriented local training is already present. |
| Training horizon | `models/eagle3/core.py::forward` recursively feeds draft hidden states for three TTT steps in the recipe. Ground-truth token embeddings are shifted between steps. | This includes training-time simulation; it is not fully sampled speculative rollout training. Teacher forcing alone is not evidence of a bug. |
| Depth weights | `models/eagle3/metrics.py` uses `ttt_step_loss_decay`; the recipe sets it to `1.0`. | The three depths have equal configured multipliers, rather than a learned prefix-survival weighting. |
| Generator conditioning | `generator/online_activations.py::_generate` takes the last prompt token at target layers `[2,18,33]`. `generator/model.py` pools encoded conditioning examples by mean/std/count. | Pooling more examples and pooling more tokens within a prompt are different experiments. |
| Generated parameters | Full A/B factors for V and O projections; the generator has about 293 million trainable parameters at the logged rank-32 configuration. | The number of trainable hypernetwork parameters differs substantially from a directly trained static LoRA. |
| Generation policy | The serving generator produces an adapter from the prompt, with request-lifetime adapter state. | The adapter does not adapt to the evolving response. EAGLE3 still receives current target features during ordinary speculative decoding. |
| Data | Online extraction replays saved prompt/response token IDs. `bank/workflow.py::regenerate` generates responses using the target; existing generation YAMLs configure temperature 0.7. | Online activation construction does not mean fresh draft-policy trajectories. Verify the released data's provenance before assigning it a temperature. |
| Vocabulary | Archived bundle metadata records 32,000 draft tokens and 151,936 target tokens. `model.py::_load_verifier_weights` restricts the training teacher head to the draft vocabulary. | The training softmax renormalizes the teacher on that support; its agreement/overlap can hide missing target vocabulary. |
| Configuration | `generator/campaign_config.py` currently requires `lk_hybrid`, last-token conditioning, and context-4 validation. | Loss/summary ablations need deliberate configuration and identity changes; editing a YAML value alone will fail. |
| Alternative draft models | The library has P-EAGLE, DFlash, and DSpark model implementations. | Their existence is not evidence that conditional generator bundles or this serving controller support them. |

Code paths in the table are relative to `src/speculators/`. Recipe: `examples/eagle3-lora/generator_training_recipe.yaml`.

## Literature and applicability

### Training trajectories and target alignment

[EAGLE3](https://arxiv.org/html/2503.01840) removes feature-regression constraints and trains on recursively generated draft features. [HASS](https://arxiv.org/abs/2408.15766) also addresses training/inference context and objective mismatch. Our recursive hidden-state training already incorporates part of this family of ideas. Replacing teacher-forced tokens with sampled tokens requires matching target supervision to the resulting prefixes; it is not a safe one-line substitution.

[DistillSpec](https://arxiv.org/abs/2310.08461) emphasizes both the data distribution and the choice of divergence. The newer [Draft-OPD](https://arxiv.org/html/2605.29343) collects target-assisted rollouts, records draft anchors, and replays accepted and rejected draft proposals under teacher supervision. Its objective uses forward KL for accepted positions and decayed reverse KL for rejected positions. Discarding rejected proposals and training only on the final corrected continuation loses that signal.

**Proposed use:** post-train an existing generator on a bounded collection of actual speculative blocks. Retain prompt identity, committed prefix, proposed token IDs, accepted prefix length, teacher evaluations at matching draft prefixes, and checkpoint identity. Hold out the benchmark prompts. Use actual replays rather than pretending that stored target-only activations supervise a different sampled prefix.

**Counterevidence to a universal rule:** [Training Domain Draft Models: Best Practices](https://arxiv.org/abs/2503.07807) finds offline distillation stronger than its online alternatives in its domain-adaptation setup. Architectures and protocols differ from Draft-OPD. Treat replay as a controlled hypothesis, not an assured improvement.

[Target Regeneration Shrinks the Gains of Draft-Training Objectives](https://arxiv.org/html/2610.29013) finds that objective gains over externally written responses shrink after target regeneration. Our pipeline already regenerates responses. Check generation provenance, thinking mode, chat template, temperature, and length before attributing plateaus to the objective. A small greedy-regenerated training pool is a useful comparison for the current greedy serving evaluation.

### Acceptance-aware objectives and prefix survival

[LK Losses](https://arxiv.org/abs/2602.23881) motivates our existing objective. Keep it as the reference. Small local-loss comparisons can use teacher-top-1 CE, forward KL, and the current hybrid. The existing `ce` implementation takes the argmax of the supplied teacher logits, which are vocabulary-restricted here: audit full-target top-1 coverage before interpreting it as true target-top-1 supervision.

[D-PAL](https://arxiv.org/html/2610.28355) combines detached overlap-derived position weights with distributional losses, including Renyi-1/2. Its limitations explicitly say the weights assume proposals do not condition on each other; EAGLE/DSpark require a dependence-aware adaptation. The paper's Qwen3-4B headline mean acceptance length changes from 4.50 to 4.57 at temperature zero. This is an incremental lead, not evidence of a large gain over our existing hybrid.

**Proposed use:** test a local divergence separately from position weighting. For EAGLE, first compare equal depth multipliers with a mild early-depth emphasis, then investigate a prefix-aware objective derived for recursive proposals. Do not feed hybrid loss values into `dpace_loss_decay`, which expects CE-derived token confidences.

[Expected Decoding Rounds](https://arxiv.org/html/2610.10411) models cross-round occupancy for parallel/semi-AR proposals. Its exact round-count evaluator and objective are interesting if we move to that family. Its applicability and derivation must be checked for EAGLE's recursive feature policy; it is not an immediately available campaign loss.

Under standard rejection sampling with common prefix and correctly specified distributions, one-token expected acceptance is `sum_v min(p_v,q_v)`. Greedy top-1 agreement is a different metric. Neither distributional overlap nor teacher-forced agreement should be presented as measured serving acceptance.

### Conditional information and inexpensive adaptation

[Surfing the Activation Wave](https://arxiv.org/pdf/2610.12939) selects target layers using matrix entropy and CKA, rather than fixed depth heuristics. It reports modest, model-dependent gains; its selected Qwen3-4B layers should not be copied to Qwen3-8B without measurement.

**Proposed use:** first keep the drafter's target-layer interface fixed and compare the generator's last-token summary with a small recent-prompt-token pool or learned pooling, plus per-layer normalization/gating. Use only prompt/prefix information available at the moment of adapter creation. Changing the EAGLE3 input layer set also requires compatible feature projection training, extraction, and serving metadata.

[SD2: Steering Pretrained Drafters](https://arxiv.org/html/2511.09844) computes a verifier-state steering vector and injects it as MLP bias. It refreshes the signal after verification, with the draft computation depending on prior steering through its cache. Its studied standalone drafters differ from EAGLE3.

**Proposed use:** compare generated V/O factors with a small conditional feature/activation modulation at matched training data and token exposure. This asks whether the full factor generator provides value beyond a smaller conditioning mechanism. It is an architectural experiment, not an already implemented equivalent.

Two additional original design hypotheses are worth testing after the controls below:

- Separate a shared adapter update from an input-dependent residual, and measure the marginal value of the residual.
- Generate a small coefficient vector for jointly learned factor bases, for example `B diag(c(h)) A`, instead of generating every A/B entry. This constrains the adaptation family and may reduce generator size; acceptance and actual runtime must be measured.

These are proposals inspired by the cost/conditioning trade-off, not claims that SD2 establishes their effectiveness. Useful variation is the objective; a diversity penalty can create adapter motion without improving proposals.

### Vocabulary as an adaptation ceiling

[FR-Spec](https://arxiv.org/abs/2502.14856) reduces output-head cost using a fixed vocabulary subset. [SpecVocab](https://arxiv.org/html/2602.13836) selects a context-dependent subset and also studies the impact of training with reduced vocabulary. This is especially relevant to the repository's 32K/151,936 split.

**First diagnostic:** measure, per category and response position, how often the true full-target greedy next token is in the draft vocabulary. If feasible on a bounded sample, also measure target probability mass retained by the subset. These require full-target information; the restricted teacher head cannot reveal missing mass.

With stochastic proposals supported on a set S, actual overlap is at most `p(S)`. With greedy verification, a target argmax outside S cannot be proposed correctly. A low restricted-vocabulary validation loss does not remove either limit.

Only after measuring the gap should we change vocabulary mappings, train a full-vocabulary draft head, or add a vocabulary predictor. Remapping vocabulary and resizing the head are model/serving changes; V/O LoRA fine-tuning alone cannot implement them.

### Choosing the speculation budget

[SpecDec++](https://arxiv.org/html/2405.19715) learns acceptance predictions to decide when to stop drafting. [APEX](https://arxiv.org/html/2610.07780) combines request-level mechanism selection with block-level depth decisions, including censored first-rejection survival supervision and measured execution costs.

**Proposed use:** sweep fixed draft lengths on existing checkpoints before building a controller. Begin within the three-step training horizon; label larger lengths as extrapolation. Use actual accepted-prefix distributions and measured block costs to decide whether adaptive depth is worthwhile. Fully accepted blocks are censored observations, not evidence that the next position would reject.

[EAGLE2](https://arxiv.org/abs/2406.16858) uses confidence-guided trees; [Sequoia](https://arxiv.org/abs/2402.12374) optimizes trees with hardware cost. Trees are a later proposal/verification experiment. Confirm support in the pinned vLLM integration rather than assuming the paper's EAGLE tree implementation is active in our three-token chain.

[Understanding Target Probability Reuse](https://arxiv.org/abs/2610.29210) cautions that exact reuse needs the distribution at the complete next context; its speedups depend on overlapping extra target work with standalone drafting. Avoid adding a target pass under the assumption that cached logits from another prefix are interchangeable.

### Larger forks and deferred alternatives

| Direction | Primary source and idea | Decision for this project |
| --- | --- | --- |
| Parallel drafting | [P-EAGLE](https://arxiv.org/abs/2602.01469) predicts multiple positions in parallel; [DFlash](https://arxiv.org/html/2602.06036) uses block diffusion with target-feature KV injection. | Credible larger fork. Train/evaluate the plain alternative first, then add conditional adaptation as a separate variable. |
| Step specialization | [PRISM](https://arxiv.org/html/2602.01762) uses different parameter sets at different draft steps. | Interesting if later-step survival remains weak; broader than increasing LoRA rank. Requires explicit cache/training semantics. |
| Continual feedback | [Online Speculative Decoding](https://arxiv.org/abs/2310.07177) updates drafts from observed queries; [EvoSpec](https://arxiv.org/html/2605.27390) combines vocabulary adaptation with buffered online LoRA alignment. | Keep separate from immediate offline generator experiments; deployment learning adds update cost and changing-policy evaluation. |
| Preference/reward training | [RCSD](https://arxiv.org/html/2610.29527) uses accepted/rejected signals for DPO/GRPO. | Promising recent preprint; comparisons are against HASS, not our EAGLE3 + hybrid + generator. Replay distillation is the simpler first experiment. |
| Online diffusion adaptation | [AdaFlash](https://arxiv.org/html/2607.19223) combines online mixed distillation and adaptive length. | Revisit with a diffusion draft fork; reverse-KL superiority is not universal. |
| Evolving/long-context conditioning | [DLite](https://arxiv.org/html/2610.23787) reconstructs local proposals from current target states without historical draft KV; [bounded long-context drafting](https://arxiv.org/abs/2610.20865) bounds cache and rebases positions. | Relevant if adaptation must follow response evolution or long agent histories. A separate architecture/workload question. |
| Retrieval proposals | [REST](https://arxiv.org/abs/2311.08252) retrieves draft continuations. | Useful for repetitive/code workloads, but does not test the conditional-parameter hypothesis directly. |
| Self speculation | [LayerSkip](https://arxiv.org/abs/2404.16710) trains early-exit/self-speculative models. | Requires a different target-training/checkpoint setup. |
| Overlapped speculation | [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251) speculates on verification outcomes to overlap future drafting. | Systems work beyond the current single-GPU mechanism study. |
| Relaxed verification | [Judge Decoding](https://arxiv.org/abs/2501.19309) accepts useful continuations beyond exact target alignment. | Changes the quality/equivalence question. Keep outside the present exact-verification comparison. |

A changing adapter need not represent the same fixed-weight drafter at every time step. However, refreshing V/O weights while treating historical KV as if it had been produced by the new fixed adapter is not a complete design. Specify the causal stateful proposal policy, train against it, and handle rejection/cache rollback consistently; alternatively rebuild affected state. SD2 and DLite illustrate different designs, not automatic support for arbitrary weight swaps.

## Experiment order and decision rules

### Stage A: establish the mechanism, with existing checkpoints

Run singleton conditioning and single-request serving to isolate draft quality. Preserve original EAGLE3 on original vLLM as the external baseline. Include the existing static LoRA trained on matching data/budget where available.

| Control | What it isolates |
| --- | --- |
| Correct prompt-conditioned adapter | Reference generated proposal policy. |
| Adapter generated from a fixed training prompt | Whether conditioning changes are needed beyond one learned adapter. Select the prompt before examining validation performance. |
| Conditioning swapped within the same category | Whether useful information is request-specific within domains. |
| Conditioning swapped across categories | Whether the main benefit is coarse domain selection. |
| Generator trained with constant conditioning, if needed later | Controls the hypernetwork's overparameterized optimization while removing input dependence. A static LoRA alone does not match its trainable parameter count. |

Use the same adapter execution path for mechanism comparisons. For throughput, separately report the practical fixed-LoRA baseline with its efficient merge path and vanilla EAGLE3. A fixed generated adapter control should not gain an undocumented execution-path advantage.

Measure changes in `Delta W = (alpha/r) B A`, or in resulting activations/logits. A/B factors have gauge freedom; distance between factors need not mean a different effective update. Averaging A and B separately does not produce the average update. An exact average update may exceed the original rank, so a fixed representative adapter is the simpler first control.

**Decision:** if within-category swaps do not hurt while cross-category swaps do, pursue domain-conditioned adaptation first. If both swaps preserve acceptance, investigate mostly static adaptation before adding conditional complexity. If correct per-request conditioning consistently wins, prioritize richer summaries or residual conditioning.

### Stage B: inexpensive diagnostics and budget calibration

- Measure full-target vocabulary coverage and first rejection position.
- Sweep draft length; use a separate tuning split so the reported test set does not choose the winning budget.
- Slice accepted prefixes by category, prompt length, and response position. Keep thinking mode and chat template fixed.
- Recheck source response temperature and whether a bounded greedy-regenerated pool improves the greedy regime.

### Stage C: capped training comparisons

Keep the same 24K pool, data order, optimizer, supervised token exposure, and evaluation prompts. Cap each initial new arm at 1,000 optimizer updates; also report actual tokens, examples, and GPU time because step counts alone do not equalize changed rollout workloads.

1. Compare the current summary with one recent-token/learned summary, holding loss, rank, and training horizon fixed.
2. Compare a bounded replay post-training stage with continued ordinary training from the same starting checkpoint. Hold the supervised-token budget fixed and record teacher-query cost separately.
3. Compare the current hybrid with one simple local-loss alternative on the same trajectories. Avoid simultaneously changing the data source and the objective.
4. If the conditional signal is useful, compare a static-plus-dynamic residual or compact steering alternative. Tune the adapter target modules separately; the current V/O-only restriction is an implementation choice, not a literature-established optimum.

Use paired prompt comparisons, repeated serving measurements, and additional training seeds for finalists. Expand to a small OOD confirmation only after selecting on held-out ID data; retain the current classification that Nemotron is ID. Report uncertainty before calling small differences equivalent or superior.

## Metrics that answer the hypothesis

For a maximum proposal length gamma and accepted draft prefix J, measure the full prefix survival curve `S_k = P(J >= k)`. Then `E[J] = sum_{k=1}^gamma S_k`, without assuming independent token acceptance. Define whether reported progress includes the target correction/bonus token.

An approximate decode progress rate is `(1 + E[J]) / (draft_seconds + verifier_seconds + adapter_seconds_amortized)`, with endpoint/EOS exceptions and any non-overlapped serving overhead accounted for. This is a diagnostic model, not a replacement for measured request throughput.

Record:

- Mean accepted draft tokens per verification and per-position survival/rejection.
- Accepted/proposed token ratio, with its exact denominator and proposal length.
- Aggregate output tokens/s and decode-only rate where timestamps support it.
- TTFT, TPOT, request latency, and output lengths/EOS behavior.
- One-time generator/adapter preparation costs and repeated drafting/verification costs.
- Vocabulary coverage/mass, plus prompt/response-length slices.
- Paired per-prompt acceptance differences under correct and swapped conditioning.
- Model/data revisions, token budget, seeds, hardware, full commands, patches, and checkpoint/source hashes.

The controlled study [When Does Better Acceptance Pay?](https://arxiv.org/html/2610.24993) reinforces matching training, inference, and evaluation conditions and separating acceptance returns from execution costs. Our [SYS-002 measurements](results/SYS-002-generator-runtime-profiling.md) already show why both matter, particularly for chat.

Existing throughput results with unchecked correctness remain exploratory. Keep the known target-output equality issue visible; draft acceptance and token rate are not answer-accuracy measurements. Changing the verifier would require a separate quality/equivalence protocol.

## Open questions

- How much of the measured gain survives fixed/scrambled conditioning controls?
- How much target mass and greedy next-token coverage is missing from the current draft vocabulary?
- Are weak later-step results caused by the local objective, training trajectories, conditioning, or limited adapter targets?
- Does a compact steering/residual model retain the generated-factor model's acceptance gain?
- Is prompt-level adaptation sufficient, or would a properly trained evolving-prefix policy add useful acceptance?
- Which improvements persist under repeated, matched-budget ID and OOD evaluation?

No implementation or adoption decision beyond this prioritized investigation is implied by the literature review.
