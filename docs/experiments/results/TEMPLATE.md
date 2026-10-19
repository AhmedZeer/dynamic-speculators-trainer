# <ID> — <experiment title>

## Record

- Observation date: YYYY-MM-DD
- Status: preliminary / confirmed / superseded
- Evidence: screenshots / exported metrics / repeated runs / serving benchmark
- Related runbook or earlier result:

## Question and hypothesis

State the variable being tested and the expected effect.

## Setup

| Arm | Changed variable | Training data | Update/token/time budget | Seeds | Checkpoint |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

Record shared model and data revisions, architecture, loss, optimizer, context
sizes, hardware, effective configuration, and code revision. State which budget
is held constant and how validation/test examples are separated from training.

## Evidence and measurements

Link exact W&B run URLs, metric exports, figures, and provenance artifacts.
State metric definitions, aggregation, evaluation membership, checkpoint step,
and uncertainty across repetitions. Keep approximate visual readings distinct
from exported numeric measurements. Mark missing evidence explicitly.

## Interpretation and limits

State what the evidence supports, plausible confounders, and claims it cannot
support. Do not infer statistical equivalence from overlapping curves alone.

## Decision and next check

Record the practical decision and the smallest follow-up that would test the
remaining uncertainty.

## Updates

- YYYY-MM-DD: Initial record.
