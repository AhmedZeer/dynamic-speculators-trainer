# Generator module and architecture studies

Eleven independent launches using `speculators generator train-structure --config`.
The [runbook](../../../docs/experiments/eagle3_generator_structure_v1.md)
contains setup, all eleven commands, checkpoints, recovery, memory estimates,
and the shared historical baseline. `common.yaml` holds shared settings;
`modules/` changes the adapter targets and `architecture/` changes one dimension.

Each config starts a fresh generator, uses 1K training rows per category,
runs 1,500 optimizer updates, validates every 250 updates, and saves a resumable
Hub checkpoint every 500 updates. Run the configs separately, in any order.
