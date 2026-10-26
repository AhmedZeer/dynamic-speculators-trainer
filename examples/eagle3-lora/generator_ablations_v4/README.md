# Generator experiments v4

Every leaf YAML is an independently runnable training or inference experiment.
`extends` shares settings only. There is no multi-run ablation command.

See [the experiment runbook](../../../docs/experiments/eagle3_generator_ablations_v4.md)
for setup, individual commands, the dependency order, W&B projects and limitations.
Export `SPECULATORS_STUDY_ROOT` before loading a config.
