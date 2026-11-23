"""Export next-arms trajectories from W&B into one JSON for the dynamics report.

Per run: final validation, validation per step (per-depth agreement/loss,
categories, tau_hat), probe snapshots, and dynamics samples (LCA share,
update ratio, gradient share per component). Duplicate steps from resumed
runs keep the last row. Usage: export_dynamics.py OUT.json
"""

import json
import sys
from datetime import UTC, datetime

import wandb

ENTITY = "ahmadzeer2005-y-ld-z-technical-university"
PROJECT = "eagle3-generator-next-arms"
RUNS = {
    "R01": ("static", "Static LoRA", "new 4K"),
    "R02": ("current", "Current generator", "new 4K"),
    "R03": ("a1", "A1 · core r×r", "new 4K"),
    "R04": ("a2", "A2 · rank gates", "new 4K"),
    "R05": ("a4", "A4 · static + gated core", "new 4K, from R01@750"),
    "R05n": ("a4null", "A4 · null condition", "new 4K, from R01@750"),
    "R06": ("a5", "A5 · perceiver + core", "new 4K"),
    "R07": ("lr4", "Encoder LR ×4", "new 4K"),
    "R08": ("lr16", "Encoder LR ×16", "new 4K"),
    "R09": ("drop", "Condition dropout 0.15", "new 4K"),
    "R10": ("lr4drop", "LR ×4 + dropout", "new 4K"),
    "R11": ("static", "Static LoRA", "Nemotron 24K"),
    "R13": ("static", "Static LoRA", "new 24K"),
    "R14": ("current", "Current generator", "new 24K"),
}
VAL = ([f"validation_scores/context_4_agreement_step_{d}" for d in range(3)]
       + [f"validation_losses/context_4_loss_step_{d}" for d in range(3)]
       + [f"validation_scores/{c}_agreement" for c in ("math", "code", "stem", "chat")])


def numeric(row, key):
    value = row.get(key)
    return value if isinstance(value, (int, float)) else None


def rows(run, keys, probe_key):
    out = {}
    for row in run.scan_history(keys=["optimizer_step", *keys]):
        if numeric(row, probe_key) is None:
            continue
        out[int(row["optimizer_step"])] = {
            k: numeric(row, k) for k in keys if numeric(row, k) is not None
        }
    return [{"step": s, **v} for s, v in sorted(out.items())]


def main():
    api = wandb.Api()
    data = {"exported": datetime.now(UTC).isoformat(timespec="minutes"),
            "project": f"{ENTITY}/{PROJECT}", "runs": {}}
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        rid = run.name.split("-", 1)[0]
        if rid not in RUNS:
            continue
        arm, label, data_label = RUNS[rid]
        validation = rows(run, VAL, VAL[0])
        for v in validation:
            a = [v.get(VAL[d]) for d in range(3)]
            if None not in a:
                v["tau_hat"] = 1 + a[0] + a[0] * a[1] + a[0] * a[1] * a[2]
        probe_keys = [k for k in run.summary.keys() if k.startswith("probe/")]
        dyn_keys = [k for k in run.summary.keys() if k.startswith(
            ("dynamics/lca_share/", "dynamics/update_ratio/", "dynamics/grad_share/"))]
        data["runs"][rid] = {
            "arm": arm, "label": label, "data": data_label, "state": run.state,
            "step": run.summary.get("optimizer_step"), "url": run.url,
            "validation": validation,
            "probe": rows(run, probe_keys, "probe/swap_gap/within") if probe_keys else [],
            "dynamics": rows(run, dyn_keys, dyn_keys[0]) if dyn_keys else [],
        }
        print(rid, run.state, len(validation), "val rows", flush=True)
    with open(sys.argv[1], "w") as handle:
        json.dump(data, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
