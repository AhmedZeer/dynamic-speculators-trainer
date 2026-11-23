"""Offline readout of next-arms runs from W&B (G1 / E1 decision tables).

Offline acceptance = teacher-forced per-depth draft agreement on the fixed
validation set (context-4 conditioning); tau_hat = 1 + a0 + a0*a1 + a0*a1*a2
is the implied expected accepted length incl. the bonus token under the
approximation that per-depth agreement equals conditional acceptance. It is not
a serving measurement. Single seed: differences are not yet uncertainty-rated.

Usage: python scripts/next_arms/readout.py [--runs R01 R02 ...] [--json out.json]
"""

import argparse
import json
import math

import wandb

ENTITY = "ahmadzeer2005-y-ld-z-technical-university"
PROJECT = "eagle3-generator-next-arms"
DEPTHS = [f"validation_scores/context_4_agreement_step_{d}" for d in range(3)]
CATS = [f"validation_scores/{c}_agreement" for c in ("math", "code", "stem", "chat")]


def final_validation(run):
    keys = ["optimizer_step", *DEPTHS, *CATS, "validation_scores/overall_agreement"]
    rows = [r for r in run.scan_history(keys=keys) if r.get(DEPTHS[0]) is not None]
    return max(rows, key=lambda r: r["optimizer_step"]) if rows else None


def probe_final(run):
    rows = [r for r in run.scan_history(keys=["optimizer_step", "probe/swap_gap/within"])]
    if not rows:
        return {}
    step = max(r["optimizer_step"] for r in rows)
    keys = [k for k in run.summary.keys() if k.startswith("probe/")]
    return {"step": step, **{k: run.summary.get(k) for k in keys}}


def lca_shares(run):
    keys = [k for k in run.summary.keys() if k.startswith("dynamics/lca/")
            and k != "dynamics/lca/total"]
    if not keys:
        return {}
    totals = dict.fromkeys(keys, 0.0)
    for row in run.scan_history(keys=keys):
        for k in keys:
            if isinstance(row.get(k), (int, float)) and math.isfinite(row[k]):
                totals[k] += row[k]
    norm = sum(abs(v) for v in totals.values()) or 1.0
    return {k.rsplit("/", 1)[1]: v / norm for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()
    api = wandb.Api()
    table = {}
    for run in api.runs(f"{ENTITY}/{PROJECT}"):
        rid = run.name.split("-", 1)[0]
        if rid.startswith("P0") or (args.runs and rid not in args.runs):
            continue
        val = final_validation(run)
        if val is None:
            continue
        a = [val[k] for k in DEPTHS]
        entry = {
            "name": run.name, "state": run.state, "step": val["optimizer_step"],
            "acc": a, "tau_hat": 1 + a[0] + a[0] * a[1] + a[0] * a[1] * a[2],
            "categories": {k.split("/")[1].split("_")[0]: val[k] for k in CATS},
            "probe": probe_final(run), "lca_share": lca_shares(run),
        }
        table[rid] = entry
    base = table.get("R01")
    print(f"{'run':6} {'step':>5} {'a0':>6} {'a1':>6} {'a2':>6} {'tau':>6} "
          f"{'dtau':>7} {'within':>8} {'cross':>8} {'null':>8} {'rho_v':>7} "
          f"{'effrank':>7}  lca_share")
    for rid in sorted(table):
        e = table[rid]
        p = e["probe"]
        dtau = e["tau_hat"] - base["tau_hat"] if base else float("nan")

        def g(k, p=p):
            v = p.get(k)
            return f"{v:8.4f}" if isinstance(v, (int, float)) else f"{'-':>8}"

        shares = ", ".join(f"{k}={v:.2f}" for k, v in sorted(
            e["lca_share"].items(), key=lambda kv: -abs(kv[1])))
        print(f"{rid:6} {e['step']:5d} {e['acc'][0]:6.4f} {e['acc'][1]:6.4f} "
              f"{e['acc'][2]:6.4f} {e['tau_hat']:6.3f} {dtau:+7.3f} "
              f"{g('probe/swap_gap/within')} {g('probe/swap_gap/cross')} "
              f"{g('probe/swap_gap/null')} "
              f"{(p.get('probe/rho/v_proj') or float('nan')):7.4f} "
              f"{(p.get('probe/condition_effective_rank') or float('nan')):7.2f}  "
              f"{shares}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(table, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
