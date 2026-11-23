"""Launch one next-arms run (scripts/next_arms/runs.yaml) on a single GPU.

Owns its vLLM hidden-state extractor (online activations, nothing written to
disk beyond a transient shared dir), pins the trajectory release, writes the
effective training config, restores the latest Hub checkpoint of this run when
the local output directory is fresh (new Colab VM), and trains with
``speculators generator train-full-data``. Rerunning resumes.

Paths are deterministic (no attempt suffixes) so a run can move between VMs.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from train_attention_100k import (  # noqa: E402
    process_environments,
    stop_owned,
    wait_ready,
    write_json,
)

TEMPLATE = REPO / "examples/eagle3-lora/attention_100k_2k_v1/train.yaml"
RECIPE = REPO / "examples/eagle3-lora/attention_100k_2k_v1/training_recipe.yaml"
DOMAINS = ("math", "code", "stem", "chat")
RUNS = Path(__file__).with_name("runs.yaml")


def load_spec(run_id):
    table = yaml.safe_load(RUNS.read_text())
    if run_id not in table["runs"]:
        raise SystemExit(f"Unknown run {run_id}; see {RUNS}")
    spec = {**table["defaults"], **table["runs"][run_id]}
    spec["data_spec"] = table["data"][spec["data"]]
    spec["run_id"] = run_id
    spec["hub_repo"] = f"{spec['hub_prefix']}-{run_id.lower()}"
    if spec["subsets"] > spec["data_spec"]["available_subsets"]:
        raise SystemExit("Run asks for more subsets than the release has")
    return spec


def pinned_revision(data, state):
    if "revision" in data:
        return data["revision"]
    if state.get("trajectories_revision"):
        return state["trajectories_revision"]
    return HfApi().repo_info(data["repo"], repo_type="dataset").sha


def release_records(data, revision, subsets):
    records = {}
    for domain in DOMAINS:
        path = f"releases/{data['release']}/{domain}/data.json"
        try:
            file = hf_hub_download(data["repo"], path, repo_type="dataset",
                                   revision=revision)
        except EntryNotFoundError as exc:
            raise ValueError(f"Missing {path}; finish generation and data stage") from exc
        record = json.loads(Path(file).read_text())
        ids = {s["id"] for s in record["manifest"]["subsets"]}
        need = {f"{domain}-{i:05d}" for i in range(subsets)}
        if not need <= ids:
            raise ValueError(f"{domain}: release lacks {sorted(need - ids)}")
        records[domain] = record
    return records


def build_configs(spec, root, revision, records, port, static_init, *,
                  download=True):
    data = spec["data_spec"]
    template = yaml.safe_load(TEMPLATE.read_text())
    template.pop("validation_examples_per_category", None)
    recipe = yaml.safe_load(RECIPE.read_text())
    pins = next(iter(records.values()))["manifest"]["revisions"]
    recipe["hub"].update(
        trajectories_repo=data["repo"], trajectories_revision=revision,
        release=data["release"], selection=data["selection"],
    )
    recipe["partition"]["subset_ids"] = list(range(data["available_subsets"]))
    model_paths = {}
    for role in ("target", "drafter"):
        model_paths[role] = (
            snapshot_download(recipe["models"][role], revision=pins[f"{role}_sha"])
            if download else f"/unresolved/{role}"
        )
        recipe["models"][f"{role}_revision"] = pins[f"{role}_sha"]
    launch = root / "launcher"
    recipe_path = launch / "training_recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    updates = spec["updates"]
    template.update(
        training_recipe=str(recipe_path),
        output_root=str(root),
        input_root=str(root / "work/inputs"),
        trajectories_path=str(root / "work/trajectories"),
        staging_dir=f"/tmp/nextarms-{spec['run_id']}",  # noqa: S108
        experiment_subsets=[f"{d}-{i:05d}" for d in DOMAINS
                            for i in range(spec["subsets"])],
        max_training_updates=updates,
        training_subset_limit=spec["subsets"],
        hub_upload_interval=spec.get("hub_upload_interval", 500),
    )
    if spec["arm"] == "generator":
        template["dynamics_interval"] = spec["dynamics_interval"]
    template["optimization"].update(
        pretraining_updates=updates,
        checkpoint_interval=min(250, updates),
        warmup_updates=min(100, max(1, updates // 10)),
        **spec.get("optimization", {}),
    )
    template["architecture"].update(spec.get("architecture", {}))
    template["wandb"].update(
        pretraining_project=spec["wandb_project"],
        group=spec["wave"],
        name_prefix=f"{spec['run_id']}-",
    )
    template["online"].update(
        target_model_path=model_paths["target"],
        drafter_model_path=model_paths["drafter"],
        endpoint=f"http://127.0.0.1:{port}/v1",
        shared_dir=f"/tmp/nextarms-online-{spec['run_id']}",  # noqa: S108
        summary_cache_dir=str(root / "work/summary-cache"),
        compiler_cache_dir=str(root / "work/compiler-cache"),
        ram_cache_gib=spec.get("ram_cache_gib", 100),
    )
    if static_init is not None:
        template["static_init"] = static_init
    config = launch / "train.yaml"
    config.write_text(yaml.safe_dump(template, sort_keys=False))
    return template, config, model_paths


def download_trajectories(spec, revision, records, root):
    data = spec["data_spec"]
    patterns = [f"releases/{data['release']}/{d}/data.json" for d in DOMAINS]
    patterns += [ref["path"] for r in records.values() for ref in r["artifacts"]]
    snapshot_download(data["repo"], repo_type="dataset", revision=revision,
                      local_dir=root / "work/trajectories", allow_patterns=patterns)


def resolve_static_init(spec, root):
    source = spec.get("static_init")
    if source is None:
        return None
    parent = load_spec(source["run"])
    folder = f"step-{source['step']:08d}"
    path = Path(hf_hub_download(parent["hub_repo"], f"{folder}/checkpoint.pt"))
    local = root / "launcher" / "static_init.pt"
    if not local.exists():
        shutil.copy2(path, local)
    return {"path": str(local), "sha256": sha256(local),
            "source": f"{parent['hub_repo']}/{folder}"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore_from_hub(spec, root, label):
    """On a fresh VM, install this run's newest Hub snapshot for resumption."""
    output = root / "pretraining" / label
    if (output / "checkpoint.pt").exists():
        return None
    api = HfApi()
    try:
        files = api.list_repo_files(spec["hub_repo"])
    except Exception:  # noqa: BLE001 -- repo absent: fresh run
        return None
    steps = sorted({f.split("/", 1)[0] for f in files if f.startswith("step-")
                    and f.endswith("/checkpoint.pt")})
    if not steps:
        return None
    folder = steps[-1]
    revision = api.repo_info(spec["hub_repo"]).sha
    archive = output / "provenance" / "hf-continuation-source"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("resolved_experiment.json", "result.json", "configuration.json"):
        target = archive / name
        source = hf_hub_download(spec["hub_repo"], f"{folder}/{name}",
                                 revision=revision)
        shutil.copy2(source, target)
    result = json.loads((archive / "result.json").read_text())
    checkpoint = Path(hf_hub_download(spec["hub_repo"], f"{folder}/checkpoint.pt",
                                      revision=revision))
    if sha256(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("Hub checkpoint checksum differs from its result.json")
    shutil.copy2(archive / "resolved_experiment.json",
                 output / "resolved_experiment.json")
    shutil.copy2(archive / "configuration.json", output / "configuration.json")
    write_json(output / "hf_continuation.json", {
        "repo_id": spec["hub_repo"], "revision": revision, "subfolder": folder,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "identity_sha256": sha256(archive / "resolved_experiment.json"),
        "completed_updates": result["completed_updates"],
    })
    write_json(output / "hub_checkpoints.json", {folder: {
        "checkpoint_sha256": result["checkpoint_sha256"],
        "completed_updates": result["completed_updates"], "commit_sha": revision,
    }})
    pending = output / "checkpoint.pt.pending"
    shutil.copy2(checkpoint, pending)
    pending.replace(output / "checkpoint.pt")
    print(f"Restored {spec['hub_repo']}/{folder} "
          f"(update {result['completed_updates']})", flush=True)
    return folder


def run(args):
    spec = load_spec(args.run)
    if args.max_updates:
        spec["updates"] = args.max_updates
    root = (args.output_root / spec["run_id"]).resolve()
    launch = root / "launcher"
    launch.mkdir(parents=True, exist_ok=True)
    state_path = launch / "run.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    revision = pinned_revision(spec["data_spec"], state)
    state.update(run=spec["run_id"], trajectories_revision=revision,
                 hub_repo=spec["hub_repo"], spec=spec)
    write_json(state_path, state)
    records = release_records(spec["data_spec"], revision, spec["subsets"])
    download_trajectories(spec, revision, records, root)
    static_init = None if args.prepare_only else resolve_static_init(spec, root)
    cfg, config, _ = build_configs(spec, root, revision, records, args.port,
                                   static_init, download=not args.prepare_only)
    provenance = launch / "vllm"
    provenance.mkdir(exist_ok=True)
    for source in (config, launch / "training_recipe.yaml", state_path):
        shutil.copy2(source, provenance / source.name)

    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415
    from speculators.generator.epoch_training import train_full_data  # noqa: PLC0415

    arm = spec["arm"]
    plan = train_full_data(CampaignConfig.load(config), root, spec["hub_repo"],
                           provenance, epochs=args.epochs,
                           max_updates=spec["updates"],
                           hub_upload_interval=cfg["hub_upload_interval"], arm=arm,
                           private=True, dry_run=True)
    expected = 4 * spec["subsets"] * 1000
    if plan["job"]["training_examples"] != expected or (
        plan["job"]["updates"] != spec["updates"]
    ):
        raise ValueError(f"Plan mismatch: {plan['job']['training_examples']} rows, "
                         f"{plan['job']['updates']} updates")
    write_json(provenance / "training_plan.json", plan)
    if args.prepare_only:
        print(json.dumps({"run": spec["run_id"], "label": plan["job"]["label"],
                          "rows": expected, "updates": spec["updates"]}), flush=True)
        return 0
    restore_from_hub(spec, root, plan["job"]["label"])
    server_env, trainer_env = process_environments()
    server = [
        sys.executable, str(REPO / "scripts/launch_vllm.py"), "train",
        cfg["online"]["target_model_path"], "--target-layer-ids", "2", "18", "33",
        "--include-last-layer", "--hidden-states-backend", "file",
        "--hidden-states-path", cfg["online"]["shared_dir"],
        "--provenance-dir", str(provenance), "--",
        "--served-model-name", "Qwen/Qwen3-8B", "--host", "127.0.0.1",
        "--port", str(args.port), "--dtype", "bfloat16", "--max-model-len", "8193",
        "--max-num-batched-tokens", "8193", "--max-num-seqs", "128",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager", "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
    ]
    train = [
        sys.executable, "-m", "speculators", "generator", "train-full-data",
        "--arm", arm, "--config", str(config), "--output-root", str(root),
        "--epochs", str(args.epochs), "--max-updates", str(spec["updates"]),
        "--hub-upload-interval", str(cfg["hub_upload_interval"]),
        "--hub-repo", spec["hub_repo"], "--vllm-provenance-dir", str(provenance),
        "--private",
    ]
    write_json(provenance / "launcher_commands.json",
               {"server": server, "train": train})
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", args.port)) == 0:
            raise ValueError(f"Port {args.port} is occupied")
    server_process = trainer = None
    try:
        with (launch / "vllm.log").open("a") as log:
            server_process = subprocess.Popen(  # noqa: S603
                server, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=server_env)
            wait_ready(server_process, args.port, args.startup_timeout)
            print(f"vLLM ready; training {spec['run_id']} for "
                  f"{spec['updates']} updates", flush=True)
            trainer = subprocess.Popen(train, cwd=REPO, start_new_session=True,  # noqa: S603
                                       env=trainer_env)
            while trainer.poll() is None:
                if server_process.poll() is not None:
                    raise RuntimeError("vLLM exited during training; see vllm.log")
                time.sleep(2)
            return trainer.returncode
    finally:
        stop_owned(trainer)
        stop_owned(server_process)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--output-root", type=Path, default=Path("/content/runs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Stopped; rerun to resume.", flush=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
