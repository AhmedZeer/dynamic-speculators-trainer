"""Launch pinned scaling/conditioning suites and collect figure-ready measurements."""

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
GENERATOR = ("azeeer/eagle3-generator-attention-100k-2k-v1",
             "3edcce5e4d2fbfc56dbfa71ae0cd96fb8dcc1cba")
STATIC = ("azeeer/eagle3-static-lora-attention-100k-10k-v1",
          "8b5ef5792fcd5e3ca7e49a4959edc0b347b496e6")
CATEGORIES = ("math", "stem", "chat")
PHASES = {
    "scaling": ("correct", "static"),
    "dynamism": ("correct", "fixed", "cross_category"),
    "all": ("correct", "static", "fixed", "cross_category"),
}


@contextmanager
def launch_lock(root, dry_run):
    if dry_run:
        yield
        return
    from filelock import FileLock  # noqa: PLC0415

    with FileLock(str(root / ".launch.lock"), timeout=0):
        yield


@contextmanager
def termination_handler():
    previous = signal.getsignal(signal.SIGTERM)

    def stop(*_):
        raise KeyboardInterrupt("Benchmark launch terminated")

    signal.signal(signal.SIGTERM, stop)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def frozen(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != value:
        raise ValueError(f"Configuration changed: {path}; choose a new output root")
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def suite_config(root, policy, ttt, gpu, port):
    static = policy == "static"
    repo, revision = STATIC if static else GENERATOR
    models = []
    for step in range(1, 6 if static else 7):
        model = {
            "name": f"{'static_lora' if static else 'generator'}_{step}k_{policy}",
            "type": "lora" if static else "generator",
            "model": repo, "revision": revision,
            "subfolder": f"step-{step * 1000:08d}",
            "adapter_policy": "shared",
        }
        if not static:
            model["conditioning_examples"] = 1
            model["adapter_experiment"] = {
                "conditioning": policy, "scale": 2.0, "record_rounds": False,
            }
            if policy != "correct":
                model["adapter_experiment"]["donor_bank"] = str(
                    root / "study/inputs/conditioning.safetensors"
                )
        models.append(model)
    return {
        "models": models,
        "benchmarks": [{
            "dataset": "nemotron", "categories": list(CATEGORIES),
            "validation_manifest": str(root / "study/inputs/test/manifest.json"),
        }],
        "settings": {
            "batching_policy": "fixed_shared_adapter", "concurrency": 1,
            "max_samples": 50, "max_tokens": 256, "warmups": 3, "seed": 42,
            "speculative_tokens": ttt, "gpu": gpu, "port": port,
            "continue_on_error": False,
        },
        "wandb": {
            "enabled": True, "project": "eagle3-attention-scaling-dynamism-v1",
            "group": f"{policy}-ttt{ttt}", "mode": "online",
        },
    }


def run(command, env):
    print("Launching: " + " ".join(str(x) for x in command), flush=True)
    process = subprocess.Popen(command, cwd=REPO, env=env, start_new_session=True)  # noqa: S603
    try:
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise


def prepare_inputs(root, gpu, port, env):
    """Use the existing managed extractor and training-only donor preparation."""
    from speculators.benchmark.processes import (  # noqa: PLC0415
        ensure_port_free, owned_process, termination_cleanup,
    )

    data_root = root / "data-preparation"
    study_path = root / "configs/prepare.yaml"
    frozen(study_path, {
        "workspace": str(root / "study"),
        "campaign_config": str(data_root / "launcher/train.yaml"),
        "training_subset": 1, "validation_subset": 0,
        "train_per_category": 1000, "tune_per_category": 50,
        "test_per_category": 50, "max_output_tokens": 256, "seed": 42,
    })
    manifest = root / "study/inputs/test/manifest.json"
    donors = root / "study/inputs/conditioning.safetensors"
    if manifest.is_file() and donors.is_file() and donors.with_suffix(".json").is_file():
        metadata = json.loads(donors.with_suffix(".json").read_text())
        if hashlib.sha256(donors.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Donor checksum changed")
        print("Reusing frozen workload and conditioning donor bank.", flush=True)
        return
    run([
        sys.executable, str(REPO / "scripts/train_attention_100k.py"),
        "--variant", "10k-lora", "--prepare-only", "--output-root", str(data_root),
        "--port", str(port),
    ], env)
    attempts = sorted((data_root / "launcher").glob("vllm-attempt-*"))
    attempt = attempts[-1]
    command = json.loads((attempt / "launcher_commands.json").read_text())["server"]
    ensure_port_free(port)
    with termination_cleanup(), owned_process(
        command, attempt / "vllm.log", cwd=REPO,
        env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)}, timeout=30,
    ) as process:
        deadline = time.monotonic() + 1200
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Extractor exited; see {attempt / 'vllm.log'}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except (urllib.error.URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Extractor startup timed out") from None
                time.sleep(1)
        run([
            sys.executable, "-m", "speculators", "generator", "prepare-experiments",
            "--config", str(study_path),
        ], env)


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join("---" for _ in keys) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(
            f"{row[k]:.4f}" if isinstance(row[k], float) else str(row[k]) for k in keys
        ) + " |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n")


def collect(root, ttt, phase):
    summaries = {}
    for policy in PHASES[phase]:
        path = root / f"suites/ttt{ttt}/{policy}/summary.json"
        summaries[policy] = json.loads(path.read_text())

    def metrics(policy, step, category):
        name = f"{'static_lora' if policy == 'static' else 'generator'}_{step}k_{policy}"
        return summaries[policy]["results"][name][f"nemotron/nemo-{category}"]

    # Every subtraction/comparison must use identical workload content.
    for category in CATEGORIES:
        hashes = {
            metrics(policy, step, category)["dataset_sha256"]
            for policy in PHASES[phase]
            for step in range(1, 6 if policy == "static" else 7)
        }
        if len(hashes) != 1:
            raise ValueError(f"Workload hashes differ for {category}")
    if phase in ("scaling", "all"):
        rows = []
        for step in range(1, 6):
            for policy in ("correct", "static"):
                for category in CATEGORIES:
                    group = metrics(policy, step, category)
                    rows.append({
                        "steps": step * 1000, "model": "generator" if policy == "correct" else "static_lora",
                        "category": category, "serving_ttt": ttt,
                        "tokens_per_second": group["performance"][0]["output_tokens_per_second"],
                        "acceptance_percent": 100 * group["acceptance"]["draft_acceptance_rate"],
                    })
        write_table(root / f"tables/scaling_ttt{ttt}", rows)
    if phase in ("dynamism", "all"):
        rows = []
        for step in range(1, 7):
            for category in CATEGORIES:
                rates = {
                    p: 100 * metrics(p, step, category)["acceptance"]["draft_acceptance_rate"]
                    for p in ("correct", "fixed", "cross_category")
                }
                rows.append({
                    "steps": step * 1000, "category": category, "serving_ttt": ttt,
                    "correct_acceptance_percent": rates["correct"],
                    "fixed_acceptance_percent": rates["fixed"],
                    "cross_category_acceptance_percent": rates["cross_category"],
                    "correct_minus_fixed_pp": rates["correct"] - rates["fixed"],
                    "cross_category_minus_fixed_pp": rates["cross_category"] - rates["fixed"],
                })
        write_table(root / f"tables/dynamism_ttt{ttt}", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output/attention-scaling-dynamism-v1"))
    parser.add_argument("--phase", choices=tuple(PHASES), default="all")
    parser.add_argument("--ttt", nargs="+", type=int, choices=(3, 7), default=[3])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with launch_lock(root, args.dry_run), termination_handler():
        for ttt in dict.fromkeys(args.ttt):
            for policy in PHASES[args.phase]:
                frozen(root / f"configs/{policy}_ttt{ttt}.yaml",
                       suite_config(root, policy, ttt, args.gpu, args.port))
        if args.dry_run:
            print(json.dumps({"output_root": str(root), "phase": args.phase,
                              "serving_ttt": args.ttt, "measurements_per_ttt": sum(
                                  15 if p == "static" else 18 for p in PHASES[args.phase]
                              )}, indent=2))
            return 0
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu)}
        if not args.collect_only:
            prepare_inputs(root, args.gpu, args.port, env)
        for ttt in dict.fromkeys(args.ttt):
            if not args.collect_only:
                for policy in PHASES[args.phase]:
                    run([
                        sys.executable, "-m", "speculators", "benchmark", "suite",
                        "--config", str(root / f"configs/{policy}_ttt{ttt}.yaml"),
                        "--output-root", str(root / f"suites/ttt{ttt}/{policy}"), "--resume",
                    ], env)
            collect(root, ttt, args.phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
