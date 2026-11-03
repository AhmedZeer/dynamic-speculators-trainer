"""Pin/download the 100K release, own its extractor, and train a generator."""

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import suppress
from http import HTTPStatus
from pathlib import Path

import yaml
from filelock import FileLock
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples/eagle3-lora/attention_100k_2k_v1"
DOMAINS = ("math", "code", "stem", "chat")
TRAIN_ROWS = 100000
VALIDATION_ROWS = 1000
UPDATES = 2000
PILOT_TRAIN_ROWS = 1000
PILOT_UPDATES = 10
SUBSETS_PER_DOMAIN = 25
TRAIN_PER_SUBSET = 1000
VALIDATION_PER_SUBSET = 100
MAX_PORT = 65535


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def release_records(repo, revision):
    """Fail before allocating GPUs if generation/finalization is incomplete."""
    records = {}
    for domain in DOMAINS:
        path = f"releases/r0002-100k/{domain}/data.json"
        try:
            file = hf_hub_download(repo, path, repo_type="dataset", revision=revision)
        except EntryNotFoundError as exc:
            raise ValueError(
                f"Missing {path}. Finish generation, then run nemotron_100k_v1/"
                "run.sh finalize and run.sh data before training."
            ) from exc
        record = json.loads(Path(file).read_text())
        subsets = record["manifest"]["subsets"]
        expected = {f"{domain}-{i:05d}" for i in range(SUBSETS_PER_DOMAIN)}
        if len(subsets) != SUBSETS_PER_DOMAIN or {s["id"] for s in subsets} != expected:
            raise ValueError(f"Expected 25 complete subsets in {domain}")
        if any(
            len(s["train_indices"]) != TRAIN_PER_SUBSET
            or len(s["validation_indices"]) != VALIDATION_PER_SUBSET
            for s in subsets
        ):
            raise ValueError(f"Expected 25K train + 2.5K validation in {domain}")
        records[domain] = record
    revisions = [r["manifest"]["revisions"] for r in records.values()]
    if any(value != revisions[0] for value in revisions[1:]):
        raise ValueError("Categories use different model/source revisions")
    return records


def limits(args):
    """Return training membership, optimizer steps, and upload interval."""
    return (
        (PILOT_TRAIN_ROWS, PILOT_UPDATES, PILOT_UPDATES)
        if getattr(args, "pilot", False)
        else (TRAIN_ROWS, 10000 if getattr(args, "variant", "2k") == "10k" else UPDATES, 500)
    )


def configure_pilot(template, pilot):
    """Use a small data view and exercise validation/checkpoint/upload at step ten."""
    if pilot:
        template["training_examples_per_category"] = PILOT_TRAIN_ROWS // len(DOMAINS)
        template["max_training_updates"] = PILOT_UPDATES
        template["hub_upload_interval"] = PILOT_UPDATES
        template["optimization"].update(
            pretraining_updates=PILOT_UPDATES,
            warmup_updates=1,
            pretraining_validation_interval=PILOT_UPDATES,
            checkpoint_interval=PILOT_UPDATES,
        )
        template["wandb"].update(
            group="attention-100k-2k-v1-pilot", name_prefix="attention-100k-pilot-"
        )
        template["input_root"] = str(
            Path(template["input_root"]).parent / "pilot/inputs"
        )


def resource_change_matches(state, training_template, recipe, launch):
    """Allow only cache/queue sizing to change in an already-prepared run."""
    path = launch / "train.yaml"
    if not path.is_file():
        return False
    saved = yaml.safe_load(path.read_text())
    candidate = {**training_template, "online": dict(training_template["online"])}
    for key in ("ram_cache_gib", "concurrency", "buffered_examples"):
        if key not in saved.get("online", {}):
            return False
        candidate["online"][key] = saved["online"][key]
    signature = hashlib.sha256(
        json.dumps({"training": candidate, "recipe": recipe}, sort_keys=True).encode()
    ).hexdigest()
    return signature == state.get("training_signature")


def prepare(args):
    """Resolve moving refs once, then reuse the same revision on every resume."""
    variant = getattr(args, "variant", "2k")
    train_path = EXAMPLE / ("train_10k.yaml" if variant == "10k" else "train.yaml")
    template = yaml.safe_load(train_path.read_text())
    recipe = yaml.safe_load((EXAMPLE / "training_recipe.yaml").read_text())
    pilot = getattr(args, "pilot", False)
    mode = ("pilot" if pilot else "full") + ("-10k" if variant == "10k" else "")
    configure_pilot(template, pilot)
    if variant == "10k" and pilot:
        template["wandb"].update(
            group="attention-100k-10k-v1-pilot", name_prefix="attention-100k-10k-pilot-"
        )
    root = args.output_root.resolve()
    launch = root / "launcher"
    launch.mkdir(parents=True, exist_ok=True)
    state_path = launch / "run.json"
    recipe_hash = hashlib.sha256(
        train_path.read_bytes()
        + (EXAMPLE / "training_recipe.yaml").read_bytes()
    ).hexdigest()
    training_template = dict(template)
    validation_quota = training_template.pop("validation_examples_per_category", None)
    training_signature = hashlib.sha256(
        json.dumps(
            {"training": training_template, "recipe": recipe}, sort_keys=True
        ).encode()
    ).hexdigest()
    # Accept the old launcher's pre-validation-cap template for a prepared run.
    legacy_bytes = (
        train_path
        .read_bytes()
        .split(b"\n# Runtime validation only;", 1)[0]
    )
    legacy_hash = hashlib.sha256(
        legacy_bytes + (EXAMPLE / "training_recipe.yaml").read_bytes()
    ).hexdigest()
    api = HfApi()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("mode", "full") != mode:
            raise ValueError(
                "Pilot and full training require separate --output-root directories"
            )
        signature_matches = state.get(
            "training_signature"
        ) == training_signature or resource_change_matches(
            state, training_template, recipe, launch
        )
        legacy_matches = state["template_sha256"] in (recipe_hash, legacy_hash)
        if (not signature_matches and not legacy_matches) or state[
            "hub_repo"
        ] != args.hub_repo:
            raise ValueError("Run configuration changed; use a new --output-root")
        if state.get("validation_examples_per_category") != validation_quota and any(
            root.glob("pretraining/**/checkpoint.pt")
        ):
            raise ValueError(
                "Validation membership changed after training started; "
                "use a new --output-root"
            )
        if args.trajectories_revision not in ("main", state["trajectories_revision"]):
            raise ValueError("Resume must retain the pinned trajectory revision")
        revision = state["trajectories_revision"]
    else:
        revision = api.repo_info(
            recipe["hub"]["trajectories_repo"],
            repo_type="dataset",
            revision=args.trajectories_revision,
        ).sha
        state = {
            "trajectories_revision": revision,
            "template_sha256": recipe_hash,
            "hub_repo": args.hub_repo,
        }
    state.update(
        template_sha256=recipe_hash,
        mode=mode,
        training_signature=training_signature,
        validation_examples_per_category=validation_quota,
    )
    records = release_records(recipe["hub"]["trajectories_repo"], revision)
    pins = next(iter(records.values()))["manifest"]["revisions"]
    for record in records.values():
        models = record["manifest"]["cache_spec"]["models"]
        if any(
            models[role] != recipe["models"][role] for role in ("target", "drafter")
        ):
            raise ValueError("Release model IDs differ from the training recipe")
    # Persist the pin before potentially lengthy/retriable file downloads.
    write_json(state_path, state)
    work = Path(template["input_root"]).resolve().parent
    trajectories = Path(template["trajectories_path"]).resolve()
    patterns = [f"releases/r0002-100k/{d}/data.json" for d in DOMAINS]
    patterns += [ref["path"] for r in records.values() for ref in r["artifacts"]]
    print(f"Restoring 100K trajectories at {revision}", flush=True)
    snapshot_download(
        recipe["hub"]["trajectories_repo"],
        repo_type="dataset",
        revision=revision,
        local_dir=trajectories,
        allow_patterns=patterns,
    )
    model_paths = {}
    for role in ("target", "drafter"):
        print(f"Restoring pinned {role}: {recipe['models'][role]}", flush=True)
        model_paths[role] = snapshot_download(
            recipe["models"][role], revision=pins[f"{role}_sha"]
        )
        recipe["models"][f"{role}_revision"] = pins[f"{role}_sha"]
    recipe["hub"]["trajectories_revision"] = revision
    recipe_path = launch / "training_recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False))
    template.update(
        training_recipe=str(recipe_path),
        output_root=str(root),
        trajectories_path=str(trajectories),
        input_root=str(work / "inputs"),
    )
    template["online"].update(
        target_model_path=model_paths["target"],
        drafter_model_path=model_paths["drafter"],
        endpoint=f"http://127.0.0.1:{args.port}/v1",
        shared_dir=f"/tmp/speculators-attention-100k-{hashlib.sha256(str(root).encode()).hexdigest()[:12]}",  # noqa: S108 -- shared local extractor directory
    )
    config = launch / "train.yaml"
    config.write_text(yaml.safe_dump(template, sort_keys=False))
    provenance = launch / f"vllm-attempt-{len(list(launch.glob('vllm-attempt-*'))):05d}"
    provenance.mkdir()
    # Co-locate the effective recipes and launcher pin with extraction provenance.
    for source in (config, recipe_path, state_path):
        (provenance / source.name).write_bytes(source.read_bytes())
    return template, config, provenance


def commands(args, cfg, config, provenance):
    _, updates, upload_interval = limits(args)
    server = [
        sys.executable,
        str(REPO / "scripts/launch_vllm.py"),
        "train",
        cfg["online"]["target_model_path"],
        "--target-layer-ids",
        "2",
        "18",
        "33",
        "--include-last-layer",
        "--hidden-states-backend",
        "file",
        "--hidden-states-path",
        cfg["online"]["shared_dir"],
        "--provenance-dir",
        str(provenance),
        "--",
        "--served-model-name",
        "Qwen/Qwen3-8B",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8193",
        "--max-num-batched-tokens",
        "8193",
        "--max-num-seqs",
        str(getattr(args, "max_num_seqs", 128)),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--no-enable-chunked-prefill",
    ]
    train = [
        sys.executable,
        "-m",
        "speculators",
        "generator",
        "train-full-data",
        "--config",
        str(config),
        "--output-root",
        str(args.output_root.resolve()),
        "--epochs",
        "10",
        "--max-updates",
        str(updates),
        "--hub-upload-interval",
        str(upload_interval),
        "--hub-repo",
        args.hub_repo,
        "--vllm-provenance-dir",
        str(provenance),
    ]
    return server, train


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("vLLM exited during startup; see vllm.log")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(1)
    raise TimeoutError("vLLM readiness timed out; see vllm.log")


def stop_owned(process):
    """Only terminate the process group started by this launcher."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run(args):
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output_root / ".launcher.lock"), timeout=0):
        if not args.prepare_only:
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", args.port)) == 0:
                    raise ValueError(
                        f"Port {args.port} is occupied; "
                        "stop the old vLLM server or use --port"
                    )
        cfg, config, provenance = prepare(args)
        server_command, train_command = commands(args, cfg, config, provenance)
        write_json(
            provenance / "launcher_commands.json",
            {"server": server_command, "train": train_command},
        )
        # Existing dry-run checks every local row, exact membership, and schedule
        # without contacting vLLM or allocating training tensors on the GPU.
        from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415
        from speculators.generator.epoch_training import train_full_data  # noqa: PLC0415

        train_rows, updates, upload_interval = limits(args)
        plan = train_full_data(
            CampaignConfig.load(config),
            args.output_root,
            args.hub_repo,
            provenance,
            epochs=10,
            max_updates=updates,
            hub_upload_interval=upload_interval,
            dry_run=True,
        )
        if (
            plan["job"]["training_examples"] != train_rows
            or plan["job"]["updates"] != updates
            or plan["job"].get("validation_examples") != VALIDATION_ROWS
        ):
            raise ValueError(
                f"Expected {train_rows} training rows, 1K validation rows, "
                f"and {updates} updates"
            )
        write_json(provenance / "training_plan.json", plan)
        if args.prepare_only:
            print(
                f"Prepared {train_rows} training rows; plan: "
                f"{provenance / 'training_plan.json'}",
                flush=True,
            )
            return 0
        server = trainer = None
        print(f"Starting vLLM; log: {provenance / 'vllm.log'}", flush=True)
        try:
            with (provenance / "vllm.log").open("a") as log:
                server = subprocess.Popen(  # noqa: S603 -- explicit argv, no shell
                    server_command,
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                wait_ready(server, args.port, args.startup_timeout)
                print(
                    f"vLLM ready; starting/resuming {updates} updates "
                    f"on {train_rows} training samples",
                    flush=True,
                )
                trainer = subprocess.Popen(  # noqa: S603 -- explicit argv, no shell
                    train_command, cwd=REPO, start_new_session=True
                )
                while trainer.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError("vLLM exited during training; see vllm.log")
                    time.sleep(1)
                return trainer.returncode
        finally:
            stop_owned(trainer)
            stop_owned(server)


def interrupt(_signum, _frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("2k", "10k"), default="2k")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--hub-repo", default=None)
    parser.add_argument(
        "--trajectories-revision",
        default="main",
        help="Resolved once; resume reuses saved SHA",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.40)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--startup-timeout", type=float, default=1200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="End-to-end pilot: 1,000 training samples, 10 updates; "
        "isolated output and HF repo",
    )
    args = parser.parse_args()
    suffix = "-pilot" if args.pilot else ""
    if args.output_root is None:
        args.output_root = Path(f"output/dyn-spec-attention-100k-{args.variant}-v1" + suffix)
    if args.hub_repo is None:
        args.hub_repo = f"azeeer/eagle3-generator-attention-100k-{args.variant}-v1" + suffix
    if (
        not 0 < args.gpu_memory_utilization < 1
        or not 1 <= args.port <= MAX_PORT
        or args.startup_timeout <= 0
        or args.max_num_seqs < 1
    ):
        parser.error(
            "GPU memory utilization must be between 0 and 1; "
            "port must be 1..65535; startup-timeout and max-num-seqs must be positive"
        )
    os.chdir(REPO)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        return run(args)
    except KeyboardInterrupt:
        print(
            "Stopped; rerun the same command to resume saved checkpoints.", flush=True
        )
        return 130
    except (ValueError, RuntimeError, TimeoutError) as error:
        print(f"Training launcher stopped: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
