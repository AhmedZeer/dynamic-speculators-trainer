"""One durable W&B run per completed model/category measurement."""

import json

from speculators.bank.artifacts import digest, write_json
from speculators.generator.tracking import wandb_module


def add_round_evidence(artifact, root, task, destination):
    source = root / task.get("server_provenance", "missing") / "adapter_rounds"
    before_path = destination / task["category"] / "server_before.json"
    after_path = destination / task["category"] / "server_after.json"
    if not source.is_dir() or not before_path.exists() or not after_path.exists():
        return
    left, right = (json.loads(p.read_text()) for p in (before_path, after_path))
    start = left["workers"][0]["adapter_totals"]["prepared_requests"]
    end = right["workers"][0]["adapter_totals"]["prepared_requests"]
    for path in sorted(source.glob("*.json")):
        row = json.loads(path.read_text())
        if start < row.get("prepared_sequence_number", 0) <= end:
            artifact.add_file(str(path), name="adapter_rounds/" + path.name)


def publish_completed(root, plan, state):
    """Retry tracking independently: a W&B outage must never repeat GPU work."""
    if not plan.get("configuration", {}).get("wandb", {}).get("enabled", False):
        return
    for entry in state["tasks"].values():
        if entry.get("status") != "completed" or entry.get("wandb_published"):
            continue
        destination = root / entry["output"]
        results = json.loads((destination / "results.json").read_text())
        publish_measurement(
            root, plan, entry, results["groups"][entry["category"]], destination
        )
        entry["wandb_published"] = True
        write_json(root / "state.json", state)


def publish_measurement(root, plan, task, result, destination):  # noqa: C901 -- metric sections and provenance artifact
    settings = plan.get("configuration", {}).get("wandb", {})
    if not settings.get("enabled", False):
        return
    module = wandb_module()
    identity = {
        "suite": plan["identity"],
        "task": {key: task[key] for key in ("model", "benchmark", "category")},
        "project": settings["project"],
    }
    run = module.init(
        project=settings["project"],
        entity=settings.get("entity"),
        mode=settings.get("mode", "online"),
        id=digest(identity)[:16],
        resume="allow" if settings.get("mode", "online") == "online" else None,
        name=task["model"] + "/" + task["category"],
        group=settings.get("group") or root.name,
        job_type="inference",
        tags=[task["category"]],
        config={
            "task": task,
            "configuration": plan["configuration"],
            "correctness": "not_checked",
            "resolved_models": plan.get("models", []),
            "workloads": plan.get("workloads", {}),
        },
    )
    try:
        for section in ("throughput", "acceptance", "latency", "adapter_performance"):
            run.define_metric(section + "/*")
        metrics = {}
        for row in result.get("performance", []):
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    section = (
                        "latency"
                        if any(s in key for s in ("latency", "ttft", "itl", "tpot"))
                        else "throughput"
                    )
                    metrics[section + "/" + key] = value
        for name, prefix in (
            ("acceptance", "acceptance"),
            ("adapter_metrics", "adapter_performance"),
        ):
            for key, value in (result.get(name) or {}).items():
                if isinstance(value, (int, float)):
                    metrics[prefix + "/" + key] = value
        run.log(metrics)
        run.summary.update(metrics)
        evidence = module.Artifact(
            digest(identity)[:16] + "-measurement", type="benchmark"
        )
        for name in ("results.json", "benchmark_identity.json", "eval_command.txt"):
            if (destination / name).is_file():
                evidence.add_file(str(destination / name), name=name)
        for path in sorted((destination / "server_provenance").rglob("*")):
            if (
                path.is_file()
                and path.suffix in (".json", ".txt", ".patch")
                and "vanilla-vllm" not in path.parts
            ):
                evidence.add_file(
                    str(path),
                    name="server_provenance/"
                    + str(path.relative_to(destination / "server_provenance")),
                )
        add_round_evidence(evidence, root, task, destination)
        run.log_artifact(evidence)
        write_json(
            destination / "wandb.json", {"id": run.id, "url": run.url, **settings}
        )
    finally:
        run.finish()
