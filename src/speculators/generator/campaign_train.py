"""Update-based reconstruction and behavioral training on matched episodes."""

import json
import random
import time
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.bank.progress import report
from speculators.generator.campaign_data import EpisodeSchedule
from speculators.generator.campaign_metrics import ProcessMonitor
from speculators.generator.data import atomic_torch_save, groups
from speculators.generator.engine import clip_gradients, set_learning_rate
from speculators.generator.model import (
    backward_factors,
    detached_factors,
    factor_loss,
    factor_parameters,
)
from speculators.generator.tracking import ExperimentTracker, replay_campaign_history
from speculators.train.utils import normalize_counted_metrics


@contextmanager
def compute_timer(runtime, events):
    if runtime.device.type == "cuda":
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        yield
        stop.record()
        events.append((start, stop))
    else:
        start = time.perf_counter()
        yield
        events.append(time.perf_counter() - start)


def synchronize(runtime):
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)


@torch.no_grad()
def evaluate_subset(runtime, subset, generator):
    synchronize(runtime)
    started = time.perf_counter()
    runtime.drafter.eval()
    if generator is not None:
        generator.eval()
    indices = runtime.corpus.indices(subset, validation=True)
    cohorts = list(groups(indices, runtime.cfg.seed + 9000, 4, 4))
    accumulated = {}
    report(f"Validation {subset}: starting; contexts={len(cohorts)}")
    for position, cohort in enumerate(cohorts):
        upcoming = [i for group in cohorts[position : position + 8] for i in group]
        refs = runtime.corpus.references(upcoming, full=True)
        runtime.corpus.cache.protect(refs)
        runtime.corpus.cache.prefetch(refs)
        factors = (
            generator(*runtime.corpus.context(cohort, "last", runtime.device))
            if generator is not None
            else None
        )
        for batch, _ in runtime.corpus.microbatches(
            cohort, runtime.device, runtime.dtype
        ):
            _, _, metrics = runtime.forward(batch, factors)
            for key, value in metrics.items():
                accumulated[key] = accumulated.get(key, 0.0) + float(value)
        if position == 0 or (position + 1) % 10 == 0 or position + 1 == len(cohorts):
            synchronize(runtime)
            report(
                f"Validation {subset}: contexts={position + 1}/{len(cohorts)}; "
                f"elapsed={time.perf_counter() - started:.2f}s"
            )
    normalized = normalize_counted_metrics(accumulated)
    return {
        "score": normalized["full_acc_0"],
        "validation_examples": len(indices),
        "elapsed_seconds": time.perf_counter() - started,
        "contexts": {"4": normalized},
    }


def evaluate(runtime, subsets, generator):
    results = {name: evaluate_subset(runtime, name, generator) for name in subsets}
    by_category = {}
    for domain in sorted({s.rsplit("-", 1)[0] for s in subsets}):
        selected = [m for s, m in results.items() if s.rsplit("-", 1)[0] == domain]
        by_category[domain] = sum(m["score"] for m in selected) / len(selected)
    keys = next(iter(results.values()))["contexts"]["4"]
    return {
        "score": sum(by_category.values()) / len(by_category),
        "validation_examples": sum(m["validation_examples"] for m in results.values()),
        "contexts": {
            "4": {
                k: sum(m["contexts"]["4"][k] for m in results.values()) / len(results)
                for k in keys
            }
        },
        "subsets": results,
        "categories": by_category,
    }


def select_target(entries, seed, update, episode):
    rng = random.Random(seed + 7000001 + update * 100003 + episode * 1009)
    return rng.choice(entries)


def train_update(runtime, generator, optimizer, episodes, snapshots, update, objective):
    cfg = runtime.cfg
    runtime.drafter.train()
    if generator is not None:
        generator.train()
    optimizer.zero_grad(set_to_none=True)
    set_learning_rate(optimizer, cfg, update + 1)
    losses, events = [], []
    data_seconds = 0.0
    response_tokens = 0
    examples = 0
    for episode, item in enumerate(episodes):
        cohort = item["indices"]
        started = time.perf_counter()
        if hasattr(runtime.corpus, "prepare_episode"):
            runtime.corpus.prepare_episode(cohort, full=objective != "bank")
        data_seconds += time.perf_counter() - started
        examples += len(cohort)
        # Stable augmentation/dropout RNG independent of snapshot sampling.
        torch.manual_seed(cfg.seed + update * 100003 + episode * 1009)
        started = time.perf_counter()
        context = (
            runtime.corpus.context(cohort, "last", runtime.device)
            if generator is not None
            else None
        )
        data_seconds += time.perf_counter() - started
        with compute_timer(runtime, events):
            factors = generator(*context) if generator is not None else None
        if objective == "bank":
            target = select_target(snapshots[item["subset"]], cfg.seed, update, episode)
            started = time.perf_counter()
            reference = runtime.cache.get(target["path"])
            data_seconds += time.perf_counter() - started
            with compute_timer(runtime, events):
                loss = factor_loss(factors, reference)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite reconstruction loss")
                (loss / len(episodes)).backward()
            losses.append(loss.detach().item())
            continue
        leaves = detached_factors(factors) if factors is not None else None
        total = sum(sum(runtime.corpus.rows[i]["loss_mask"][1:]) for i in cohort)
        if not total:
            raise ValueError("Episode has no supervised response tokens")
        response_tokens += total
        loss_value = 0.0
        for batch, _ in runtime.corpus.microbatches(
            cohort, runtime.device, runtime.dtype, training=True
        ):
            data_seconds += runtime.corpus.last_batch_data_seconds
            with compute_timer(runtime, events):
                _, loss, _ = runtime.forward(batch, leaves)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite behavioral loss")
                weight = batch["loss_mask"].sum().item() / total
                (loss * weight / len(episodes)).backward()
                loss_value += loss.detach().item() * weight
        if factors is not None:
            with compute_timer(runtime, events):
                backward_factors(factors, leaves)
        losses.append(loss_value)
    with compute_timer(runtime, events):
        gradients = clip_gradients(optimizer, cfg.optimization.max_grad_norm)
        optimizer.step()
    synchronize(runtime)
    compute_seconds = (
        sum(a.elapsed_time(b) for a, b in events) / 1000
        if runtime.device.type == "cuda"
        else sum(events)
    )
    return {
        "train/reconstruction_l1" if objective == "bank" else "train/drafter_loss": sum(
            losses
        )
        / len(losses),
        "train/data_wait_seconds": data_seconds,
        "train/gpu_compute_seconds": compute_seconds,
        "train/response_tokens": response_tokens,
        "train/examples": examples,
        "train/lr": optimizer.param_groups[0]["lr"],
        **gradients,
    }


def source_weights(runtime, generator, job):
    source = job.get("pretrained_path")
    if source is None:
        return
    source = Path(source)
    identity = json.loads((source.parent / "resolved_experiment.json").read_text())
    if (
        identity["lora"] != runtime.bank.lora.model_dump(mode="json")
        or identity["bank_data"]["revisions"] != runtime.corpus.identity["revisions"]
        or identity["configuration"]["architecture"]
        != runtime.cfg.architecture.model_dump(mode="json")
        or identity["job"].get("objective") != "bank"
    ):
        raise ValueError("Incompatible bank-warmed source checkpoint")
    if set(identity["job"]["subsets"]) & set(job["subsets"]):
        raise ValueError("Adaptation overlaps the source warmup subsets")
    if file_digest(source) != job["source_sha256"]:
        raise ValueError("Source checkpoint checksum changed")
    state = torch.load(source, map_location="cpu", weights_only=True)
    generator.load_state_dict(state["model"])


def _prefetch(runtime, schedule, step, updates, lookahead, objective):
    if hasattr(runtime.corpus, "prepare_episode"):
        runtime.corpus.prepare_episode(
            schedule.update(step)[0]["indices"], full=objective != "bank"
        )
        return
    indices = [
        i
        for number in range(step, min(updates, step + lookahead))
        for episode in schedule.update(number)
        for i in episode["indices"]
    ]
    refs = runtime.corpus.references(indices, full=objective != "bank")
    runtime.corpus.cache.protect(refs)
    runtime.corpus.cache.prefetch(refs)


def run_training(runtime, job, output, identity, *, lookahead, started=None):  # noqa: C901 -- durable update loop
    entered = time.perf_counter()
    started = entered if started is None else started
    setup_seconds = entered - started
    cfg = runtime.cfg
    generator = None if job["arm"] == "fresh_lora" else runtime.new_generator("last")
    if generator is None:
        runtime.initialize_direct()
        parameters = [p for p in runtime.drafter.parameters() if p.requires_grad]
    else:
        runtime.drafter.requires_grad_(False)
        source_weights(runtime, generator, job)
        parameters = list(generator.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.optimization.lr, weight_decay=cfg.optimization.weight_decay
    )
    checkpoint = output / "checkpoint.pt"
    schedule = EpisodeSchedule(
        cfg, runtime.corpus, job["subsets"], long_first=job.get("pilot", False)
    )
    history, timing, progress = [], [], 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["identity"] != digest(identity):
            raise ValueError("Recovery identity differs")
        if generator is None:
            runtime.drafter.load_state_dict(saved["model"], strict=False)
        else:
            generator.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        progress, history, timing = saved["progress"], saved["history"], saved["timing"]
        torch.set_rng_state(saved["torch_rng"])
        if runtime.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved

    def save():
        checkpoint_bytes = sum(p.numel() for p in parameters) * 16 + 1024**2
        runtime.corpus.cache.protect([])
        with runtime.corpus.cache.reserve_output(checkpoint_bytes):
            persist_checkpoint()

    def persist_checkpoint():
        state = (
            generator.state_dict()
            if generator is not None
            else factor_parameters(runtime.paths, runtime.adapter_state())
        )
        atomic_torch_save(
            checkpoint,
            {
                "identity": digest(identity),
                "model": {k: v.detach().cpu() for k, v in state.items()},
                "optimizer": optimizer.state_dict(),
                "progress": progress,
                "history": history,
                "timing": timing,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if runtime.device.type == "cuda"
                else [],
            },
        )

    cache_start = runtime.corpus.cache.snapshot()
    if runtime.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(runtime.device)
    with (
        ProcessMonitor(runtime.device) as monitor,
        ExperimentTracker(cfg, job, output, identity) as tracker,
    ):
        if progress and tracker.new_destination and cfg.wandb.layout == "sections":
            replay_campaign_history(tracker, history, timing)

        def validate():
            synchronize(runtime)
            validation_started = time.perf_counter()
            report(f"{job['label']}: validation starting at update={progress}")
            runtime.corpus.cache.protect([])
            metrics = evaluate(runtime, job["validation_subsets"], generator)
            synchronize(runtime)
            row = {
                "update": progress,
                "validation_seconds": time.perf_counter() - validation_started,
                "training_seconds": sum(t["train/update_seconds"] for t in timing),
                **metrics,
            }
            history.append(row)
            tracker.validation(
                metrics, progress, training_seconds=row["training_seconds"]
            )
            tracker.log(
                {
                    "optimizer_step": progress,
                    "validation/seconds": row["validation_seconds"],
                    "validation/examples": metrics["validation_examples"],
                    "validation/cumulative_seconds": sum(
                        r["validation_seconds"] for r in history
                    ),
                    "time/session_elapsed_seconds": time.perf_counter() - started,
                    **{
                        f"validation/subset_{name}/seconds": value["elapsed_seconds"]
                        for name, value in metrics["subsets"].items()
                    },
                    **{
                        f"validation/category_{d}/offline_agreement": v
                        for d, v in metrics["categories"].items()
                    },
                }
            )
            write_json(output / "metrics.json", history)
            report(
                f"{job['label']}: validation completed at update={progress}; "
                f"wall={row['validation_seconds']:.2f}s; "
                f"session_elapsed={time.perf_counter() - started:.2f}s"
            )

        if not history:
            validate()
            save()
        examples_seen = sum(t["train/examples"] for t in timing)
        compute_total = sum(t["train/gpu_compute_seconds"] for t in timing)
        wall_total = sum(t["train/update_seconds"] for t in timing)
        for update in range(progress, job["updates"]):
            tick = time.perf_counter()
            _prefetch(
                runtime, schedule, update, job["updates"], lookahead, job["objective"]
            )
            scheduling_seconds = time.perf_counter() - tick
            values = train_update(
                runtime,
                generator,
                optimizer,
                schedule.update(update),
                job.get("snapshots_by_subset", {}),
                update,
                job["objective"],
            )
            elapsed = time.perf_counter() - tick
            values["train/data_wait_seconds"] += scheduling_seconds
            examples_seen += values["train/examples"]
            compute_total += values["train/gpu_compute_seconds"]
            wall_total += elapsed
            progress = update + 1
            synchronize(runtime)
            cache = runtime.corpus.cache.snapshot()
            values.update(
                {
                    "train/update_seconds": elapsed,
                    "train/scheduling_seconds": scheduling_seconds,
                    "train/examples_seen": examples_seen,
                    "train/cumulative_gpu_compute_seconds": compute_total,
                    "train/cumulative_update_seconds": wall_total,
                    "time/session_elapsed_seconds": time.perf_counter() - started,
                    "train/data_wait_fraction": values["train/data_wait_seconds"]
                    / max(elapsed, 1e-9),
                    "train/host_rss_bytes": psutil.Process().memory_info().rss,
                    "train/gpu_compute_fraction": values["train/gpu_compute_seconds"]
                    / max(elapsed, 1e-9),
                    **{f"train/{k}": v for k, v in monitor.snapshot().items()},
                    **{f"cache/{k}": v for k, v in cache.items()},
                    "cache/conditioning_hits": runtime.corpus.summary_hits,
                    "cache/conditioning_misses": runtime.corpus.summary_misses,
                    "cache/conditioning_hit_rate": runtime.corpus.summary_hits
                    / max(
                        1, runtime.corpus.summary_hits + runtime.corpus.summary_misses
                    ),
                    "cache/conditioning_seconds": runtime.corpus.summary_seconds,
                }
            )
            if runtime.device.type == "cuda":
                free_bytes, total_bytes = torch.cuda.mem_get_info(runtime.device)
                values.update(
                    {
                        "train/gpu_device_used_bytes": total_bytes - free_bytes,
                        "train/gpu_peak_allocated_bytes": (
                            torch.cuda.max_memory_allocated(runtime.device)
                        ),
                        "train/gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(
                            runtime.device
                        ),
                    }
                )
            timing.append({"update": progress, **values})
            tracker.training(progress, values, final=progress == job["updates"])
            if progress == 1 or progress % cfg.optimization.log_interval == 0:
                report(
                    f"{job['label']}: update={progress}/{job['updates']}; "
                    f"wall={elapsed:.2f}s; "
                    f"data={values['train/data_wait_seconds']:.2f}s; "
                    f"training_total={wall_total:.2f}s; "
                    f"session_elapsed={time.perf_counter() - started:.2f}s; "
                    f"cache={cache['cache_bytes'] / 2**30:.2f} GiB"
                )
            if (
                progress % cfg.optimization.pretraining_validation_interval == 0
                or progress == job["updates"]
            ):
                validate()
            if (
                progress % cfg.optimization.checkpoint_interval == 0
                or progress == job["updates"]
            ):
                save()
                write_json(output / "timing.json", timing)
        cache_end = runtime.corpus.cache.snapshot()
        result = {
            "job": job,
            "completed_updates": progress,
            "history": history,
            "score": history[-1]["score"],
            "metric": "offline agreement (full_acc_0), context size 4",
            "elapsed_seconds": time.perf_counter() - started,
            "setup_seconds": setup_seconds,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_digest(checkpoint),
            "process_metrics": monitor.snapshot(),
            "cache_start": cache_start,
            "cache_end": cache_end,
            "training_seconds": sum(t["train/update_seconds"] for t in timing),
            "validation_seconds": sum(r["validation_seconds"] for r in history),
            "data_wait_seconds": sum(t["train/data_wait_seconds"] for t in timing),
            "summary_hits": runtime.corpus.summary_hits,
            "summary_misses": runtime.corpus.summary_misses,
        }
        if job["stage"] == "adaptation":
            result["score_change"] = result["score"] - history[0]["score"]
        write_json(output / "result.json", result)
        tracker.result(result)
        report(
            f"{job['label']}: completed; "
            f"session_wall={result['elapsed_seconds']:.2f}s; "
            f"setup={setup_seconds:.2f}s; "
            f"training_total={result['training_seconds']:.2f}s; "
            f"validation_total={result['validation_seconds']:.2f}s"
        )
    return result
