"""Update-based behavioral training on saved token episodes."""

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


class UpdateTimings:
    """Disjoint stream intervals, resolved after the existing update-end sync."""

    COMPONENTS = (
        "generator_forward",
        "generator_backward",
        "drafter_forward",
        "drafter_backward",
        "optimizer",
    )

    def __init__(self, runtime):
        self.runtime = runtime
        self.events = {name: [] for name in self.COMPONENTS}

    def measure(self, name):
        return compute_timer(self.runtime, self.events[name])

    def seconds(self):
        return {
            name: (
                sum(start.elapsed_time(stop) for start, stop in events) / 1000
                if self.runtime.device.type == "cuda"
                else sum(events)
            )
            for name, events in self.events.items()
        }


def conditioning_subset(cohort, size, seed):
    """Choose conditioning members without changing the supervised loss cohort."""
    if size is None or size >= len(cohort):
        return cohort
    return random.Random(seed).sample(cohort, size)


@torch.no_grad()
def evaluate_subset(runtime, subset, generator, conditioning_examples=None):
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
            generator(
                *runtime.corpus.context(
                    conditioning_subset(
                        cohort,
                        conditioning_examples,
                        runtime.cfg.seed + 9000 + position,
                    ),
                    "last",
                    runtime.device,
                )
            )
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


def evaluate(runtime, subsets, generator, conditioning_examples=None):
    results = {
        name: evaluate_subset(runtime, name, generator, conditioning_examples)
        for name in subsets
    }
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


def generator_batch(
    runtime, generator, episodes, *, update, conditioning_examples, timers
):
    """Keep each cohort's conditioning independent while batching the hypernetwork."""
    started = time.perf_counter()
    contexts = [
        runtime.corpus.context(
            conditioning_subset(
                item["indices"],
                conditioning_examples,
                runtime.cfg.seed + update * 100003 + episode * 1009,
            ),
            "last",
            runtime.device,
        )
        for episode, item in enumerate(episodes)
    ]
    data_seconds = time.perf_counter() - started
    with timers.measure("generator_forward"):
        factors = generator.forward_cohorts(contexts)
    return (
        factors,
        detached_factors(factors),
        data_seconds,
        sum(len(c[0]) for c in contexts),
    )


def train_update(  # noqa: C901, PLR0917 -- shared serial/batched update API
    runtime,
    generator,
    optimizer,
    episodes,
    update,
    objective,
    *,
    conditioning_examples=None,
):
    if objective != "direct":
        raise ValueError("Campaign training supports only the direct objective")
    cfg = runtime.cfg
    runtime.drafter.train()
    if generator is not None:
        generator.train()
    optimizer.zero_grad(set_to_none=True)
    set_learning_rate(optimizer, cfg, update + 1)
    losses, timers = [], UpdateTimings(runtime)
    drafter_microbatches = 0
    data_seconds = 0.0
    response_tokens = 0
    examples = 0
    conditioning_count = 0
    batched = generator is not None and cfg.generator_batching
    if batched:
        generated_batch, leaves_batch, data_seconds, conditioning_count = (
            generator_batch(
                runtime,
                generator,
                episodes,
                update=update,
                conditioning_examples=conditioning_examples,
                timers=timers,
            )
        )
    for episode, item in enumerate(episodes):
        cohort = item["indices"]
        started = time.perf_counter()
        if hasattr(runtime.corpus, "prepare_episode"):
            runtime.corpus.prepare_episode(cohort, full=True)
        data_seconds += time.perf_counter() - started
        examples += len(cohort)
        # Stable augmentation/dropout RNG for each scheduled episode.
        torch.manual_seed(cfg.seed + update * 100003 + episode * 1009)
        started = time.perf_counter()
        context = (
            runtime.corpus.context(
                conditioning_subset(
                    cohort,
                    conditioning_examples,
                    cfg.seed + update * 100003 + episode * 1009,
                ),
                "last",
                runtime.device,
            )
            if generator is not None and not batched
            else None
        )
        data_seconds += time.perf_counter() - started
        if context is not None:
            conditioning_count += context[0].shape[0]
        factors = None
        if context is not None:
            with timers.measure("generator_forward"):
                factors = generator(*context)
        leaves = (
            {
                name: tuple(value[episode] for value in pair)
                for name, pair in leaves_batch.items()
            }
            if batched
            else detached_factors(factors)
            if factors is not None
            else None
        )
        total = sum(sum(runtime.corpus.rows[i]["loss_mask"][1:]) for i in cohort)
        if not total:
            raise ValueError("Episode has no supervised response tokens")
        response_tokens += total
        loss_value = 0.0
        for batch, _ in runtime.corpus.microbatches(
            cohort, runtime.device, runtime.dtype, training=True
        ):
            data_seconds += runtime.corpus.last_batch_data_seconds
            drafter_microbatches += 1
            with timers.measure("drafter_forward"):
                _, loss, _ = runtime.forward(batch, leaves)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite behavioral loss")
            weight = batch["loss_mask"].sum().item() / total
            with timers.measure("drafter_backward"):
                (loss * weight / len(episodes)).backward()
            loss_value += loss.detach().item() * weight
        if factors is not None:
            with timers.measure("generator_backward"):
                backward_factors(factors, leaves)
        losses.append(loss_value)
    if batched:
        with timers.measure("generator_backward"):
            backward_factors(generated_batch, leaves_batch)
    with timers.measure("optimizer"):
        gradients = clip_gradients(optimizer, cfg.optimization.max_grad_norm)
        optimizer.step()
    synchronize(runtime)
    component_seconds = timers.seconds()
    generator_calls = (1 if batched else len(episodes)) if generator is not None else 0
    return {
        "train/drafter_loss": sum(losses) / len(losses),
        "train/data_wait_seconds": data_seconds,
        "train/gpu_compute_seconds": sum(component_seconds.values()),
        **{
            f"train/{name}_seconds": seconds
            for name, seconds in component_seconds.items()
        },
        "train/drafter_microbatches": drafter_microbatches,
        "train/generator_forward_ms_per_call": 1000
        * component_seconds["generator_forward"]
        / max(generator_calls, 1),
        "train/generator_forward_ms_per_cohort": 1000
        * component_seconds["generator_forward"]
        / len(episodes),
        "train/drafter_forward_ms_per_microbatch": 1000
        * component_seconds["drafter_forward"]
        / max(drafter_microbatches, 1),
        "train/response_tokens": response_tokens,
        "train/examples": examples,
        "train/conditioning_examples": conditioning_count,
        "train/conditioning_size_mean": conditioning_count / len(episodes),
        "train/generator_forward_calls": generator_calls,
        "train/generator_backward_calls": generator_calls,
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
        or identity["job"].get("objective") != "direct"
    ):
        raise ValueError("Incompatible direct-pretrained source checkpoint")
    if set(identity["job"]["subsets"]) & set(job["subsets"]):
        raise ValueError("Adaptation overlaps the source pretraining subsets")
    if file_digest(source) != job["source_sha256"]:
        raise ValueError("Source checkpoint checksum changed")
    state = torch.load(source, map_location="cpu", weights_only=True)
    generator.load_state_dict(state["model"])


def _prefetch(runtime, schedule, step, updates, lookahead, objective):  # noqa: ARG001, PLR0917
    if hasattr(runtime.corpus, "set_training_window"):
        runtime.corpus.set_training_window(
            [
                episode["indices"]
                for number in range(
                    step,
                    min(
                        updates,
                        step + runtime.corpus.cache.settings.lookahead_updates + 1,
                    ),
                )
                for episode in schedule.update(number)
            ],
            full=True,
        )
        return
    indices = [
        i
        for number in range(step, min(updates, step + lookahead))
        for episode in schedule.update(number)
        for i in episode["indices"]
    ]
    refs = runtime.corpus.references(indices, full=True)
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
        if "adapter_dtype" in job:
            for path in runtime.paths.values():
                adapter = runtime.drafter.get_submodule(path)
                for name in ("a", "b"):
                    factor = getattr(adapter, name)
                    setattr(
                        adapter,
                        name,
                        torch.nn.Parameter(
                            factor.detach().to(
                                dtype=getattr(torch, job["adapter_dtype"])
                            ),
                            requires_grad=factor.requires_grad,
                        ),
                    )
        parameters = [p for p in runtime.drafter.parameters() if p.requires_grad]
    else:
        runtime.drafter.requires_grad_(False)
        source_weights(runtime, generator, job)
        parameters = list(generator.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.optimization.lr, weight_decay=cfg.optimization.weight_decay
    )
    checkpoint = output / "checkpoint.pt"
    epoch_schedule = "epochs" in job
    hub_backups = "hub_repo" in job
    if hub_backups:
        from speculators.generator.epoch_training import backup_checkpoint  # noqa: PLC0415

    if epoch_schedule:
        from speculators.generator.epoch_training import (  # noqa: PLC0415
            EpochSchedule,
        )

        schedule = EpochSchedule(
            cfg,
            runtime.corpus,
            job["subsets"],
            job["epochs"],
            cohort_size=job.get("training_cohort_size"),
        )
        if schedule.epoch_ends != job["epoch_ends"]:
            raise ValueError("Epoch schedule differs from the saved training plan")
    else:
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

    def backup_if_due():
        if not hub_backups or not progress:
            return
        epoch = schedule.completed_epochs(progress) if epoch_schedule else None
        if progress % job["hub_upload_interval"] == 0 or progress == job["updates"]:
            backup_checkpoint(output, job, identity, epoch, progress)

    # Retry an interrupted boundary upload before doing any further training.
    if progress:
        backup_if_due()

    def save():
        report(f"{job['label']}: saving local checkpoint at update={progress}")
        checkpoint_bytes = sum(p.numel() for p in parameters) * 16 + 1024**2
        runtime.corpus.cache.protect([])
        with runtime.corpus.cache.reserve_output(checkpoint_bytes):
            persist_checkpoint()
        report(f"{job['label']}: local checkpoint saved at update={progress}")

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
        preparation = {"seconds": 0.0}
        if generator is not None and hasattr(runtime.corpus, "prepare_summaries"):
            report(f"{job['label']}: preparing persistent prompt summaries")
            preparation = runtime.corpus.prepare_summaries(
                job["subsets"] + job["validation_subsets"]
            )
            tracker.log(
                {
                    "optimizer_step": progress,
                    **{
                        f"preparation_performance/{k}": v
                        for k, v in preparation.items()
                    },
                }
            )
            report(
                f"{job['label']}: prompt summaries ready; "
                f"wall={preparation['seconds']:.2f}s"
            )

        def validate():
            synchronize(runtime)
            validation_started = time.perf_counter()
            report(f"{job['label']}: validation starting at update={progress}")
            runtime.corpus.cache.protect([])
            metrics = evaluate(
                runtime,
                job["validation_subsets"],
                generator,
                job.get("training_conditioning_examples"),
            )
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

        if not history or (
            progress > 0
            and progress % cfg.optimization.pretraining_validation_interval == 0
            and history[-1]["update"] != progress
        ):
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
                update,
                job["objective"],
                **(
                    {"conditioning_examples": job["training_conditioning_examples"]}
                    if "training_conditioning_examples" in job
                    else {}
                ),
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
            if epoch_schedule:
                values["train/epoch"] = examples_seen / job["training_examples"]
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
                report(
                    f"{job['label']}: timings/update; "
                    + "; ".join(
                        f"{name}={values[f'train/{name}_seconds'] * 1000:.2f}ms"
                        for name in UpdateTimings.COMPONENTS
                    )
                    + f"; generator_calls={values['train/generator_forward_calls']}; "
                    + f"drafter_microbatches={values['train/drafter_microbatches']}"
                )
            hub_checkpoint = hub_backups and (
                progress % job["hub_upload_interval"] == 0 or progress == job["updates"]
            )
            validation_due = (
                progress % cfg.optimization.pretraining_validation_interval == 0
            )
            if validation_due or progress == job["updates"]:
                validate()
            if (
                progress % cfg.optimization.checkpoint_interval == 0
                or hub_checkpoint
                or progress == job["updates"]
            ):
                save()
                write_json(output / "timing.json", timing)
                backup_if_due()
        cache_end = runtime.corpus.cache.snapshot()
        result = {
            "job": job,
            "completed_updates": progress,
            "preparation_seconds": preparation["seconds"],
            "preparation": preparation,
            "compiler_cache_dir": getattr(runtime, "compiler_cache_dir", None),
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
        if epoch_schedule:
            result["completed_epochs"] = schedule.completed_epochs(progress)
        if hub_backups:
            result["hub_checkpoints"] = json.loads(
                (output / "hub_checkpoints.json").read_text()
            )
        if "training_conditioning_examples" in job:
            result["training_conditioning_examples"] = job[
                "training_conditioning_examples"
            ]
            result["validation_conditioning_examples"] = min(
                4, job["training_conditioning_examples"]
            )
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
