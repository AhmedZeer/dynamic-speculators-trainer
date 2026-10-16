"""Metric names that keep quality, speed, and resource charts in separate sections."""

SECTIONS = (
    "validation_scores",
    "validation_losses",
    "training_losses",
    "training_performance",
    "validation_performance",
    "cache_performance",
    "resources",
    "optimization",
    "run_summary",
    "validation_details",
)
OPTIMIZER_KEYS = {"lr", "grad_norm", "gradient_clipped", "clip_scale", "grad_rms"}


def resource_metric(key, value):
    if key.endswith("_bytes"):
        return "resources/" + key.removesuffix("_bytes") + "_gib", value / 2**30
    return "resources/" + key, value


def sectioned_metrics(values):  # noqa: C901 -- routing existing metric families
    result = {}
    for original_key, original_value in values.items():
        key, value = original_key, original_value
        if key.startswith("train/"):
            name = key.removeprefix("train/")
            if name in ("reconstruction_l1", "drafter_loss"):
                key = "training_losses/" + name
            elif name in OPTIMIZER_KEYS:
                key = "optimization/" + name
            elif (
                name.startswith(("host_", "disk_", "gpu_peak_", "gpu_device_"))
                or name == "gpu_utilization_percent"
            ):
                key, value = resource_metric(name, value)
            else:
                key = "training_performance/" + name
        elif key.startswith("cache/"):
            name = key.removeprefix("cache/")
            if name in ("disk_free_bytes", "temporary_disk_free_bytes"):
                key, value = resource_metric(name, value)
            else:
                key = "cache_performance/" + name
                if key.endswith("_bytes"):
                    key = key.removesuffix("_bytes") + "_gib"
                    value /= 2**30
        elif key.startswith("validation/category_"):
            name = key.split("/")[1].removeprefix("category_")
            key = f"validation_scores/{name}_agreement"
        elif key.startswith("validation/subset_"):
            # Subset details live in the table, rather than scalar panels.
            continue
        elif key == "validation/seconds":
            key = "validation_performance/pass_seconds"
        elif key.startswith("validation/"):
            key = "validation_performance/" + key.removeprefix("validation/")
        elif key == "time/session_elapsed_seconds":
            key = "run_summary/session_elapsed_seconds"
        if key.endswith("_fraction"):
            key = key.removesuffix("_fraction") + "_percent"
            value *= 100
        elif key.startswith("cache_performance/") and key.endswith("hit_rate"):
            key = key.removesuffix("hit_rate") + "hit_percent"
            value *= 100
        result[key] = value
    elapsed = values.get("train/update_seconds", 0)
    if elapsed > 0:
        result["training_performance/examples_per_second"] = (
            values.get("train/examples", 0) / elapsed
        )
        if values.get("train/response_tokens", 0):
            result["training_performance/supervised_tokens_per_second"] = (
                values["train/response_tokens"] / elapsed
            )
    seconds = values.get("validation/seconds", 0)
    if seconds > 0 and "validation/examples" in values:
        result["validation_performance/examples_per_second"] = (
            values["validation/examples"] / seconds
        )
    return result


def validation_values(metrics, step):
    values = {
        "optimizer_step": step,
        "validation_scores/overall_agreement": metrics["score"],
    }
    for size, context in metrics["contexts"].items():
        for key, value in context.items():
            if key.startswith("loss_"):
                name = (
                    f"validation_losses/context_{size}_loss_step_"
                    f"{key.removeprefix('loss_')}"
                )
            elif key.startswith("full_acc_"):
                name = (
                    f"validation_scores/context_{size}_agreement_step_"
                    f"{key.removeprefix('full_acc_')}"
                )
            else:
                name = f"diagnostics/context_{size}_{key}"
            values[name] = value
    for category, score in metrics.get("categories", {}).items():
        values[f"validation_scores/{category}_agreement"] = score
    return values


def subset_table(metrics, step):
    rows = []
    for name, result in metrics.get("subsets", {}).items():
        for size, context in result["contexts"].items():
            rows.extend(
                [step, name, size, key, value] for key, value in context.items()
            )
        rows.append([step, name, "4", "seconds", result.get("elapsed_seconds")])
    return rows
