"""Prepare target data once and train independent subset/seed banks."""

import json
import random
import subprocess
import sys
from pathlib import Path

import yaml

from speculators.bank.artifacts import digest, file_digest, load_subset, write_json
from speculators.bank.config import BankConfig

TARGET_HIDDEN_SIZE = 4096
HIDDEN_STATE_NDIM = 3


def prompt_messages(row: dict) -> list[dict]:
    messages = row.get("messages") or row.get("conversations") or row.get("input")
    if not isinstance(messages, list):
        raise ValueError("Source rows must contain messages, conversations, or input")
    result = []
    for message in messages:
        role = message.get("role", message.get("from"))
        if role in ("assistant", "gpt"):
            break
        role = "user" if role == "human" else role
        if role not in ("system", "user"):
            raise ValueError("Bank v1 supports text prompts without tools")
        content = message.get("content", message.get("value", ""))
        if not isinstance(content, str):
            raise ValueError("Bank v1 requires text content")
        result.append({"role": role, "content": content})
    if sum(m["role"] == "user" for m in result) != 1:
        raise ValueError("Bank v1 uses one user prompt per source example")
    return result


def partition_rows(rows, cfg: BankConfig):
    """Deduplicate identical prompts before splitting; retain all source aliases."""
    unique = {}
    for index, row in enumerate(rows):
        if cfg.dataset.sample_limit is not None and index >= cfg.dataset.sample_limit:
            break
        messages = prompt_messages(row)
        identity = digest(messages)
        source_id = str(row.get("uuid") or row.get("id") or digest(row))
        if identity not in unique:
            unique[identity] = {"id": identity, "messages": messages, "source_ids": []}
        unique[identity]["source_ids"].append(source_id)
    for item in unique.values():
        item["source_ids"] = sorted(set(item["source_ids"]))
    ordered = [unique[key] for key in sorted(unique)]
    random.Random(cfg.partition.seed).shuffle(ordered)
    size = cfg.partition.train_examples + cfg.partition.validation_examples
    groups = len(ordered) // size
    if max(cfg.partition.subset_ids) >= groups:
        raise ValueError(f"Only {groups} complete subsets available for selection")
    selected, subsets = [], []
    for subset_number in cfg.partition.subset_ids:
        subset_id = f"{cfg.dataset.category}-{subset_number:05d}"
        group = ordered[subset_number * size : (subset_number + 1) * size]
        start = len(selected)
        selected.extend({**row, "subset_id": subset_id} for row in group)
        subsets.append(
            {
                "id": subset_id,
                "train_ids": [r["id"] for r in group[: cfg.partition.train_examples]],
                "validation_ids": [
                    r["id"] for r in group[cfg.partition.train_examples :]
                ],
                "train_indices": list(
                    range(start, start + cfg.partition.train_examples)
                ),
                "validation_indices": list(
                    range(start + cfg.partition.train_examples, start + size)
                ),
            }
        )
    return (
        selected,
        subsets,
        {
            "unique_prompts": len(ordered),
            "complete_subsets": groups,
            "incomplete_group_examples": len(ordered) % size,
        },
    )


def cache_spec(cfg: BankConfig, revisions: dict) -> dict:
    return {
        "models": {**cfg.models.model_dump(), **revisions},
        "dataset": cfg.dataset.model_dump(),
        "partition": cfg.partition.model_dump(),
        "responses": cfg.responses.model_dump(),
        "hidden_states": cfg.hidden_states.model_dump(),
    }


def select_data(cfg: BankConfig) -> dict:
    from datasets import load_dataset  # noqa: PLC0415
    from huggingface_hub import HfApi  # noqa: PLC0415

    root = cfg.output_root.resolve()
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        spec = cache_spec(cfg, manifest["revisions"])
        if digest(spec) != manifest["cache_fingerprint"]:
            raise ValueError(
                "Incompatible cache configuration; use another output_root"
            )
        return manifest
    api = HfApi()
    revisions = {
        "target_sha": api.model_info(
            cfg.models.target, revision=cfg.models.target_revision
        ).sha,
        "drafter_sha": api.model_info(
            cfg.models.drafter, revision=cfg.models.drafter_revision
        ).sha,
        "dataset_sha": api.dataset_info(
            cfg.dataset.source, revision=cfg.dataset.revision
        ).sha,
    }
    dataset = load_dataset(
        cfg.dataset.source,
        name=cfg.dataset.configuration,
        split=cfg.dataset.split,
        revision=revisions["dataset_sha"],
    )
    selected, subsets, stats = partition_rows(dataset, cfg)
    root.mkdir(parents=True, exist_ok=True)
    prompts = root / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in selected))
    spec = cache_spec(cfg, revisions)
    manifest = {
        "version": 1,
        "category": cfg.dataset.category,
        "cache_fingerprint": digest(spec),
        "cache_spec": spec,
        "revisions": revisions,
        "root": str(root),
        "data_path": str(root / "data"),
        "hidden_states_path": str(root / "hidden_states"),
        "subsets": subsets,
        "partition_stats": stats,
        "prompts_sha256": file_digest(prompts),
        "prepared": False,
    }
    write_json(manifest_path, manifest)
    return manifest


def regenerate(cfg: BankConfig, manifest: dict):
    from speculators.cli.regenerate_responses import (  # noqa: PLC0415
        regenerate_responses,
    )
    from speculators.train.utils import save_train_command  # noqa: PLC0415

    root = Path(manifest["root"])
    provenance = root / "provenance" / "responses"
    provenance.mkdir(parents=True, exist_ok=True)
    save_train_command(str(provenance))
    regenerate_responses(
        endpoint=cfg.execution.generation_endpoint,
        model=cfg.models.target,
        dataset=str(root / "prompts.jsonl"),
        split=None,
        subset=None,
        limit=None,
        concurrency=cfg.execution.concurrency,
        max_tokens=cfg.responses.max_tokens,
        sampling_params=json.dumps(
            {
                "temperature": cfg.responses.temperature,
                "top_p": cfg.responses.top_p,
                "top_k": cfg.responses.top_k,
                "seed": cfg.responses.seed,
                "chat_template_kwargs": {"enable_thinking": cfg.responses.thinking},
                "truncate_prompt_tokens": cfg.responses.context_length
                - cfg.responses.max_tokens,
            }
        ),
        outfile=str(root / "responses.jsonl"),
        resume=True,
        response_staging_dir=cfg.execution.response_staging_dir,
        response_sync_interval=cfg.execution.response_sync_interval,
        language_filter=None,
        max_retries=cfg.execution.max_retries,
        reasoning_effort=None,
        temperature=None,
        seed=cfg.responses.seed,
    )


def prepare_rows(
    responses: list[dict], manifest: dict, sequence_length: int
) -> list[dict]:
    """Reorder async responses by identity and keep exact generation boundaries."""
    by_id = {}
    for response in responses:
        primary = response["primary_id"]
        if primary in by_id:
            raise ValueError(f"Multiple responses for prompt {primary}")
        by_id[primary] = response
    prompts = [
        json.loads(line)
        for line in (Path(manifest["root"]) / "prompts.jsonl").read_text().splitlines()
    ]
    expected = {p["id"] for p in prompts}
    if set(by_id) != expected:
        raise ValueError("Response identities do not match the selected corpus")
    result = []
    for prompt in prompts:
        response = by_id[prompt["id"]]
        ids, mask = response["input_ids"], response["loss_mask"]
        if len(ids) != len(mask) or not ids or len(ids) > sequence_length:
            raise ValueError(
                "Response exceeds sequence limit or has malformed boundaries"
            )
        boundary = next((i for i, bit in enumerate(mask) if bit == 1), None)
        if (
            boundary is None
            or boundary == 0
            or mask != [0] * boundary + [1] * (len(ids) - boundary)
        ):
            raise ValueError(
                "Expected one prompt prefix followed by supervised target tokens"
            )
        result.append(
            {
                "input_ids": ids,
                "loss_mask": mask,
                "seq_len": len(ids),
                "primary_id": prompt["id"],
                "source_ids": prompt["source_ids"],
                "category": manifest["category"],
                "subset_id": prompt["subset_id"],
                "prompt_length": boundary,
                "metadata": response["metadata"],
            }
        )
    return result


def prepare_arrow(cfg: BankConfig, manifest: dict):
    from datasets import Dataset, load_from_disk  # noqa: PLC0415

    root = Path(manifest["root"])
    responses_path = root / "responses.jsonl"
    rows = prepare_rows(
        [json.loads(line) for line in responses_path.read_text().splitlines()],
        manifest,
        cfg.hidden_states.sequence_length,
    )
    fingerprint = digest(rows)
    path = Path(manifest["data_path"])
    if path.exists():
        existing = load_from_disk(str(path)).to_list()
        if digest(existing) != fingerprint:
            raise ValueError("Prepared rows differ from the cached dataset")
    else:
        pending = root / "data.pending"
        if pending.exists():
            import shutil  # noqa: PLC0415

            shutil.rmtree(pending)
        Dataset.from_list(rows).with_format("torch").save_to_disk(str(pending))
        pending.rename(path)
    manifest.update(
        prepared=True,
        prepared_fingerprint=fingerprint,
        responses_sha256=file_digest(responses_path),
        examples=len(rows),
        tokens=sum(r["seq_len"] for r in rows),
        supervised_tokens=sum(sum(r["loss_mask"]) for r in rows),
    )
    write_json(root / "manifest.json", manifest)


def validate_cache(manifest: dict, cfg: BankConfig | None = None):  # noqa: C901
    from datasets import load_from_disk  # noqa: PLC0415
    from safetensors.torch import load_file  # noqa: PLC0415

    from speculators.data_generation.offline import check_hidden_states  # noqa: PLC0415

    if not manifest.get("prepared"):
        raise ValueError("Run prepare --stage data before extracting/training")
    root = Path(manifest["root"])
    if file_digest(root / "prompts.jsonl") != manifest["prompts_sha256"]:
        raise ValueError("Prompt identities changed after partitioning")
    data = load_from_disk(manifest["data_path"]).with_format(None)
    if digest(data.to_list()) != manifest["prepared_fingerprint"]:
        raise ValueError("Prepared token content differs from the cache fingerprint")
    for subset in manifest["subsets"]:
        for split, identity_key in (
            ("train_indices", "train_ids"),
            ("validation_indices", "validation_ids"),
        ):
            selected = [data[index]["primary_id"] for index in subset[split]]
            if selected != subset[identity_key]:
                raise ValueError("Subset membership does not match prepared identities")
        load_subset(root / "manifest.json", subset["id"])
    if (
        cfg
        and digest(cache_spec(cfg, manifest["revisions"]))
        != manifest["cache_fingerprint"]
    ):
        raise ValueError("Experiment configuration does not match cached data")
    files = []
    for index in range(len(data)):
        path = Path(manifest["hidden_states_path"]) / f"hs_{index}.safetensors"
        stat = path.stat()
        files.append([index, stat.st_size, stat.st_mtime_ns])
    validation_identity = digest(
        {
            "prepared": manifest["prepared_fingerprint"],
            "cache": manifest["cache_fingerprint"],
            "files": files,
        }
    )
    receipt = root / "hidden_states_validation.json"
    if (
        receipt.exists()
        and json.loads(receipt.read_text()).get("identity") == validation_identity
    ):
        return
    layers = manifest["cache_spec"]["hidden_states"]["layer_ids"]
    expected_dtype = manifest["cache_spec"]["hidden_states"]["dtype"]
    for index, row in enumerate(data):
        path = Path(manifest["hidden_states_path"]) / f"hs_{index}.safetensors"
        states = load_file(str(path))
        check_hidden_states(states, row["input_ids"])
        hidden = states["hidden_states"]
        if (
            hidden.ndim != HIDDEN_STATE_NDIM
            or hidden.shape[1] != len(layers)
            or hidden.shape[2] != TARGET_HIDDEN_SIZE
        ):
            raise ValueError(f"Invalid target-state shape at row {index}")
        if str(hidden.dtype).split(".")[-1] != expected_dtype:
            raise ValueError(f"Invalid target-state dtype at row {index}")
    write_json(
        receipt, {"identity": validation_identity, "validated_files": len(files)}
    )


def extract(cfg: BankConfig, manifest: dict):
    from speculators.cli.generate_offline_data import (  # noqa: PLC0415
        generate_offline_data,
    )

    if not manifest.get("prepared"):
        raise ValueError("Run prepare --stage data first")
    generate_offline_data(
        model=cfg.models.target,
        endpoint=cfg.execution.extraction_endpoint,
        preprocessed_data=manifest["data_path"],
        output=manifest["hidden_states_path"],
        max_samples=None,
        concurrency=cfg.execution.concurrency,
        validate_outputs=True,
        request_timeout=cfg.execution.request_timeout,
        max_retries=cfg.execution.max_retries,
        fail_on_error=True,
        max_consecutive_errors=None,
        world_size=1,
        rank=0,
    )
    validate_cache(manifest, cfg)


def prepare(cfg: BankConfig, stage: str):
    manifest = select_data(cfg)
    stages = ("responses", "data", "hidden") if stage == "all" else (stage,)
    for current in stages:
        if current == "responses":
            regenerate(cfg, manifest)
        elif current == "data":
            prepare_arrow(cfg, manifest)
        elif current == "hidden":
            extract(cfg, manifest)
    write_json(Path(manifest["root"]) / "experiment.json", cfg.model_dump(mode="json"))
    return manifest


def train_config(
    cfg: BankConfig,
    manifest: dict,
    subset_id: str,
    seed: int,
    target_path: str,
    drafter_path: str,
):
    root = Path(manifest["root"])
    run = root / "runs" / subset_id / f"seed-{seed}"
    return {
        "train": {
            "speculator_type": "eagle3",
            "seed": seed,
            "verifier": {"verifier_name_or_path": target_path},
            "draft": {
                "from_pretrained": drafter_path,
                "target_layer_ids": cfg.hidden_states.layer_ids[:-1],
                "draft_attn_impl": cfg.execution.draft_attn_impl,
            },
            "data": {
                "data_path": manifest["data_path"],
                "total_seq_len": cfg.training.token_budget,
                "hidden_states_dtype": cfg.hidden_states.dtype,
                "noise_std": cfg.training.noise_std,
                "num_workers": 0,
            },
            "generation": {"on_missing": "raise"},
            "backend": {"hidden_states_path": manifest["hidden_states_path"]},
            "optimizer": {
                "optimizer": "adamw",
                "lr": cfg.training.warmup_lr,
                "weight_decay": cfg.training.weight_decay,
            },
            "scheduler": {"scheduler_type": "none"},
            "lora": {
                "lora_r": cfg.lora.rank,
                "lora_alpha": cfg.lora.alpha,
                "lora_dropout": cfg.lora.dropout,
                "lora_target_modules": cfg.lora.target_modules,
                "lora_save_merged": False,
            },
            "bank": {
                "bank_manifest": str(root / "manifest.json"),
                "bank_subset_id": subset_id,
                "bank_warmup_epochs": cfg.training.warmup_epochs,
                "bank_collect_epochs": cfg.training.collect_epochs,
                "bank_collect_lr": cfg.training.collect_lr,
                "bank_save_interval": cfg.training.save_interval,
            },
            "trainer": {
                "epochs": cfg.training.warmup_epochs + cfg.training.collect_epochs,
                "save_path": str(run),
                "checkpoint_freq": 1,
                "log_freq": cfg.training.log_freq,
                "gradient_checkpointing": cfg.training.gradient_checkpointing,
            },
            "loss": {
                "ttt_steps": cfg.training.ttt_steps,
                "ttt_step_loss_decay": cfg.training.ttt_step_loss_decay,
                "loss_fn": cfg.training.loss_fn,
            },
        }
    }


def train(cfg: BankConfig):
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    manifest = select_data(cfg)
    validate_cache(manifest, cfg)
    paths = {
        role: snapshot_download(
            getattr(cfg.models, role), revision=manifest["revisions"][f"{role}_sha"]
        )
        for role in ("target", "drafter")
    }
    for subset in manifest["subsets"]:
        for seed in cfg.training.seeds:
            settings = train_config(
                cfg, manifest, subset["id"], seed, paths["target"], paths["drafter"]
            )
            run = Path(settings["train"]["trainer"]["save_path"])
            run.mkdir(parents=True, exist_ok=True)
            config_path = run / "bank_train.yaml"
            text = yaml.safe_dump(settings, sort_keys=False)
            if config_path.exists() and config_path.read_text() != text:
                raise ValueError("Run settings changed; choose a new output root")
            config_path.write_text(text)
            command = [
                sys.executable,
                "-m",
                "speculators.train",
                "--config",
                str(config_path),
            ]
            if cfg.execution.processes > 1:
                command = [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc-per-node={cfg.execution.processes}",
                    *command[2:],
                ]
            subprocess.run(command, check=True)  # noqa: S603 -- argv, never a shell
    return inspect_bank(cfg)


def bank_context(train_cfg):
    manifest, subset = load_subset(
        Path(train_cfg.bank.bank_manifest), train_cfg.bank.bank_subset_id
    )
    values = train_cfg.flatten()
    # Operational resume switches do not change the experiment identity.
    values.pop("no_resume_from_checkpoint", None)
    context = {
        "cache_fingerprint": manifest["cache_fingerprint"],
        "prepared_fingerprint": manifest["prepared_fingerprint"],
        "models": manifest["cache_spec"]["models"],
        "category": manifest["category"],
        "subset_id": subset["id"],
        "membership_sha256": digest(subset),
        "seed": train_cfg.seed,
        "rank": train_cfg.lora.lora_r,
        "alpha": train_cfg.lora.lora_alpha,
        "target_modules": train_cfg.lora.lora_target_modules,
        "train_config_sha256": digest(values),
    }
    return manifest, subset, context


def inspect_bank(cfg: BankConfig):
    from speculators.bank.inspect import inspect_adapters  # noqa: PLC0415

    manifest = select_data(cfg)
    root = Path(manifest["root"])
    entries, runs = [], []
    for subset in manifest["subsets"]:
        for seed in cfg.training.seeds:
            run = root / "runs" / subset["id"] / f"seed-{seed}"
            diagnostics = inspect_adapters(run, cfg.lora)
            entries.extend(diagnostics.pop("entries"))
            runs.append({"subset_id": subset["id"], "seed": seed, **diagnostics})
    report = {
        "cache_fingerprint": manifest["cache_fingerprint"],
        "runs": runs,
        "entries": entries,
        "candidate_count": len(entries),
        "unique_training_examples": sum(
            len(s["train_indices"]) for s in manifest["subsets"]
        ),
        "tokens": manifest.get("tokens"),
        "supervised_tokens": manifest.get("supervised_tokens"),
        "conditioning": cfg.conditioning.model_dump(),
    }
    write_json(root / "bank.json", report)
    return report
