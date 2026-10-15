"""Read-only pinned restoration of campaign metadata and first-epoch adapters."""

import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from speculators.bank.artifacts import digest, write_json
from speculators.bank.config import BankConfig
from speculators.bank.hub import fetch, prepare
from speculators.generator.campaign_config import DOMAINS, SNAPSHOTS_PER_SUBSET


def campaign_banks(cfg):
    banks = {}
    for path in [cfg.bank_config, *cfg.additional_bank_configs]:
        bank = BankConfig.load(path).model_copy(deep=True)
        if bank.hub is None or bank.hub.trajectories_revision == "main":
            raise ValueError("Campaign requires pinned consumer bank configs")
        domain = bank.dataset.category
        if domain in banks:
            raise ValueError(f"Duplicate campaign category: {domain}")
        bank.hub.mode = "consumer"
        bank.hub.selection_root = cfg.input_root / "selection"
        bank.execution.activation_local_dir = None
        bank.output_root = cfg.input_root / domain
        banks[domain] = bank
    if set(banks) != set(DOMAINS):
        raise ValueError(f"Campaign requires categories {DOMAINS}")
    first = next(iter(banks.values()))
    for bank in banks.values():
        if (
            bank.hub.trajectories_repo,
            bank.hub.trajectories_revision,
            bank.hub.banks_repo,
            bank.hub.bank_release,
        ) != (
            first.hub.trajectories_repo,
            first.hub.trajectories_revision,
            first.hub.banks_repo,
            first.hub.bank_release,
        ):
            raise ValueError(
                "Campaign bank configs must share pinned data and bank releases"
            )
    return banks


def read_record(bank, subset, seed, revision):
    prefix = (
        f"releases/{bank.hub.bank_release}/{bank.dataset.category}/{subset}/seed-{seed}"
    )
    record = json.loads(
        Path(
            hf_hub_download(
                bank.hub.banks_repo,
                prefix + "/completion.json",
                repo_type="dataset",
                revision=revision,
            )
        ).read_text()
    )
    return prefix, record


def restore_targets(cfg, bank, manifest):  # noqa: C901 -- pinned metadata and minimal portable restore
    records = []
    for subset in manifest["subsets"]:
        if subset["id"] not in [f"{bank.dataset.category}-{i:05d}" for i in range(4)]:
            continue
        seed = subset.get("training_seed")
        if seed is None:
            raise ValueError("Campaign bank subsets require catalog-assigned seeds")
        prefix, record = read_record(bank, subset["id"], seed, cfg.bank_revision)
        identity = record["identity"]
        for key in ("prepared_fingerprint", "cache_fingerprint", "revisions"):
            if identity[key] != manifest[key]:
                raise ValueError(f"Published bank source differs: {subset['id']}/{key}")
        if identity["subset"] != subset or identity["lora"] != bank.lora.model_dump(
            mode="json"
        ):
            raise ValueError(
                f"Published bank membership or LoRA differs: {subset['id']}"
            )
        first_epoch = identity["training"]["warmup_epochs"]
        entries = []
        # Read all small entry files; download weights only for the first epoch.
        for name, metadata in record["files"].items():
            if "/snapshots/" not in name or not name.endswith("/entry.json"):
                continue
            entry_path = (bank.output_root / name).resolve()
            if not entry_path.is_relative_to(bank.output_root.resolve()):
                raise ValueError("Published snapshot path escapes input directory")
            fetch(
                {
                    "repo": bank.hub.banks_repo,
                    "revision": cfg.bank_revision,
                    "path": prefix + "/" + name,
                    **metadata,
                },
                entry_path,
            )
            entry = json.loads(entry_path.read_text())
            if entry["epoch"] != first_epoch:
                entry_path.unlink()
                continue
            directory = str(Path(name).parent) + "/"
            for member, details in record["files"].items():
                if member.startswith(directory):
                    destination = (bank.output_root / member).resolve()
                    if not destination.is_relative_to(bank.output_root.resolve()):
                        raise ValueError("Published bank path escapes input directory")
                    fetch(
                        {
                            "repo": bank.hub.banks_repo,
                            "revision": cfg.bank_revision,
                            "path": prefix + "/" + member,
                            **details,
                        },
                        destination,
                    )
            entries.append(entry)
        if len(entries) != SNAPSHOTS_PER_SUBSET:
            raise ValueError(f"Expected ten first-epoch adapters for {subset['id']}")
        records.append(
            {
                "subset": subset["id"],
                "record_identity": digest(identity),
                "bank_revision": cfg.bank_revision,
                "training": identity["training"],
            }
        )
    write_json(bank.output_root / "campaign_bank_sources.json", records)


def prepare_inputs(cfg):
    banks = campaign_banks(cfg)
    cfg.input_root.mkdir(parents=True, exist_ok=True)
    for domain, bank in banks.items():
        manifest = prepare(bank, "hidden")
        for number in range(6):
            name = f"{domain}-{number:05d}"
            subset = next((s for s in manifest["subsets"] if s["id"] == name), None)
            if subset is None:
                raise ValueError(f"Missing campaign subset {name}")
            for split, count in (
                ("train", cfg.expected_train_examples),
                ("validation", cfg.expected_validation_examples),
            ):
                if len(subset[f"{split}_indices"]) != count:
                    raise ValueError(f"Unexpected {split} count for {name}")
        restore_targets(cfg, bank, manifest)
        write_json(cfg.input_root / f"bank_{domain}.json", bank.model_dump(mode="json"))
    return banks
