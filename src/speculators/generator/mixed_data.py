"""Combine metadata and route activation reads, retaining subset boundaries."""

import bisect
import json
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

from speculators.bank.artifacts import load_subset
from speculators.bank.config import BankConfig
from speculators.bank.transfer import BankFileTransfer


class MixedTransfer:
    direct_reads = True

    def __init__(self, sources):
        self.offsets = [offset for offset, _ in sources]
        self.transfers = [transfer for _, transfer in sources]
        self.hub_store = self if all(t.hub_store for t in self.transfers) else None

    def locate(self, index):
        source = bisect.bisect_right(self.offsets, index) - 1
        if source < 0:
            raise IndexError(index)
        return self.transfers[source], index - self.offsets[source]

    def read(self, index, prompt_length=None, summary=False):
        transfer, local_index = self.locate(index)
        return transfer.hub_store.read(local_index, prompt_length, summary)

    def get_cached(self, index):
        transfer, local_index = self.locate(index)
        return transfer.get_cached(local_index)

    def get_prompt_states(self, index, prompt_length):
        transfer, local_index = self.locate(index)
        return transfer.get_prompt_states(local_index, prompt_length)


def mixed_inputs(cfg, primary_bank):  # noqa: C901 -- compatibility and row remapping across banks
    banks = [primary_bank, *map(BankConfig.load, cfg.additional_bank_configs)]
    manifests = [
        json.loads((b.output_root / "manifest.json").read_text()) for b in banks
    ]
    primary = manifests[0]
    for bank, manifest in zip(banks[1:], manifests[1:], strict=True):
        if (
            any(
                manifest["revisions"][key] != primary["revisions"][key]
                for key in ("target_sha", "drafter_sha")
            )
            or bank.hidden_states != primary_bank.hidden_states
            or bank.lora != primary_bank.lora
            or bank.training.ttt_steps != primary_bank.training.ttt_steps
            or bank.training.noise_std != primary_bank.training.noise_std
            or bank.training.ttt_step_loss_decay
            != primary_bank.training.ttt_step_loss_decay
            or bank.execution.draft_attn_impl != primary_bank.execution.draft_attn_impl
            or (
                cfg.optimization.loss_fn is None
                and bank.training.loss_fn != primary_bank.training.loss_fn
            )
        ):
            raise ValueError(
                "Mixed banks require compatible models, activations, LoRAs and losses"
            )
    available = {}
    for bank, manifest in zip(banks, manifests, strict=True):
        for subset in manifest["subsets"]:
            if subset["id"] in available:
                raise ValueError(f"Duplicate subset across banks: {subset['id']}")
            available[subset["id"]] = (bank, subset)
    requested = cfg.experiment_subsets or cfg.pretraining_subsets
    selected = (
        [name for name, (_, s) in available.items() if s.get("role", "bank") == "bank"]
        if requested == "all"
        else requested or []
    )
    unknown = set(selected) - available.keys()
    if unknown:
        raise ValueError(f"Unknown pretraining subsets: {sorted(unknown)}")
    if cfg.experiment_subsets is None and any(
        available[name][1].get("role", "bank") != "bank" for name in selected
    ):
        raise ValueError("Reserved adaptation subsets cannot enter bank pretraining")
    needed = set(selected)
    subsets, datasets, transfers, sources, roots = {}, [], [], [], {}
    offset = 0
    for bank, manifest in zip(banks, manifests, strict=True):
        rows = load_from_disk(manifest["data_path"]).with_format(None)
        transfer = BankFileTransfer(
            Path(manifest["hidden_states_path"]),
            cfg.activation_local_dir or bank.execution.activation_local_dir,
        )
        transfers.append((offset, transfer))
        datasets.append(rows)
        for original in manifest["subsets"]:
            name = original["id"]
            if name not in needed:
                continue
            _, subset = load_subset(bank.output_root / "manifest.json", name)
            subsets[name] = {
                **subset,
                **{
                    key: [i + offset for i in subset[key]]
                    for key in ("train_indices", "validation_indices")
                },
            }
            roots[name] = bank.output_root
        sources.append(
            {
                "prepared_fingerprint": manifest["prepared_fingerprint"],
                "cache_spec": manifest["cache_spec"],
                "offset": offset,
            }
        )
        offset += len(rows)
    if needed - subsets.keys():
        raise ValueError(
            f"Missing experiment subsets: {sorted(needed - subsets.keys())}"
        )
    return {
        "subsets": subsets,
        "rows": concatenate_datasets(datasets),
        "transfer": MixedTransfer(transfers),
        "sources": sources,
        "pretraining_subsets": selected,
        "bank_roots": roots,
    }
