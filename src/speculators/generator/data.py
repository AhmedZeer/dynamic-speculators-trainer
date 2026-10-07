"""Compact prompt summaries and bounded, buffered bank reads."""

import random
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from datasets import load_from_disk
from safetensors.torch import load

from speculators.bank.artifacts import digest, load_subset, write_json
from speculators.bank.progress import report, stage_progress
from speculators.bank.transfer import BankFileTransfer
from speculators.generator.config import CONDITIONS
from speculators.models.attention import FLEX_ATTENTION_BLOCK_SIZE
from speculators.models.eagle3.data import shift_batch
from speculators.train.data import CollateFn


def atomic_torch_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    try:
        torch.save(value, pending)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


@torch.no_grad()
def prompt_summary(states, prompt_length, model):
    """Summarize before the response boundary, including exact frozen FC norms."""
    if not 1 <= prompt_length <= states.shape[0]:
        raise ValueError("Invalid prompt boundary")
    prompt = states[:prompt_length, :-1].flatten(1)
    raw_last, raw_mean = prompt[-1].float().clone(), prompt.float().mean(0)
    projected_sum = None
    projected_last = None
    # Keep projection activations bounded even for long prompts.
    for chunk in prompt.split(256):
        x = chunk.to(device=model.fc.weight.device, dtype=model.fc.weight.dtype)
        if model.input_norm is not None:
            x = model.input_norm(x)
        if model.fc_norm is not None:
            x = torch.cat(
                [
                    norm(part)
                    for norm, part in zip(
                        model.fc_norm, x.chunk(len(model.fc_norm), dim=-1), strict=True
                    )
                ],
                dim=-1,
            )
        projected = model.fc(x).float().cpu()
        value = projected.sum(0)
        projected_sum = value if projected_sum is None else projected_sum + value
        projected_last = projected[-1]
    return {
        "last": raw_last,
        "last_mean": torch.cat([raw_last, raw_mean]),
        "projected": torch.cat([projected_last, projected_sum / prompt_length]),
    }


class Corpus:
    def __init__(self, cfg, bank):
        self.cfg, self.bank = cfg, bank
        self.manifest_path = bank.output_root / "manifest.json"
        self.manifest, first = load_subset(self.manifest_path, cfg.bank_subset)
        _, second = load_subset(self.manifest_path, cfg.conditioning_subset)
        self.subsets = {first["id"]: first, second["id"]: second}
        if any(
            cfg.context.max_examples > len(s["train_indices"])
            for s in self.subsets.values()
        ):
            raise ValueError("Conditioning context exceeds subset training size")
        all_indices = [
            i
            for s in self.subsets.values()
            for key in ("train_indices", "validation_indices")
            for i in s[key]
        ]
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("Experiment subsets must have disjoint memberships")
        self.rows = load_from_disk(self.manifest["data_path"]).with_format(None)
        self.transfer = BankFileTransfer(Path(self.manifest["hidden_states_path"]))
        self.identity = {
            "version": 1,
            "prepared_fingerprint": self.manifest["prepared_fingerprint"],
            "cache_spec": self.manifest["cache_spec"],
            "revisions": self.manifest["revisions"],
            "subsets": self.subsets,
        }
        self.cache_key = digest(self.identity)
        self.local = cfg.staging_dir / self.cache_key
        self.summaries = None

    def indices(self, subset, validation=False):
        return list(
            self.subsets[subset][
                "validation_indices" if validation else "train_indices"
            ]
        )

    def prepare(self, model):
        path = self.local / "summaries.pt"
        remote = self.cfg.output_root / "prepared" / self.cache_key
        if not path.exists() and (remote / "summaries.pt").exists():
            with stage_progress("Restoring compact prompt-summary cache"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((remote / "summaries.pt").read_bytes())
        if path.exists():
            with stage_progress("Loading compact prompt-summary cache"):
                self.summaries = torch.load(path, map_location="cpu", weights_only=True)
            return
        indices = sorted(
            {
                i
                for s in self.subsets.values()
                for key in ("train_indices", "validation_indices")
                for i in s[key]
            }
        )
        values = {condition: [] for condition in CONDITIONS}
        lengths = []
        started = time.monotonic()
        read_bytes = 0
        workers = min(self.cfg.preprocessing_workers, len(indices))
        report(
            f"Prompt preprocessing: {workers} read workers; prompt-only buffered reads"
        )
        with stage_progress(f"Preparing prompt summaries for {len(indices)} examples"):
            for position, (index, length, states) in enumerate(
                self.prompt_states(indices), 1
            ):
                read_bytes += states.numel() * states.element_size()
                summaries = prompt_summary(states, length, model)
                for condition in CONDITIONS:
                    values[condition].append(summaries[condition])
                lengths.append(length)
                del states
                if position == 1 or position % 10 == 0 or position == len(indices):
                    elapsed = time.monotonic() - started
                    rate = position / max(elapsed, 1e-6)
                    eta = (len(indices) - position) / rate
                    report(
                        f"Prompt summaries: {position}/{len(indices)}; row={index}; "
                        f"{rate:.2f} examples/s; ETA={eta:.0f}s; "
                        f"prompt tensors={read_bytes / 2**30:.2f} GiB"
                    )
        self.summaries = {
            "indices": torch.tensor(indices),
            "lengths": torch.tensor(lengths),
            **{key: torch.stack(value) for key, value in values.items()},
        }
        atomic_torch_save(path, self.summaries)
        remote.mkdir(parents=True, exist_ok=True)
        # One copy after preparation; response payloads are never retained here.
        temporary = remote / "summaries.pt.pending"
        temporary.write_bytes(path.read_bytes())
        temporary.replace(remote / "summaries.pt")
        write_json(remote / "metadata.json", self.identity)

    def prompt_states(self, indices):
        """Bound outstanding reads; keep model/GPU work in the calling thread."""
        remaining = iter(indices)
        pending = deque()

        def submit(pool, index):
            length = self.rows[index]["prompt_length"]
            pending.append(
                (
                    index,
                    length,
                    pool.submit(self.transfer.get_prompt_states, index, length),
                )
            )

        with ThreadPoolExecutor(max_workers=self.cfg.preprocessing_workers) as pool:
            for _ in range(min(len(indices), self.cfg.preprocessing_workers)):
                submit(pool, next(remaining))
            while pending:
                index, length, future = pending.popleft()
                states = future.result()
                if states is None:
                    raise FileNotFoundError(f"Missing hidden-state row {index}")
                yield index, length, states
                del states, future
                index = next(remaining, None)
                if index is not None:
                    submit(pool, index)

    def context(self, indices, condition, device):
        if self.summaries is None:
            raise RuntimeError("Run generator prepare first")
        locations = torch.searchsorted(self.summaries["indices"], torch.tensor(indices))
        if torch.any(locations >= len(self.summaries["indices"])) or not torch.equal(
            self.summaries["indices"][locations], torch.tensor(indices)
        ):
            raise ValueError("Conditioning row is outside the prepared cache")
        return (
            self.summaries[condition][locations].to(device),
            self.summaries["lengths"][locations].to(device),
        )

    def attention_block_size(self):
        if self.bank.execution.draft_attn_impl != "simple_flex_attention":
            return 1
        if self.cfg.optimization.token_budget % FLEX_ATTENTION_BLOCK_SIZE:
            raise ValueError(
                "FlexAttention token_budget must be a multiple of "
                f"{FLEX_ATTENTION_BLOCK_SIZE}"
            )
        return FLEX_ATTENTION_BLOCK_SIZE

    def microbatches(self, indices, device, dtype, training=False):
        chunks, current, length = [], [], 0
        budget = self.cfg.optimization.token_budget
        block_size = self.attention_block_size()
        for index in indices:
            size = self.rows[index]["seq_len"] - 1
            if size < 1 or size > budget:
                raise ValueError(f"Row {index} cannot fit token budget {budget}")
            if current and length + size > budget:
                chunks.append(current)
                current, length = [], 0
            current.append(index)
            length += size
        if current:
            chunks.append(current)
        for chunk in chunks:
            samples = []
            for index in chunk:
                row = self.rows[index]
                payload = self.transfer.get_cached(index)
                if payload is None:
                    raise FileNotFoundError(f"Missing hidden-state row {index}")
                ids = payload["token_ids"]
                states = payload["hidden_states"]
                sample = {
                    "input_ids": ids,
                    "hidden_states": states[:, :-1].flatten(1),
                    "verifier_last_hidden_states": states[:, -1],
                    "loss_mask": torch.tensor(row["loss_mask"], dtype=torch.bool),
                    "lengths": torch.tensor([len(ids)]),
                    "position_ids": torch.arange(len(ids)),
                }
                samples.append(shift_batch(sample))
            total = sum(len(s["input_ids"]) for s in samples)
            # Draft KV segments are concatenated at each TTT step. Their
            # starts must align to blocks for extend_mask_for_draft_tokens.
            total = (total + block_size - 1) // block_size * block_size
            width = samples[0]["verifier_last_hidden_states"].shape[-1]
            batch = CollateFn(total, width, dtype=dtype)(samples)
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            if training and self.bank.training.noise_std:
                x = batch["hidden_states"]
                batch["hidden_states"] = (
                    x + 2 * (torch.rand_like(x) - 0.5) * self.bank.training.noise_std
                )
            yield batch, chunk == chunks[-1]


def groups(indices, seed, minimum, maximum):
    rng = random.Random(seed)
    indices = list(indices)
    rng.shuffle(indices)
    offset = 0
    while offset < len(indices):
        count = rng.randint(minimum, maximum)
        yield indices[offset : offset + count]
        offset += count


def discover_snapshots(root, subset):
    result = []
    for path in sorted(
        (root / "runs" / subset).glob("seed-*/snapshots/step-*/entry.json")
    ):
        import json  # noqa: PLC0415

        entry = json.loads(path.read_text())
        adapter = path.parent / "adapter" / "adapter_model.safetensors"
        if not adapter.exists():
            raise FileNotFoundError(adapter)
        result.append(
            {
                "seed": int(path.parents[2].name.removeprefix("seed-")),
                "collection_step": entry["collection_step"],
                "path": str(adapter),
                "entry": entry,
            }
        )
    if not result:
        raise ValueError(f"No collected adapters for {subset}")
    return result


def filter_snapshots(entries, seeds, stride, collection_window=None):
    selected_runs = {seed: [e for e in entries if e["seed"] == seed] for seed in seeds}
    if any(not run for run in selected_runs.values()):
        raise ValueError(f"Missing bank seed in {seeds}")
    # Use the overlapping collection interval across the requested seeds.
    if collection_window is None:
        start = max(
            min(e["collection_step"] for e in run) for run in selected_runs.values()
        )
        stop = min(
            max(e["collection_step"] for e in run) for run in selected_runs.values()
        )
    else:
        start, stop = collection_window
    selected = [
        e
        for e in entries
        if e["seed"] in seeds
        and start <= e["collection_step"] <= stop
        and e["collection_step"] % stride == 0
    ]
    if any(not any(e["seed"] == seed for e in selected) for seed in seeds):
        raise ValueError(f"No checkpoints retained for every seed at stride {stride}")
    return selected


class FactorCache:
    def __init__(self, limit_mib, shapes, rank):
        self.limit = limit_mib * 1024**2
        self.shapes, self.rank = shapes, rank
        self.items = OrderedDict()
        self.size = 0

    def get(self, path):
        if path in self.items:
            self.items.move_to_end(path)
            return self.items[path]
        with stage_progress(f"Reading bank adapter {path}", quiet=True):
            state = load(Path(path).read_bytes())
        factors = {}
        for name, (in_features, out_features) in self.shapes.items():
            pair = []
            for suffix, shape in (
                ("A", (self.rank, in_features)),
                ("B", (out_features, self.rank)),
            ):
                matches = [
                    v
                    for key, v in state.items()
                    if key.endswith(f".{name}.lora_{suffix}.weight")
                ]
                if len(matches) != 1 or matches[0].shape != shape:
                    raise ValueError(f"Incompatible {name}/{suffix} in {path}")
                pair.append(matches[0])
            factors[name] = tuple(pair)
        size = sum(
            v.numel() * v.element_size() for pair in factors.values() for v in pair
        )
        if size <= self.limit:
            while self.items and self.size + size > self.limit:
                _, old = self.items.popitem(last=False)
                self.size -= sum(
                    v.numel() * v.element_size() for p in old.values() for v in p
                )
            self.items[path] = factors
            self.size += size
        return factors
