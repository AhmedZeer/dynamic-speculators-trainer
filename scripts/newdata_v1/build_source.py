"""Fold the 6-domain decontaminated prompt set into the trainer's 4 categories.

The generator trainer and the trajectory producer assume the four Nemotron
categories (math, code, stem, chat). The new prompt set has six domains, so we
build one private Hugging Face dataset whose splits are those four categories:

    math <- math
    stem <- science
    code <- coding  + formal_structured (SQL, Lean 4, TikZ), 50/50
    chat <- general_chat + persona_roleplay, 50/50

Inside each source domain rows are taken in the dataset's own diversity-first
``sample_order`` (stratified by sub-domain and topic cluster), so any prefix is
balanced. Two-domain categories interleave their sources one row at a time.
Only the dataset's ``train`` split is used; its validation/test splits stay
untouched for evaluation. The result is then consumed unchanged by
``speculators lora-bank prepare`` (see ``examples/eagle3-lora/newdata_v1``).
"""

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

SOURCE = "erdem-erdem/dyn-spec-train-1.4M-decontaminated"
CATEGORIES = {
    "math": ("math",),
    "code": ("coding", "formal_structured"),
    "stem": ("science",),
    "chat": ("general_chat", "persona_roleplay"),
}
COLUMNS = ["id", "domain", "sub_domain", "source_dataset", "messages", "license",
           "sample_order", "prompt_tokens"]


def valid_prompt(messages):
    """Mirror bank.workflow.prompt_messages: optional system + exactly one user."""
    roles = [m["role"] for m in messages]
    return (
        roles.count("user") == 1
        and set(roles) <= {"system", "user"}
        and all(isinstance(m["content"], str) and m["content"] for m in messages)
    )


def take(table, domain, count):
    rows = table.filter(pc.equal(table["domain"], domain))
    rows = rows.sort_by([("sample_order", "ascending")]).to_pylist()
    kept = [r for r in rows if valid_prompt(r["messages"])]
    if len(kept) < count:
        raise ValueError(f"{domain}: only {len(kept)} usable rows, need {count}")
    return kept[:count], len(rows) - len(kept)


def interleave(groups):
    out = []
    for index in range(max(len(g) for g in groups)):
        out.extend(g[index] for g in groups if index < len(g))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Folder with the source train-*.parquet shards")
    parser.add_argument("--per-category", type=int, default=27500,
                        help="25 subsets x (1,000 train + 100 validation)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--push", default=None, help="Private HF dataset repo id")
    args = parser.parse_args()

    shards = sorted(args.raw_dir.glob("train-*.parquet"))
    if not shards:
        raise SystemExit(f"No train-*.parquet in {args.raw_dir}")
    table = pq.ParquetDataset(shards).read(columns=COLUMNS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"source": SOURCE, "per_category": args.per_category, "categories": {}}
    for category, domains in CATEGORIES.items():
        share = [args.per_category // len(domains)] * len(domains)
        share[0] += args.per_category - sum(share)
        groups, dropped = [], {}
        for domain, count in zip(domains, share, strict=True):
            rows, bad = take(table, domain, count)
            groups.append(rows)
            dropped[domain] = bad
        rows = interleave(groups)
        records = [
            {
                "uuid": r["id"],
                "messages": [{"role": m["role"], "content": m["content"]}
                             for m in r["messages"]],
                "license": r["license"] or "",
                "domain": r["domain"],
                "sub_domain": r["sub_domain"],
                "source_dataset": r["source_dataset"],
                "prompt_tokens": r["prompt_tokens"],
            }
            for r in rows
        ]
        path = args.out_dir / f"{category}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        subs = {}
        for r in records:
            subs[r["sub_domain"]] = subs.get(r["sub_domain"], 0) + 1
        stats["categories"][category] = {
            "rows": len(records),
            "domains": dict(zip(domains, share, strict=True)),
            "dropped_unsupported": dropped,
            "sub_domains": dict(sorted(subs.items())),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(category, len(records), dropped, flush=True)
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    if args.push:
        from datasets import DatasetDict, load_dataset  # noqa: PLC0415

        splits = DatasetDict({
            c: load_dataset("json", data_files=str(args.out_dir / f"{c}.jsonl"),
                            split="train")
            for c in CATEGORIES
        })
        splits.push_to_hub(args.push, private=True)
        from huggingface_hub import HfApi  # noqa: PLC0415

        HfApi().upload_file(path_or_fileobj=str(args.out_dir / "stats.json"),
                            path_in_repo="stats.json", repo_id=args.push,
                            repo_type="dataset")
        print("pushed", args.push, flush=True)


if __name__ == "__main__":
    main()
