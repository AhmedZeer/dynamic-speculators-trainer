"""Freeze inspected Red Hat benchmark files into the HTTP workload format."""

import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.campaign_inputs import campaign_banks
from speculators.generator.local_campaign import trajectory_record_path
from speculators.generator.serving.validation import sample_domain

REPO = "RedHatAI/speculator_benchmarks"
REVISION = "2ae86affa2cb97a972b7fc681dd51c04fbff083e"
FILES = (
    "HumanEval",
    "math_reasoning",
    "qa",
    "question",
    "rag",
    "summarization",
    "tool_call",
    "translation",
)


def export_external_validation(  # noqa: C901 -- explicit source and prompt checks
    cfg,
    output,
    *,
    tokenizer_path=None,
    max_tokens=256,
    categories=None,
    max_samples=50,
    seed=42,
    prompt_format="auto",
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new validation export directory: {output}")
    if max_tokens < 1 or max_samples < 0:
        raise ValueError("max_tokens must be positive and max_samples nonnegative")
    if prompt_format not in ("auto", "raw", "chat"):
        raise ValueError("prompt_format must be auto, raw, or chat")
    aliases = {name.lower(): name for name in FILES}
    # The inspected writing file is byte-identical to question/MT-Bench.
    aliases["writing"] = "question"
    selected = []
    for name in categories or FILES:
        key = name.removeprefix("specbench-").lower().replace("-", "_")
        if key not in aliases:
            raise ValueError(f"Unknown Red Hat benchmark category: {name}")
        if aliases[key] not in selected:
            selected.append(aliases[key])
    bank = campaign_banks(cfg)["math"]
    metadata = trajectory_record_path(cfg, bank)
    manifest = json.loads(metadata.read_text())["manifest"]
    models = {
        role: {
            "id": manifest["cache_spec"]["models"][role],
            "revision": manifest["revisions"][f"{role}_sha"],
        }
        for role in ("target", "drafter")
    }
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path or cfg.online.target_model_path
    )
    groups, sources, records = {}, [], {}
    for name in selected:
        path = Path(
            hf_hub_download(
                REPO, f"{name}.jsonl", repo_type="dataset", revision=REVISION
            )
        )
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        group = "specbench-" + name.lower().replace("_", "-")
        formatted = (
            "raw"
            if prompt_format == "auto" and name == "HumanEval"
            else ("chat" if prompt_format == "auto" else prompt_format)
        )
        available, excluded = [], 0
        for index, row in enumerate(rows):
            text = row["prompt"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Empty or non-text prompt: {name}/{index}")
            if formatted == "chat":
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            tokens = tokenizer.encode(text, add_special_tokens=False)
            cap = min(max_tokens, bank.hidden_states.sequence_length - len(tokens))
            if not tokens or cap < 1:
                excluded += 1
                continue
            source_category = row.get("category", name)
            source_id = row.get("task_id", row.get("question_id", index))
            available.append(
                {
                    "row_id": f"{group}/{index:05d}/{source_id}",
                    "subset": f"{group}-{str(source_category).replace('/', '_')}",
                    "category": group,
                    "dataset": REPO,
                    "source_category": source_category,
                    "source_id": source_id,
                    "prompt_format": formatted,
                    "prompt": text,
                    "prompt_token_ids": tokens,
                    "prompt_tokens_count": len(tokens),
                    "output_tokens_count": cap,
                }
            )
        chosen = sample_domain(available, max_samples, seed)
        if not chosen:
            raise ValueError(f"No prompts fit the context budget: {name}")
        records[group] = chosen
        sources.append(
            {
                "file": f"{name}.jsonl",
                "sha256": file_digest(path),
                "source_rows": len(rows),
                "eligible_rows": len(available),
                "excluded_context_rows": excluded,
                "selected_rows": len(chosen),
                "prompt_format": formatted,
            }
        )
    pending = output.with_name(output.name + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    try:
        for group, rows in records.items():
            path = pending / f"{group}.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            )
            groups[group] = {
                "file": path.name,
                "sha256": file_digest(path),
                "rows": len(rows),
            }
        identity = {
            "version": 1,
            "models": models,
            "dataset": {"id": REPO, "revision": REVISION},
            "model_reference_metadata_sha256": file_digest(metadata),
            "subsets": sorted(
                {row["subset"] for rows in records.values() for row in rows}
            ),
            "sources": sources,
            "max_tokens": max_tokens,
            "selection": {
                "max_samples_per_category": max_samples,
                "seed": seed,
                "prompt_format": prompt_format,
                "method": "seeded_source_category_round_robin",
            },
            "groups": groups,
        }
        write_json(
            pending / "manifest.json", {**identity, "identity": digest(identity)}
        )
        pending.rename(output)
    except BaseException:
        shutil.rmtree(pending)
        raise
    return {
        "directory": str(output),
        "examples": sum(g["rows"] for g in groups.values()),
        "groups": groups,
    }
