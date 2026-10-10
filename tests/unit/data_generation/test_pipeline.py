"""Real artifact pairing and recovery without a model download or GPU."""

import json
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock

import httpx
import openai
import pytest
import torch
from datasets import load_from_disk
from safetensors.torch import save_file

from speculators.bank.artifacts import digest, read_jsonl
from speculators.bank.config import BankConfig
from speculators.bank.workflow import partition_rows, prepare_arrow
from speculators.data_generation import pipeline
from speculators.data_generation.vllm_client import _handle_retry_error


def config():
    return pipeline.PrepareConfig.model_validate(
        {
            "subset_ids": [0, 1],
            "bank": {
                "partition": {"train_examples": 2, "validation_examples": 1},
                "conditioning": {"max_examples": 2},
            },
        }
    )


def rows(domain):
    shared = [
        {"id": f"shared-{i}", "messages": [{"role": "user", "content": f"Shared {i}"}]}
        for i in range(4)
    ]
    distinct = [
        {
            "id": f"{domain}-{i}",
            "messages": [{"role": "user", "content": f"{domain} prompt {i}"}],
        }
        for i in range(18)
    ]
    return shared + distinct + [{**distinct[0], "id": "alias"}]


def initialize(tmp_path, cfg=None):
    return pipeline.initialize(
        cfg or config(),
        tmp_path / "run",
        tmp_path / "models",
        {"git_sha": "test", "patch": ""},
        loader=rows,
        revisions={
            "target_sha": "target",
            "drafter_sha": "draft",
            "dataset_sha": "dataset",
        },
    )


def test_disk_selection_matches_existing_shuffle_and_aliases(tmp_path):
    bank = config().bank
    bank.partition.subset_ids = [0, 1]
    excluded = {digest([{"role": "user", "content": "Shared 0"}])}
    source = rows("math")
    expected, _, _ = partition_rows(
        (r for r in source if digest(pipeline.prompt_messages(r)) not in excluded), bank
    )
    actual, _ = pipeline.selected_rows(source, bank, excluded, tmp_path / "select.db")
    assert actual == expected


def test_full_selection_is_disjoint_and_indices_stay_fixed(tmp_path):
    result = initialize(tmp_path)
    assert [(e["domain"], e["subset"]) for e in result["banks"]] == [
        (d, s) for s in (0, 1) for d in pipeline.DOMAINS
    ]
    ids, indices = [], []
    for entry in result["banks"]:
        root = Path(entry["root"])
        prompts = read_jsonl(root / "prompts.jsonl")
        manifest = json.loads((root / "manifest.json").read_text())
        assert len(prompts) == 3
        assert manifest["subsets"][0]["train_indices"] == [0, 1]
        assert manifest["subsets"][0]["validation_indices"] == [2]
        ids.extend(p["id"] for p in prompts)
        indices.extend(manifest["global_indices"])
    assert len(ids) == len(set(ids)) == 24
    assert sorted(indices) == list(range(24))
    changed = config()
    changed.batch_size = 1
    changed.bank.execution.concurrency = 1
    changed.gpu_hours_limit = 1
    assert initialize(tmp_path, changed) == result
    changed.bank.responses.temperature = 0.2
    with pytest.raises(ValueError, match="new run_id"):
        initialize(tmp_path, changed)


@pytest.mark.parametrize("content", ["ordinary", "line\u2028separator\u0085here\u2029"])
def test_torn_response_recovery_preserves_unicode(tmp_path, content):
    path = tmp_path / "responses.jsonl"
    row = {"primary_id": "a", "text": content, "input_ids": [1, 2], "loss_mask": [0, 1]}
    path.write_text(
        json.dumps(row, ensure_ascii=False) + '\n{"primary_id": "b", "text": "torn',
        encoding="utf-8",
    )
    assert pipeline.repair_responses(path, {"a", "b"}) == [row]
    assert read_jsonl(path) == [row]


@pytest.mark.parametrize(
    "payload",
    [
        '{"primary_id":"a"}\n{"broken":\n',
        '{"primary_id":"a"}\n{"primary_id":"a"}\n',
        '{"primary_id":"unknown"}\n',
    ],
)
def test_corruption_and_duplicate_identity_fail(tmp_path, payload):
    path = tmp_path / "responses.jsonl"
    path.write_text(payload)
    with pytest.raises(ValueError):
        pipeline.repair_responses(path, {"a"})


def test_server_command_reserves_output_token_and_provenance(tmp_path):
    bank = config().bank
    bank.output_root = tmp_path
    response = pipeline.server_command(
        "/repo", bank, "/models/target", "responses", tmp_path
    )
    hidden = pipeline.server_command(
        "/repo", bank, "/models/target", "hidden", tmp_path
    )
    assert response[response.index("--max-model-len") + 1] == "8192"
    assert hidden[hidden.index("--max-model-len") + 1] == "8193"
    assert "--provenance-dir" in response
    assert "--provenance-dir" in hidden
    assert hidden[
        hidden.index("--target-layer-ids") + 1 : hidden.index("--hidden-states-path")
    ] == ["2", "18", "33", "36"]


def test_subprocess_failure_cleans_group_and_commits():
    commit = Mock()
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_child(
            [sys.executable, "-c", "raise SystemExit(3)"], time.monotonic() + 10, commit
        )
    commit.assert_called_once()


def test_stop_process_stops_descendants(monkeypatch):
    process = Mock(pid=123)
    kill = Mock()
    monkeypatch.setattr(pipeline.os, "killpg", kill)
    pipeline.stop_process(process)
    assert [c.args for c in kill.call_args_list] == [
        (123, signal.SIGTERM),
        (123, signal.SIGKILL),
    ]


def test_subset_pairs_real_arrow_and_hidden_artifacts_and_resumes(
    tmp_path, monkeypatch
):
    cfg = config()
    cfg.batch_size = 1
    selection = initialize(tmp_path, cfg)
    directory = Path(selection["banks"][0]["root"])
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"hidden_size": 2}')
    events = []

    @contextmanager
    def server(command, _log, _deadline):
        mode = command[2]
        events.append("start-" + mode)
        yield
        events.append("stop-" + mode)

    def child(command, _deadline, commit):
        stage = command[3]
        bank = BankConfig.load(Path(command[command.index("--config") + 1]))
        manifest = json.loads((directory / "manifest.json").read_text())
        if stage == "responses":
            prompts = read_jsonl(Path(command[command.index("--prompts") + 1]))
            with (directory / "responses.jsonl").open("a") as stream:
                for row in prompts:
                    stream.write(
                        json.dumps(
                            {
                                "primary_id": row["id"],
                                "input_ids": [11, 12, 13],
                                "loss_mask": [0, 0, 1],
                                "metadata": {"text": "math\u2028code\u0085stem"},
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        elif stage == "data":
            prepare_arrow(bank, manifest)
        else:
            hidden = directory / "hidden_states"
            hidden.mkdir(exist_ok=True)
            data = load_from_disk(manifest["data_path"]).with_format(None)
            for index, row in enumerate(data):
                save_file(
                    {
                        "token_ids": torch.tensor(row["input_ids"]),
                        "hidden_states": torch.zeros(3, 4, 2, dtype=torch.bfloat16),
                    },
                    hidden / f"hs_{index}.safetensors",
                )
        commit()

    monkeypatch.setattr(pipeline, "server", server)
    monkeypatch.setattr(pipeline, "run_child", child)
    commit = Mock()
    first = pipeline.prepare_subset(
        cfg, directory, model, "/repo", time.monotonic() + 100, commit=commit
    )
    assert first["complete"]
    assert first["hidden_states"] == first["responses"] == 3
    assert events == ["start-responses", "stop-responses", "start-train", "stop-train"]
    prepared = load_from_disk(str(directory / "data"))
    assert [r["primary_id"] for r in prepared] == [
        r["id"] for r in read_jsonl(directory / "prompts.jsonl")
    ]
    assert (
        json.loads((directory / "metrics.json").read_text())["responses"][
            "generated_tokens"
        ]
        == 3
    )
    events.clear()
    second = pipeline.prepare_subset(
        cfg, directory, model, "/repo", time.monotonic() + 100, commit=commit
    )
    assert second == first
    assert events == []
    (directory / "hidden_states/hs_1.safetensors").unlink()
    assert not pipeline.bank_status(directory)["complete"]


@pytest.mark.parametrize(
    ("status", "retry"),
    [(400, False), (404, False), (408, True), (425, True), (429, True), (500, True)],
)
def test_http_errors_only_retry_when_transient(status, retry):
    request = httpx.Request("POST", "http://localhost/completions")
    response = httpx.Response(status, request=request)
    error = openai.APIStatusError("failure", response=response, body={})
    if retry:
        assert _handle_retry_error(error, 1, 2) == 2
    else:
        with pytest.raises(openai.APIStatusError):
            _handle_retry_error(error, 1, 2)
