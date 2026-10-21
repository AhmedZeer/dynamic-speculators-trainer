"""Pinned external categories, prompt formatting, and frozen workload membership."""

import json
from types import SimpleNamespace

import pytest

from speculators.generator.serving import external_validation as external
from speculators.generator.serving.endpoint import selected_rows
from speculators.generator.serving.validation import load_validation


@pytest.fixture
def external_inputs(tmp_path, monkeypatch):
    cfg = SimpleNamespace(online=SimpleNamespace(target_model_path="local-tokenizer"))
    bank = SimpleNamespace(hidden_states=SimpleNamespace(sequence_length=8192))
    metadata = tmp_path / "data.json"
    metadata.write_text(
        json.dumps(
            {
                "manifest": {
                    "cache_spec": {
                        "models": {"target": "target", "drafter": "drafter"}
                    },
                    "revisions": {"target_sha": "a" * 40, "drafter_sha": "b" * 40},
                }
            }
        )
    )
    monkeypatch.setattr(external, "campaign_banks", lambda _: {"math": bank})
    monkeypatch.setattr(external, "trajectory_record_path", lambda *_: metadata)
    for name in external.FILES:
        rows = [
            {
                "prompt": f"{name} prompt {i}",
                "question_id": i,
                "category": "math" if i % 2 else "writing",
            }
            for i in range(6)
        ]
        if name == "HumanEval":
            rows.append({"prompt": "x" * 8192})
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    downloads = []

    def download(repo, name, **kwargs):
        assert repo == external.REPO
        assert kwargs == {"repo_type": "dataset", "revision": external.REVISION}
        downloads.append(name)
        return str(tmp_path / name)

    class Tokenizer:
        def encode(self, text, **kwargs):
            assert kwargs == {"add_special_tokens": False}
            return list(text.encode())

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return "<user>" + messages[0]["content"] + "<assistant>"

    monkeypatch.setattr(external, "hf_hub_download", download)
    monkeypatch.setattr(
        external.AutoTokenizer, "from_pretrained", lambda *_: Tokenizer()
    )
    return cfg, downloads


def test_external_export_unique_groups_and_formatting(external_inputs, tmp_path):
    cfg, downloads = external_inputs
    result = external.export_external_validation(
        cfg, tmp_path / "export", max_samples=2
    )
    assert result["examples"] == 16
    manifest, groups = load_validation(tmp_path / "export/manifest.json")
    assert len(groups) == 8
    assert "writing.jsonl" not in downloads
    assert manifest["dataset"]["revision"] == external.REVISION
    assert len({row["source_category"] for row in groups["specbench-question"]}) == 2
    assert all(row["prompt_format"] == "raw" for row in groups["specbench-humaneval"])
    assert all(row["prompt"].startswith("<user>") for row in groups["specbench-qa"])
    assert manifest["sources"][0]["excluded_context_rows"] == 1
    assert set(selected_rows(groups, categories=["specbench-qa"])) == {"specbench-qa"}
    again = tmp_path / "again"
    external.export_external_validation(cfg, again, max_samples=2)
    assert load_validation(again / "manifest.json")[1] == groups


def test_external_aliases_and_raw_prompts(external_inputs, tmp_path):
    cfg, downloads = external_inputs
    result = external.export_external_validation(
        cfg,
        tmp_path / "raw",
        categories=["question", "writing"],
        prompt_format="raw",
        max_samples=2,
    )
    assert result["examples"] == 2
    assert downloads == ["question.jsonl"]
    _, groups = load_validation(tmp_path / "raw/manifest.json")
    assert all(row["prompt_format"] == "raw" for row in groups["specbench-question"])
    with pytest.raises(ValueError, match="Unknown"):
        external.export_external_validation(cfg, tmp_path / "bad", categories=["math"])


def test_cli_selects_redhat_exporter(external_inputs, tmp_path, monkeypatch):
    from typer.testing import CliRunner  # noqa: PLC0415

    from speculators.cli import app  # noqa: PLC0415
    from speculators.generator.campaign_config import CampaignConfig  # noqa: PLC0415

    cfg, _ = external_inputs
    config = tmp_path / "config.yaml"
    config.write_text("{}")
    monkeypatch.setattr(CampaignConfig, "load", lambda _: cfg)
    result = CliRunner().invoke(
        app,
        [
            "generator",
            "export-validation",
            "--config",
            str(config),
            "--output",
            str(tmp_path / "cli"),
            "--dataset",
            external.REPO,
            "--categories",
            "HumanEval,qa",
            "--max-samples",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["examples"] == 4
    _, groups = load_validation(tmp_path / "cli/manifest.json")
    assert set(groups) == {"specbench-humaneval", "specbench-qa"}
