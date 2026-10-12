import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import datasets
import huggingface_hub
import pytest
import torch
import yaml
from datasets import load_from_disk
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import save_file

from speculators.bank import hub, workflow
from speculators.bank.artifacts import read_jsonl
from speculators.bank.config import BankConfig
from speculators.bank.transfer import BankFileTransfer
from speculators.cli import lora_bank


def test_preparation_progress_setting_is_scoped(preparation, monkeypatch):
    cfg, _, _, _ = preparation
    disabled = [False]
    monkeypatch.setattr(hub, "are_progress_bars_disabled", lambda: disabled[0])
    monkeypatch.setattr(
        hub, "disable_progress_bars", lambda: disabled.__setitem__(0, True)
    )
    monkeypatch.setattr(
        hub, "enable_progress_bars", lambda: disabled.__setitem__(0, False)
    )

    def interrupt():
        with hub.hub_progress(cfg):
            assert disabled[0]
            raise RuntimeError("interrupted upload")

    with pytest.raises(RuntimeError):
        interrupt()
    assert not disabled[0]
    disabled[0] = True
    with hub.hub_progress(cfg):
        assert disabled[0]
    assert disabled[0]


generate_offline_data = importlib.import_module("speculators.cli.generate_offline_data")


def test_serve_uses_frozen_target_and_server_provenance(
    preparation, tmp_path, monkeypatch
):
    cfg, remote, requests, extractions = preparation
    workflow.prepare(cfg, "select")
    config = tmp_path / "bank.yaml"
    config.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    resolved = []

    def snapshot(model, revision):
        resolved.append((model, revision))
        return str(tmp_path / "models" / revision)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    commands = []
    monkeypatch.setattr(
        lora_bank.subprocess, "run", lambda command, **kwargs: commands.append(command)
    )
    lora_bank.serve(config, mode="hidden", port=8000)
    command = commands[0]
    assert resolved == [(cfg.models.target, "target")]
    assert command[2] == "train"
    assert "--provenance-dir" in command
    assert command[command.index("--served-model-name") + 1] == cfg.models.target
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert command[
        command.index("--target-layer-ids") + 1 : command.index("--include-last-layer")
    ] == ["2", "18", "33"]


class FakeHub:
    def __init__(self, root):
        self.root, self.repos, self.history, self.commits = root, {}, {}, 0
        self.fail = None

    def create_repo(self, repo, **kwargs):
        assert kwargs["private"] is False
        self.repos.setdefault(repo, ("0" * 40, {}))

    def repo_info(self, repo, **kwargs):
        return SimpleNamespace(private=False, sha=self.repos[repo][0])

    def model_info(self, model, **kwargs):
        return SimpleNamespace(sha="target" if model.startswith("Qwen/") else "draft")

    def dataset_info(self, *args, **kwargs):
        return SimpleNamespace(sha="source")

    def list_repo_files(self, repo, **kwargs):
        return list(self.repos[repo][1])

    def create_commit(self, repo, operations, **kwargs):
        parent, files = self.repos[repo]
        assert kwargs["parent_commit"] == parent
        if self.fail and any(
            self.fail in operation.path_in_repo for operation in operations
        ):
            self.fail = None
            raise OSError("interrupted upload")
        files = dict(files)
        for operation in operations:
            source = operation.path_or_fileobj
            files[operation.path_in_repo] = (
                Path(source).read_bytes() if isinstance(source, (str, Path)) else source
            )
        self.commits += 1
        revision = f"{self.commits:040x}"
        self.repos[repo] = revision, files
        self.history[repo, revision] = files
        return SimpleNamespace(oid=revision)

    def download(self, repo, path, revision="main", local_dir=None, **kwargs):
        revision = self.repos[repo][0] if revision == "main" else revision
        files = self.history.get((repo, revision), self.repos[repo][1])
        if path not in files:
            raise EntryNotFoundError(path)
        target = (
            Path(local_dir) / path if local_dir else self.root / repo / revision / path
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[path])
        return str(target)


@pytest.fixture
def preparation(tmp_path, monkeypatch):
    remote = FakeHub(tmp_path / "remote")
    monkeypatch.setattr(hub, "HfApi", lambda: remote)
    monkeypatch.setattr(hub, "hf_hub_download", remote.download)
    monkeypatch.setattr(hub, "provenance", lambda *args: [])

    def source(dataset_name, **kwargs):
        domain = kwargs["split"]
        rows = [
            {
                "uuid": f"{domain}-{i}",
                "license": "cc-by-4.0",
                "messages": [{"role": "user", "content": f"{domain} example {i}"}],
            }
            for i in range(8)
        ]
        rows.append(
            {
                "uuid": f"{domain}-shared",
                "license": "cc-by-4.0",
                "messages": [{"role": "user", "content": "shared"}],
            }
        )
        return rows

    monkeypatch.setattr(datasets, "load_dataset", source)
    requests = []

    def regenerate(cfg, manifest, prompts_path):
        path = cfg.output_root / "responses.jsonl"
        existing = (
            {r["primary_id"] for r in read_jsonl(path)} if path.exists() else set()
        )
        with path.open("a") as stream:
            for prompt in read_jsonl(prompts_path):
                if prompt["id"] in existing:
                    continue
                requests.append(prompt["id"])
                stream.write(
                    json.dumps(
                        {
                            "primary_id": prompt["id"],
                            "input_ids": [1, 2, 3],
                            "loss_mask": [0, 0, 1],
                            "metadata": {"finish_reason": "stop"},
                        }
                    )
                    + "\n"
                )

    monkeypatch.setattr(workflow, "regenerate", regenerate)
    extractions = []

    def extract(**kwargs):
        rows = load_from_disk(kwargs["preprocessed_data"])
        root = Path(kwargs["output"])
        root.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            if (root / f"hs_{i}.safetensors").exists():
                continue
            extractions.append(row["primary_id"])
            save_file(
                {
                    "token_ids": torch.tensor(row["input_ids"]),
                    "hidden_states": torch.arange(36, dtype=torch.bfloat16).reshape(
                        3, 4, 3
                    ),
                },
                root / f"hs_{i}.safetensors",
            )

    monkeypatch.setattr(generate_offline_data, "generate_offline_data", extract)
    cfg = BankConfig(
        output_root=tmp_path / "producer",
        partition={"train_examples": 2, "validation_examples": 1, "subset_ids": [0, 1]},
        conditioning={"max_examples": 2},
        hub={
            "selection_root": tmp_path / "selection",
            "domains": ["chat", "math"],
            "adaptation_subset_ids": [1],
            "response_chunk_size": 2,
            "extraction_batch_size": 2,
            "cache_gib": 0.001,
            "cache_reserve_gib": 0,
            "shard_bytes": 1024,
        },
    )
    return cfg, remote, requests, extractions


def test_public_pipeline_resume_and_relocatable_lazy_restore(preparation, tmp_path):
    cfg, remote, requests, extractions = preparation
    selection = workflow.prepare(cfg, "select")
    assert selection["subsets"][1]["role"] == "adaptation"
    remote.fail = "/responses/"
    with pytest.raises(OSError, match="interrupted"):
        workflow.prepare(cfg, "responses")
    assert len(requests) == 2
    workflow.prepare(cfg, "responses")
    assert len(requests) == 6
    workflow.prepare(cfg, "data")
    workflow.prepare(cfg, "hidden")
    assert len(extractions) == 6
    workflow.prepare(cfg, "hidden")
    assert len(extractions) == 6
    assert not list((cfg.output_root / "hidden_states").glob("hs_*.safetensors"))
    revision = remote.repo_info(cfg.hub.trajectories_repo).sha
    consumer = cfg.model_copy(deep=True)
    consumer.output_root = tmp_path / "consumer"
    consumer.hub.mode = "consumer"
    consumer.hub.trajectories_revision = revision
    manifest = workflow.prepare(consumer, "hidden")
    assert manifest["root"] == str(consumer.output_root)
    assert (
        manifest["prepared_fingerprint"]
        == json.loads((cfg.output_root / "manifest.json").read_text())[
            "prepared_fingerprint"
        ]
    )
    workflow.validate_cache(manifest, consumer)
    reader = BankFileTransfer(consumer.output_root / "hidden_states")
    assert not list(reader.hub_store.cache.glob("*.safetensors"))
    payload = reader.get_cached(0)
    assert payload["token_ids"].tolist() == [1, 2, 3]
    assert reader.get_prompt_states(0, 2).shape == (2, 4, 3)
    assert reader.hub_store.read(0, summary=True).shape == (9,)
    assert len(requests) == len(extractions) == 6


def test_remote_chunks_resume_after_losing_producer_disk(preparation):
    cfg, remote, requests, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    shutil.rmtree(cfg.output_root)
    workflow.prepare(cfg, "responses")
    assert len(requests) == 6
    assert len(read_jsonl(cfg.output_root / "responses.jsonl")) == 6


def test_immutable_paths_and_selection_compatibility(preparation):
    cfg, remote, requests, extractions = preparation
    workflow.prepare(cfg, "select")
    publisher = hub.Publisher(cfg)
    file = cfg.output_root / "payload"
    file.write_text("first")
    publisher.put(cfg.hub.trajectories_repo, "immutable", file)
    file.write_text("changed")
    with pytest.raises(ValueError, match="Immutable"):
        publisher.put(cfg.hub.trajectories_repo, "immutable", file)
    changed = cfg.model_copy(deep=True)
    changed.output_root = cfg.output_root / "other"
    changed.partition.seed = 2
    with pytest.raises(ValueError, match="Frozen selection changed"):
        workflow.prepare(changed, "select")


def test_frozen_catalog_survives_sampler_change(preparation, monkeypatch):
    cfg, remote, requests, extractions = preparation
    digest = hub.digest

    def old_digest(value):
        if isinstance(value, dict) and "selection_policy" in value:
            value = {k: v for k, v in value.items() if k != "selection_policy"}
        return digest(value)

    monkeypatch.setattr(hub, "digest", old_digest)
    original = workflow.prepare(cfg, "select")
    prompts = read_jsonl(cfg.output_root / "prompts.jsonl")
    workflow.prepare(cfg, "responses")
    workflow.prepare(cfg, "data")
    workflow.prepare(cfg, "hidden")
    generated = (len(requests), len(extractions))
    monkeypatch.setattr(hub, "digest", digest)
    shutil.rmtree(cfg.output_root)
    shutil.rmtree(cfg.hub.selection_root)

    def forbid_resampling(*args, **kwargs):
        pytest.fail("Frozen membership must not be resampled")

    monkeypatch.setattr(datasets, "load_dataset", forbid_resampling)
    resumed = workflow.prepare(cfg, "select")
    assert resumed["subsets"] == original["subsets"]
    assert read_jsonl(cfg.output_root / "prompts.jsonl") == prompts
    # Simulate a replacement GPU machine: recover trajectories and all sealed
    # activation batches without issuing another response/extraction request.
    shutil.rmtree(cfg.output_root)
    workflow.prepare(cfg, "hidden")
    assert (len(requests), len(extractions)) == generated


def test_hub_retains_same_text_in_each_domain(preparation, monkeypatch):
    cfg, remote, requests, extractions = preparation

    def repeated_source(*args, **kwargs):
        return [
            {
                "uuid": "same-source-id",
                "license": "cc-by-4.0",
                "messages": [{"role": "user", "content": "Repeated prompt"}],
            }
        ] * 6

    monkeypatch.setattr(datasets, "load_dataset", repeated_source)
    workflow.prepare(cfg, "select")
    catalog = json.loads((cfg.hub.selection_root / "catalog.json").read_text())
    chat = catalog["domains"]["chat"]["prompts"]
    math = catalog["domains"]["math"]["prompts"]
    assert len(chat) == len(math) == 6
    assert len({p["id"] for p in chat + math}) == 12
    assert len({p["prompt_hash"] for p in chat + math}) == 1


def test_hidden_shards_publish_together_in_one_commit(preparation, monkeypatch):
    cfg, remote, _, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    workflow.prepare(cfg, "data")
    hashes = {}
    file_digest = hub.file_digest

    def count_hashes(path):
        path = Path(path)
        if ".pending-extraction" in path.parts and path.suffix == ".safetensors":
            hashes[path] = hashes.get(path, 0) + 1
        return file_digest(path)

    monkeypatch.setattr(hub, "file_digest", count_hashes)
    repo = cfg.hub.activations_repo
    before = sum(key[0] == repo for key in remote.history)
    workflow.prepare(cfg, "hidden")
    after = sum(key[0] == repo for key in remote.history)
    # Four extraction batches, three objects each: one group commit + final index.
    assert after - before == 2
    assert len(extractions) == 6
    # One packaging checksum + one publication verification per file, regardless
    # of how many examples share the shard.
    assert hashes
    assert set(hashes.values()) == {2}
    paths = remote.list_repo_files(repo)
    assert len([p for p in paths if p.endswith("-complete.json")]) == 4
    index = json.loads((cfg.output_root / "hidden_states" / "index.json").read_text())
    for row in index["rows"].values():
        for kind in ("shard", "summary"):
            assert len(row[kind]["revision"]) == 40
            assert Path(
                remote.download(repo, row[kind]["path"], revision=row[kind]["revision"])
            ).exists()


def test_old_separately_published_shard_is_reused(preparation):
    cfg, remote, _, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    manifest = workflow.prepare(cfg, "data")
    data = load_from_disk(manifest["data_path"]).with_format(None)
    primary = data[0]["primary_id"]
    publisher = hub.Publisher(cfg)
    local = cfg.output_root / "old-shard.safetensors"
    save_file(
        {
            f"{primary}/token_ids": torch.tensor(data[0]["input_ids"]),
            f"{primary}/hidden_states": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
        },
        local,
    )
    location = f"{hub.profile(cfg)}/math-00000/train/old"
    shard = publisher.put(cfg.hub.activations_repo, location + ".safetensors", local)
    save_file({primary: torch.zeros(9)}, local)
    summary = publisher.put(
        cfg.hub.activations_repo, location + "-conditioning.safetensors", local
    )
    publisher.metadata(
        cfg.hub.activations_repo,
        location + "-complete.json",
        {
            "prepared_fingerprint": manifest["prepared_fingerprint"],
            "rows": {"0": {"primary_id": primary, "shard": shard, "summary": summary}},
        },
    )
    workflow.prepare(cfg, "hidden")
    assert len(extractions) == 5
    assert primary not in extractions


def test_interrupted_hidden_upload_retains_payloads_and_recovers_remote_rows(
    preparation,
):
    cfg, remote, requests, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    workflow.prepare(cfg, "data")
    remote.fail = "-complete.json"
    with pytest.raises(OSError, match="interrupted"):
        workflow.prepare(cfg, "hidden")
    assert list(
        (cfg.output_root / "hidden_states" / ".pending-extraction").rglob(
            "hs_*.safetensors"
        )
    )
    workflow.prepare(cfg, "hidden")
    assert len(extractions) == 6
    shutil.rmtree(cfg.output_root)
    workflow.prepare(cfg, "hidden")
    assert len(extractions) == 6
    assert (
        len(
            json.loads((cfg.output_root / "hidden_states" / "index.json").read_text())[
                "rows"
            ]
        )
        == 6
    )
