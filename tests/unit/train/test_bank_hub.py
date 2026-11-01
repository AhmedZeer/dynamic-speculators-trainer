import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import datasets
import httpx
import huggingface_hub
import pytest
import torch
import yaml
from datasets import load_from_disk
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from pydantic import ValidationError
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


@pytest.mark.parametrize("resume_from_hf", [False, True])
def test_serve_uses_frozen_target_and_server_provenance(
    preparation, tmp_path, monkeypatch, resume_from_hf
):
    cfg, remote, requests, extractions = preparation
    workflow.prepare(cfg, "select")
    commits = remote.commits
    if resume_from_hf:
        shutil.rmtree(cfg.output_root)

        def forbid_selection(*args, **kwargs):
            pytest.fail("Server startup must restore frozen data without selecting")

        monkeypatch.setattr(datasets, "load_dataset", forbid_selection)
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
    lora_bank.serve(config, mode="hidden", port=8000, resume_from_hf=resume_from_hf)
    assert remote.commits == commits
    assert (cfg.output_root / "manifest.json").is_file()
    command = commands[0]
    assert resolved == [(cfg.models.target, "target")]
    assert command[2] == "train"
    assert "--provenance-dir" in command
    assert command[command.index("--served-model-name") + 1] == cfg.models.target
    assert command[command.index("--max-model-len") + 1] == "8193"
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
        rows = load_from_disk(kwargs["preprocessed_data"]).with_format(None)
        root = Path(kwargs["output"])
        root.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(rows):
            if i in (kwargs.get("skip_indices") or []):
                continue
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
            "extraction_scope": "all",
            "activation_format": "sharded",
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
        if ".packaged" in path.parts and path.suffix == ".safetensors":
            hashes[path] = hashes.get(path, 0) + 1
        return file_digest(path)

    monkeypatch.setattr(hub, "file_digest", count_hashes)
    repo = cfg.hub.activations_repo
    before = sum(key[0] == repo for key in remote.history)
    workflow.prepare(cfg, "hidden")
    after = sum(key[0] == repo for key in remote.history)
    # Four subset/split shards: one group commit + final index.
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


def test_continuous_extraction_finishes_before_any_activation_upload(
    preparation, monkeypatch
):
    cfg, remote, _, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    manifest = workflow.prepare(cfg, "data")
    cfg.hub.upload_batch_examples = 1
    calls = []
    extract = generate_offline_data.generate_offline_data

    def continuous(**kwargs):
        calls.append(kwargs)
        extract(**kwargs)

    commit = remote.create_commit

    def after_extraction(repo, **kwargs):
        if repo == cfg.hub.activations_repo:
            assert len(extractions) == 6
            assert len(calls) == 1
            assert not list(
                (cfg.output_root / "hidden_states" / ".raw-states").glob(
                    "hs_*.safetensors"
                )
            )
        return commit(repo, **kwargs)

    monkeypatch.setattr(generate_offline_data, "generate_offline_data", continuous)
    monkeypatch.setattr(remote, "create_commit", after_extraction)
    workflow.prepare(cfg, "hidden")
    assert calls[0]["preprocessed_data"] == manifest["data_path"]
    assert calls[0]["validate_outputs"] is False


def test_resume_code_from_hf_without_local_inputs(preparation, tmp_path):
    cfg, remote, requests, extractions = preparation
    cfg.hub.domains.append("code")
    cfg.dataset.category = "code"
    cfg.dataset.split = "code"
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    workflow.prepare(cfg, "data")
    workflow.prepare(cfg, "hidden")
    generated = (len(requests), len(extractions))
    restored = cfg.model_copy(deep=True)
    restored.output_root = tmp_path / "new-code-machine"
    manifest = workflow.prepare(restored, "hidden", resume_from_hf=True)
    assert manifest["category"] == "code"
    assert (len(requests), len(extractions)) == generated
    assert (restored.output_root / "data").is_dir()
    assert (
        len(
            json.loads(
                (restored.output_root / "hidden_states" / "index.json").read_text()
            )["rows"]
        )
        == 6
    )


def test_machine_loss_between_final_upload_groups_recovers_published_rows(
    preparation, monkeypatch
):
    cfg, remote, _, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    workflow.prepare(cfg, "data")
    cfg.hub.upload_batch_examples = 2
    commit = remote.create_commit
    groups = [0]

    def interrupt_second_group(repo, **kwargs):
        if repo == cfg.hub.activations_repo and any(
            op.path_in_repo.endswith("-complete.json") for op in kwargs["operations"]
        ):
            groups[0] += 1
            if groups[0] == 2:
                raise OSError("rate limited")
        return commit(repo, **kwargs)

    monkeypatch.setattr(remote, "create_commit", interrupt_second_group)
    with pytest.raises(OSError, match="rate limited"):
        workflow.prepare(cfg, "hidden")
    assert len(extractions) == 6
    shutil.rmtree(cfg.output_root)
    workflow.prepare(cfg, "hidden", resume_from_hf=True)
    # Two rows in the successful group survived machine loss; only four rows
    # from uncommitted packages need to be extracted again.
    assert len(extractions) == 10


def test_old_pending_batch_is_reused_without_reextraction(preparation):
    cfg, remote, _, extractions = preparation
    workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    manifest = workflow.prepare(cfg, "data")
    old = cfg.output_root / "hidden_states" / ".pending-extraction" / "old-batch"
    data = load_from_disk(manifest["data_path"]).with_format(None)
    data.select([0, 1]).save_to_disk(old / "data")
    states = old / "states"
    states.mkdir()
    for i in range(2):
        save_file(
            {
                "token_ids": torch.tensor(data[i]["input_ids"]),
                "hidden_states": torch.zeros(3, 4, 3, dtype=torch.bfloat16),
            },
            states / f"hs_{i}.safetensors",
        )
    workflow.prepare(cfg, "hidden")
    assert len(extractions) == 4
    assert data[0]["primary_id"] not in extractions
    assert data[1]["primary_id"] not in extractions


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
        (cfg.output_root / "hidden_states" / ".packaged").glob("*/shard.safetensors")
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


def test_append_selection_preserves_splits_reuses_responses_and_resumes(
    preparation, tmp_path
):
    cfg, remote, requests, _ = preparation
    original = workflow.prepare(cfg, "select")
    workflow.prepare(cfg, "responses")
    old_prompts = read_jsonl(cfg.output_root / "prompts.jsonl")
    assert len(requests) == 6
    revision = remote.repo_info(cfg.hub.trajectories_repo).sha
    expanded = cfg.model_copy(deep=True)
    expanded.output_root = tmp_path / "expanded"
    expanded.partition.subset_ids = [0, 1, 2]
    expanded.hub.selection = "expanded"
    expanded.hub.release = "expanded"
    expanded.hub.selection_root = tmp_path / "expanded-selection"
    expanded.hub.parent_selection = cfg.hub.selection
    expanded.hub.parent_release = cfg.hub.release
    expanded.hub.parent_revision = revision
    manifest = workflow.prepare(expanded, "select")
    prompts = read_jsonl(expanded.output_root / "prompts.jsonl")
    assert prompts[:6] == old_prompts
    assert manifest["subsets"][:2] == original["subsets"]
    assert len(prompts) == 9
    inherited = {s for p in old_prompts for s in p["source_ids"]}
    assert not inherited.intersection(s for p in prompts[6:] for s in p["source_ids"])
    remote.fail = "/responses/"
    with pytest.raises(OSError, match="interrupted"):
        workflow.prepare(expanded, "responses")
    workflow.prepare(expanded, "responses")
    assert len(requests) == 9  # Six inherited responses never hit the target again.
    assert len(read_jsonl(expanded.output_root / "responses.jsonl")) == 9
    workflow.prepare(expanded, "responses")
    assert len(requests) == 9
    changed = expanded.model_copy(deep=True)
    changed.responses.temperature = 0.5
    with pytest.raises(ValueError, match="recipe differs"):
        hub.parent_inputs(changed)
    changed = expanded.model_copy(deep=True)
    changed.partition.validation_examples = 2
    with pytest.raises(ValueError, match="train/validation split"):
        hub.parent_inputs(changed)


def test_parent_expansion_config_requires_pinned_distinct_parent():
    with pytest.raises(ValidationError, match="required together"):
        BankConfig(hub={"parent_selection": "s0001"})
    with pytest.raises(ValidationError, match="complete Hub commit"):
        BankConfig(
            hub={
                "parent_selection": "old",
                "parent_release": "old",
                "parent_revision": "main",
            }
        )
    with pytest.raises(ValidationError, match="new selection and release"):
        BankConfig(
            hub={
                "parent_selection": "s0001",
                "parent_release": "r0001",
                "parent_revision": "a" * 40,
            }
        )


def test_two_nodes_resume_single_node_work_and_finalize(preparation, tmp_path):
    cfg, remote, requests, _ = preparation
    cfg.hub.domains = ["math"]
    workflow.prepare(cfg, "select")
    # Stop the old single-node process after its first chunk is published and
    # the following chunk was generated locally but its upload failed.
    original_commit = remote.create_commit
    response_uploads = 0

    def interrupt_second(repo, operations, **kwargs):
        nonlocal response_uploads
        if any("/responses/responses-" in op.path_in_repo for op in operations):
            response_uploads += 1
            if response_uploads == 2:
                raise OSError("old process interrupted")
        return original_commit(repo, operations, **kwargs)

    remote.create_commit = interrupt_second
    with pytest.raises(OSError, match="old process interrupted"):
        workflow.prepare(cfg, "responses")
    remote.create_commit = original_commit
    assert len(requests) == 4
    initial = list(requests)
    # Simulate stochastic responses on the old unfinished chunk. The new owner
    # may produce different tokens; node 0 must not compare its stale local copy.
    local = cfg.output_root / "responses.jsonl"
    rows = read_jsonl(local)
    for row in rows[2:]:
        row["input_ids"] = [9, 8, 7]
    local.write_text("".join(json.dumps(row) + "\n" for row in rows))
    cfg.execution.response_num_machines = 2
    cfg.execution.response_machine_index = 0
    # Fresh node 1 restores the selection directly from HF.
    other = cfg.model_copy(deep=True)
    other.execution.response_machine_index = 1
    other.output_root = tmp_path / "other-machine"
    workflow.prepare(other, "responses")
    assert requests[4:] == initial[2:4]  # Only unpublished chunk is regenerated.
    with pytest.raises(ValueError, match="Cannot finalize"):
        workflow.prepare(other, "finalize-responses")
    workflow.prepare(cfg, "responses")
    assert (cfg.output_root / "responses-before-partition.jsonl").exists()
    # The old local copy of node 1's chunk cannot conflict with its published copy.
    assert len(requests) == 8  # 4 old calls + 2 node1 calls + 2 remaining node0 calls.
    workflow.prepare(other, "responses")
    workflow.prepare(cfg, "responses")
    assert len(requests) == 8
    paths = remote.list_repo_files(cfg.hub.trajectories_repo)
    prefix = hub.profile(cfg)
    assert f"{prefix}/responses.json" not in paths
    assert f"{prefix}/responses-machine-00000-of-00002.json" in paths
    assert f"{prefix}/responses-machine-00001-of-00002.json" in paths
    workflow.prepare(cfg, "finalize-responses")
    assert len(requests) == 8
    assert len(read_jsonl(cfg.output_root / "responses.jsonl")) == 6
    workflow.prepare(cfg, "data")
    assert len(load_from_disk(cfg.output_root / "data")) == 6
    wrong = cfg.model_copy(deep=True)
    wrong.execution.response_num_machines = 1
    with pytest.raises(ValueError, match="different num-machines"):
        workflow.prepare(wrong, "responses")


def test_publisher_retries_competing_head_and_checks_immutable_objects(
    preparation, tmp_path, monkeypatch
):
    cfg, remote, _, _ = preparation
    publisher = hub.Publisher(cfg)
    source = tmp_path / "object.txt"
    source.write_text("same payload")
    original = remote.create_commit
    calls = []

    def conflict_once(repo, operations, **kwargs):
        calls.append(kwargs["parent_commit"])
        if len(calls) == 1:
            # A competing writer advances HEAD just before our commit.
            original(
                repo,
                [
                    huggingface_hub.CommitOperationAdd(
                        path_in_repo="other-node.txt", path_or_fileobj=b"other"
                    )
                ],
                parent_commit=kwargs["parent_commit"],
            )
            response = httpx.Response(
                412, request=httpx.Request("POST", "https://huggingface.co")
            )
            raise HfHubHTTPError("parent commit changed", response=response)
        return original(repo, operations, **kwargs)

    monkeypatch.setattr(remote, "create_commit", conflict_once)
    monkeypatch.setattr(hub.time, "sleep", lambda _: None)
    reference = publisher.put(cfg.hub.trajectories_repo, "object.txt", source)
    assert len(calls) == 2
    assert calls[0] != calls[1]
    assert reference["revision"] == remote.repo_info(cfg.hub.trajectories_repo).sha
    fresh = cfg.model_copy(deep=True)
    fresh.output_root = tmp_path / "fresh-publisher"
    second = hub.Publisher(fresh)
    source.write_text("different payload")
    with pytest.raises(ValueError, match="Immutable Hub object differs"):
        second.put(cfg.hub.trajectories_repo, "object.txt", source)
