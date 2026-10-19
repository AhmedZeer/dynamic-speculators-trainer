"""Local packed inputs and online extraction exercised with real tiny training."""

import json
import os
import shlex
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import torch
from datasets import load_from_disk
from safetensors.torch import load_file, save_file

from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.generator import campaign, campaign_train, online_activations
from speculators.generator.campaign_config import CampaignConfig
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.online_activations import (
    ActivationRAM,
    OnlineActivations,
    OnlineCorpus,
)
from speculators.generator.persistent_summaries import PersistentSummaries
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401


@pytest.fixture
def online_campaign(tiny_campaign, tmp_path, monkeypatch):  # noqa: F811
    cfg, downloads = tiny_campaign
    cfg.activation_source = "vllm"
    cfg.data_path = tmp_path / "packed"
    cfg.trajectories_path = tmp_path / "trajectories"
    cfg.online.shared_dir = tmp_path / "shared"
    cfg.online.reserve_gib = 0
    cfg.online.summary_cache_dir = tmp_path / "persistent-summaries"
    cfg.online.compiler_cache_dir = tmp_path / "compiler-cache"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path / "compiler-cache"))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "compiler-cache/triton"))
    cfg.pilot.updates = 2
    catalog = {
        "format": "speculators-packed-lora-bank",
        "version": 1,
        "release": "b0002",
        "runs": [],
    }
    for bank_path in [cfg.bank_config, *cfg.additional_bank_configs]:
        bank = BankConfig.load(bank_path)
        bank.hidden_states.dtype = "float32"
        write_json(bank_path, bank.model_dump(mode="json"))
        domain = bank.dataset.category
        manifest = json.loads((bank.output_root / "manifest.json").read_text())
        manifest["cache_fingerprint"] = domain + "-cache"
        manifest["cache_spec"] = {
            "hidden_states": bank.hidden_states.model_dump(mode="json"),
            "models": {
                role: getattr(bank.models, role) for role in ("target", "drafter")
            },
        }
        rows = load_from_disk(manifest["data_path"])
        # Distinct train/validation prompts exercise prompt-only extraction;
        # the underlying shared fixture otherwise uses identical prompts.
        rows = rows.map(
            lambda row, i: {
                "input_ids": [1, 3 if i % 12 < 8 else 2, *row["input_ids"][2:]]
            },
            with_indices=True,
        )
        artifacts = []
        for subset in manifest["subsets"]:
            for split in ("train", "validation"):
                indices = subset[f"{split}_indices"]
                subset[f"{split}_ids"] = list(rows.select(indices)["primary_id"])
                relative = (
                    f"releases/{bank.hub.release}/{domain}/"
                    f"{subset['id']}/{split}.parquet"
                )
                path = cfg.trajectories_path / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                rows.select(indices).to_parquet(path)
                artifacts.append({"path": relative, "bytes": path.stat().st_size})
            snapshots, tensors = [], {}
            if int(subset["id"].rsplit("-", 1)[1]) >= 4:
                continue
            for step in range(1, 11):
                sid = f"step-{step:08d}"
                path = (
                    bank.output_root
                    / "runs"
                    / subset["id"]
                    / "seed-42"
                    / "snapshots"
                    / sid
                )
                original = load_file(path / "adapter" / "adapter_model.safetensors")
                keys = {key: f"{sid}.{key}" for key in original}
                tensors.update({keys[k]: v for k, v in original.items()})
                snapshots.append(
                    {
                        "snapshot_id": sid,
                        "tensor_keys": keys,
                        "entry": {"epoch": 1, "collection_step": step},
                    }
                )
            relative = f"{domain}/{subset['id']}/seed-42"
            output = cfg.data_path / "releases/b0002" / relative
            output.mkdir(parents=True)
            save_file(tensors, output / "epoch-01.safetensors")
            packed = {
                "source_identity": {
                    k: manifest[k]
                    for k in ("prepared_fingerprint", "cache_fingerprint", "revisions")
                },
                "epochs": [
                    {
                        "collection_epoch": 1,
                        "file": "epoch-01.safetensors",
                        "bytes": (output / "epoch-01.safetensors").stat().st_size,
                        "snapshots": snapshots,
                    }
                ],
            }
            packed["source_identity"].update(
                subset=subset, seed=42, lora=bank.lora.model_dump(mode="json")
            )
            write_json(output / "manifest.json", packed)
            catalog["runs"].append(
                {
                    "path": relative,
                    "manifest_sha256": file_digest(output / "manifest.json"),
                }
            )
        write_json(
            cfg.trajectories_path / f"releases/{bank.hub.release}/{domain}/data.json",
            {"manifest": manifest, "artifacts": artifacts},
        )
    write_json(cfg.data_path / "releases/b0002/catalog.json", catalog)
    requests = []

    class Client:
        models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id=bank.models.target)])
        )

        def close(self):
            pass

    def extract(client, model, data, **kwargs):
        tokens = data["input_ids"]
        requests.append(list(tokens))
        path = cfg.online.shared_dir / f"{uuid.uuid4()}.safetensors"
        save_file(
            {
                "token_ids": torch.tensor(tokens),
                "hidden_states": torch.arange(
                    len(tokens) * 32, dtype=torch.float32
                ).reshape(-1, 4, 8)
                / 100,
            },
            path,
        )
        return str(path)

    monkeypatch.setattr(online_activations.openai, "OpenAI", lambda **_: Client())
    monkeypatch.setattr(online_activations, "generate_hidden_states", extract)
    return cfg, downloads, requests


def test_local_online_four_jobs_and_pilot(online_campaign):
    cfg, downloads, requests = online_campaign
    result = campaign.run_campaign(cfg, "all")
    assert result["runs"] == 4
    assert not downloads
    assert requests
    assert any(len(r) == 2 for r in requests)
    assert any(len(r) > 2 for r in requests)
    assert not list(cfg.online.shared_dir.iterdir())
    report = json.loads((cfg.output_root / "pilot/report.json").read_text())
    assert report["passed"]
    assert len(report["results"]) == 2
    assert all(r["cache_end"]["temporary_bytes"] == 0 for r in report["results"])
    # Existing seven-arm campaigns can resume without touching retired artifacts.
    manifest_path = cfg.output_root / "campaign.json"
    manifest = json.loads(manifest_path.read_text())
    retired = ("direct_1", "direct_2", "fresh_generator")
    for arm in retired:
        manifest["jobs"].append({"label": arm, "stage": "adaptation"})
        artifact = cfg.output_root / "adaptation" / arm / "retained.txt"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("existing artifact")
    write_json(manifest_path, manifest)
    before = len(requests)
    assert campaign.run_campaign(cfg, "all")["runs"] == 4
    assert len(requests) == before
    assert len(json.loads(manifest_path.read_text())["jobs"]) == 4
    for arm in retired:
        assert (cfg.output_root / "adaptation" / arm / "retained.txt").read_text() == (
            "existing artifact"
        )


def test_online_full_prefetch_supplies_summary_without_second_request(online_campaign):
    cfg, downloads, requests = online_campaign
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    try:
        cache.prefetch([{"index": 0, "summary": False}])
        summary = cache.get_summary(0)
        payload = cache.get_full(0)
        torch.testing.assert_close(summary, payload["hidden_states"][1, :-1].flatten())
        assert len(requests) == 1
        assert not downloads
        assert not list(cfg.online.shared_dir.iterdir())
    finally:
        cache.close()


def test_online_rejects_wrong_tokens_and_cleans_file(online_campaign, monkeypatch):
    cfg, _, _ = online_campaign
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])

    def bad(*args, **kwargs):
        path = cfg.online.shared_dir / "bad.safetensors"
        save_file(
            {"token_ids": torch.tensor([99]), "hidden_states": torch.zeros(1, 4, 8)},
            path,
        )
        return str(path)

    monkeypatch.setattr(online_activations, "generate_hidden_states", bad)
    try:
        with pytest.raises(ValueError, match="Token ids"):
            cache.get_full(0)
        assert not list(cfg.online.shared_dir.iterdir())
    finally:
        cache.close()


def test_local_input_rejects_changed_membership(online_campaign):
    cfg, _, _ = online_campaign
    path = cfg.trajectories_path / "releases/r0001/math/data.json"
    record = json.loads(path.read_text())
    record["manifest"]["subsets"][0]["train_ids"][0] = "different"
    write_json(path, record)
    with pytest.raises(ValueError, match="identity/order"):
        load_local_inputs(cfg)


def test_parquet_only_download_reports_metadata_repair_command(online_campaign):
    cfg, downloads, requests = online_campaign
    (cfg.trajectories_path / "releases/r0001/math/data.json").unlink()
    with pytest.raises(FileNotFoundError, match="Missing trajectory metadata") as error:
        load_local_inputs(cfg)
    command = shlex.split(str(error.value).splitlines()[-1])
    assert command[:3] == ["hf", "download", "azeeer/eagle3-qwen3-8b-trajectories-v1"]
    assert command[3:7] == [
        f"releases/r0001/{domain}/data.json" for domain in campaign.DOMAINS
    ]
    assert command[command.index("--local-dir") + 1] == str(cfg.trajectories_path)
    assert command[command.index("--revision") + 1] == "a" * 40
    assert not downloads
    assert not requests


def test_online_configuration_dry_run():
    cfg = CampaignConfig.load(
        Path("examples/eagle3-lora/generator_campaign_online.yaml")
    )
    result = campaign.run_campaign(cfg, dry_run=True)
    assert len(result["jobs"]) == 4
    assert cfg.online.concurrency == 16
    assert cfg.online.buffered_examples == 64
    assert cfg.online.ram_cache_gib == 96
    assert cfg.wandb.pretraining_project == "eagle3-generator-ablations-v3"
    assert {campaign.job_config(cfg, j).wandb.group for j in result["jobs"]} == {
        "experiment-1",
        "experiment-2",
    }


def test_existing_campaign_accepts_budget_change_and_keeps_pilot(online_campaign):
    cfg, _, _ = online_campaign
    campaign.run_campaign(cfg, "prepare")
    previous_id = campaign.campaign_identity(cfg)
    write_json(
        cfg.output_root / "pilot/report.json",
        {
            "campaign_identity": previous_id,
            "profile": campaign.pilot_profile(cfg),
            "passed": True,
        },
    )
    cfg.optimization.pretraining_updates = 1
    campaign.run_campaign(cfg, "prepare")
    report = json.loads((cfg.output_root / "pilot/report.json").read_text())
    assert report["campaign_identity"] == campaign.campaign_identity(cfg)
    assert report["passed"]
    assert list((cfg.output_root / "provenance").glob("resume-*/campaign.json"))
    cfg.optimization.lr *= 2
    with pytest.raises(ValueError, match="Campaign or local data changed"):
        campaign.run_campaign(cfg, "prepare")


def test_ram_cache_enforces_bytes_and_prioritizes_validation():
    payload = {
        "token_ids": torch.ones(2, dtype=torch.int64),
        "hidden_states": torch.zeros(2, 4, 8),
    }
    size = sum(t.numel() * t.element_size() for t in payload.values())
    ram = ActivationRAM(2 * size / 2**30, validation_indices=[0, 4])
    ram.put(0, payload)
    ram.put(1, payload)
    ram.put(2, payload)
    assert list(ram.full) == [0, 2]
    assert ram.bytes == 2 * size
    ram.put(4, payload)
    ram.put(5, payload)
    assert list(ram.full) == [0, 4]
    assert ram.peak_bytes <= ram.limit
    assert ram.evictions == 2
    disabled = ActivationRAM(0)
    disabled.put(0, payload)
    assert not disabled.full


def test_ram_cache_reuses_full_states_across_jobs(online_campaign):
    cfg, _, requests = online_campaign
    inputs = load_local_inputs(cfg)
    ram = ActivationRAM(1, [0])
    first = OnlineActivations(cfg, inputs.rows, inputs.banks["math"], ram=ram)
    try:
        expected = first.get_full(0)
        assert first.get_full(0) is expected
    finally:
        first.close()
    second = OnlineActivations(cfg, inputs.rows, inputs.banks["math"], ram=ram)
    try:
        assert second.get_full(0) is expected
        assert len(requests) == 1
        assert second.snapshot()["hits"] == 1
        assert second.get_summary(0).shape == (24,)
    finally:
        second.close()


def test_persistent_summaries_reload_without_extraction(online_campaign):
    cfg, _, requests = online_campaign
    inputs = load_local_inputs(cfg)
    store = PersistentSummaries(cfg, inputs.banks["math"], inputs)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    try:
        stats = store.prepare(["math-00000"], cache, 24)
        expected = cache.get_summary(0).clone()
        assert stats["generated_examples"] == 12
    finally:
        cache.close()
    before = len(requests)
    # A new owner/client models restarting the training process.
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    fresh = PersistentSummaries(cfg, inputs.banks["math"], inputs)
    try:
        stats = fresh.prepare(["math-00000"], cache, 24)
        assert stats["loaded_examples"] == 12
        assert stats["generated_examples"] == 0
        assert len(requests) == before
        torch.testing.assert_close(cache.get_summary(0), expected)
        assert not list(fresh.root.glob("*.pending"))
        changed = inputs.banks["math"].model_copy(deep=True)
        changed.hidden_states.layer_ids = [3, 18, 33, 36]
        assert PersistentSummaries(cfg, changed, inputs).root != fresh.root
    finally:
        cache.close()


def test_persistent_summaries_import_legacy_and_reject_bad_shard(online_campaign):
    cfg, _, requests = online_campaign
    inputs = load_local_inputs(cfg)
    store = PersistentSummaries(cfg, inputs.banks["math"], inputs)
    old = store.legacy_path(0)
    old.parent.mkdir(parents=True, exist_ok=True)
    expected = torch.ones(24)
    torch.save(expected, old)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    try:
        stats = store.prepare(["math-00000"], cache, 24)
        # All eight training rows in this fixture share the same prompt.
        assert stats["legacy_examples"] == 8
        assert len(requests) == 4
        torch.testing.assert_close(cache.get_summary(0), expected)
        payload = load_file(store.root / "math-00000.safetensors")
        payload["lengths"][0] += 1
        save_file(payload, store.root / "math-00000.safetensors")
        fresh = PersistentSummaries(cfg, inputs.banks["math"], inputs)
        with pytest.raises(ValueError, match="membership differs"):
            fresh.prepare(["math-00000"], cache, 24)
    finally:
        cache.close()


def test_lookahead_refills_before_next_episode_and_keeps_inflight_work(online_campaign):
    cfg, _, _ = online_campaign
    cfg.online.buffered_examples = 8
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    ahead_started, release = Event(), Event()
    generate = cache._generate

    def delayed(index, summary):
        if index == 8:
            ahead_started.set()
            assert release.wait(10)
        return generate(index, summary)

    cache._generate = delayed
    try:
        refs = [{"index": i, "summary": False} for i in range(12)]
        cache.protect(refs)
        cache.prefetch(refs)
        cache.get_full(0)
        # Consuming one current result automatically starts future work while
        # the caller is free to compute on that result.
        assert ahead_started.wait(5)
        future = cache.futures[(8, False)]
        assert len(cache.futures) <= cfg.online.buffered_examples
        cache.protect(refs[8:])
        cache.prefetch(refs[8:])
        assert cache.futures[(8, False)] is future
        release.set()
        assert cache.get_full(8)["hidden_states"].shape[1:] == (4, 8)
    finally:
        release.set()
        cache.close()


def test_online_episode_keeps_future_episode_prefetched(online_campaign):
    cfg, _, _ = online_campaign
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    corpus = OnlineCorpus(
        campaign.job_config(cfg, campaign.jobs(cfg)[1]),
        inputs.banks["math"],
        cache,
        inputs,
    )
    try:
        corpus.set_training_window([[0], [1]], full=True)
        future = cache.futures[(1, False)]
        corpus.prepare_episode([0], full=True)
        cache.get_full(0)
        corpus.prepare_episode([1], full=True)
        assert cache.futures[(1, False)] is future
    finally:
        cache.close()


def test_online_lookahead_resume_matches_uninterrupted(online_campaign, monkeypatch):
    cfg, _, _ = online_campaign
    job = campaign.jobs(cfg)[1]
    original = campaign_train.train_update

    def interrupt(*args, **kwargs):
        if args[4] == 1:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign_train, "train_update", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.run_one(cfg, job)
    monkeypatch.setattr(campaign_train, "train_update", original)
    resumed = campaign.run_one(cfg, job)
    clean_cfg = cfg.model_copy(update={"output_root": cfg.output_root / "clean"})
    uninterrupted = campaign.run_one(clean_cfg, job)
    left = torch.load(resumed["checkpoint_path"], weights_only=True)
    right = torch.load(uninterrupted["checkpoint_path"], weights_only=True)
    assert left["progress"] == right["progress"] == 2
    for name in left["model"]:
        torch.testing.assert_close(
            left["model"][name], right["model"][name], rtol=0, atol=0
        )
    assert resumed["score"] == uninterrupted["score"]


def test_configured_compiler_cache_overrides_import_time_default(
    online_campaign, monkeypatch
):
    cfg, _, _ = online_campaign
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(cfg.output_root / "old-default"))
    result = campaign.run_one(cfg, campaign.jobs(cfg)[0])
    expected = str(cfg.online.compiler_cache_dir.resolve())
    assert result["compiler_cache_dir"] == expected
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == expected
    assert os.environ["TRITON_CACHE_DIR"] == str(Path(expected) / "triton")
    assert result["preparation_seconds"] > 0


def test_window_change_cancels_queued_requests_before_draining_workers(online_campaign):
    cfg, _, requests = online_campaign
    cfg.online.concurrency = 1
    inputs = load_local_inputs(cfg)
    cache = OnlineActivations(cfg, inputs.rows, inputs.banks["math"])
    started, release, cancelled = Event(), Event(), Event()
    generate = cache._generate

    def delayed(index, summary):
        if index == 0:
            started.set()
            assert release.wait(10)
        return generate(index, summary)

    cache._generate = delayed
    control = ThreadPoolExecutor(1)
    try:
        refs = [{"index": i, "summary": False} for i in range(8)]
        cache.prefetch(refs)
        assert started.wait(5)
        cache.futures[(7, False)].add_done_callback(
            lambda future: cancelled.set() if future.cancelled() else None
        )
        drained = control.submit(cache.protect, [])
        assert cancelled.wait(5)
        release.set()
        drained.result(timeout=5)
        assert len(requests) == 1
    finally:
        release.set()
        control.shutdown(wait=True)
        cache.close()


def test_online_prepare_needs_no_bank_files(online_campaign):
    import shutil  # noqa: PLC0415

    cfg, downloads, requests = online_campaign
    recipe_path = cfg.output_root.parent / "training-recipe.json"
    recipe = BankConfig.load(cfg.bank_config)
    write_json(recipe_path, recipe.model_dump(mode="json"))
    cfg.training_recipe = recipe_path
    for path in [cfg.bank_config, *cfg.additional_bank_configs]:
        path.unlink()
    cfg.additional_bank_configs = []
    shutil.rmtree(cfg.data_path)
    cfg.data_path = None
    result = campaign.run_campaign(cfg, "prepare")
    assert "packed_catalog_sha256" not in result["local_inputs"]
    assert all(j["objective"] == "direct" for j in result["jobs"])
    assert not downloads
    assert not requests


def test_adaptation_rejects_bank_source(online_campaign):
    cfg, _, _ = online_campaign
    pretraining = campaign.jobs(cfg)[0]
    result = campaign.run_one(cfg, pretraining)
    source = Path(result["checkpoint_path"]).parent / "resolved_experiment.json"
    identity = json.loads(source.read_text())
    identity["job"]["objective"] = "bank"
    write_json(source, identity)
    # Source validation remains available for historical/custom adaptation jobs.
    adaptation = {
        **campaign.jobs(cfg)[-1],
        "arm": "direct_1",
        "label": "direct_1",
        "source_job": "scale-1-direct",
    }
    with pytest.raises(ValueError, match="requested direct campaign arm"):
        campaign.run_one(cfg, adaptation)
