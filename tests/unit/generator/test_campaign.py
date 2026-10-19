"""Campaign membership, asynchronous IO, and real tiny-model training checks."""

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock

import pytest
import torch
from datasets import Dataset
from safetensors.torch import save_file

from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.cli import app
from speculators.generator import campaign, campaign_cache, campaign_train
from speculators.generator.campaign_config import CampaignConfig, DownloadSettings
from speculators.generator.campaign_data import CampaignCorpus, EpisodeSchedule
from speculators.generator.campaign_resume import compatible_campaign
from speculators.generator.data import FactorCache
from speculators.generator.engine import Runtime
from speculators.generator.model import install_adapters
from speculators.losses import resolve_loss_config
from tests.unit.generator.test_experiments import small_architecture, tiny_drafter


def config():
    return CampaignConfig.load(Path("examples/eagle3-lora/generator_campaign.yaml"))


def test_design_and_dry_run_have_exact_memberships_and_no_io(monkeypatch):
    cfg = config()
    monkeypatch.setattr(
        campaign, "prepare_inputs", lambda *_: pytest.fail("dry run performed IO")
    )
    manifest = campaign.run_campaign(cfg, dry_run=True)
    jobs = manifest["jobs"]
    assert len(jobs) == 7
    assert [j["training_examples"] for j in jobs[:3]] == [4000, 8000, 16000]
    assert all(j["objective"] == "direct" for j in jobs)
    assert {j["arm"] for j in jobs[3:]} == {
        "fresh_generator",
        "direct_1",
        "direct_2",
        "fresh_lora",
    }
    for job in jobs[3:]:
        assert job["training_examples"] == 16000
        assert {int(s.rsplit("-", 1)[1]) for s in job["subsets"]} == {2, 3, 4, 5}
        if "source_job" in job:
            source = next(j for j in jobs if j["label"] == job["source_job"])
            assert not set(source["subsets"]) & set(job["subsets"])
        assert campaign.job_config(cfg, job).context.evaluation_sizes == [4]
    from typer.testing import CliRunner  # noqa: PLC0415

    result = CliRunner().invoke(
        app,
        [
            "generator",
            "campaign",
            "--config",
            "examples/eagle3-lora/generator_campaign.yaml",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)["jobs"]) == 7


def reference(source, name):
    return {
        "repo": "synthetic",
        "revision": "a" * 40,
        "path": name,
        "sha256": file_digest(source),
        "bytes": source.stat().st_size,
    }


def test_cache_downloads_are_deduplicated_and_do_not_block_other_files(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.write_bytes(b"a" * 128)
    slow, fast = reference(source, "slow"), reference(source, "fast")
    started, release = Event(), Event()
    calls, mutex = [], Lock()

    def download(ref, local_dir, **kwargs):
        with mutex:
            calls.append(ref["path"])
        if ref["path"] == "slow":
            started.set()
            assert release.wait(10)
        target = local_dir / ref["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr(campaign_cache, "_download_with_retries", download)
    cache = campaign_cache.CampaignCache(
        tmp_path / "cache",
        DownloadSettings(capacity_gib=0.01, reserve_gib=0, workers=2),
    )
    try:
        cache.prefetch([slow, slow])
        assert started.wait(5)
        with cache.open(fast) as path:
            assert path.read_bytes() == source.read_bytes()
        release.set()
        with ThreadPoolExecutor(4) as pool:

            def read():
                with cache.open(slow) as path:
                    return path.read_bytes()

            assert all(
                v == source.read_bytes() for v in pool.map(lambda _: read(), range(8))
            )
        assert calls.count("slow") == calls.count("fast") == 1
        assert cache.snapshot()["download_bytes"] == 256
    finally:
        release.set()
        cache.close()


def test_cache_eviction_protection_and_receipt_reuse(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"a" * 128)
    refs = [reference(source, str(i)) for i in range(3)]
    hashes = []
    original_digest = campaign_cache.file_digest
    monkeypatch.setattr(
        campaign_cache,
        "file_digest",
        lambda p: (hashes.append(p), original_digest(p))[1],
    )

    def download(ref, local_dir, **kwargs):
        target = local_dir / ref["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return target

    monkeypatch.setattr(campaign_cache, "_download_with_retries", download)
    settings = DownloadSettings(capacity_gib=256 / 2**30, reserve_gib=0, workers=1)
    cache = campaign_cache.CampaignCache(tmp_path / "cache", settings)
    try:
        cache.protect(refs[:1])
        with cache.open(refs[0]):
            pass
        with cache.open(refs[0]):
            pass
        assert len(hashes) == 1
        with cache.open(refs[1]):
            pass
        with cache.open(refs[2]):
            pass
        assert cache.path(refs[0]).is_file()
        assert not cache.path(refs[1]).exists()
        assert cache.snapshot()["evictions"] == 1
        cache.path(refs[0]).write_bytes(b"b" * 128)
        with pytest.raises(ValueError, match="checksum"), cache.open(refs[0]):
            pass
    finally:
        cache.close()


def test_cache_rejects_bad_download_and_resumes_partial(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_bytes(b"a" * 128)
    ref = reference(source, "payload")

    def download(ref, local_dir, **kwargs):
        target = local_dir / ref["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"b" * 128)
        return target

    monkeypatch.setattr(campaign_cache, "_download_with_retries", download)
    cache = campaign_cache.CampaignCache(
        tmp_path / "cache", DownloadSettings(capacity_gib=0.01, reserve_gib=0)
    )
    try:
        with pytest.raises(ValueError, match="checksum"), cache.open(ref):
            pass
        assert not cache.path(ref).exists()
    finally:
        cache.close()


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    root = tmp_path / "inputs"
    sources = tmp_path / "sources"
    template = tiny_drafter()
    paths = install_adapters(template, 2, 4, 0)
    shapes = {
        name: (
            template.get_submodule(path).base.in_features,
            template.get_submodule(path).base.out_features,
        )
        for name, path in paths.items()
    }
    configs = []
    for domain in campaign.DOMAINS:
        output = root / domain
        rows, indexed, subsets = [], {}, []
        for number in range(6):
            start = len(rows)
            for local in range(12):
                index = len(rows)
                size = 6 + local % 2
                ids = [1, 2, 3, 4, 5, 6, 7][:size]
                primary = f"{domain}-{index}"
                rows.append(
                    {
                        "input_ids": ids,
                        "seq_len": size,
                        "prompt_length": 2,
                        "primary_id": primary,
                        "loss_mask": [0, 0] + [1] * (size - 2),
                    }
                )
                source = sources / domain / f"hs_{index}.safetensors"
                source.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {
                        "token_ids": torch.tensor(ids),
                        "hidden_states": torch.randn(size, 4, 8),
                    },
                    source,
                )
                indexed[str(index)] = {
                    "format": "raw",
                    "primary_id": primary,
                    "shard": reference(source, f"{domain}/hs_{index}.safetensors"),
                }
            name = f"{domain}-{number:05d}"
            subsets.append(
                {
                    "id": name,
                    "role": "adaptation" if number == 5 else "bank",
                    "train_indices": list(range(start, start + 8)),
                    "validation_indices": list(range(start + 8, start + 12)),
                }
            )
            if number < 4:
                for step in range(1, 11):
                    path = (
                        output
                        / "runs"
                        / name
                        / "seed-42"
                        / "snapshots"
                        / f"step-{step:08d}"
                    )
                    (path / "adapter").mkdir(parents=True)
                    save_file(
                        {
                            f"base.{name}.lora_{label}.weight": torch.randn(*shape)
                            * 0.02
                            for name, (inputs, outputs) in shapes.items()
                            for label, shape in (
                                ("A", (2, inputs)),
                                ("B", (outputs, 2)),
                            )
                        },
                        path / "adapter" / "adapter_model.safetensors",
                    )
                    write_json(
                        path / "entry.json", {"epoch": 1, "collection_step": step}
                    )
        Dataset.from_list(rows).save_to_disk(output / "data")
        write_json(
            output / "hidden" / "index.json",
            {"rows": indexed, "cache_gib": 1, "reserve_gib": 0},
        )
        write_json(
            output / "manifest.json",
            {
                "prepared_fingerprint": domain,
                "cache_spec": {},
                "revisions": {"target_sha": "target", "drafter_sha": "draft"},
                "subsets": subsets,
                "data_path": str(output / "data"),
                "hidden_states_path": str(output / "hidden"),
            },
        )
        write_json(
            output / "campaign_bank_sources.json",
            [
                {"subset": s["id"], "training": {"warmup_epochs": 1}}
                for s in subsets[:4]
            ],
        )
        bank = BankConfig(
            output_root=output,
            dataset={"category": domain},
            lora={"rank": 2, "alpha": 4, "dropout": 0},
            execution={"draft_attn_impl": "eager"},
            hub={"mode": "consumer", "trajectories_revision": "a" * 40},
        )
        bank_path = root / f"bank_{domain}.json"
        write_json(bank_path, bank.model_dump(mode="json"))
        configs.append(bank_path)
    cfg = CampaignConfig(
        bank_config=configs[0],
        additional_bank_configs=configs[1:],
        input_root=root,
        output_root=tmp_path / "output",
        staging_dir=tmp_path / "staging",
        expected_train_examples=8,
        expected_validation_examples=4,
        device="cpu",
        dtype="float32",
        architecture=small_architecture(),
        optimization={
            "loss_fn": "lk_hybrid",
            "pretraining_updates": 2,
            "episodes_per_update": 4,
            "token_budget": 8,
            "checkpoint_interval": 1,
            "warmup_updates": 0,
        },
        adaptation_updates=2,
        downloads={"capacity_gib": 0.02, "reserve_gib": 0, "lookahead_updates": 2},
    )
    calls = []

    def download(ref, local_dir, **kwargs):
        calls.append(ref["path"])
        target = local_dir / ref["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources / ref["path"], target)
        return target

    monkeypatch.setattr(campaign_cache, "_download_with_retries", download)

    def factory(config, bank, *, corpus_factory, model_paths=None):
        runtime = Runtime.__new__(Runtime)
        runtime.cfg, runtime.bank = config, bank
        runtime.device, runtime.dtype = torch.device("cpu"), torch.float32
        runtime.corpus = corpus_factory(config, bank)
        torch.manual_seed(config.seed)
        runtime.drafter = tiny_drafter().requires_grad_(False)
        runtime.corpus.prepare(runtime.drafter)
        runtime.paths = install_adapters(runtime.drafter, 2, 4, 0)
        runtime.shapes = shapes
        runtime.cache = FactorCache(1, shapes, 2)
        runtime.call_kwargs = {
            "ttt_steps": 3,
            "loss_config": resolve_loss_config("lk_hybrid", "eager"),
        }
        return runtime

    monkeypatch.setattr(campaign, "Runtime", factory)
    return cfg, calls


def test_lazy_summaries_only_read_requested_examples(tiny_campaign):
    cfg, calls = tiny_campaign
    job = campaign.jobs(cfg)[0]
    local = campaign.job_config(cfg, job)
    cache = campaign_cache.CampaignCache(cfg.staging_dir / "files", cfg.downloads)
    try:
        bank = BankConfig.load(local.bank_config)
        corpus = CampaignCorpus(local, bank, cache)
        corpus.prepare(tiny_drafter())
        assert not calls
        index = corpus.indices("math-00000")[0]
        ref, metadata = corpus.reference(index, summary=True)
        assert "prompt_length" not in metadata
        # Exercise the initial-validation prefetch path from the reported crash.
        assert corpus.references([index], full=True) == [ref]
        first = corpus.context([index], "last", "cpu")
        second = corpus.context([index], "last", "cpu")
        assert len(calls) == 1
        assert corpus.summary_misses == corpus.summary_hits == 1
        torch.testing.assert_close(first[0], second[0])
        payload = corpus.transfer.get_cached(index)
        boundary = corpus.rows[index]["prompt_length"]
        torch.testing.assert_close(
            first[0][0], payload["hidden_states"][boundary - 1, :-1].flatten().float()
        )
        schedule = EpisodeSchedule(local, corpus, job["subsets"])
        assert schedule.update(5) == schedule.update(5)
        assert schedule.update(5) != schedule.update(6)
    finally:
        cache.close()


def test_seven_jobs_train_resume_and_reject_changed_sources(tiny_campaign):
    cfg, calls = tiny_campaign
    results = [campaign.run_one(cfg, job) for job in campaign.jobs(cfg)]
    assert len(results) == 7
    assert all(len(result["history"]) == 2 for result in results)
    assert all(set(result["history"][-1]["contexts"]) == {"4"} for result in results)
    for result in results:
        assert result["validation_seconds"] == sum(
            row["validation_seconds"] for row in result["history"]
        )
        assert result["elapsed_seconds"] >= result["validation_seconds"] > 0
        assert result["training_seconds"] > 0
        assert all(
            subset["elapsed_seconds"] > 0
            for row in result["history"]
            for subset in row["subsets"].values()
        )
    assert len(calls) == len(set(calls))
    before = len(calls)
    assert campaign.run_one(cfg, campaign.jobs(cfg)[0])["score"] == results[0]["score"]
    assert len(calls) == before
    for result in results:
        path = Path(result["checkpoint_path"])
        assert path.is_file()
        assert (path.parent / "train_command.txt").is_file()
        assert (path.parent / "speculators.patch").is_file()
    campaign.plot_comparisons(cfg, results[:3], "pretraining")
    assert (cfg.output_root / "pretraining" / "comparison.png").is_file()


def test_resume_matches_uninterrupted_training(tiny_campaign, monkeypatch):
    cfg, _ = tiny_campaign
    job = campaign.jobs(cfg)[1]
    original = campaign_train.train_update

    def interrupt(runtime, generator, optimizer, episodes, update, objective):
        if update == 1:
            raise RuntimeError("simulated interruption")
        return original(runtime, generator, optimizer, episodes, update, objective)

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


def test_resume_after_intentional_update_budget_change(tiny_campaign, monkeypatch):
    cfg, _ = tiny_campaign
    cfg.optimization.pretraining_updates = 3
    job = campaign.jobs(cfg)[0]
    original = campaign_train.train_update

    def interrupt(*args, **kwargs):
        if args[4] == 1:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign_train, "train_update", interrupt)
    with pytest.raises(RuntimeError, match="interruption"):
        campaign.run_one(cfg, job)
    output = cfg.output_root / "pretraining" / job["label"]
    saved_identity = json.loads((output / "resolved_experiment.json").read_text())
    before = torch.load(output / "checkpoint.pt", weights_only=True)
    assert before["progress"] == 1
    cfg.optimization.pretraining_updates = 2
    monkeypatch.setattr(campaign_train, "train_update", original)
    result = campaign.run_one(cfg, campaign.jobs(cfg)[0])
    after = torch.load(output / "checkpoint.pt", weights_only=True)
    assert result["completed_updates"] == 2
    assert after["identity"] == before["identity"]
    assert after["progress"] == 2
    assert (
        json.loads((output / "resolved_experiment.json").read_text()) == saved_identity
    )
    assert list((output / "provenance").glob("resume-*/configuration.json"))
    # A training change beyond budgets still fails before checkpoint reuse.
    cfg.optimization.lr *= 2
    with pytest.raises(ValueError, match="configuration changed"):
        campaign.run_one(cfg, campaign.jobs(cfg)[0])


def test_campaign_compatibility_allows_only_budget_and_output_changes():
    cfg = config()
    previous = {"configuration": cfg.model_dump(mode="json")}
    cfg.output_root = cfg.output_root / "moved"
    cfg.optimization.pretraining_updates = 1500
    cfg.adaptation_updates = 500
    assert compatible_campaign(previous, cfg)
    cfg.seed += 1
    assert not compatible_campaign(previous, cfg)


def test_pilot_profiles_worker_selection_and_failure_reports(
    tiny_campaign, monkeypatch
):
    cfg, _ = tiny_campaign
    seen = []

    def fake_run(config, job, **kwargs):
        seen.append(job)
        seconds = 100 if job["workers"] == 4 else 50
        return {
            "job": job,
            "training_seconds": seconds,
            "data_wait_seconds": seconds * 0.1,
        }

    monkeypatch.setattr(campaign, "run_one", fake_run)
    report = campaign.pilot(cfg)
    assert report["passed"]
    assert report["selected_download_workers"] == 8
    assert len(seen) == 8
    assert all(job["pilot"] for job in seen)
    assert len(report["projected_training_seconds"]) == 7
    assert report["working_sets"]["fresh_lora"]["training_full_activation_bytes"] > 0
    original_profile = campaign.pilot_profile(cfg)
    cfg.downloads.lookahead_updates += 1
    assert campaign.pilot_profile(cfg) != original_profile

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(campaign, "run_one", fail)
    with pytest.raises(OSError, match="disk full"):
        campaign.pilot(cfg)
    failed = json.loads((cfg.output_root / "pilot" / "report.json").read_text())
    assert not failed["passed"]
    assert failed["error"] == "disk full"


def test_wandb_training_and_scale_comparison(tiny_campaign, monkeypatch):
    import sys  # noqa: PLC0415
    from types import SimpleNamespace  # noqa: PLC0415

    cfg, _ = tiny_campaign
    cfg.wandb.enabled = True
    runs = []

    def initialize(**kwargs):
        run = SimpleNamespace(summary={}, logs=[], url=None, kwargs=kwargs)
        run.log = run.logs.append
        run.define_metric = lambda *args, **kwargs: None
        run.finish = lambda **kwargs: None
        runs.append(run)
        return run

    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            init=initialize,
            Image=lambda path: path,
            Table=lambda **kwargs: kwargs,
        ),
    )
    results = [campaign.run_one(cfg, job) for job in campaign.jobs(cfg)[:2]]
    campaign.plot_comparisons(cfg, results, "pretraining")
    assert len(runs) == 3
    assert all(r.kwargs["group"].endswith("-pretraining") for r in runs)
    chart = runs[-1].logs[-1]
    rows = chart["comparison/data"]["data"]
    assert {row[3] for row in rows} == {campaign.COLORS[scale] for scale in (1, 2)}
    assert "comparison/curves" in chart
    assert any(
        "validation/category_math/offline_agreement" in log for log in runs[0].logs
    )


def test_standard_prepare_supports_indexes_without_prompt_lengths(tiny_campaign):
    from speculators.generator.data import Corpus  # noqa: PLC0415

    cfg, _ = tiny_campaign
    local = campaign.job_config(cfg, campaign.jobs(cfg)[0])
    cache = campaign_cache.CampaignCache(cfg.staging_dir / "files", cfg.downloads)
    try:
        corpus = Corpus(local, BankConfig.load(local.bank_config))
        for transfer in corpus.transfer.transfers:
            transfer.hub_store.shared_cache = cache
        corpus.prepare(tiny_drafter())
        assert corpus.summaries["last"].shape == (48, 24)
        index = int(corpus.summaries["indices"][0])
        boundary = corpus.rows[index]["prompt_length"]
        states = corpus.transfer.get_cached(index)["hidden_states"]
        torch.testing.assert_close(
            corpus.summaries["last"][0], states[boundary - 1, :-1].flatten().float()
        )
    finally:
        cache.close()
