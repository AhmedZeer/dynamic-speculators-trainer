"""Original random-cohort training with configurable heads and durable backups."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from typer.testing import CliRunner

from speculators.cli import app
from speculators.generator import campaign, publish, structure_study
from speculators.generator.campaign_data import EpisodeSchedule
from speculators.generator.config import Architecture
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.serving.bundle import bundle_state, load_generator
from speculators.generator.structure_study import StructureStudy, training_job
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401

ROOT = Path("examples/eagle3-lora/structure_ablations_v1")
MODULE_SETS = {
    "mlp": {"up_proj", "down_proj", "gate_proj"},
    "attention": {"q_proj", "k_proj", "v_proj", "o_proj"},
    "all": {
        "up_proj",
        "down_proj",
        "gate_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    },
}


def test_eleven_configs_share_original_budget_data_and_schedule():
    paths = sorted(ROOT.glob("*/*.yaml"))
    assert len(paths) == 11
    schedules, repos, names = [], set(), set()
    for path in paths:
        cfg = StructureStudy.load(path)
        plan = structure_study.train_structure(cfg, dry_run=True)
        job = plan["job"]
        assert job["training_examples"] == 4000
        assert job["updates"] == 1500
        assert job["subsets"] == job["validation_subsets"]
        assert all(name.endswith("-00000") for name in job["subsets"])
        assert "epochs" not in job
        assert "pretrained_path" not in job
        assert cfg.optimization.pretraining_validation_interval == 250
        assert cfg.optimization.checkpoint_interval == cfg.hub_upload_interval == 500
        assert cfg.optimization.episodes_per_update == 4
        assert cfg.optimization.warmup_updates == 100
        assert cfg.optimization.token_budget == 8192
        assert cfg.wandb.pretraining_project == "eagle3-generator-ablations-v3"
        assert cfg.wandb.enabled
        assert cfg.wandb.mode == "online"
        assert cfg.training_recipe.is_file()
        assert cfg.generator_dtype == "bfloat16"
        assert cfg.generator_batching
        baseline = Architecture().model_dump()
        if cfg.family == "architecture":
            field, number = path.stem.rsplit("-", 1)
            baseline[field] = int(number)
            assert cfg.target_modules == ["o_proj", "v_proj"]
        else:
            assert set(cfg.target_modules) == MODULE_SETS[path.stem]
        assert cfg.architecture.model_dump() == baseline
        corpus = SimpleNamespace(indices=lambda _: list(range(1000)))
        schedule = EpisodeSchedule(cfg, corpus, job["subsets"])
        schedules.append([schedule.update(i) for i in range(1500)])
        repos.add(cfg.repo_id)
        names.add(cfg.name)
    assert len(repos) == len(names) == 11
    assert all(schedule == schedules[0] for schedule in schedules)
    result = CliRunner().invoke(
        app, ["generator", "train-structure", "--config", str(paths[0]), "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["job"]["updates"] == 1500


@pytest.mark.parametrize(
    ("variant", "migrate"), [*((name, False) for name in MODULE_SETS), ("mlp", True)]
)
def test_module_training_exports_and_resumes_failed_upload(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
    variant,
    migrate,
):
    cfg, _, _ = online_campaign
    values = cfg.model_dump()
    provenance = tmp_path / "extractor"
    provenance.mkdir()
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text("test provenance")
    drafter = tmp_path / "drafter"
    drafter.mkdir()
    (drafter / "config.json").write_text("{}")
    values.update(
        name=f"test-{variant}",
        family="modules",
        repo_id="example/structure",
        provenance_dir=provenance,
        target_modules=sorted(MODULE_SETS[variant]),
    )
    study = StructureStudy.model_validate(values)
    study.generator_dtype = "float32" if migrate else "bfloat16"
    study.generator_batching = not migrate
    factory = campaign.Runtime

    def bf16_runtime(*args, **kwargs):
        runtime = factory(*args, **kwargs)
        runtime.dtype = getattr(torch, runtime.cfg.generator_dtype)
        return runtime

    monkeypatch.setattr(campaign, "Runtime", bf16_runtime)
    study.online.drafter_model_path = drafter
    study.optimization.pretraining_updates = 4
    study.optimization.pretraining_validation_interval = 2
    study.optimization.checkpoint_interval = study.hub_upload_interval = 2
    study.context.min_examples = 1
    study.context.max_examples = 8
    job = training_job(study)
    uploaded, states = [], []
    fail = [True]

    class API:
        def whoami(self):
            return {"name": "test"}

        def create_repo(self, **kwargs):
            return None

        def upload_folder(self, **kwargs):
            path = Path(kwargs["folder_path"])
            metadata, weights = bundle_state(path)
            assert set(metadata["shapes"]) == MODULE_SETS[variant]
            assert all(
                torch.count_nonzero(weights[f"heads.{name}.B.bias"]) > 0
                for name in MODULE_SETS[variant]
            )
            generator = load_generator(metadata, weights)
            factors = generator(
                torch.randn(2, metadata["input_dim"]), torch.tensor([2, 3])
            )
            assert set(factors) == MODULE_SETS[variant]
            saved = torch.load(path / "checkpoint.pt", weights_only=True)
            assert saved["optimizer"]["state"]
            expected_dtype = (
                torch.float32 if migrate and saved["progress"] == 2 else torch.bfloat16
            )
            assert all(t.dtype == expected_dtype for t in weights.values())
            assert all(t.dtype == expected_dtype for t in saved["model"].values())
            assert all(
                state["exp_avg"].dtype == state["exp_avg_sq"].dtype == expected_dtype
                for state in saved["optimizer"]["state"].values()
            )
            assert saved["torch_rng"].numel()
            assert (path / "provenance/vllm/vllm_command.txt").is_file()
            if fail[0]:
                fail[0] = False
                raise RuntimeError("upload interrupted")
            uploaded.append(saved["progress"])
            states.append(weights)
            return SimpleNamespace(oid="test", commit_url="https://hf.test/commit")

    monkeypatch.setattr(structure_study, "HfApi", API)
    monkeypatch.setattr(publish, "HfApi", API)
    with pytest.raises(RuntimeError, match="upload interrupted"):
        structure_study.train_structure(study)
    run = study.output_root / "pretraining" / study.name
    interrupted = torch.load(run / "checkpoint.pt", weights_only=True)
    assert interrupted["progress"] == 2
    study.generator_dtype = "bfloat16"
    study.generator_batching = True
    result = structure_study.train_structure(study)
    assert uploaded == [2, 4]
    assert result["completed_updates"] == 4
    timing = json.loads((run / "timing.json").read_text())
    for row in timing:
        expected_calls = 4 if migrate and row["update"] <= 2 else 1
        assert row["train/generator_forward_calls"] == expected_calls
        assert row["train/generator_backward_calls"] == expected_calls
    if migrate:
        resumed = torch.load(run / "checkpoint.pt", weights_only=True)
        assert resumed["identity"] == interrupted["identity"]
        assert list((run / "provenance").glob("resume-*/configuration.json"))
    assert [record["update"] for record in result["history"]] == [0, 2, 4]
    assert result["hub_checkpoints"]["step-00000004"]["commit_sha"] == "test"
    assert any(not torch.equal(states[0][key], states[1][key]) for key in states[0])
    identity = json.loads((run / "resolved_experiment.json").read_text())
    assert set(identity["lora"]["target_modules"]) == MODULE_SETS[variant]
    assert identity["job"]["tracking_group"] == "structure-v1/modules"
    configuration = json.loads((run / "configuration.json").read_text())
    assert configuration["wandb"]["group"] == "structure-v1/modules"
    assert job["subsets"] == identity["job"]["subsets"]


def test_module_override_is_part_of_checkpoint_identity(online_campaign):  # noqa: F811
    from speculators.generator.engine import run_identity  # noqa: PLC0415

    cfg, _, _ = online_campaign
    inputs = load_local_inputs(cfg)
    job = campaign.jobs(cfg)[0]
    corpus = SimpleNamespace(identity=inputs.identity)
    first = run_identity(cfg, inputs.banks["math"], corpus, job)
    assert "target_modules" not in first["configuration"]
    cfg.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    second = run_identity(cfg, inputs.banks["math"], corpus, job)
    assert first != second
    assert second["lora"]["target_modules"] == sorted(cfg.target_modules)


def test_hub_snapshot_is_resumable_despite_snapshot_result(tmp_path, monkeypatch):
    from speculators.generator import engine  # noqa: PLC0415

    identity = {"test": "identity"}
    (tmp_path / "resolved_experiment.json").write_text(json.dumps(identity))
    (tmp_path / "result.json").write_text(json.dumps({"status": "training_snapshot"}))
    with pytest.raises(FileNotFoundError, match="checkpoint.pt"):
        engine.begin_run(tmp_path, identity)
    (tmp_path / "checkpoint.pt").touch()
    monkeypatch.setattr(engine, "save_train_command", lambda _: None)
    assert engine.begin_run(tmp_path, identity)
    (tmp_path / "result.json").write_text(json.dumps({"completed_updates": 1500}))
    assert not engine.begin_run(tmp_path, identity)
