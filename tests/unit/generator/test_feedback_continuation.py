"""Current-policy feedback, projector isolation, source loading and recovery."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from typer.testing import CliRunner

from speculators.bank.artifacts import file_digest, write_json
from speculators.cli import app
from speculators.generator import (
    feedback_training,
    online_activations,
    study,
    target_verifier,
)
from speculators.generator.feedback import AdapterExperiment
from speculators.generator.feedback_config import FeedbackStudy
from speculators.generator.local_campaign import load_local_inputs
from speculators.generator.model import LoRAGenerator
from speculators.generator.online_activations import ActivationRAM
from speculators.generator.online_feedback import OnlineFeedbackTrainer
from speculators.generator.serving.bundle import read_bundle
from speculators.generator.serving.endpoint import validate_adapter_delta
from speculators.generator.serving.experiments import ExperimentalAdapterController
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_experiments import small_architecture, tiny_drafter
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401
from tests.unit.generator.test_serving import draft, write_bundle
from tests.unit.generator.test_studies import propose, streaming_runtime

ROOT = Path(__file__).resolve().parents[3]


def test_two_configs_share_4k_budget_and_parent_but_separate_trainable_weights(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SPECULATORS_FEEDBACK_ROOT", str(tmp_path))
    root = ROOT / "examples/eagle3-lora/feedback_continuation_v1"
    regular = FeedbackStudy.load(root / "regular.yaml")
    projector = FeedbackStudy.load(root / "projector_only.yaml")
    for cfg in (regular, projector):
        assert cfg.steps == 1000
        assert cfg.train_per_category == 1000
        assert cfg.examples_per_update == 8
        assert cfg.parent.optimizer_step == 4000
        assert cfg.replay is False
        assert cfg.conditioning == "prompt"
        assert cfg.rollout_tokens == 64
        assert cfg.wandb.project == "eagle3-generator-online-feedback-v1"
    assert regular.parent == projector.parent
    assert regular.trainable == "generator"
    assert projector.trainable == "projector"
    assert regular.feedback is False
    assert projector.feedback is True
    assert regular.output != projector.output
    assert regular.repo_id != projector.repo_id
    result = CliRunner().invoke(
        app,
        [
            "generator",
            "train-feedback",
            "--config",
            str(root / "projector_only.yaml"),
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    values = json.loads(result.output)
    assert values["training_examples"] == 4000
    assert values["final_lineage_step"] == 5000


def test_feedback_collection_keeps_prompt_fixed_and_never_supervises_invalid_suffix(
    tmp_path,
):
    runtime, generator = streaming_runtime(
        feedback=True, backend="simple_flex_attention"
    )
    runtime.bank = SimpleNamespace(hidden_states=SimpleNamespace(sequence_length=64))
    pool = ThreadPoolExecutor(max_workers=2)
    runtime.corpus = SimpleNamespace(
        rows=[{"input_ids": [1, 2, 3], "prompt_length": 2, "primary_id": "p"}],
        cache=SimpleNamespace(ram=ActivationRAM(0.01), pool=pool),
    )
    cfg = FeedbackStudy(
        workspace=tmp_path,
        parent={"model": str(tmp_path)},
        trainable="projector",
        feedback=True,
        rollout_tokens=8,
        loss_implementation="eager",
    )
    trainer = OnlineFeedbackTrainer.__new__(OnlineFeedbackTrainer)
    trainer.runtime, trainer.study, trainer.generator = runtime, cfg, generator
    from speculators.generator.streaming import StreamingDraft  # noqa: PLC0415
    from speculators.losses import resolve_loss_config  # noqa: PLC0415

    trainer.stream = StreamingDraft(runtime)
    trainer.eos, trainer.prompt_futures = [], {}
    trainer.loss = next(iter(resolve_loss_config("lk_hybrid", "eager").values()))[0]
    trainer.extract = lambda tokens: (
        torch.arange(len(tokens) * 32).reshape(-1, 4, 8).float() / 100
    )
    trainer.teacher = lambda states: torch.tensor([[4.0] + [0.0] * 15]).expand(
        len(states), -1
    )
    native = trainer.stream.propose
    forced_inputs = []

    def wrong_proposals(*args, **kwargs):
        proposed, logits = native(*args, **kwargs)
        if kwargs.get("forced") is None:
            return [1] * kwargs["count"], logits
        forced_inputs.append(kwargs["forced"])
        return proposed, logits

    trainer.stream.propose = wrong_proposals
    generator.requires_grad_(False)
    generator.feedback_encoder.requires_grad_(True)
    initial = {k: v.clone() for k, v in generator.state_dict().items()}
    optimizer = torch.optim.AdamW(generator.feedback_encoder.parameters(), lr=0.01)
    try:
        loss, count, trace = trainer.train_episode(0, 1)
        assert loss > 0
        assert count > 0
        assert trace["rounds"][0]["feedback"] == [0, 0, 0]
        assert trace["rounds"][1]["feedback"] == [0, 0, 1]
        assert all(tokens == [0] for tokens in forced_inputs)
        assert all(row["accepted"] == 0 for row in trace["rounds"])
        assert generator.feedback_encoder.weight.grad.abs().sum() > 0
        optimizer.step()
        for key, value in generator.state_dict().items():
            if key != "feedback_encoder.weight":
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        seen = []
        hook = generator.register_forward_pre_hook(
            lambda m, args, kwargs: seen.append(
                (
                    args[0].clone(),
                    args[1].item(),
                    kwargs["feedback"].clone(),
                    m.feedback_encoder.weight.clone(),
                )
            ),
            with_kwargs=True,
        )
        trainer.collect(0, gradient_blocks=False)
        hook.remove()
        assert all(length == 2 for _, length, _, _ in seen)
        assert all(torch.equal(summary, seen[0][0]) for summary, _, _, _ in seen)
        torch.testing.assert_close(seen[0][3], generator.feedback_encoder.weight)
        assert not torch.equal(seen[0][3], initial["feedback_encoder.weight"])
    finally:
        pool.shutdown(wait=True)


def test_serving_feedback_refresh_retains_original_prompt(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    metadata, _ = write_bundle(bundle)
    state = load_file(bundle / "model.safetensors")
    state["feedback_encoder.weight"] = torch.zeros(
        metadata["architecture"]["condition_dim"], 3
    )
    save_file(state, bundle / "model.safetensors")
    metadata["feedback_dim"] = 3
    metadata["weights_sha256"] = file_digest(bundle / "model.safetensors")
    write_json(bundle / "bundle.json", metadata)
    policy = AdapterExperiment(feedback=True, refresh="prompt").model_dump(mode="json")
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG",
        json.dumps(
            {
                "adapter_experiment": policy,
                "adapter_rounds_dir": str(tmp_path / "rounds"),
            }
        ),
    )
    ctl = ExperimentalAdapterController(bundle, draft())
    before = {
        "configuration": {"kind": "generator", "adapter_experiment": policy},
        "workers": [{"adapter_totals": dict(ctl.totals)}],
    }
    seen = []
    ctl.generator.register_forward_pre_hook(
        lambda m, args, kwargs: seen.append(
            (args[0].clone(), args[1].clone(), kwargs["feedback"].clone())
        ),
        with_kwargs=True,
    )
    prompt = torch.randn(2, 12)
    propose(ctl, torch.arange(2), prompt)
    propose(ctl, torch.arange(2, 6), torch.randn(4, 12), torch.tensor([2]))
    torch.testing.assert_close(seen[1][0], prompt[-1:])
    assert seen[1][1].item() == 2
    assert seen[1][2].tolist() == pytest.approx([1 / 3, 1 / 3, 1])
    ctl.finish("request")
    after = {**before, "workers": [{"adapter_totals": dict(ctl.totals)}]}
    assert validate_adapter_delta(before, after)["generator_invocations"] == 2
    assert ctl.prompt_summary is None


@pytest.fixture
def feedback_fixture(online_campaign, tmp_path, monkeypatch):  # noqa: F811
    cfg, _, requests = online_campaign
    cfg.online.drafter_model_path = tmp_path / "source-model"
    cfg.online.drafter_model_path.mkdir()
    write_json(
        cfg.online.drafter_model_path / "config.json", tiny_drafter().config.to_dict()
    )

    def settings(s):
        local = cfg.model_copy(deep=True)
        local.output_root, local.input_root = s.output, s.inputs / "metadata"
        local.staging_dir = s.workspace / "cache"
        local.wandb.enabled = False
        local.optimization.episodes_per_update = s.examples_per_update
        return local

    monkeypatch.setattr(study, "campaign_settings", settings)
    monkeypatch.setattr(feedback_training, "campaign_settings", settings)
    from speculators.generator import campaign  # noqa: PLC0415

    monkeypatch.setattr(study, "Runtime", campaign.Runtime)
    monkeypatch.setattr(
        target_verifier,
        "load_model_layers",
        lambda *a: {"lm_head.weight": torch.zeros(16, 8)},
    )
    monkeypatch.setattr(
        target_verifier,
        "generate_hidden_states",
        online_activations.generate_hidden_states,
    )
    parent = tmp_path / "parent"
    parent.mkdir()
    definition = FeedbackStudy(
        workspace=tmp_path / "online-study",
        name="regular",
        parent={"model": str(parent)},
        trainable="generator",
        steps=2,
        train_per_category=8,
        tune_per_category=1,
        test_per_category=1,
        examples_per_update=4,
        rollout_tokens=8,
        validation_interval=1,
        checkpoint_interval=1,
        loss_implementation="eager",
        wandb={"enabled": False},
    )
    provenance = definition.workspace / "provenance/vllm-extractor"
    provenance.mkdir(parents=True)
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text("synthetic provenance")
    inputs = load_local_inputs(settings(definition))
    runtime, _, _ = study.build_runtime(
        definition, inputs, architecture=small_architecture()
    )
    generator = LoRAGenerator(
        24,
        runtime.shapes,
        runtime.bank.lora.rank,
        small_architecture(),
        runtime.cfg.context.max_examples,
    )
    with torch.no_grad():
        for head in generator.heads.values():
            head["A"].weight.normal_(std=0.03)
            head["B"].weight.normal_(std=0.03)
    save_file(generator.state_dict(), parent / "model.safetensors")
    manifest = inputs.manifests["math"]
    metadata = {
        "version": 1,
        "kind": "generator",
        "condition": "last",
        "input_dim": 24,
        "layer_ids": runtime.bank.hidden_states.layer_ids[:-1],
        "feedback_dim": 0,
        "architecture": small_architecture().model_dump(),
        "rank": runtime.bank.lora.rank,
        "alpha": runtime.bank.lora.alpha,
        "shapes": {k: list(v) for k, v in runtime.shapes.items()},
        "max_examples": runtime.cfg.context.max_examples,
        "models": {
            role: {
                "id": manifest["cache_spec"]["models"][role],
                "revision": manifest["revisions"][role + "_sha"],
            }
            for role in ("target", "drafter")
        },
        "weights_sha256": file_digest(parent / "model.safetensors"),
    }
    write_json(parent / "bundle.json", metadata)
    runtime.corpus.cache.close()
    return definition, generator.state_dict(), requests


@pytest.mark.parametrize("trainable", ["generator", "projector"])
def test_full_online_continuation_freeze_export_and_resume(
    feedback_fixture, trainable, monkeypatch
):
    definition, parent_state, requests = feedback_fixture
    cfg = definition.model_copy(
        update={
            "name": trainable,
            "trainable": trainable,
            "feedback": trainable == "projector",
        }
    )
    original_upload = study.upload_snapshot
    triggered = []

    def interruption(*args):
        if not triggered:
            triggered.append(True)
            raise RuntimeError("interrupted after durable checkpoint")
        return original_upload(*args)

    monkeypatch.setattr(study, "upload_snapshot", interruption)
    with pytest.raises(RuntimeError, match="durable checkpoint"):
        feedback_training.train_feedback(cfg)
    saved = torch.load(cfg.output / "checkpoint.pt", weights_only=True)
    assert saved["progress"] == 1
    result = feedback_training.train_feedback(cfg, resume=True)
    assert result["completed_updates"] == 2
    assert result["lineage_optimizer_step"] == 4002
    assert [r["optimizer_step"] for r in result["history"]] == [0, 1, 2]
    saved = torch.load(cfg.output / "checkpoint.pt", weights_only=True)
    assert saved["optimizer"]["state"]
    assert {int(slot["step"]) for slot in saved["optimizer"]["state"].values()} == {2}
    if trainable == "projector":
        assert saved["parameter_names"] == ["feedback_encoder.weight"]
        assert saved["model"]["feedback_encoder.weight"].abs().sum() > 0
        for key in parent_state:
            torch.testing.assert_close(
                saved["model"][key], parent_state[key], rtol=0, atol=0
            )
    bundle = read_bundle(cfg.output / "checkpoints/step-00000002/bundle")
    assert bundle["feedback_dim"] == (3 if trainable == "projector" else 0)
    before = len(requests)
    assert feedback_training.train_feedback(cfg, resume=True)["completed_updates"] == 2
    assert len(requests) == before
    with pytest.raises(FileExistsError, match="resume"):
        feedback_training.train_feedback(cfg)
    trace = json.loads((cfg.output / "online_traces/update-00000002.json").read_text())
    assert {row["policy_optimizer_step"] for row in trace} == {1}
    membership = json.loads((cfg.inputs / "membership.json").read_text())
    assert {len(g["train"]) for g in membership["groups"].values()} == {8}
