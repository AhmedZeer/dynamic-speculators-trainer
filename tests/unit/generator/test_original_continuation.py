"""Original-pipeline parity, causal saved-prefix feedback and optimizer inheritance."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.bank.artifacts import file_digest, write_json
from speculators.generator import feedback_training, saved_response_feedback, study
from speculators.generator.continuation_state import restore_parent_optimizer
from speculators.generator.engine import Runtime
from speculators.generator.feedback_config import FeedbackStudy
from speculators.generator.online_feedback import OnlineFeedbackTrainer
from speculators.generator.original_continuation import original_schedule
from speculators.generator.serving.bundle import bundle_state, load_generator
from speculators.losses import resolve_loss_config
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_experiments import small_architecture
from tests.unit.generator.test_feedback_continuation import feedback_fixture  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401
from tests.unit.generator.test_studies import streaming_runtime

ROOT = Path(__file__).resolve().parents[3]


def test_v3_restores_original_pipeline_and_does_not_restart_warmup(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SPECULATORS_FEEDBACK_ROOT", str(tmp_path))
    folder = ROOT / "examples/eagle3-lora/feedback_continuation_v3"
    regular, feedback = (
        FeedbackStudy.load(folder / name) for name in ("regular.yaml", "feedback.yaml")
    )
    for cfg in (regular, feedback):
        assert cfg.inherit_parent_optimizer
        assert cfg.warmup_steps == 0
        assert cfg.steps == 1000
        assert cfg.train_per_category == 1000
        assert cfg.validation_interval == 125
        assert cfg.examples_per_update == 4
        assert cfg.wandb.project == "eagle3-generator-original-feedback-v3"
    assert regular.parent == feedback.parent
    assert not regular.feedback
    assert feedback.trainable == "generator_and_projector"


@pytest.fixture
def saved_batch(tmp_path):
    runtime, generator = streaming_runtime(feedback=True)
    runtime.drafter.requires_grad_(False)
    runtime.call_kwargs = {
        "ttt_steps": 3,
        "ttt_step_loss_decay": 1.0,
        "loss_config": resolve_loss_config("lk_hybrid", "eager"),
    }
    length = 20
    batch = {
        "hidden_states": torch.randn(1, length, 24),
        "input_ids": torch.randint(0, 16, (1, length)),
        "document_ids": torch.zeros(1, length, dtype=torch.long),
        "position_ids": torch.arange(1, length + 1)[None],
        "verifier_last_hidden_states": torch.randn(1, length, 8),
        "loss_mask": (torch.arange(length) >= 3)[None],
    }
    context = (torch.randn(1, 24), torch.tensor([3]))
    runtime.corpus = SimpleNamespace(
        context=lambda *args: context,
        microbatches=lambda *args, **kwargs: iter([(batch, True)]),
        attention_block_size=lambda: 1,
    )
    runtime.forward = Runtime.forward.__get__(runtime)
    definition = FeedbackStudy(
        workspace=tmp_path,
        parent={"model": str(tmp_path)},
        trainable="generator_and_projector",
        feedback=True,
        feedback_prefix_tokens=4,
        feedback_prefix_blocks=2,
    )
    return runtime, generator, definition, batch, context


@pytest.mark.parametrize("packed", [False, True])
def test_feedback_partition_preserves_original_loss_and_generator_gradients(
    saved_batch,
    packed,
):
    runtime, generator, cfg, batch, context = saved_batch
    if packed:
        batch["document_ids"][:, 10:] = 1
        batch["position_ids"][:, 10:] -= 10
        batch["loss_mask"][:, 10:13] = False
        context = (torch.cat([context[0], context[0] + 0.2]), torch.tensor([3, 3]))
        runtime.corpus.context = lambda *args: context
    factors = generator(*context, feedback=torch.zeros(3))
    _, expected, expected_metrics = runtime.forward(batch, factors)
    expected.backward()
    gradients = {
        name: p.grad.clone()
        for name, p in generator.named_parameters()
        if p.grad is not None and name != "feedback_encoder.weight"
    }
    generator.zero_grad(set_to_none=True)
    actual, count, actual_metrics, trace = saved_response_feedback.feedback_sample(
        runtime, cfg, generator, 0, training=True
    )
    assert actual == pytest.approx(float(expected.detach()), rel=1e-5)
    assert count == int(batch["loss_mask"].sum())
    assert sum(row["supervised_positions"] for row in trace) == count
    assert actual_metrics["full_acc_0_sum"] == expected_metrics["full_acc_0_sum"]
    assert trace[0]["feedback"] == [0, 0, 0]
    assert trace[1]["feedback"][2] == 1
    for name, p in generator.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=1e-4, atol=1e-6)
    assert generator.feedback_encoder.weight.grad.abs().sum() > 0


def test_feedback_never_reads_future_response_positions(saved_batch):
    runtime, generator, cfg, batch, _ = saved_batch
    with torch.no_grad():
        _, _, _, first = saved_response_feedback.feedback_sample(
            runtime, cfg, generator, 0, training=False
        )
        start = first[0]["stop"]
        batch["input_ids"][:, start:] = (batch["input_ids"][:, start:] + 3) % 16
        batch["hidden_states"][:, start:] += 100
        batch["verifier_last_hidden_states"][:, start:] *= -10
        _, _, _, second = saved_response_feedback.feedback_sample(
            runtime, cfg, generator, 0, training=False
        )
    assert first[0] == second[0]
    assert first[1]["feedback"] == second[1]["feedback"]


def test_regular_arm_calls_the_original_training_update(saved_batch, monkeypatch):
    runtime, generator, cfg, _, _ = saved_batch
    regular = cfg.model_copy(update={"feedback": False, "trainable": "generator"})
    seen = []
    expected = {"train/drafter_loss": 0.5}

    def original(*args):
        seen.append(args)
        return expected

    monkeypatch.setattr(saved_response_feedback, "train_update", original)
    episodes, optimizer = [{"indices": [0]}], object()
    values, observations = saved_response_feedback.update(
        runtime, regular, generator, optimizer, episodes, 7
    )
    assert values is expected
    assert observations == []
    assert seen == [(runtime, generator, optimizer, episodes, 7, "direct")]


def test_parent_adam_moments_are_mapped_by_name_and_projector_starts_fresh(
    saved_batch, tmp_path
):
    _, generator, cfg, _, _ = saved_batch
    source = type(generator)(
        24, generator.shapes, generator.rank, architecture=small_architecture()
    )
    old = torch.optim.AdamW(
        source.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    for p in source.parameters():
        old.state[p] = {
            "step": torch.tensor(4000.0),
            "exp_avg": torch.full_like(p, 0.2),
            "exp_avg_sq": torch.full_like(p, 0.3),
        }
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": source.state_dict(),
            "optimizer": old.state_dict(),
            "progress": 4000,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
        },
        path,
    )
    optimizer = torch.optim.AdamW(
        generator.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    receipt = restore_parent_optimizer(
        cfg, {"source_checkpoint_sha256": file_digest(path)}, generator, optimizer
    )
    assert receipt["new_parameters"] == ["feedback_encoder.weight"]
    for name, p in generator.named_parameters():
        if name == "feedback_encoder.weight":
            assert p not in optimizer.state
        else:
            assert optimizer.state[p]["step"].item() == 4000
            torch.testing.assert_close(
                optimizer.state[p]["exp_avg"], torch.full_like(p, 0.2)
            )
    with pytest.raises(ValueError, match="checksum"):
        restore_parent_optimizer(
            cfg, {"source_checkpoint_sha256": "bad"}, generator, optimizer
        )


def test_original_continuation_cannot_launch_rollouts(feedback_fixture, monkeypatch):  # noqa: F811
    definition, _, requests = feedback_fixture

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Original saved-response training must not construct a rollout trainer"
        )

    monkeypatch.setattr(OnlineFeedbackTrainer, "__init__", forbidden)
    cfg = definition.model_copy(update={"name": "original-no-rollouts"})
    result = feedback_training.train_feedback(cfg)
    assert result["job"]["training_mode"] == "original_saved_response"
    assert not (cfg.output / "online_traces").exists()
    inputs = study.load_local_inputs(study.campaign_settings(cfg))
    membership = json.loads((cfg.inputs / "membership.json").read_text())
    selected = {
        index
        for split in membership["groups"].values()
        for role in ("train", "tune")
        for index in split[role]
    }
    rows = [inputs.rows[index] for index in selected]
    allowed = {tuple(row["input_ids"]) for row in rows}
    allowed |= {tuple(row["input_ids"][: row["prompt_length"]]) for row in rows}
    assert all(tuple(tokens) in allowed for tokens in requests)
    execution = json.loads((cfg.output / "execution.json").read_text())
    assert execution["target_requests"] == "saved_sequences_only"


def test_original_schedule_keeps_original_cohorts_within_selected_membership(tmp_path):
    cfg = FeedbackStudy(
        workspace=tmp_path,
        parent={"model": str(tmp_path)},
        trainable="generator",
        examples_per_update=4,
        steps=1000,
    )
    runtime = SimpleNamespace(
        cfg=SimpleNamespace(
            seed=42,
            context=SimpleNamespace(min_examples=1, max_examples=8),
            optimization=SimpleNamespace(episodes_per_update=4),
        )
    )
    members = {
        name: {"train": list(range(n * 1000, (n + 1) * 1000))}
        for n, name in enumerate(("math", "code", "chat", "stem"))
    }
    subsets = [name + "-00001" for name in members]
    schedule = original_schedule(cfg, runtime, members, subsets)
    sizes = set()
    for step in range(1000):
        for episode in schedule.update(step):
            selected = episode["indices"]
            assert set(selected) <= set(
                members[episode["subset"].rsplit("-", 1)[0]]["train"]
            )
            sizes.add(len(selected))
    assert sizes == set(range(1, 9))


def test_full_original_stage_really_continues_parent_optimizer(feedback_fixture):  # noqa: F811
    definition, _, _ = feedback_fixture
    metadata, weights = bundle_state(definition.parent.model)
    source = load_generator(metadata, weights).requires_grad_(True)
    optimizer = torch.optim.AdamW(
        source.parameters(), lr=definition.lr, weight_decay=definition.weight_decay
    )
    for parameter in source.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(4000.0),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    checkpoint = Path(definition.parent.model) / "checkpoint.pt"
    torch.save(
        {
            "progress": 4000,
            "model": source.state_dict(),
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
        },
        checkpoint,
    )
    metadata["source_checkpoint_sha256"] = file_digest(checkpoint)
    write_json(Path(definition.parent.model) / "bundle.json", metadata)
    cfg = definition.model_copy(
        update={
            "name": "inherited-original",
            "feedback": True,
            "trainable": "generator_and_projector",
            "inherit_parent_optimizer": True,
            "warmup_steps": 0,
        }
    )
    result = feedback_training.train_feedback(cfg)
    assert result["completed_updates"] == 2
    saved = torch.load(cfg.output / "checkpoint.pt", weights_only=True)
    mapping = dict(
        zip(
            saved["parameter_names"],
            saved["optimizer"]["param_groups"][0]["params"],
            strict=True,
        )
    )
    for name, identifier in mapping.items():
        expected = 2 if name == "feedback_encoder.weight" else 4002
        assert saved["optimizer"]["state"][identifier]["step"].item() == expected
