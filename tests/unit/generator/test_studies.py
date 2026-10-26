"""Causal refresh, replay alignment, checkpoint branching and independent configs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.func import functional_call
from transformers import DynamicCache

from speculators.bank.artifacts import digest, write_json
from speculators.benchmark import tracking as benchmark_tracking
from speculators.benchmark.config import Suite
from speculators.experiment_config import load_experiment_yaml
from speculators.generator.feedback import AdapterExperiment, VerifierFeedback
from speculators.generator.model import (
    LoRAGenerator,
    factor_parameters,
    install_adapters,
)
from speculators.generator.serving.endpoint import validate_adapter_delta
from speculators.generator.serving.experiments import ExperimentalAdapterController
from speculators.generator.streaming import StreamingDraft, detached_cache
from speculators.generator.study import initialize_parent, update_members
from speculators.generator.study_config import GeneratorStudy
from speculators.generator.study_replay import ReplayTrainer
from speculators.losses import resolve_loss_config
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_experiments import small_architecture, tiny_drafter
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401
from tests.unit.generator.test_serving import draft, write_bundle

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

ROOT = Path(__file__).resolve().parents[3]


def test_all_study_configs_are_independent_and_budgets_are_explicit(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SPECULATORS_STUDY_ROOT", str(tmp_path))
    root = ROOT / "examples/eagle3-lora/generator_ablations_v4"
    configs = [
        GeneratorStudy.load(path) for path in sorted((root / "training").glob("*.yaml"))
    ]
    assert len(configs) == 7
    assert {c.steps for c in configs} == {1000}
    assert {c.train_per_category for c in configs} == {1000}
    assert {c.examples_per_update for c in configs} == {8}
    assert (
        sum(c.steps - (c.initial_step if c.initial_checkpoint else 0) for c in configs)
        == 5500
    )
    assert len({c.repo_id for c in configs}) == 7
    for path in (root / "inference").rglob("*.yaml"):
        suite = Suite.load(path)
        assert len(suite.models) == 1
        assert suite.settings.concurrency_values == [1]
        assert suite.benchmarks[0].categories == ["math", "chat"]
        assert suite.wandb.enabled
    assert not (root / "ablations").exists()


def test_config_inheritance_replaces_lists_and_rejects_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\nitems: [1, 2]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 3}\nitems: [4]\n")
    assert load_experiment_yaml(tmp_path / "child.yaml") == {
        "a": {"x": 1, "y": 3},
        "items": [4],
    }
    (tmp_path / "base.yaml").write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_experiment_yaml(tmp_path / "child.yaml")


def test_schedule_visits_the_4k_pool_twice_without_shared_adapters(tmp_path):
    cfg = GeneratorStudy(workspace=tmp_path)
    members = {
        d: {"train": list(range(k * 1000, (k + 1) * 1000))}
        for k, d in enumerate(sorted(("math", "code", "stem", "chat")))
    }
    for first in (0, 500):
        indices = [
            e["indices"][0]
            for step in range(first, first + 500)
            for e in update_members(cfg, members, step)
        ]
        assert sorted(indices) == list(range(4000))
    assert update_members(cfg, members, 99) == update_members(cfg, members, 99)


def test_feedback_is_causal_bounded_and_resettable():
    policy = AdapterExperiment(
        scale_policy="acceptance_ema", min_feedback_rounds=1, ema_decay=0
    )
    state = VerifierFeedback(policy)
    assert state.tensor("cpu").tolist() == [0, 0, 0]
    state.observe(0, 3)
    assert state.scale == 1.5
    state.observe(3, 3)
    assert state.scale == 2
    for _ in range(20):
        state.observe(3, 3)
    assert state.scale == 4
    assert VerifierFeedback(policy).rounds == 0
    with pytest.raises(ValueError, match="counts"):
        state.observe(4, 3)


def experimental_controller(tmp_path, monkeypatch, policy):
    bundle = tmp_path / "bundle"
    write_bundle(bundle)
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG",
        json.dumps(
            {
                "adapter_experiment": policy,
                "adapter_rounds_dir": str(tmp_path / "rounds"),
            }
        ),
    )
    return ExperimentalAdapterController(bundle, draft())


def propose(ctl, positions, states, rejected=None):
    ctl.prepare_proposal(
        ["request"],
        [2],
        positions,
        states,
        {
            "target_token_ids": torch.tensor([1, 2]),
            "num_rejected_tokens_gpu": rejected,
            "num_speculative_tokens": 3,
        },
    )


def test_refresh_uses_only_committed_states_and_resets_on_finish(tmp_path, monkeypatch):
    ctl = experimental_controller(tmp_path, monkeypatch, {"refresh": "prefix"})
    before = {
        "configuration": {
            "kind": "generator",
            "adapter_experiment": {"refresh": "prefix"},
        },
        "workers": [{"adapter_totals": dict(ctl.totals)}],
    }
    seen = []
    ctl.generator.register_forward_pre_hook(
        lambda m, args: seen.append(args[0].clone())
    )
    propose(ctl, torch.arange(2), torch.randn(2, 12))
    states = torch.randn(4, 12)
    states[2:] = 10000  # Rejected suffix must never reach the generator.
    propose(ctl, torch.arange(2, 6), states, torch.tensor([2]))
    torch.testing.assert_close(seen[-1], states[1:2])
    assert ctl.feedback_state.acceptance == pytest.approx(1 / 3)
    assert ctl.totals["generator_invocations"] == 2
    assert ctl.totals["adapter_refreshes"] == 1
    ctl.finish("request")
    after = {**before, "workers": [{"adapter_totals": dict(ctl.totals)}]}
    assert validate_adapter_delta(before, after)["feedback_rounds"] == 1
    assert ctl.feedback_state.rounds == 0
    assert ctl.last_position is None
    assert len(list((tmp_path / "rounds").glob("*.json"))) == 1


def test_scale_only_changes_strength_without_regenerating(tmp_path, monkeypatch):
    ctl = experimental_controller(
        tmp_path,
        monkeypatch,
        {
            "scale_policy": "acceptance_ema",
            "min_feedback_rounds": 1,
            "ema_decay": 0,
        },
    )
    propose(ctl, torch.arange(2), torch.randn(2, 12))
    factors = ctl.current_factors
    propose(ctl, torch.arange(2, 6), torch.randn(4, 12), torch.tensor([3]))
    assert ctl.current_factors is factors
    assert ctl.totals["generator_invocations"] == 1
    assert ctl.merger.scale == 1.5
    for name, weight in ctl.merger.weights.items():
        a, b = factors[name]
        torch.testing.assert_close(weight, ctl.merger.base[name] + 1.5 * b @ a)
    ctl.finish("request")
    for name, weight in ctl.merger.weights.items():
        torch.testing.assert_close(weight, ctl.merger.base[name], rtol=0, atol=0)


def streaming_runtime(feedback=False, backend="eager"):
    model = tiny_drafter(backend=backend).eval().requires_grad_(False)
    paths = install_adapters(model, 2, 4, 0)
    shapes = {
        name: (
            model.get_submodule(path).base.in_features,
            model.get_submodule(path).base.out_features,
        )
        for name, path in paths.items()
    }
    runtime = SimpleNamespace(
        drafter=model, paths=paths, device=torch.device("cpu"), dtype=torch.float32
    )
    generator = LoRAGenerator(
        24, shapes, 2, small_architecture(), feedback_dim=3 if feedback else 0
    )
    with torch.no_grad():
        for head in generator.heads.values():
            head["A"].weight.normal_(std=0.03)
            head["B"].weight.normal_(std=0.03)
    return runtime, generator


@pytest.mark.parametrize("backend", ["eager", "simple_flex_attention"])
def test_streaming_prefill_matches_core_and_replays_identical_logits(backend):
    runtime, generator = streaming_runtime(backend=backend)
    summary = torch.randn(1, 24)
    factors = generator(summary, torch.tensor([3]))
    states, ids = torch.randn(3, 24), torch.tensor([[1, 2, 3]])
    captured = []
    hook = runtime.drafter.lm_head.register_forward_hook(
        lambda m, a, out: captured.append(out)
    )
    with torch.no_grad():
        functional_call(
            runtime.drafter,
            factor_parameters(runtime.paths, factors),
            (),
            {
                "hidden_states": states[None],
                "input_ids": ids,
                "document_ids": torch.zeros_like(ids),
                "position_ids": torch.arange(3)[None],
                "ttt_steps": 1,
            },
        )
    expected = captured[-1][:, -1:]
    hook.remove()
    stream, cache = StreamingDraft(runtime), DynamicCache()
    proposals, logits = stream.propose(states, ids[0].tolist(), 0, cache, factors)
    torch.testing.assert_close(logits[:, :1], expected)
    _, repeated = stream.propose(
        states, ids[0].tolist(), 0, DynamicCache(), factors, forced=proposals
    )
    torch.testing.assert_close(repeated, logits)
    logits.square().mean().backward()
    assert generator.heads["v_proj"]["B"].weight.grad.abs().sum() > 0
    cache.crop(-(cache.get_seq_length() - 3))
    detached = detached_cache(cache)
    assert detached.get_seq_length() == 3
    assert not detached.layers[0].values.requires_grad


def test_replay_uses_actual_draft_prefixes_and_trains_feedback(tmp_path):
    runtime, generator = streaming_runtime(feedback=True)
    runtime.bank = SimpleNamespace(hidden_states=SimpleNamespace(sequence_length=64))
    runtime.corpus = SimpleNamespace(
        rows=[{"input_ids": [1, 2, 3], "prompt_length": 2, "primary_id": "train"}]
    )
    study = GeneratorStudy(
        name="replay",
        workspace=tmp_path,
        replay=True,
        conditioning="prefix",
        feedback=True,
        initial_checkpoint=tmp_path / "parent.pt",
        loss_implementation="eager",
        replay_rounds_per_episode=2,
    )
    trainer = ReplayTrainer.__new__(ReplayTrainer)
    trainer.runtime, trainer.study, trainer.generator = runtime, study, generator
    trainer.stream = StreamingDraft(runtime)
    trainer.head = runtime.drafter.verifier_lm_head.weight
    trainer.loss = next(iter(resolve_loss_config("lk_hybrid", "eager").values()))[0]
    trainer.eos, trainer.trace_records = [], []
    extracted = []

    def extract(tokens):
        extracted.append(list(tokens))
        # A causal synthetic teacher: token t's state never reads later tokens.
        return torch.stack(
            [torch.arange(32).reshape(4, 8).float() / 100 + t / 10 for t in tokens]
        )

    trainer.extract = extract
    result = trainer.episode(0, 501, 0, 1)
    assert result["rounds"] == 2
    assert result["tokens"] == 6
    trace = trainer.trace_records[0]["rounds"]
    assert trace[0]["feedback_input"] == [0, 0, 0]
    assert trace[1]["feedback_input"][2] == 1
    assert extracted[1][-3:] == trace[0]["proposed"]
    assert generator.feedback_encoder.weight.grad.abs().sum() > 0


def test_branch_maps_optimizer_moments_by_name_and_preserves_initial_function(tmp_path):
    runtime, parent = streaming_runtime()
    optimizer = torch.optim.AdamW(parent.parameters(), lr=1e-4)
    sum(p.square().sum() for p in parent.parameters()).backward()
    optimizer.step()
    study = GeneratorStudy(
        workspace=tmp_path,
        replay=True,
        conditioning="prefix",
        feedback=True,
        initial_checkpoint=tmp_path / "checkpoint.pt",
    )
    source_study = study.model_dump(mode="json")
    source_study.update(loss="lk_hybrid", replay=False, arm="generator")
    write_json(tmp_path / "resolved_experiment.json", {"study": source_study})
    torch.save(
        {
            "progress": 500,
            "model": parent.state_dict(),
            "optimizer": optimizer.state_dict(),
            "parameter_names": list(parent.state_dict()),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": [],
        },
        study.initial_checkpoint,
    )
    child = LoRAGenerator(24, parent.shapes, 2, small_architecture(), feedback_dim=3)
    child_optimizer = torch.optim.AdamW(child.parameters(), lr=1e-4)
    assert initialize_parent(study, child, child_optimizer) == 500
    for name, value in parent.state_dict().items():
        torch.testing.assert_close(value, child.state_dict()[name], rtol=0, atol=0)
    parent_parameters = dict(parent.named_parameters())
    for name, param in child.named_parameters():
        if name in parent_parameters:
            torch.testing.assert_close(
                child_optimizer.state[param]["exp_avg"],
                optimizer.state[parent_parameters[name]]["exp_avg"],
            )
    assert not child.feedback_encoder.weight.count_nonzero()


def test_wandb_retry_keeps_completed_measurement(tmp_path, monkeypatch):
    entry = {
        "status": "completed",
        "output": "measurement",
        "model": "model",
        "benchmark": "nemotron",
        "category": "nemo-math",
    }
    state = {"tasks": {"key": entry}}
    write_json(
        tmp_path / "measurement/results.json",
        {"groups": {"nemo-math": {"performance": []}}},
    )
    plan = {"configuration": {"wandb": {"enabled": True}}}

    def failed(*args):
        raise RuntimeError("tracking unavailable")

    monkeypatch.setattr(benchmark_tracking, "publish_measurement", failed)
    with pytest.raises(RuntimeError, match="tracking unavailable"):
        benchmark_tracking.publish_completed(tmp_path, plan, state)
    assert entry["status"] == "completed"
    calls = []
    monkeypatch.setattr(
        benchmark_tracking, "publish_measurement", lambda *args: calls.append(args)
    )
    benchmark_tracking.publish_completed(tmp_path, plan, state)
    benchmark_tracking.publish_completed(tmp_path, plan, state)
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("arm", "loss"),
    [
        ("generator", "lk_hybrid"),
        ("generator", "kl_div"),
        ("generator", "ce"),
        ("lora", "lk_hybrid"),
    ],
)
def test_full_single_experiment_checkpoint_export_and_recovery(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
    arm,
    loss,
):
    from speculators.generator import campaign, study_inputs  # noqa: PLC0415
    from speculators.generator import study as module  # noqa: PLC0415
    from speculators.generator.local_campaign import load_local_inputs  # noqa: PLC0415
    from speculators.generator.serving.bundle import read_bundle  # noqa: PLC0415

    cfg, _, requests = online_campaign
    definition = GeneratorStudy(
        workspace=tmp_path / "study",
        name="prompt_lk",
        arm=arm,
        loss=loss,
        steps=2,
        train_per_category=8,
        tune_per_category=2,
        test_per_category=2,
        examples_per_update=4,
        validation_interval=1,
        checkpoint_interval=1,
        loss_implementation="eager",
        wandb={"enabled": False},
    )
    cfg.online.drafter_model_path = tmp_path / "source-model"
    cfg.online.drafter_model_path.mkdir()
    write_json(
        cfg.online.drafter_model_path / "config.json", tiny_drafter().config.to_dict()
    )
    provenance = definition.workspace / "provenance/vllm-extractor"
    provenance.mkdir(parents=True)
    for name in ("vllm_command.txt", "vllm.patch", "checkpoint_sha256.txt"):
        (provenance / name).write_text("synthetic provenance")

    def settings(s):
        local = cfg.model_copy(deep=True)
        local.output_root = s.output
        local.input_root = s.inputs / "metadata"
        local.staging_dir = s.workspace / "cache"
        local.wandb.enabled = False
        local.optimization.episodes_per_update = s.examples_per_update
        return local

    monkeypatch.setattr(module, "campaign_settings", settings)
    monkeypatch.setattr(study_inputs, "campaign_settings", settings)
    monkeypatch.setattr(module, "Runtime", campaign.Runtime)
    inputs = load_local_inputs(settings(definition))
    split = study_inputs.membership(definition, inputs)
    split["identity"] = digest(split)
    write_json(definition.inputs / "membership.json", split)
    result = module.train_study(definition)
    assert result["completed_updates"] == 2
    bundle = definition.output / "checkpoints/step-00000002/bundle"
    assert read_bundle(bundle)["kind"] == (
        "generator" if arm == "generator" else "static_lora"
    )
    checkpoint = torch.load(definition.output / "checkpoint.pt", weights_only=True)
    assert checkpoint["progress"] == 2
    assert checkpoint["optimizer"]["state"]
    assert (definition.output / "checkpoints/step-00000001/checkpoint.pt").exists()
    with pytest.raises(FileExistsError, match="resume"):
        module.train_study(definition)
    assert module.train_study(definition, resume=True)["completed_updates"] == 2
    # Simulate interruption after final durable checkpoint, before result.json.
    (definition.output / "result.json").unlink()
    import shutil  # noqa: PLC0415

    shutil.rmtree(bundle)
    before = len(requests)
    assert module.train_study(definition, resume=True)["completed_updates"] == 2
    assert len(requests) == before

    if arm == "generator" and loss == "lk_hybrid":
        from speculators.generator import online_activations, study_replay  # noqa: PLC0415

        monkeypatch.setattr(
            study_replay,
            "load_model_layers",
            lambda *args: {"lm_head.weight": torch.zeros(16, 8)},
        )
        monkeypatch.setattr(
            study_replay,
            "generate_hidden_states",
            online_activations.generate_hidden_states,
        )
        branch = definition.model_copy(
            update={
                "name": "prefix_feedback",
                "replay": True,
                "feedback": True,
                "conditioning": "prefix",
                "initial_step": 1,
                "initial_checkpoint": definition.output
                / "checkpoints/step-00000001/checkpoint.pt",
            }
        )
        continued = module.train_study(branch)
        assert continued["initial_step"] == 1
        assert continued["completed_updates"] == 2
        assert (
            read_bundle(branch.output / "checkpoints/step-00000002/bundle")[
                "feedback_dim"
            ]
            == 3
        )
        records = json.loads(
            (branch.output / "replay/update-00000002.json").read_text()
        )
        assert len(records) == branch.examples_per_update
        assert records[0]["rounds"]


def test_nested_policy_inheritance_is_relative_to_leaf(tmp_path):
    (tmp_path / "policy.yaml").write_text("scale_min: 0\nscale_max: 4\n")
    (tmp_path / "run.yaml").write_text(
        "models: [{adapter_experiment: {extends: policy.yaml, scale: 2}}]\n"
    )
    assert load_experiment_yaml(tmp_path / "run.yaml")["models"][0] == {
        "adapter_experiment": {"scale_min": 0, "scale_max": 4, "scale": 2}
    }


def test_round_artifact_excludes_warmups(tmp_path):
    source = tmp_path / "server/adapter_rounds"
    source.mkdir(parents=True)
    for number in range(1, 5):
        write_json(source / f"{number}.json", {"prepared_sequence_number": number})
    destination = tmp_path / "measurement"
    for name, count in (("before", 2), ("after", 4)):
        write_json(
            destination / "nemo-math" / f"server_{name}.json",
            {"workers": [{"adapter_totals": {"prepared_requests": count}}]},
        )
    paths = []
    artifact = SimpleNamespace(
        add_file=lambda path, **kwargs: paths.append(Path(path).name)
    )
    benchmark_tracking.add_round_evidence(
        artifact,
        tmp_path,
        {"server_provenance": "server", "category": "nemo-math"},
        destination,
    )
    assert paths == ["3.json", "4.json"]


def test_preparation_freezes_disjoint_manifests_and_training_donors(
    online_campaign,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from speculators.benchmark.prepare import prepare_workload  # noqa: PLC0415
    from speculators.generator import study_inputs  # noqa: PLC0415
    from speculators.generator.local_campaign import load_local_inputs  # noqa: PLC0415
    from speculators.generator.serving.validation import load_validation  # noqa: PLC0415

    cfg, _, _ = online_campaign
    inputs = load_local_inputs(cfg)
    inputs.rows = list(inputs.rows)
    definition = GeneratorStudy(
        workspace=tmp_path / "study",
        train_per_category=8,
        tune_per_category=2,
        test_per_category=2,
    )
    split = study_inputs.membership(definition, inputs)
    for number, members in enumerate(split["groups"].values(), 1):
        for index in sum(members.values(), []):
            inputs.rows[index]["input_ids"] = [
                number,
                *inputs.rows[index]["input_ids"][1:],
            ]

    class Tokenizer:
        @staticmethod
        def decode(tokens, **kwargs):
            return ",".join(map(str, tokens))

        @staticmethod
        def encode(text, **kwargs):
            return list(map(int, text.split(",")))

    cache = SimpleNamespace(
        check_server=lambda: None,
        prefetch=lambda refs: None,
        get_summary=lambda index: torch.ones(24) * index,
        close=lambda: None,
    )
    monkeypatch.setattr(study_inputs, "campaign_settings", lambda study: cfg)
    monkeypatch.setattr(study_inputs, "load_local_inputs", lambda settings: inputs)
    monkeypatch.setattr(study_inputs, "OnlineActivations", lambda *args: cache)
    monkeypatch.setattr(
        study_inputs.AutoTokenizer, "from_pretrained", lambda *a: Tokenizer()
    )
    assert not study_inputs.prepare_study(definition)["reused"]
    assert study_inputs.prepare_study(definition)["reused"]
    manifest, groups = load_validation(definition.inputs / "test/manifest.json")
    assert len(groups) == 4
    assert {len(rows) for rows in groups.values()} == {2}
    donors = json.loads((definition.inputs / "conditioning.json").read_text())
    assert len(donors["groups"]["math"]) == 8
    suite = Suite.model_validate(
        {
            "models": [{"name": "base", "type": "drafter", "model": "base"}],
            "benchmarks": [
                {
                    "dataset": "nemotron",
                    "categories": ["math", "chat"],
                    "validation_manifest": str(
                        definition.inputs / "test/manifest.json"
                    ),
                }
            ],
            "settings": {"max_samples": 2, "max_tokens": 256},
        }
    )
    result = prepare_workload(
        suite.benchmarks[0], cfg, tmp_path / "frozen", suite.settings, None
    )
    assert set(result["groups"]) == {"nemo-math", "nemo-chat"}
    assert result["identity"] != manifest["identity"]
    assert (
        prepare_workload(
            suite.benchmarks[0], cfg, tmp_path / "frozen", suite.settings, None
        )
        == result
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_streaming_flex_uses_sdpa_across_lengths_and_grad_modes(monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from speculators.models import attention  # noqa: PLC0415

    def forbidden(*args, **kwargs):
        raise AssertionError("Streaming replay entered FlexAttention compilation")

    monkeypatch.setattr(attention, "_standalone_flex_attention", forbidden)
    monkeypatch.setattr(attention, "flex_attention", forbidden)
    runtime, generator = streaming_runtime(backend="simple_flex_attention")
    runtime.device = torch.device(device)
    runtime.drafter.to(device)
    generator.to(device)
    stream = StreamingDraft(runtime)
    for length in range(2, 12):
        generator.zero_grad(set_to_none=True)
        factors = generator(
            torch.randn(1, 24, device=device), torch.tensor([length], device=device)
        )
        states = torch.randn(length, 24)
        tokens = [1] * length
        with torch.no_grad():
            proposals, _ = stream.propose(states, tokens, 0, DynamicCache(), factors)
        _, logits = stream.propose(
            states, tokens, 0, DynamicCache(), factors, forced=proposals
        )
        logits.square().mean().backward()
        assert generator.heads["v_proj"]["B"].weight.grad.abs().sum() > 0
