import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from datasets import Dataset
from safetensors.torch import save_file
from torch import nn
from torch.func import functional_call
from transformers import Qwen3Config
from typer.testing import CliRunner

from speculators.bank.artifacts import write_json
from speculators.bank.config import BankConfig
from speculators.cli import app
from speculators.generator import workflow
from speculators.generator.config import Architecture, Context, ExperimentConfig
from speculators.generator.data import (
    Corpus,
    FactorCache,
    discover_snapshots,
    filter_snapshots,
    groups,
    prompt_summary,
)
from speculators.generator.engine import Runtime, begin_run, run_job
from speculators.generator.model import (
    AdapterLinear,
    LoRAGenerator,
    factor_loss,
    factor_parameters,
    install_adapters,
)
from speculators.losses import resolve_loss_config
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.train.utils import normalize_counted_metrics


def small_architecture():
    return Architecture(
        encoder_hidden=8,
        example_dim=4,
        condition_dim=4,
        descriptor_dim=2,
        decoder_width=8,
        residual_blocks=1,
    )


def tiny_drafter():
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            intermediate_size=16,
            vocab_size=16,
            max_position_embeddings=64,
            attn_implementation="eager",
        ),
        draft_vocab_size=16,
        eagle_aux_hidden_state_layer_ids=[2, 18, 33],
    )
    config.transformer_layer_config._attn_implementation = "eager"
    model = Eagle3DraftModel(config)
    with torch.no_grad():
        model.embed_tokens.weight.normal_(std=0.2)
        model.lm_head.weight.normal_(std=0.2)
        model.verifier_lm_head.weight.copy_(model.lm_head.weight)
    return model


def test_context_range_and_config_reference(tmp_path):
    with pytest.raises(ValueError, match="less than or equal to 8"):
        Context(max_examples=9)
    with pytest.raises(ValueError, match="context range"):
        Context(evaluation_sizes=[1, 9])
    config = tmp_path / "experiment.yaml"
    config.write_text("bank_config: bank.yaml\n")
    assert ExperimentConfig.load(config).bank_config == tmp_path / "bank.yaml"


def test_set_encoder_is_permutation_invariant_and_singleton_finite():
    generator = LoRAGenerator(5, {"o_proj": (4, 4)}, 2, small_architecture())
    # Exercise conditioning beyond the zero-initialized heads.
    nn.init.normal_(generator.heads["o_proj"]["B"].weight)
    summaries, lengths = torch.randn(8, 5), torch.arange(1, 9)
    left = generator(summaries, lengths)
    order = torch.randperm(8)
    right = generator(summaries[order], lengths[order])
    for a, b in zip(left["o_proj"], right["o_proj"], strict=True):
        torch.testing.assert_close(a, b)
    assert all(
        torch.isfinite(t).all() for t in generator(summaries[:1], lengths[:1])["o_proj"]
    )
    with pytest.raises(ValueError, match="conditioning examples"):
        generator(torch.randn(9, 5), torch.ones(9))


def test_functional_adapters_preserve_base_and_backpropagate():
    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = nn.Linear(4, 4)
            self.o_proj = nn.Linear(4, 4)

        def forward(self, x):
            return self.o_proj(self.v_proj(x))

    model = Draft()
    x = torch.randn(3, 4)
    original = model(x).detach()
    paths = install_adapters(model, 2, 4, 0)
    model.requires_grad_(False)
    generator = LoRAGenerator(5, dict.fromkeys(paths, (4, 4)), 2, small_architecture())
    factors = generator(torch.randn(2, 5), torch.tensor([3, 4]))
    output = functional_call(model, factor_parameters(paths, factors), (x,))
    torch.testing.assert_close(output, original)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        factors = generator(torch.randn(2, 5), torch.tensor([3, 4]))
        loss = (
            functional_call(model, factor_parameters(paths, factors), (x,))
            .square()
            .mean()
        )
        loss.backward()
        optimizer.step()
    assert generator.encoder[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(not p.requires_grad for p in model.parameters())


def test_prompt_only_summaries_use_actual_frozen_normalization():
    model = SimpleNamespace(
        fc=nn.Linear(6, 2, bias=False),
        input_norm=nn.LayerNorm(6),
        fc_norm=None,
    )
    states = torch.randn(6, 4, 2)
    before = prompt_summary(states, 3, model)
    states[3:] = 1000
    after = prompt_summary(states, 3, model)
    for name in before:
        torch.testing.assert_close(before[name], after[name])
    projected = model.fc(model.input_norm(states[:3, :-1].flatten(1))).detach()
    torch.testing.assert_close(
        before["projected"], torch.cat([projected[-1], projected.mean(0)])
    )


def test_bfloat16_drafter_keeps_float32_adapter_updates():
    base = nn.Linear(4, 4).to(torch.bfloat16)
    adapter = AdapterLinear(base, 2, 4, 0)
    assert adapter.a.dtype == adapter.b.dtype == torch.float32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = adapter(torch.randn(3, 4).to(torch.bfloat16))
        loss = output.float().square().mean()
    loss.backward()
    assert adapter.b.grad.dtype == torch.float32
    assert adapter.b.grad.abs().sum() > 0
    assert base.weight.grad is None


def test_stride_is_collection_relative_and_uses_a_shared_window():
    entries = [
        {"seed": seed, "collection_step": step, "global_step": step + 262}
        for seed in [42, 43, 44]
        for step in range(10, 61, 10)
    ]
    retained = filter_snapshots(entries, [42, 44], 30)
    assert [(e["seed"], e["collection_step"]) for e in retained] == [
        (42, 30),
        (42, 60),
        (44, 30),
        (44, 60),
    ]
    with pytest.raises(ValueError, match="Missing bank seed"):
        filter_snapshots(entries, [45], 10)
    sparse = [
        e for e in entries if not (e["seed"] == 44 and e["collection_step"] == 60)
    ]
    retained = filter_snapshots(sparse, [42, 44], 30, (10, 60))
    assert (42, 60) in [(e["seed"], e["collection_step"]) for e in retained]


def test_grouping_covers_epoch_and_counted_metrics():
    cohorts = list(groups(range(101), 42, 1, 8))
    assert sorted(i for cohort in cohorts for i in cohort) == list(range(101))
    assert all(1 <= len(cohort) <= 8 for cohort in cohorts)
    assert cohorts == list(groups(range(101), 42, 1, 8))
    metrics = normalize_counted_metrics({"full_acc_0_sum": 9, "full_acc_0_total": 12})
    assert metrics == {"full_acc_0": 0.75}


def test_factor_cache_and_raw_factor_loss(tmp_path):
    state = {
        "base.o_proj.lora_A.weight": torch.ones(2, 4),
        "base.o_proj.lora_B.weight": torch.ones(3, 2),
    }
    path = tmp_path / "adapter.safetensors"
    save_file(state, path)
    cache = FactorCache(1, {"o_proj": (4, 3)}, 2)
    target = cache.get(str(path))
    assert cache.get(str(path)) is target
    assert factor_loss({"o_proj": (torch.zeros(2, 4), torch.zeros(3, 2))}, target) == 1
    no_cache = FactorCache(0, {"o_proj": (4, 3)}, 2)
    no_cache.get(str(path))
    assert not no_cache.items


def test_resume_rejects_changes_and_skips_completed(tmp_path):
    run = tmp_path / "run"
    assert begin_run(run, {"value": 1})
    write_json(run / "result.json", {"score": 0.5})
    assert not begin_run(run, {"value": 1})
    with pytest.raises(ValueError, match="configuration changed"):
        begin_run(run, {"value": 2})


@pytest.fixture
def synthetic_experiment(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    root = tmp_path / "bank"
    hidden = root / "hidden_states"
    hidden.mkdir(parents=True)
    rows = []
    for index in range(40):
        size = 6 + index % 2
        rows.append(
            {
                "input_ids": [1, 2, 3, 4, 5, 6, 7][:size],
                "seq_len": size,
                "prompt_length": 2,
                "loss_mask": [0, 0] + [1] * (size - 2),
            }
        )
        save_file(
            {
                "token_ids": torch.tensor(rows[-1]["input_ids"]),
                "hidden_states": torch.randn(size, 4, 8),
            },
            hidden / f"hs_{index}.safetensors",
        )
    Dataset.from_list(rows).save_to_disk(root / "data")
    manifest = {
        "prepared_fingerprint": "synthetic",
        "cache_spec": {},
        "revisions": {},
        "data_path": str(root / "data"),
        "hidden_states_path": str(hidden),
        "subsets": [
            {
                "id": "math-00000",
                "train_indices": list(range(10)),
                "validation_indices": list(range(10, 20)),
            },
            {
                "id": "math-00001",
                "train_indices": list(range(20, 30)),
                "validation_indices": list(range(30, 40)),
            },
        ],
    }
    write_json(root / "manifest.json", manifest)
    template = tiny_drafter()
    paths = install_adapters(template, 2, 4, 0)
    shapes = {
        name: (
            template.get_submodule(path).base.in_features,
            template.get_submodule(path).base.out_features,
        )
        for name, path in paths.items()
    }
    for seed in [42, 43, 44]:
        for step in range(10, 61, 10):
            path = (
                root
                / "runs"
                / "math-00000"
                / f"seed-{seed}"
                / "snapshots"
                / f"step-{step:08d}"
            )
            (path / "adapter").mkdir(parents=True)
            state = {
                f"base.{name}.lora_{suffix}.weight": torch.randn(*shape) * 0.02
                for name, (inputs, outputs) in shapes.items()
                for suffix, shape in [("A", (2, inputs)), ("B", (outputs, 2))]
            }
            save_file(state, path / "adapter" / "adapter_model.safetensors")
            write_json(
                path / "entry.json",
                {"collection_step": step, "global_step": 100 + step},
            )
    bank = BankConfig(output_root=root, lora={"rank": 2, "alpha": 4, "dropout": 0})
    bank_path = tmp_path / "bank.yaml"
    bank_path.write_text(yaml.safe_dump(bank.model_dump(mode="json")))
    cfg = ExperimentConfig(
        bank_config=bank_path,
        output_root=tmp_path / "results",
        staging_dir=tmp_path / "local",
        device="cpu",
        dtype="float32",
        architecture=small_architecture(),
        optimization={
            "conditioning_epochs": 1,
            "pretraining_updates": 2,
            "episodes_per_update": 1,
            "checkpoint_interval": 1,
            "token_budget": 8,
        },
    )

    def runtime_factory(config, bank_config):
        runtime = Runtime.__new__(Runtime)
        runtime.cfg, runtime.bank = config, bank_config
        runtime.device, runtime.dtype = torch.device("cpu"), torch.float32
        runtime.corpus = Corpus(config, bank_config)
        torch.manual_seed(config.seed)
        runtime.drafter = tiny_drafter().requires_grad_(False)
        runtime.corpus.prepare(runtime.drafter)
        runtime.paths = install_adapters(runtime.drafter, 2, 4, 0)
        runtime.shapes = shapes
        runtime.cache = FactorCache(1, shapes, 2)
        runtime.call_kwargs = {
            "ttt_steps": 3,
            "loss_config": resolve_loss_config("kl_div", "eager"),
        }
        return runtime

    monkeypatch.setattr(workflow, "Runtime", runtime_factory)
    return cfg, runtime_factory


def test_all_three_workflows_and_completed_resume(synthetic_experiment, monkeypatch):
    cfg, _ = synthetic_experiment
    experiment_path = cfg.output_root.parent / "experiment.yaml"
    experiment_path.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    runner = CliRunner()
    for command in ["prepare", "conditioning", "heatmap", "adaptation"]:
        result = runner.invoke(
            app, ["generator", command, "--config", str(experiment_path)]
        )
        assert result.exit_code == 0, (result.exception, result.output)
    conditioning = json.loads(
        (cfg.output_root / "conditioning" / "selection.json").read_text()
    )["results"]
    assert len(conditioning) == 3
    assert all(len(result["history"]) == 2 for result in conditioning)
    heatmap = json.loads((cfg.output_root / "heatmap" / "selection.json").read_text())[
        "results"
    ]
    assert len(heatmap) == 18
    assert (cfg.output_root / "heatmap" / "score.png").exists()
    arms = json.loads((cfg.output_root / "adaptation" / "results.json").read_text())
    assert {result["job"]["arm"] for result in arms} == {
        "pretrained_generator",
        "fresh_generator",
        "fresh_lora",
        "transferred_lora",
    }
    assert all(len(result["history"]) == 2 for result in arms)
    # Completed runs must skip before any model is initialized.
    monkeypatch.setattr(
        workflow, "Runtime", lambda *args: pytest.fail("Completed runs loaded a model")
    )
    assert workflow.conditioning(cfg) == conditioning
    assert workflow.heatmap(cfg) == heatmap
    assert workflow.adaptation(cfg) == arms


def test_reconstruction_resumes_exactly_after_interrupted_update(
    synthetic_experiment, monkeypatch
):
    cfg, factory = synthetic_experiment
    bank = BankConfig.load(cfg.bank_config)
    job = {
        "kind": "heatmap",
        "condition": "last",
        "seeds": [42],
        "stride": 10,
        "snapshots": filter_snapshots(
            discover_snapshots(bank.output_root, cfg.bank_subset), [42], 10
        ),
    }
    uninterrupted = cfg.output_root / "uninterrupted"
    uninterrupted.mkdir(parents=True)
    run_job(factory(cfg, bank), job, uninterrupted)
    expected = torch.load(uninterrupted / "checkpoint.pt", weights_only=True)["model"]
    interrupted = cfg.output_root / "interrupted"
    interrupted.mkdir(parents=True)
    runtime = factory(cfg, bank)
    original = runtime.cache.get
    calls = 0

    def fail_second_read(path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(path)

    monkeypatch.setattr(runtime.cache, "get", fail_second_read)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_job(runtime, job, interrupted)
    assert torch.load(interrupted / "checkpoint.pt", weights_only=True)["progress"] == 1
    run_job(factory(cfg, bank), job, interrupted)
    resumed = torch.load(interrupted / "checkpoint.pt", weights_only=True)["model"]
    for name, tensor in expected.items():
        torch.testing.assert_close(resumed[name], tensor, rtol=0, atol=0)


def test_production_runtime_uses_pinned_models_and_frozen_verifier(
    synthetic_experiment, monkeypatch
):
    cfg, _ = synthetic_experiment
    bank = BankConfig.load(cfg.bank_config)
    manifest_path = bank.output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["revisions"] = {"target_sha": "target-sha", "drafter_sha": "draft-sha"}
    write_json(manifest_path, manifest)
    model = tiny_drafter()
    model.config.speculators_config = SimpleNamespace(
        verifier=SimpleNamespace(name_or_path="old-target")
    )
    snapshots = []

    def snapshot(model_name, revision):
        snapshots.append((model_name, revision))
        return revision

    monkeypatch.setattr("speculators.generator.engine.snapshot_download", snapshot)
    monkeypatch.setattr(
        "speculators.convert.entrypoints.maybe_convert_external_checkpoint",
        lambda path, verifier: path,
    )
    monkeypatch.setattr(
        "speculators.model.SpeculatorModel.config_class.from_pretrained",
        lambda path: model.config,
    )
    monkeypatch.setattr(
        "speculators.model.SpeculatorModel.from_pretrained",
        lambda *args, **kwargs: model,
    )
    runtime = Runtime(cfg, bank)
    assert snapshots == [
        (bank.models.target, "target-sha"),
        (bank.models.drafter, "draft-sha"),
    ]
    assert model.config.speculators_config.verifier.name_or_path == "target-sha"
    assert set(runtime.paths) == {"o_proj", "v_proj"}
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if not name.endswith((".a", ".b"))
    )


def test_parallel_dispatch_writes_worker_requests(synthetic_experiment, monkeypatch):
    cfg, _ = synthetic_experiment
    cfg.n_workers = 2
    dispatched = []

    def parallel(runs, workers):
        assert workers == 2
        for run in runs:
            payload = json.loads(Path(run.command[-1]).read_text())
            dispatched.append(payload["job"])
            assert run.command[-3:-1] == ["-m", "speculators.generator.worker"]
            write_json(
                run.output / "result.json", {"score": 0.1, "job": payload["job"]}
            )

    monkeypatch.setattr(workflow, "run_parallel", parallel)
    jobs = [
        ({"kind": "conditioning", "condition": name}, cfg.output_root / name)
        for name in ["last", "projected"]
    ]
    results = workflow.execute(cfg, jobs)
    assert len(results) == len(dispatched) == 2


def test_token_microbatching_keeps_every_example(synthetic_experiment):
    cfg, factory = synthetic_experiment
    runtime = factory(cfg, BankConfig.load(cfg.bank_config))
    indices = runtime.corpus.indices(cfg.conditioning_subset)
    batches = list(runtime.corpus.microbatches(indices, "cpu", torch.float32))
    assert len(batches) == len(indices)
    assert sum(batch["loss_mask"].sum().item() for batch, _ in batches) == sum(
        sum(runtime.corpus.rows[i]["loss_mask"]) for i in indices
    )
    assert [last for _, last in batches] == [False] * (len(indices) - 1) + [True]


def test_drafter_loss_updates_generator_and_preserves_frozen_weights(
    synthetic_experiment,
):
    cfg, factory = synthetic_experiment
    runtime = factory(cfg, BankConfig.load(cfg.bank_config))
    generator = runtime.new_generator("last")
    before = copy.deepcopy(runtime.drafter.state_dict())
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0.01)
    runtime.train_epoch(cfg.conditioning_subset, "last", generator, optimizer, 0)
    assert generator.heads["o_proj"]["B"].weight.abs().sum() > 0
    for key, value in runtime.drafter.state_dict().items():
        torch.testing.assert_close(value, before[key])
