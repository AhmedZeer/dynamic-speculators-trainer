import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Lock
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
from speculators.bank.transfer import BankFileTransfer
from speculators.cli import app
from speculators.generator import workflow
from speculators.generator.config import (
    Architecture,
    Context,
    ExperimentConfig,
    Optimization,
    WandbSettings,
)
from speculators.generator.data import (
    Corpus,
    FactorCache,
    discover_snapshots,
    filter_snapshots,
    groups,
    prompt_summary,
)
from speculators.generator.engine import (
    Runtime,
    begin_run,
    clip_gradients,
    run_identity,
    run_job,
)
from speculators.generator.model import (
    AdapterLinear,
    LoRAGenerator,
    backward_factors,
    detached_factors,
    factor_loss,
    factor_parameters,
    install_adapters,
)
from speculators.generator.tracking import ExperimentTracker, run_name
from speculators.losses import resolve_loss_config
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.models.eagle3.attention import extend_mask_for_draft_tokens
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


def tiny_drafter(hidden_size=8, backend="eager"):
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=Qwen3Config(
            hidden_size=hidden_size,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=hidden_size // 2,
            intermediate_size=2 * hidden_size,
            vocab_size=16,
            max_position_embeddings=64,
            attn_implementation="eager",
        ),
        draft_vocab_size=16,
        eagle_aux_hidden_state_layer_ids=[2, 18, 33],
    )
    config.transformer_layer_config._attn_implementation = backend
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
    assert Optimization().max_grad_norm == 0.85
    with pytest.raises(ValueError, match="greater than 0"):
        Optimization(max_grad_norm=0)


def test_gradient_clipping_uses_global_norm_and_reports_before_clipping():
    first, second = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([{"params": [first]}, {"params": [second]}])
    first.grad, second.grad = torch.tensor([3.0]), torch.tensor([4.0])
    metrics = clip_gradients(optimizer, 0.85)
    assert metrics["train/grad_norm"] == 5.0
    assert metrics["train/gradient_clipped"] == 1.0
    assert metrics["train/clip_scale"] == pytest.approx(0.17)
    assert metrics["train/grad_rms"] == pytest.approx(5 / 2**0.5)
    assert torch.cat([first.grad, second.grad]).norm().item() == pytest.approx(0.85)
    first.grad, second.grad = torch.tensor([0.03]), torch.tensor([0.04])
    metrics = clip_gradients(optimizer, 0.85)
    assert metrics["train/grad_norm"] == pytest.approx(0.05)
    assert metrics["train/gradient_clipped"] == 0
    torch.testing.assert_close(first.grad, torch.tensor([0.03]))
    torch.testing.assert_close(second.grad, torch.tensor([0.04]))
    first.grad.fill_(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        clip_gradients(optimizer, 0.85)


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


@pytest.mark.parametrize("compiled", [False, True])
def test_factor_gradient_accumulation_matches_combined_loss(compiled):
    torch.manual_seed(42)
    reference = LoRAGenerator(
        5, {"o_proj": (4, 4), "v_proj": (4, 4)}, 2, small_architecture()
    )
    for heads in reference.heads.values():
        nn.init.normal_(heads["B"].weight, std=0.01)
    accumulated = copy.deepcopy(reference)
    summaries, lengths = torch.randn(3, 5), torch.tensor([3, 4, 5])
    chunks = [(torch.randn(count, 4), torch.randn(count, 4)) for count in [2, 3]]
    frozen_v, frozen_o = torch.randn(4, 4), torch.randn(4, 4)

    def loss_fn(x, target, a_v, b_v, a_o, b_o):
        hidden = x @ frozen_v.T + 2 * (x @ a_v.T) @ b_v.T
        output = hidden @ frozen_o.T + 2 * (hidden @ a_o.T) @ b_o.T
        return (output - target).square().mean()

    expected_factors = reference(summaries, lengths)
    combined = sum(
        loss_fn(x, target, *expected_factors["v_proj"], *expected_factors["o_proj"])
        * len(x)
        / 5
        for x, target in chunks
    )
    combined.backward()
    generated = accumulated(summaries, lengths)
    leaves = detached_factors(generated)
    kernel = (
        torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
        if compiled
        else loss_fn
    )
    for x, target in chunks:
        loss = kernel(x, target, *leaves["v_proj"], *leaves["o_proj"])
        (loss * len(x) / 5).backward()
    backward_factors(generated, leaves)
    for expected, actual in zip(
        reference.parameters(), accumulated.parameters(), strict=True
    ):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-6)


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


def test_resume_records_new_provenance_and_preserves_previous_attempt(
    tmp_path, monkeypatch
):
    attempts = iter(["old code", "new code"])

    def save(path):
        value = next(attempts)
        (Path(path) / "train_command.txt").write_text(value)
        (Path(path) / "speculators.patch").write_text(value)

    monkeypatch.setattr("speculators.generator.engine.save_train_command", save)
    run = tmp_path / "run"
    assert begin_run(run, {"value": 1})
    assert begin_run(run, {"value": 1})
    assert (run / "train_command.txt").read_text() == "new code"
    archive = list((run / "provenance").glob("attempt-*"))
    assert len(archive) == 1
    assert (archive[0] / "train_command.txt").read_text() == "old code"
    assert (archive[0] / "speculators.patch").read_text() == "old code"


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
    bank = BankConfig(
        output_root=root,
        lora={"rank": 2, "alpha": 4, "dropout": 0},
        execution={"draft_attn_impl": "eager"},
    )
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
            "loss_config": resolve_loss_config(
                config.optimization.loss_fn or bank_config.training.loss_fn, "eager"
            ),
        }
        return runtime

    monkeypatch.setattr(workflow, "Runtime", runtime_factory)
    return cfg, runtime_factory


@pytest.mark.parametrize("tracking", [False, True])
def test_all_three_workflows_and_completed_resume(
    synthetic_experiment,
    monkeypatch,
    fake_wandb,
    tracking,
):
    cfg, _ = synthetic_experiment
    cfg.wandb = WandbSettings(enabled=tracking, upload_artifacts=True)
    if tracking:
        cfg.optimization.loss_fn = "lk_hybrid"
    original_step = torch.optim.AdamW.step
    applied_norms = []

    def checked_step(optimizer, *args, **kwargs):
        gradients = [
            p.grad.norm()
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        norm = torch.stack(gradients).norm().item()
        assert norm <= cfg.optimization.max_grad_norm + 1e-6
        applied_norms.append(norm)
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", checked_step)
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
    assert applied_norms
    if tracking:
        assert (
            len(fake_wandb.runs) == 26
        )  # Three conditions, 18 cells, summary, four arms.
        assert len({run.kwargs["id"] for run in fake_wandb.runs}) == 26
        assert {run.kwargs["project"] for run in fake_wandb.runs} == set(
            cfg.wandb.projects.values()
        )
        for run in fake_wandb.runs:
            assert run.exit_code == 0
            assert run.artifacts
            assert run.kwargs["dir"] == str(cfg.staging_dir / "wandb")
            if run.kwargs["job_type"] == "heatmap-summary":
                table = run.logs[0]["heatmap/cells"]
                assert len(table.data) == 18
                assert len(run.logs[0]) == 5  # Table and four heatmap images.
                continue
            loss_key = (
                "train/reconstruction_l1"
                if run.kwargs["job_type"] == "heatmap"
                else "train/drafter_loss"
            )
            losses = [row for row in run.logs if loss_key in row]
            assert losses
            assert all(torch.isfinite(torch.tensor(row[loss_key])) for row in losses)
            assert all(row["train/grad_norm"] >= 0 for row in losses)
            assert all(
                row["train/gradient_clipped"]
                == float(row["train/grad_norm"] > cfg.optimization.max_grad_norm)
                for row in losses
            )
            steps = [row["optimizer_step"] for row in losses]
            assert steps == list(range(1, len(steps) + 1))
            validation = [
                row for row in run.logs if "validation/score_mean_full_acc_0" in row
            ]
            assert validation
            assert all(
                f"validation/context_{size}/full_acc_0" in validation[-1]
                for size in [1, 4, 8]
            )
            assert (
                run.summary["final_score"]
                == validation[-1]["validation/score_mean_full_acc_0"]
            )
    # Completed runs must skip before any model is initialized.
    monkeypatch.setattr(
        workflow, "Runtime", lambda *args: pytest.fail("Completed runs loaded a model")
    )
    assert workflow.conditioning(cfg) == conditioning
    assert workflow.heatmap(cfg) == heatmap
    assert workflow.adaptation(cfg) == arms
    assert len(fake_wandb.runs) == (26 if tracking else 0)


def test_reconstruction_resumes_exactly_after_interrupted_update(
    synthetic_experiment,
    monkeypatch,
    fake_wandb,
):
    cfg, factory = synthetic_experiment
    cfg.wandb.enabled = True
    cfg.optimization.warmup_updates = 50
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
    assert fake_wandb.runs[1].exit_code == 1
    assert fake_wandb.runs[2].exit_code == 0
    assert fake_wandb.runs[1].kwargs["id"] == fake_wandb.runs[2].kwargs["id"]
    assert fake_wandb.runs[2].logs[0]["optimizer_step"] == 2
    for name, tensor in expected.items():
        torch.testing.assert_close(resumed[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("loss_fn", [None, "lk_hybrid"])
def test_production_runtime_uses_pinned_models_and_frozen_verifier(
    synthetic_experiment,
    monkeypatch,
    loss_fn,
):
    cfg, _ = synthetic_experiment
    cfg.optimization.loss_fn = loss_fn
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
    assert set(runtime.call_kwargs["loss_config"]) == {loss_fn or "kl_div"}
    assert bank.training.loss_fn == "kl_div"
    identity = run_identity(
        cfg, bank, runtime.corpus, {"kind": "conditioning", "condition": "last"}
    )
    assert identity["drafter_training"]["loss_fn"] == (loss_fn or "kl_div")
    if loss_fn is None:
        assert "loss_fn" not in identity["configuration"]["optimization"]
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


def test_flex_microbatch_padding_matches_all_draft_mask_lengths(
    synthetic_experiment, monkeypatch
):
    cfg, factory = synthetic_experiment
    runtime = factory(cfg, BankConfig.load(cfg.bank_config))
    runtime.bank.execution.draft_attn_impl = "simple_flex_attention"
    cfg.optimization.token_budget = 8192
    # Reproduce the reported q_len=1782 after EAGLE3's one-token shift.
    rows = runtime.corpus.rows.to_list()
    index = runtime.corpus.indices(cfg.conditioning_subset)[0]
    rows[index] = {
        "seq_len": 1783,
        "prompt_length": 2,
        "input_ids": [1] * 1783,
        "loss_mask": [0, 0] + [1] * 1781,
    }
    runtime.corpus.rows = Dataset.from_list(rows)
    monkeypatch.setattr(
        runtime.corpus.activations,
        "get_cached",
        lambda _: {
            "token_ids": torch.ones(1783, dtype=torch.long),
            "hidden_states": torch.randn(1783, 4, 8),
        },
    )
    batch, last = next(runtime.corpus.microbatches([index], "cpu", torch.float32))
    assert last
    assert batch["input_ids"].shape == (1, 1792)
    assert batch["loss_mask"].sum() == 1781
    assert not batch["loss_mask"][0, 1782:].any()
    assert (batch["document_ids"][0, 1782:] == -1).all()
    assert (batch["hidden_states"][0, 1782:] == 0).all()
    flex = tiny_drafter(backend="simple_flex_attention")
    mask = flex._build_attn_mask(batch["document_ids"][0], 1792, "cpu")
    for step in range(3):
        assert mask.shape[-2:] == (1792, 1792 * (step + 1))
        mask = extend_mask_for_draft_tokens(mask)
    cfg.optimization.token_budget = 1782
    with pytest.raises(ValueError, match="multiple of 128"):
        next(runtime.corpus.microbatches([index], "cpu", torch.float32))


def test_nonrecursive_disable_allows_nested_kernel_compilation():
    compiled_graphs = []

    def backend(graph, inputs):
        compiled_graphs.append(graph)
        return graph.forward

    kernel = torch.compile(lambda x: x.sin(), backend=backend, fullgraph=True)

    @torch.compiler.disable(recursive=False)
    def model(x):
        return kernel(x)

    x = torch.randn(4, requires_grad=True)
    model(x).sum().backward()
    assert compiled_graphs
    torch.testing.assert_close(x.grad, x.detach().cos())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_flex_generated_lora_matches_dense_forward_and_backward():
    torch.manual_seed(42)
    flex = tiny_drafter(32, "simple_flex_attention").cuda().to(torch.bfloat16)
    dense = tiny_drafter(32).cuda().to(torch.bfloat16)
    dense.load_state_dict(flex.state_dict())
    paths = install_adapters(flex, 2, 4, 0)
    install_adapters(dense, 2, 4, 0)
    flex.requires_grad_(False)
    dense.requires_grad_(False)
    for model in (flex, dense):
        forward = type(model).forward
        forward = getattr(forward, "_torchdynamo_orig_callable", forward)
        model.forward = torch.compiler.disable(forward.__get__(model), recursive=False)
    generator = LoRAGenerator(
        5, {"o_proj": (32, 32), "v_proj": (64, 16)}, 2, small_architecture()
    ).cuda()
    for heads in generator.heads.values():
        nn.init.normal_(heads["B"].weight, std=0.01)
    factors = generator(
        torch.randn(2, 5, device="cuda"), torch.tensor([4, 5], device="cuda")
    )
    batch = {
        "hidden_states": torch.randn(1, 256, 96, device="cuda", dtype=torch.bfloat16),
        "verifier_last_hidden_states": torch.randn(
            1, 256, 32, device="cuda", dtype=torch.bfloat16
        ),
        "input_ids": torch.randint(0, 16, (1, 256), device="cuda"),
        "document_ids": torch.zeros(1, 256, dtype=torch.long, device="cuda"),
        "loss_mask": torch.zeros(1, 256, dtype=torch.bool, device="cuda"),
        "ttt_steps": 3,
        "loss_config": resolve_loss_config("kl_div", "eager"),
    }
    batch["document_ids"][:, 131:] = -1
    batch["loss_mask"][:, 4:131] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, flex_loss, flex_metrics = functional_call(
            flex, factor_parameters(paths, factors), (), batch
        )
        _, dense_loss, dense_metrics = functional_call(
            dense, factor_parameters(paths, factors), (), batch
        )
    torch.testing.assert_close(flex_loss, dense_loss, rtol=0.02, atol=0.002)
    values = tuple(value for pair in factors.values() for value in pair)
    flex_grads = torch.autograd.grad(flex_loss, values)
    dense_grads = torch.autograd.grad(dense_loss, values)
    for actual, expected in zip(flex_grads, dense_grads, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.002)
    for step in range(3):
        torch.testing.assert_close(
            flex_metrics[f"full_acc_{step}_total"],
            dense_metrics[f"full_acc_{step}_total"],
        )


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


def test_compiled_microbatch_training_never_retains_backward_graph(
    synthetic_experiment, monkeypatch
):
    cfg, factory = synthetic_experiment
    runtime = factory(cfg, BankConfig.load(cfg.bank_config))
    cfg.context.min_examples = cfg.context.max_examples
    generator = runtime.new_generator("last")
    optimizer = torch.optim.AdamW(generator.parameters(), lr=0.01)
    compiled_loss = torch.compile(
        lambda value: value.sin().cos().square(), backend="aot_eager", fullgraph=True
    )
    original_forward, original_backward = runtime.forward, torch.Tensor.backward
    calls = []

    def forward(batch, factors):
        assert all(value.is_leaf for pair in factors.values() for value in pair)
        tokens, loss, metrics = original_forward(batch, factors)
        return tokens, compiled_loss(loss), metrics

    def backward(tensor, *args, **kwargs):
        assert not kwargs.get("retain_graph", False)
        calls.append(tensor)
        return original_backward(tensor, *args, **kwargs)

    monkeypatch.setattr(runtime, "forward", forward)
    monkeypatch.setattr(torch.Tensor, "backward", backward)
    runtime.train_epoch(cfg.conditioning_subset, "last", generator, optimizer, 0)
    assert len(calls) == len(runtime.corpus.indices(cfg.conditioning_subset))
    assert generator.encoder[0].weight.grad.abs().sum() > 0


@pytest.fixture
def fake_wandb(monkeypatch):
    module = SimpleNamespace(runs=[])

    def init(**kwargs):
        run = SimpleNamespace(
            kwargs=kwargs,
            summary={},
            logs=[],
            artifacts=[],
            url=None,
            exit_code=None,
            metrics=[],
        )
        run.log = run.logs.append
        run.define_metric = lambda *args, **kwargs: run.metrics.append((args, kwargs))
        run.log_artifact = run.artifacts.append

        def finish(exit_code=0):
            run.exit_code = exit_code

        run.finish = finish
        module.runs.append(run)
        return run

    def artifact(name, type):  # noqa: A002 -- SDK signature
        value = SimpleNamespace(name=name, type=type, files=[])
        value.add_file = lambda path, name: value.files.append((path, name))
        return value

    module.init = init
    module.Artifact = artifact
    module.Table = SimpleNamespace
    module.Image = lambda path: SimpleNamespace(path=path)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_tracking_names_and_distinct_projects(tmp_path):
    cfg = ExperimentConfig(bank_config=tmp_path / "bank.yaml")
    job = {"kind": "heatmap", "condition": "last_mean", "stride": 20, "seeds": [42, 44]}
    assert run_name(cfg, job) == (
        "heatmap-math-00000-last_mean-stride20-bankseeds42+44-gseed42-ctx1-8"
    )
    assert "fresh_lora" in run_name(
        cfg,
        {
            "kind": "adaptation",
            "condition": "last",
            "arm": "fresh_lora",
        },
    )
    with pytest.raises(ValueError, match="different"):
        WandbSettings(projects=dict.fromkeys(cfg.wandb.projects, "same"))
    with pytest.raises(ValueError, match="each"):
        WandbSettings(projects={"conditioning": "one"})


def test_tracking_preserves_training_identity_and_failed_run_id(
    synthetic_experiment,
    fake_wandb,
):
    cfg, factory = synthetic_experiment
    runtime = factory(cfg, BankConfig.load(cfg.bank_config))
    job = {"kind": "conditioning", "condition": "last"}
    identity = run_identity(cfg, runtime.bank, runtime.corpus, job)
    assert "wandb" not in identity["configuration"]
    cfg.wandb = WandbSettings(enabled=True, mode="online", name_prefix="trial-")
    assert run_identity(cfg, runtime.bank, runtime.corpus, job) == identity
    output = cfg.output_root / "tracking"
    with (
        pytest.raises(RuntimeError, match="training failure"),
        ExperimentTracker(
            cfg,
            job,
            output,
            identity,
        ),
    ):
        raise RuntimeError("training failure")
    assert fake_wandb.runs[0].exit_code == 1
    assert json.loads((output / "wandb.json").read_text())["status"] == "failed"
    with ExperimentTracker(cfg, job, output, identity) as tracker:
        tracker.training(5, {"train/drafter_loss": 0.5})
    assert fake_wandb.runs[1].exit_code == 0
    assert fake_wandb.runs[0].kwargs["id"] == fake_wandb.runs[1].kwargs["id"]
    assert fake_wandb.runs[1].kwargs["resume"] == "allow"
    assert fake_wandb.runs[1].logs[0]["optimizer_step"] == 5


def test_completed_runs_backfill_without_retraining(
    synthetic_experiment,
    fake_wandb,
    monkeypatch,
):
    cfg, _ = synthetic_experiment
    original = workflow.conditioning(cfg)
    cfg.wandb.enabled = True
    monkeypatch.setattr(
        workflow, "Runtime", lambda *args: pytest.fail("Reloaded model")
    )
    assert workflow.conditioning(cfg) == original
    assert len(fake_wandb.runs) == 3
    assert all(len(run.logs) == 2 for run in fake_wandb.runs)
    assert all(not run.artifacts for run in fake_wandb.runs)
    assert workflow.conditioning(cfg) == original
    assert len(fake_wandb.runs) == 3


def test_real_wandb_offline_smoke(tmp_path):
    pytest.importorskip("wandb")
    cfg = ExperimentConfig(
        bank_config=tmp_path / "bank.yaml",
        staging_dir=tmp_path / "local",
        wandb={"enabled": True, "mode": "offline", "upload_artifacts": True},
    )
    output = tmp_path / "run"
    with ExperimentTracker(
        cfg, {"kind": "conditioning", "condition": "last"}, output, {"smoke": True}
    ) as tracker:
        tracker.training(1, {"train/drafter_loss": 0.25})
        tracker.validation(
            {
                "score": 0.5,
                "validation_examples": 100,
                "contexts": {"1": {"full_acc_0": 0.5}},
            },
            1,
            1,
        )
        write_json(output / "result.json", {"smoke": True})
        tracker.artifact("offline-smoke", [output / "result.json"])
        assert tracker.run.offline
    assert json.loads((output / "wandb.json").read_text())["status"] == "finished"
    assert list((cfg.staging_dir / "wandb").rglob("*.wandb"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_prompt_prefix_reader_skips_response_payload(tmp_path, monkeypatch, dtype):
    states = torch.randn(100, 4, 8).to(dtype)
    path = tmp_path / "hs_0.safetensors"
    save_file({"hidden_states": states, "token_ids": torch.arange(100)}, str(path))
    original_open = Path.open
    read_bytes = []

    @contextmanager
    def tracked_open(file_path, *args, **kwargs):
        with original_open(file_path, *args, **kwargs) as handle:

            def read(size=-1):
                data = handle.read(size)
                read_bytes.append(len(data))
                return data

            yield SimpleNamespace(read=read, seek=handle.seek)

    monkeypatch.setattr(Path, "open", tracked_open)
    actual = BankFileTransfer(tmp_path).get_prompt_states(0, 5)
    torch.testing.assert_close(actual, states[:5], rtol=0, atol=0)
    assert sum(read_bytes) < path.stat().st_size / 4
    assert BankFileTransfer(tmp_path).get_prompt_states(1, 5) is None
    with pytest.raises(ValueError, match="prompt boundary"):
        BankFileTransfer(tmp_path).get_prompt_states(0, 101)


def test_prompt_read_workers_are_concurrent_bounded_and_ordered(
    synthetic_experiment,
    monkeypatch,
):
    cfg, factory = synthetic_experiment
    runtime = factory(cfg, BankConfig.load(cfg.bank_config))
    cfg.preprocessing_workers = 3
    barrier, lock = Barrier(3), Lock()
    active, maximum = 0, 0

    def read(index, length):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        if index < 3:
            barrier.wait(timeout=5)
        with lock:
            active -= 1
        return torch.full((length, 4, 8), index)

    monkeypatch.setattr(runtime.corpus.transfer, "get_prompt_states", read)
    results = list(runtime.corpus.prompt_states([0, 1, 2, 3, 4]))
    assert [index for index, _, _ in results] == [0, 1, 2, 3, 4]
    assert maximum == 3


def test_parallel_prompt_summaries_match_serial_and_preserve_run_identity(
    synthetic_experiment,
):
    cfg, factory = synthetic_experiment
    bank = BankConfig.load(cfg.bank_config)
    serial_cfg = cfg.model_copy(deep=True)
    serial_cfg.staging_dir = cfg.staging_dir / "serial"
    serial_cfg.output_root = cfg.output_root / "serial"
    serial_cfg.preprocessing_workers = 1
    serial = factory(serial_cfg, bank)
    parallel_cfg = serial_cfg.model_copy(deep=True)
    parallel_cfg.staging_dir = cfg.staging_dir / "parallel"
    parallel_cfg.output_root = cfg.output_root / "parallel"
    parallel_cfg.preprocessing_workers = 4
    parallel = factory(parallel_cfg, bank)
    for key, tensor in serial.corpus.summaries.items():
        torch.testing.assert_close(
            parallel.corpus.summaries[key], tensor, rtol=0, atol=0
        )
    job = {"kind": "conditioning", "condition": "last"}
    identity = run_identity(serial_cfg, bank, serial.corpus, job)
    serial_cfg.preprocessing_workers = 8
    serial_cfg.activation_cache_gib = 128
    serial_cfg.activation_prefetch_workers = 4
    assert run_identity(serial_cfg, bank, serial.corpus, job) == identity
