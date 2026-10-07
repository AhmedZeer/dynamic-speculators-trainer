import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from datasets import Dataset
from safetensors.torch import save_file
from torch import nn

from speculators.bank.artifacts import file_digest, write_json
from speculators.bank.config import BankConfig
from speculators.generator.config import Architecture, ExperimentConfig
from speculators.generator.model import LoRAGenerator
from speculators.generator.serving import benchmark as benchmark_module
from speculators.generator.serving import bundle, patch
from speculators.generator.serving import controller as controller_module
from speculators.generator.serving.benchmark import (
    BenchmarkConfig,
    benchmark_prompts,
    check_outputs,
    metric_delta,
    scalar_metrics,
    summarize,
)
from speculators.generator.serving.controller import (
    AdapterMerger,
    DraftAdapterController,
    validate_runtime,
)
from speculators.generator.serving.worker import run_worker

SHAPES = {"o_proj": [4, 4], "v_proj": [8, 2]}


def architecture():
    return Architecture(
        encoder_hidden=8,
        example_dim=4,
        condition_dim=4,
        descriptor_dim=2,
        decoder_width=8,
        residual_blocks=1,
    )


def draft(dtype=torch.float32):
    attention = nn.Module()
    attention.qkv_proj = nn.Linear(8, 8, bias=False, dtype=dtype)
    attention.o_proj = nn.Linear(4, 4, bias=False, dtype=dtype)
    layer = nn.Module()
    layer.self_attn = attention
    inner = nn.Module()
    inner.layers = nn.ModuleList([layer])
    model = nn.Module()
    model.model = inner
    return model


def write_bundle(path, kind="generator"):
    path.mkdir()
    generator = LoRAGenerator(12, SHAPES, 2, architecture())
    with torch.no_grad():
        for head in generator.heads.values():
            head["B"].bias.normal_(std=0.02)
    metadata = {
        "version": 1,
        "kind": kind,
        "rank": 2,
        "alpha": 4,
        "shapes": SHAPES,
        "condition": "last" if kind == "generator" else None,
        "input_dim": 12,
        "architecture": architecture().model_dump(),
        "max_examples": 8,
        "layer_ids": [2, 18, 33],
        "draft_config": {},
        "models": {
            role: {"id": role, "revision": role + "-revision"}
            for role in ("target", "drafter")
        },
    }
    if kind == "generator":
        state = generator.state_dict()
    else:
        state = {
            f"{name}.{label}": torch.randn(*shape) * 0.02
            for name, (inputs, outputs) in SHAPES.items()
            for label, shape in (("A", (2, inputs)), ("B", (outputs, 2)))
        }
    save_file(state, path / "model.safetensors")
    metadata["weights_sha256"] = file_digest(path / "model.safetensors")
    write_json(path / "bundle.json", metadata)
    return metadata, generator


@pytest.mark.parametrize("kind", ["generator", "static_lora"])
def test_export_round_trip_and_no_activation_io(tmp_path, monkeypatch, kind):
    root = tmp_path / "bank"
    root.mkdir()
    bank = BankConfig(output_root=root, lora={"rank": 2, "alpha": 4})
    bank_yaml = tmp_path / "bank.yaml"
    bank_yaml.write_text(yaml.safe_dump(bank.model_dump(mode="json")))
    cfg = ExperimentConfig(bank_config=bank_yaml, architecture=architecture())
    models = bank.models.model_dump()
    revisions = {"target_sha": "target-revision", "drafter_sha": "drafter-revision"}
    manifest = {
        "prepared_fingerprint": "test",
        "revisions": revisions,
        "cache_spec": {
            "models": models,
            "hidden_states": {"layer_ids": [2, 18, 33, 36]},
        },
        "hidden_states_path": "/does/not/exist",
    }
    write_json(root / "manifest.json", manifest)
    source_config = tmp_path / "draft-config.json"
    write_json(source_config, {})
    monkeypatch.setattr(bundle, "hf_hub_download", lambda *a, **kw: str(source_config))
    run = tmp_path / "run"
    run.mkdir()
    job = {
        "kind": "adaptation",
        "condition": "last",
        "arm": "fresh_generator" if kind == "generator" else "fresh_lora",
    }
    identity = {
        "job": job,
        "lora": bank.lora.model_dump(mode="json"),
        "configuration": cfg.model_dump(mode="json"),
        "bank_data": {"prepared_fingerprint": "test", "revisions": revisions},
    }
    write_json(run / "resolved_experiment.json", identity)
    write_json(run / "result.json", {"score": 0.5})
    (run / "train_command.txt").write_text("source command\n")
    (run / "speculators.patch").write_text("source patch\n")
    generator = LoRAGenerator(12, SHAPES, 2, architecture())
    state = (
        generator.state_dict()
        if kind == "generator"
        else {
            f"layers.0.self_attn.{name}.{suffix}": torch.randn(*shape)
            for name, (inputs, outputs) in SHAPES.items()
            for suffix, shape in (("a", (2, inputs)), ("b", (outputs, 2)))
        }
    )
    torch.save(
        {"model": state, "optimizer": {"do_not_export": torch.ones(5)}},
        run / "checkpoint.pt",
    )
    output = tmp_path / "export"
    metadata = bundle.export_run(cfg, run, output)
    exported_metadata, exported = bundle.bundle_state(output)
    assert metadata == exported_metadata
    assert not any("optimizer" in key for key in exported)
    assert (output / "train_command.txt").read_text() == "source command\n"
    assert not (output / "checkpoint.pt").exists()
    if kind == "generator":
        rebuilt = bundle.load_generator(metadata, exported)
        summaries, lengths = torch.randn(1, 12), torch.tensor([7])
        for name, pair in generator(summaries, lengths).items():
            for left, right in zip(
                pair, rebuilt(summaries, lengths)[name], strict=True
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
    else:
        for name in SHAPES:
            torch.testing.assert_close(
                exported[f"{name}.A"], state[f"layers.0.self_attn.{name}.a"]
            )
    with pytest.raises(FileExistsError):
        bundle.export_run(cfg, run, output)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_merger_parity_qk_preservation_and_exact_restoration(dtype):
    model = draft(dtype)
    attention = model.model.layers[0].self_attn
    original = attention.qkv_proj.weight.detach().clone()
    merger = AdapterMerger(model, {"shapes": SHAPES, "rank": 2, "alpha": 4})
    factors = {
        name: (torch.randn(2, inputs) * 0.02, torch.randn(outputs, 2) * 0.02)
        for name, (inputs, outputs) in SHAPES.items()
    }
    inputs = torch.randn(5, 8)
    a, b = factors["v_proj"]
    explicit = inputs @ original[-2:].float().T + 2 * (inputs @ a.T) @ b.T
    merger.apply(factors)
    merged = inputs @ attention.qkv_proj.weight[-2:].float().T
    torch.testing.assert_close(
        merged,
        explicit,
        atol=1e-5 if dtype == torch.float32 else 0.02,
        rtol=1e-5 if dtype == torch.float32 else 0.02,
    )
    assert torch.equal(attention.qkv_proj.weight[:-2], original[:-2])
    first = attention.qkv_proj.weight.detach().clone()
    merger.apply(factors)
    assert torch.equal(attention.qkv_proj.weight, first)
    merger.restore()
    assert torch.equal(attention.qkv_proj.weight, original)
    for name in SHAPES:
        assert torch.equal(merger.weights[name], merger.base[name])


@pytest.mark.parametrize("kind", ["generator", "static_lora"])
def test_request_lifetime_and_profiling(tmp_path, kind):
    metadata, generator = write_bundle(tmp_path / "bundle", kind)
    model = draft()
    original = model.model.layers[0].self_attn.qkv_proj.weight.detach().clone()
    controller = DraftAdapterController(tmp_path / "bundle", model)
    controller.profile()
    assert controller.active is None
    assert controller.record == {}
    assert torch.equal(model.model.layers[0].self_attn.qkv_proj.weight, original)
    states = torch.randn(3, 12)
    calls = []
    if controller.generator is not None:
        controller.generator.register_forward_pre_hook(
            lambda _, args: calls.append(args)
        )
    controller.prepare("first", 3, torch.arange(3), states)
    first = model.model.layers[0].self_attn.qkv_proj.weight.detach().clone()
    controller.prepare("first", 3, torch.tensor([3, 4]), torch.randn(2, 12))
    assert torch.equal(model.model.layers[0].self_attn.qkv_proj.weight, first)
    with pytest.raises(RuntimeError, match="predecessor"):
        controller.prepare("second", 3, torch.arange(3), states)
    record = controller.finish("first")
    assert record["generator_invocations"] == int(kind == "generator")
    assert controller.finish("first") == record
    assert torch.equal(model.model.layers[0].self_attn.qkv_proj.weight, original)
    controller.prepare("first", 3, torch.arange(3), states + 0.2)
    controller.finish("first")
    if kind == "generator":
        assert len(calls) == 2
        torch.testing.assert_close(calls[0][0], states[-1:].float())
        assert calls[0][1].tolist() == [3]
    with pytest.raises(ValueError, match="complete target prefill"):
        controller.prepare("invalid", 3, torch.tensor([1, 2]), states[1:])


def valid_config(metadata):
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="eagle3",
            enforce_eager=True,
            quantization=None,
            draft_tensor_parallel_size=1,
            parallel_drafting=False,
            draft_model_config=SimpleNamespace(
                model="drafter",
                revision="drafter-revision",
                dtype=torch.bfloat16,
                hf_config=SimpleNamespace(),
            ),
        ),
        model_config=SimpleNamespace(
            model="target",
            revision="target-revision",
            enforce_eager=True,
            dtype=torch.bfloat16,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=1, enable_chunked_prefill=False, async_scheduling=False
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1
        ),
    )


def test_runtime_rejects_unsupported_settings_and_revision(tmp_path):
    metadata, _ = write_bundle(tmp_path / "bundle")
    config = valid_config(metadata)
    validate_runtime(config, metadata)
    assert (
        config.speculative_config.draft_model_config.hf_config.eagle_aux_hidden_state_layer_ids
        == [2, 18, 33]
    )
    for owner, key, value in [
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "enable_chunked_prefill", True),
        ("scheduler_config", "async_scheduling", True),
        ("cache_config", "enable_prefix_caching", True),
        ("parallel_config", "tensor_parallel_size", 2),
        ("model_config", "enforce_eager", False),
    ]:
        changed = copy.deepcopy(config)
        setattr(getattr(changed, owner), key, value)
        with pytest.raises(ValueError, match="requires"):
            validate_runtime(changed, metadata)
    config.model_config.revision = "main"
    with pytest.raises(ValueError, match="pinned"):
        validate_runtime(config, metadata)


def test_patch_check_apply_revert_and_refusal(tmp_path, monkeypatch):
    # Small source fixture exercises the transactional installer without a vLLM import.
    originals = {"vllm/a.py": "old = 1\n", "vllm/b.py": "old = 2\n"}
    manifest = {"files": {}}
    patch_dir = tmp_path / "patch"
    patch_dir.mkdir()
    for name, original in originals.items():
        file = tmp_path / name
        file.parent.mkdir(exist_ok=True)
        file.write_text(original)
        modified = original.replace("old", "new")
        manifest["files"][name] = {
            "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(modified.encode()).hexdigest(),
            "replacements": [["old", "new"]],
        }
    write_json(patch_dir / "manifest.json", manifest)
    monkeypatch.setattr(patch, "PATCH_DIR", patch_dir)
    assert (
        patch.patch_sources("check", tmp_path, version="0.31.0")["state"] == "original"
    )
    assert (
        patch.patch_sources("apply", tmp_path, version="0.31.0")["state"] == "patched"
    )
    assert (
        patch.patch_sources("apply", tmp_path, version="0.31.0")["state"] == "patched"
    )
    assert (
        patch.patch_sources("revert", tmp_path, version="0.31.0")["state"] == "original"
    )
    for name, original in originals.items():
        assert (tmp_path / name).read_text() == original
    (tmp_path / "vllm/a.py").write_text("unrecognized\n")
    with pytest.raises(ValueError, match="refusing"):
        patch.patch_sources("apply", tmp_path, version="0.31.0")
    with pytest.raises(ValueError, match="only"):
        patch.patch_sources("check", tmp_path, version="0.32.0")


def test_prompt_boundaries_without_hidden_state_loading(tmp_path):
    Dataset.from_list([{"input_ids": [1, 2, 3, 4], "prompt_length": 2}]).save_to_disk(
        tmp_path / "rows"
    )
    cfg = BenchmarkConfig(
        bank_config=tmp_path / "bank.yaml",
        arms=["target_only"],
        prompts=1,
        max_model_len=4,
        max_tokens=10,
    )
    rows = benchmark_prompts(
        cfg,
        BankConfig(),
        {"data_path": str(tmp_path / "rows")},
        {"validation_indices": [0]},
    )
    assert rows == [{"row": 0, "prompt_token_ids": [1, 2], "max_tokens": 2}]


def test_metrics_deltas_exclude_warmup_and_undefined_target_acceptance():
    def metrics(drafts, proposed, accepted, positions):
        return [
            SimpleNamespace(name="vllm:spec_decode_num_drafts", value=drafts),
            SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=proposed),
            SimpleNamespace(
                name="vllm:spec_decode_num_accepted_tokens", value=accepted
            ),
            SimpleNamespace(
                name="vllm:spec_decode_num_accepted_tokens_per_pos", values=positions
            ),
        ]

    difference = metric_delta(
        scalar_metrics(metrics(3, 9, 5, [3, 2, 0])),
        scalar_metrics(metrics(5, 15, 9, [5, 3, 1])),
    )
    assert difference == {
        "num_drafts": 2,
        "num_draft_tokens": 6,
        "num_accepted_tokens": 4,
        "accepted_per_position": [2, 1, 1],
    }
    row = {
        **difference,
        "request_seconds": 2,
        "output_token_ids": [1, 2, 3, 4],
        "peak_gpu_allocated_bytes": 10,
    }
    value = summarize([row])
    assert value["accepted_length"] == 3
    assert value["acceptance_ratio"] == pytest.approx(4 / 6)
    assert value["output_tokens_per_second"] == 2
    empty = {**row, **scalar_metrics([])}
    assert summarize([empty])["acceptance_ratio"] is None


def test_worker_warmup_cleanup_and_output_comparison(tmp_path):
    class LLM:
        def __init__(self, **kwargs):
            self.calls = 0

        def generate(self, prompt, params, **kwargs):
            self.calls += 1
            return [
                SimpleNamespace(
                    request_id=str(self.calls),
                    outputs=[SimpleNamespace(token_ids=[4, 5])],
                )
            ]

        def get_metrics(self):
            return []

        def collective_rpc(self, method, args=()):
            if method.__name__ == "adapter_stats":
                return [{"peak_gpu_allocated_bytes": 1, "generator_invocations": 0}]
            return [None]

    job = {
        "arm": "target_only",
        "llm_args": {},
        "output": str(tmp_path),
        "prompts": [{"row": 7, "prompt_token_ids": [1, 2], "max_tokens": 2}],
        "warmups": 3,
        "repetitions": 2,
        "seed": 42,
    }
    value = run_worker(job, llm_type=LLM, sampling_type=lambda **kw: kw)
    assert len(value["requests"]) == 2
    assert value["summary"]["output_tokens"] == 4
    check_outputs(value, value)
    changed = copy.deepcopy(value)
    changed["requests"][0]["output_token_ids"] = [6]
    with pytest.raises(ValueError, match="disagreement"):
        check_outputs(value, changed)


def test_generator_remains_fp32_under_vllm_default_dtype(tmp_path):
    metadata, _ = write_bundle(tmp_path / "bundle")
    state = bundle.bundle_state(tmp_path / "bundle")[1]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        generator = bundle.load_generator(metadata, state)
        assert all(p.dtype == torch.float32 for p in generator.parameters())
    finally:
        torch.set_default_dtype(previous)


def test_patched_proposer_merges_before_draft_and_restores_on_finish(monkeypatch):
    events = []

    class Base:
        def __init__(self, config, *args, **kwargs):
            self.speculative_config = config.speculative_config

        def load_model(self, target):
            self.model = target
            events.append("load")

        def propose(self, **kwargs):
            events.append("draft")

        def dummy_run(self):
            events.append("dummy")

    class Controller:
        def __init__(self, *args):
            events.append("attach")

        def profile(self):
            events.append("profile")

        def finish(self, request_id):
            events.append(f"finish:{request_id}")

    def initialize(proposer, runner):
        proposer.adapter_runner = runner

    monkeypatch.setattr(controller_module, "initialize_proposer", initialize)
    monkeypatch.setattr(controller_module, "DraftAdapterController", Controller)
    monkeypatch.setattr(
        controller_module, "prepare_proposer", lambda *args: events.append("merge")
    )
    manifest = json.loads((patch.PATCH_DIR / "manifest.json").read_text())
    replacement = manifest["files"]["vllm/v1/spec_decode/eagle.py"]["replacements"][0]
    namespace = {"SpecDecodeBaseProposer": Base}
    source = (
        "class EagleProposer(SpecDecodeBaseProposer):\n"
        "    def __init__(self, vllm_config, device, runner=None):\n"
        "        super().__init__(vllm_config, device,\n" + replacement[1]
    )
    exec(compile(source, "patched_eagle.py", "exec"), namespace)  # noqa: S102
    proposer = namespace["EagleProposer"](
        SimpleNamespace(
            speculative_config=SimpleNamespace(draft_adapter_path="bundle")
        ),
        None,
    )
    proposer.load_model(None)
    proposer.dummy_run()
    proposer.propose(target_positions=None, target_hidden_states=None)
    proposer.on_requests_finished(["cancelled"])
    assert events == [
        "load",
        "attach",
        "profile",
        "dummy",
        "merge",
        "draft",
        "finish:cancelled",
    ]


def test_serial_benchmark_smoke_resume_and_output_guard(tmp_path, monkeypatch):
    bank = BankConfig(output_root=tmp_path / "bank")
    bank_file = tmp_path / "bank.yaml"
    bank_file.write_text(yaml.safe_dump(bank.model_dump(mode="json")))
    cfg = BenchmarkConfig(
        bank_config=bank_file,
        output_root=tmp_path / "results",
        staging_dir=tmp_path / "local",
        arms=["target_only", "eagle3_base"],
    )
    manifest = {
        "cache_spec": {"models": {"target": "target", "drafter": "drafter"}},
        "revisions": {"target_sha": "a", "drafter_sha": "b"},
    }
    monkeypatch.setattr(benchmark_module, "patch_sources", lambda: {"state": "patched"})
    monkeypatch.setattr(benchmark_module, "load_subset", lambda *args: (manifest, {}))

    def prompts(settings, *args):
        assert settings.prompts == 3
        assert settings.max_tokens == 32
        return [{"row": 7, "prompt_token_ids": [1, 2], "max_tokens": 32}]

    monkeypatch.setattr(benchmark_module, "benchmark_prompts", prompts)
    monkeypatch.setattr(
        benchmark_module, "snapshot_download", lambda *a, **kw: "pinned"
    )
    monkeypatch.setattr(benchmark_module, "provenance", lambda *args: None)
    monkeypatch.setattr(benchmark_module, "log_result", lambda *args: None)
    monkeypatch.setattr(benchmark_module, "export_comparison", lambda *args: None)
    monkeypatch.setattr(benchmark_module, "git_sha", lambda *args: "test-sha")
    monkeypatch.setattr(benchmark_module, "git_diff", lambda *args: "")
    jobs = []

    def worker(command, **kwargs):
        assert kwargs["check"] is True
        job = json.loads(Path(command[-1]).read_text())
        jobs.append(job)
        args = job["llm_args"]
        assert args["max_num_seqs"] == 1
        assert args["enforce_eager"] is True
        assert args["async_scheduling"] is False
        assert args["enable_prefix_caching"] is False
        assert args["enable_chunked_prefill"] is False
        assert job["warmups"] == job["repetitions"] == 1
        write_json(
            tmp_path / "results" / "smoke" / job["arm"] / "result.json",
            {
                "requests": [{"row": 7, "repetition": 0, "output_token_ids": [3]}],
                "summary": {"output_tokens_per_second": 2.0},
            },
        )

    monkeypatch.setattr(benchmark_module.subprocess, "run", worker)
    benchmark_module.benchmark(cfg, smoke=True)
    assert [job["arm"] for job in jobs] == ["target_only", "eagle3_base"]
    assert "speculative_config" not in jobs[0]["llm_args"]
    assert jobs[1]["llm_args"]["speculative_config"]["method"] == "eagle3"
    benchmark_module.benchmark(cfg, smoke=True)
    assert len(jobs) == 2
    result_file = cfg.output_root / "smoke" / "eagle3_base" / "result.json"
    result = json.loads(result_file.read_text())
    result["requests"][0]["output_token_ids"] = [9]
    write_json(result_file, result)
    with pytest.raises(ValueError, match="disagreement"):
        benchmark_module.benchmark(cfg, smoke=True)
    assert (result_file.parent / "output_disagreement.json").exists()
