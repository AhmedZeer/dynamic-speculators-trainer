"""Continuous-batch adapter isolation and original-source baseline launch checks."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.bank.artifacts import file_digest, write_json
from speculators.benchmark.config import Suite
from speculators.benchmark.suite import server_command
from speculators.generator.serving import controller as shared_module
from speculators.generator.serving import patch, per_request, vanilla
from speculators.generator.serving.controller import prepare_proposer, validate_runtime
from speculators.generator.serving.endpoint import (
    validate_adapter_delta,
    validate_status,
)
from speculators.generator.serving.multilora import NativeDraftLoRA
from speculators.generator.serving.per_request import PerRequestAdapterController
from tests.unit.generator.test_serving import draft, valid_config, write_bundle


class TensorBackend:
    """CPU reference backend; does not stand in for a native CUDA kernel test."""

    def __init__(self, model, metadata, config):
        self.capacity = config.scheduler_config.max_num_seqs
        self.resident = {}
        self.tokens = []
        self.history = []
        self.scale = metadata["alpha"] / metadata["rank"]

    def activate(self, requests):
        assert len(requests) <= self.capacity
        self.resident = dict(requests)

    def map_tokens(self, tokens, ids):
        self.tokens = list(tokens)
        self.history.append((list(tokens), list(ids)))

    def release(self, name):
        self.resident.pop(name, None)

    def linear(self, inputs, weight, name):
        output = inputs @ weight.T
        by_id = {entry[0]: entry[1] for entry in self.resident.values()}
        for index, adapter_id in enumerate(self.tokens):
            a, b = by_id[adapter_id][name]
            delta = self.scale * (inputs[index] @ a.T) @ b.T
            if name == "v_proj":
                output[index, -b.shape[0] :] += delta
            else:
                output[index] += delta
        return output


def controller(tmp_path, capacity=2):
    write_bundle(tmp_path / "bundle")
    model = draft()
    config = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=capacity))
    return PerRequestAdapterController(
        tmp_path / "bundle", model, config, backend_factory=TensorBackend
    ), model


def md(offsets):
    return SimpleNamespace(query_start_loc_cpu=torch.tensor(offsets))


def test_mixed_prefill_decode_reordering_and_per_projection_isolation(tmp_path):
    ctl, model = controller(tmp_path)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    calls = []
    ctl.generator.register_forward_pre_hook(
        lambda _, args: calls.append(args[0].shape[0])
    )
    ctl.prepare_scheduled(["a"], [2], torch.arange(2), torch.randn(2, 12), [0, 2])
    first = ctl.adapters["a"]
    # B joins while A remains active: one existing query plus a complete prefill.
    ctl.prepare_scheduled(
        ["a", "b"], [2, 3], torch.tensor([7, 0, 1, 2]), torch.randn(4, 12), [0, 1, 4]
    )
    assert ctl.adapters["a"] is first
    assert calls == [1, 1]
    ctl.map_forward(md([0, 1, 4]))
    a_id, b_id = ctl.adapters["a"][0], ctl.adapters["b"][0]
    assert ctl.backend.tokens == [a_id, b_id, b_id, b_id]
    attention = model.model.layers[0].self_attn
    for name, base in [
        ("v_proj", attention.qkv_proj.weight),
        ("o_proj", attention.o_proj.weight),
    ]:
        inputs = torch.randn(4, base.shape[1])
        mixed = ctl.backend.linear(inputs, base, name)
        expected = inputs @ base.T
        for i, request in enumerate(["a", "b", "b", "b"]):
            a, b = ctl.adapters[request][1][name]
            delta = 2 * (inputs[i] @ a.T) @ b.T
            if name == "v_proj":
                expected[i, -2:] += delta
                torch.testing.assert_close(mixed[i, :-2], (inputs @ base.T)[i, :-2])
            else:
                expected[i] += delta
        torch.testing.assert_close(mixed, expected)
    # Later draft forwards use one token per request, not the prefill mapping.
    ctl.map_forward(md([0, 1, 2]))
    assert ctl.backend.tokens == [a_id, b_id]
    ctl.prepare_scheduled(
        ["b", "a"], [3, 2], torch.tensor([8, 10]), torch.randn(2, 12), [0, 1, 2]
    )
    ctl.map_forward(md([0, 1, 2]))
    assert ctl.backend.tokens == [b_id, a_id]
    assert calls == [1, 1]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_pause_recompute_completion_slot_reuse_and_counters(tmp_path):
    ctl, _ = controller(tmp_path, capacity=1)
    before = {
        "configuration": {"kind": "generator", "adapter_policy": "per_request"},
        "workers": [{"adapter_totals": dict(ctl.totals)}],
    }
    ctl.prepare_scheduled(["a"], [2], torch.arange(2), torch.randn(2, 12), [0, 2])
    retained = ctl.adapters["a"]
    ctl.prepare_scheduled(["b"], [2], torch.arange(2), torch.randn(2, 12), [0, 2])
    assert set(ctl.backend.resident) == {"b"}
    assert set(ctl.adapters) == {"a", "b"}
    # Resumed A recomputes prompt + generated tokens; keep its original adapter.
    ctl.prepare_scheduled(["a"], [2], torch.arange(5), torch.randn(5, 12), [0, 5])
    assert ctl.adapters["a"] is retained
    assert ctl.totals["generator_invocations"] == 2
    assert set(ctl.backend.resident) == {"a"}
    record = ctl.finish("b")
    assert ctl.finish("b") == record
    assert set(ctl.backend.resident) == {"a"}
    ctl.finish("a")
    after = {**before, "workers": [{"adapter_totals": dict(ctl.totals)}]}
    delta = validate_adapter_delta(before, after)
    assert delta["generator_invocations"] == 2
    assert delta["mean_conditioning_size"] == 1
    assert ctl.active is None
    assert not ctl.adapters
    assert not ctl.backend.resident


def test_profile_does_not_count_requests_and_always_releases(tmp_path):
    ctl, _ = controller(tmp_path)
    before = dict(ctl.totals)

    def failed_forward():
        with ctl.profiling(6):
            assert len(ctl.backend.resident) == 2
            assert len(ctl.backend.tokens) == 6
            raise RuntimeError("kernel failure")

    with pytest.raises(RuntimeError, match="kernel failure"):
        failed_forward()
    assert ctl.totals == before
    assert not ctl.backend.resident
    assert not ctl.backend.tokens


def test_reject_partial_new_prefill_and_bad_mapping(tmp_path):
    ctl, _ = controller(tmp_path)
    with pytest.raises(ValueError, match="complete target prefill"):
        ctl.prepare_scheduled(["a"], [3], torch.arange(2), torch.randn(2, 12), [0, 2])
    with pytest.raises(ValueError, match="boundaries"):
        ctl.prepare_scheduled(["a"], [2], torch.arange(2), torch.randn(2, 12), [1, 2])
    assert ctl.totals["generator_invocations"] == 0


def test_proposer_continuous_runtime_and_native_scheduler(tmp_path, monkeypatch):
    ctl, _ = controller(tmp_path)
    serving = {
        "adapter_policy": "per_request",
        "batching_policy": "continuous",
        "conditioning_examples": 1,
    }
    monkeypatch.setenv("SPECULATORS_SERVER_CONFIG", json.dumps(serving))
    cfg = valid_config(ctl.metadata)
    cfg.scheduler_config.max_num_seqs = 2
    cfg.scheduler_config.scheduler_cls = None
    validate_runtime(cfg, ctl.metadata)
    p = SimpleNamespace(
        draft_adapter=ctl,
        adapter_runner=SimpleNamespace(
            input_batch=SimpleNamespace(req_ids=["a"]),
            requests={"a": SimpleNamespace(num_prompt_tokens=2)},
        ),
    )
    prepare_proposer(p, torch.arange(2), torch.randn(2, 12), md([0, 2]))
    assert ctl.totals["prepared_requests"] == 1
    cfg.lora_config = object()
    with pytest.raises(ValueError, match="target without LoRA"):
        validate_runtime(cfg, ctl.metadata)


def test_vanilla_view_restores_original_sources_without_mutating_install(
    tmp_path, monkeypatch
):
    root = tmp_path / "site"
    pkg = root / "vllm"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "unchanged.py").write_text("native = True\n")
    target = pkg / "v1/eagle.py"
    target.parent.mkdir()
    original = "native = True\n"
    modified = "native = False\n"
    target.write_text(original)
    sha = file_digest(target)
    backup = root / patch.BACKUP_NAME / "vllm/v1/eagle.py"
    backup.parent.mkdir(parents=True)
    backup.write_text(original)
    target.write_text(modified)
    patches = tmp_path / "patches"
    patches.mkdir()
    manifest = {
        "version": "0.31.0",
        "commit": "test-pin",
        "files": {
            "vllm/v1/eagle.py": {
                "before_sha256": sha,
                "after_sha256": file_digest(target),
            }
        },
    }
    write_json(patches / "manifest.json", manifest)
    monkeypatch.setattr(vanilla, "PATCH_DIR", patches)
    monkeypatch.setattr(vanilla, "patch_sources", lambda **_: {"state": "patched"})
    view = vanilla.vanilla_source_view(tmp_path / "provenance", root=root)
    assert (view / "vllm/v1/eagle.py").read_text() == original
    assert target.read_text() == modified
    assert (view / "vllm/unchanged.py").is_symlink()
    backup.write_text("corrupted")
    with pytest.raises(ValueError, match="verified original"):
        vanilla.vanilla_source_view(tmp_path / "bad", root=root)


def test_new_suite_policies_and_baseline_source_guard():
    config = Suite.load(Path("examples/eagle3-lora/benchmark_per_request_4000.yaml"))
    assert config.settings.batching_policy == "continuous"
    assert config.models[0].vanilla_vllm
    command = server_command(
        Path("/repo"),
        {
            "type": "drafter",
            "path": "drafter",
            "revision": "a" * 40,
            "vanilla_vllm": True,
        },
        {"id": "target", "revision": "b" * 40},
        config.settings,
        Path("/out"),
    )
    assert "--vanilla-vllm" in command
    assert command[command.index("--batching-policy") + 1] == "continuous"
    status = {
        "configuration": {"models": {"target": "target"}, "vllm_sources": "vanilla"},
        "workers": [{}],
    }
    with pytest.raises(ValueError, match="Vanilla baseline worker source differs"):
        validate_status(status, {"models": {"target": "target"}})


def test_native_adapter_residency_and_mutating_scaling_cannot_change_retained_factors(
    monkeypatch,
):
    class Layer:
        def __init__(self, path, rank, alpha, a, b):
            self.lora_a, self.lora_b = a, b
            self.scale = alpha / rank

    class Model:
        def __init__(self, adapter_id, rank, layers):
            self.id, self.layers = adapter_id, layers

    class Manager:
        def __init__(self):
            self.registered = {}

        def add_adapter(self, adapter):
            for layer in adapter.layers.values():
                layer.lora_b.mul_(layer.scale)  # native packing optimizes in place
            self.registered[adapter.id] = adapter

        def activate_adapter(self, adapter_id):
            assert adapter_id in self.registered

        def remove_adapter(self, adapter_id):
            del self.registered[adapter_id]

    monkeypatch.setitem(
        sys.modules, "vllm.lora.lora_model", SimpleNamespace(LoRAModel=Model)
    )
    monkeypatch.setitem(
        sys.modules, "vllm.lora.lora_weights", SimpleNamespace(LoRALayerWeights=Layer)
    )
    backend = NativeDraftLoRA.__new__(NativeDraftLoRA)
    backend.capacity, backend.resident = 1, {}
    backend.metadata = {"rank": 2, "alpha": 4}
    backend.manager = Manager()
    factors = {"v_proj": (torch.ones(2, 8), torch.ones(2, 2))}
    backend.activate({"a": (1, factors)})
    backend.activate({"b": (2, factors)})
    backend.activate({"a": (1, factors)})
    assert set(backend.manager.registered) == {1}
    assert backend.resident == {"a": 1}
    assert torch.equal(factors["v_proj"][1], torch.ones(2, 2))
    layer = backend.manager.registered[1].layers["model.layers.0.self_attn.v_proj"]
    assert torch.equal(layer.lora_b, torch.full((2, 2), 2.0, dtype=torch.bfloat16))
    backend.release("a")
    assert not backend.manager.registered


def test_installed_patch_routes_every_draft_forward_and_profiles(tmp_path, monkeypatch):
    ctl, _ = controller(tmp_path)
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_adapter_path="bundle")
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        requests={name: SimpleNamespace(num_prompt_tokens=2) for name in ["a", "b"]},
    )

    class Base:
        def __init__(self, vllm_config, *args, **kwargs):
            self.vllm_config = vllm_config
            self.speculative_config = vllm_config.speculative_config

        def load_model(self, target):
            self.model = target

        def build_per_group_and_layer_attn_metadata(self, metadata, draft_index):
            return metadata, draft_index

        def propose(self, **kwargs):
            self.build_per_group_and_layer_attn_metadata(kwargs["common_attn_metadata"])
            self.build_per_group_and_layer_attn_metadata(md([0, 1, 2]), draft_index=1)

        def dummy_run(self, num_tokens):
            assert len(ctl.backend.resident) == 2
            assert len(ctl.backend.tokens) == num_tokens

    monkeypatch.setattr(
        shared_module,
        "initialize_proposer",
        lambda p, r: setattr(p, "adapter_runner", r),
    )
    monkeypatch.setattr(per_request, "create_controller", lambda p: ctl)
    manifest = json.loads((patch.PATCH_DIR / "manifest.json").read_text())
    replacement = manifest["files"]["vllm/v1/spec_decode/eagle.py"]["replacements"][0][
        1
    ]
    source = (
        "class EagleProposer(Base):\n"
        "    def __init__(self, vllm_config, device, runner=None):\n"
        "        super().__init__(vllm_config, device,\n" + replacement
    )
    namespace = {"Base": Base}
    exec(compile(source, "patched_eagle.py", "exec"), namespace)  # noqa: S102
    proposer = namespace["EagleProposer"](config, None, runner)
    proposer.load_model(None)
    proposer.dummy_run(4)
    assert ctl.totals["generator_invocations"] == 0
    proposer.propose(
        target_positions=torch.tensor([0, 1, 0, 1]),
        target_hidden_states=torch.randn(4, 12),
        common_attn_metadata=md([0, 2, 4]),
    )
    a_id, b_id = ctl.adapters["a"][0], ctl.adapters["b"][0]
    assert ctl.backend.history[-2:] == [
        ([a_id, a_id, b_id, b_id], [a_id, b_id]),
        ([a_id, b_id], [a_id, b_id]),
    ]
    assert ctl.totals["draft_forward_calls"] == 2
    proposer.on_requests_finished(["a", "b"])
    assert ctl.active is None


@pytest.mark.parametrize("baseline", [True, False])
@pytest.mark.parametrize("profile_enabled", [True, False])
def test_continuous_launch_uses_native_scheduler_and_explicit_source_policy(
    tmp_path, monkeypatch, baseline, profile_enabled
):
    import importlib.metadata  # noqa: PLC0415
    from argparse import Namespace  # noqa: PLC0415

    from speculators.generator.serving import server  # noqa: PLC0415

    monkeypatch.setenv("SPECULATORS_SERVER_CONFIG", "{}")
    monkeypatch.setenv("PYTHONPATH", "")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda *_: "0.31.0")
    monkeypatch.setattr(
        server, "snapshot_download", lambda model, revision: f"/cache/{revision}"
    )
    monkeypatch.setattr(server, "patch_sources", lambda: {"state": "patched"})
    monkeypatch.setattr(
        vanilla, "vanilla_source_view", lambda output: Path("/original-vllm")
    )
    args = Namespace(
        model="target",
        revision="a" * 40,
        drafter="drafter",
        drafter_revision="b" * 40,
        adapter_bundle=None,
        provenance_dir=tmp_path / "server",
        no_hash_checkpoints=False,
        served_model_name="speculators",
        host="127.0.0.1",
        port=8000,
        seed=42,
        spec_tokens=3,
        max_model_len=8192,
        concurrency=8,
        batch_wait_ms=50,
        gpu_memory_utilization=0.85,
        conditioning_examples=None,
        batching_policy="continuous",
        adapter_policy="shared" if baseline else "per_request",
        vanilla_vllm=baseline,
    )
    if not baseline:
        metadata, _ = write_bundle(tmp_path / "bundle")
        metadata["models"] = {
            "target": {"id": "target", "revision": "a" * 40},
            "drafter": {"id": "drafter", "revision": "b" * 40},
        }
        write_json(tmp_path / "bundle/bundle.json", metadata)
        args.adapter_bundle = tmp_path / "bundle"
    if profile_enabled:
        from speculators.benchmark.config import Settings  # noqa: PLC0415
        from speculators.benchmark.profiling import profiler_config  # noqa: PLC0415

        args.profiler_config = profiler_config(Settings(), tmp_path / "server")
    command = server.build_server(args, [])
    assert ("--profiler-config" in command) == profile_enabled
    assert "--scheduler-cls" not in command
    assert "--enable-lora" not in command
    speculative = json.loads(command[command.index("--speculative-config") + 1])
    assert ("draft_adapter_path" in speculative) is not baseline
    saved = json.loads((tmp_path / "server/server.json").read_text())
    assert ("profiler" in saved) == profile_enabled
    assert saved["batching_policy"] == "continuous"
    assert saved["batch_wait_ms"] == 0
    assert saved["vllm_sources"] == ("vanilla" if baseline else "installed")
    if not baseline:
        assert saved["conditioning_examples"] == 1
        assert saved["deployment_context_size"] == 1
    else:
        import os  # noqa: PLC0415

        assert os.environ["PYTHONPATH"].startswith("/original-vllm")
