"""Independent serving, token-exact workloads, and HTTP request lifetime checks."""

import asyncio
import importlib.metadata
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from speculators.bank.artifacts import write_json
from speculators.generator.serving import endpoint, server, validation
from speculators.generator.serving.controller import DraftAdapterController
from speculators.generator.serving.http import AdapterMiddleware, AdapterWorkerExtension
from tests.unit.generator.test_campaign import tiny_campaign  # noqa: F401
from tests.unit.generator.test_online_campaign import online_campaign  # noqa: F401
from tests.unit.generator.test_serving import draft, write_bundle


@pytest.fixture
def frozen_validation(online_campaign, tmp_path, monkeypatch):  # noqa: F811
    cfg, downloads, requests = online_campaign
    # Export must work with no saved banks or training Parquet files.
    import shutil  # noqa: PLC0415

    shutil.rmtree(cfg.data_path)
    for path in cfg.trajectories_path.rglob("train.parquet"):
        path.unlink()

    class Tokenizer:
        def decode(self, tokens, **kwargs):
            return " ".join(map(str, tokens))

        def encode(self, text, **kwargs):
            return [int(token) for token in text.split()]

    monkeypatch.setattr(
        validation.AutoTokenizer, "from_pretrained", lambda *_: Tokenizer()
    )
    output = tmp_path / "validation"
    result = validation.export_validation(cfg, output, max_tokens=4)
    assert result["examples"] == 80  # 5 selected subsets x 4 domains x 4 rows.
    assert not downloads
    assert not requests
    return output / "manifest.json"


def test_validation_preserves_membership_tokens_and_detects_changes(frozen_validation):
    manifest, groups = validation.load_validation(frozen_validation)
    assert len(manifest["subsets"]) == 20
    assert set(groups) == {"math", "code", "stem", "chat"}
    assert all(len(rows) == 20 for rows in groups.values())
    for rows in groups.values():
        assert all(row["prompt_token_ids"] == [1, 2] for row in rows)
        assert all(row["output_tokens_count"] == 4 for row in rows)
    path = frozen_validation.parent / "math.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        validation.load_validation(frozen_validation)


@pytest.mark.parametrize(
    "kind", ["target_only", "eagle3_base", "generator", "static_lora"]
)
def test_server_builds_matched_independent_command(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("SPECULATORS_SERVER_CONFIG", "{}")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    monkeypatch.setattr(importlib.metadata, "version", lambda *_: "0.31.0")
    monkeypatch.setattr(
        server, "snapshot_download", lambda model, revision: f"/cache/{revision}"
    )
    monkeypatch.setattr(server, "patch_sources", lambda: {"state": "patched"})
    args = Namespace(
        model="target",
        revision="a" * 40,
        drafter=None,
        drafter_revision=None,
        adapter_bundle=None,
        spec_tokens=3,
        max_model_len=8192,
        provenance_dir=str(tmp_path / "server"),
        no_hash_checkpoints=False,
        served_model_name="speculators",
        host="127.0.0.1",
        port=8000,
        gpu_memory_utilization=0.85,
        seed=42,
    )
    if kind == "eagle3_base":
        args.drafter, args.drafter_revision = "drafter", "b" * 40
    elif kind in ("generator", "static_lora"):
        bundle = tmp_path / "bundle"
        metadata, _ = write_bundle(bundle, kind)
        metadata["models"] = {
            "target": {"id": "target", "revision": "a" * 40},
            "drafter": {"id": "drafter", "revision": "b" * 40},
        }
        write_json(bundle / "bundle.json", metadata)
        args.adapter_bundle = bundle
        args.model, args.revision = None, None
    command = server.build_server(args, [])
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert "--no-enable-prefix-caching" in command
    assert "--no-enable-chunked-prefill" in command
    assert "--middleware" in command
    assert (
        command[command.index("--worker-extension-cls") + 1] == server.WORKER_EXTENSION
    )
    configuration = json.loads((tmp_path / "server/server.json").read_text())
    assert configuration["kind"] == kind
    if kind == "target_only":
        assert "--speculative-config" not in command
    else:
        speculative = json.loads(command[command.index("--speculative-config") + 1])
        assert speculative["method"] == "eagle3"
        assert ("draft_adapter_path" in speculative) == (
            kind in ("generator", "static_lora")
        )


def test_status_uses_serializable_named_worker_rpc(monkeypatch):
    monkeypatch.setenv("SPECULATORS_SERVER_CONFIG", '{"kind": "target_only"}')
    worker = SimpleNamespace(model_runner=SimpleNamespace())
    messages = []

    class Client:
        async def collective_rpc(self, method, args=()):
            assert method == "speculators_status"
            json.dumps([method, args])
            return [getattr(AdapterWorkerExtension, method)(worker, *args)]

    async def app(*_):
        pytest.fail("Status must be handled by the middleware")

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "path": "/speculators/status",
        "app": SimpleNamespace(state=SimpleNamespace(engine_client=Client())),
    }
    asyncio.run(AdapterMiddleware(app)(scope, None, send))
    assert messages[0]["status"] == 200
    payload = json.loads(messages[1]["body"])
    assert payload["configuration"]["kind"] == "target_only"
    assert payload["workers"][0]["adapter_controller_present"] is False
    assert payload["workers"][0]["adapter_totals"] == {}


def test_serve_parser_is_separate_from_legacy_modes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    from scripts import launch_vllm  # noqa: PLC0415

    monkeypatch.setattr(
        launch_vllm.sys,
        "argv",
        [
            "launch_vllm.py",
            "serve",
            "--adapter-bundle",
            "bundle",
            "--provenance-dir",
            str(tmp_path),
            "--port",
            "8123",
            "--dry-run",
        ],
    )
    args, extra = launch_vllm.parse_args()
    assert args.subcommand == "serve"
    assert args.port == 8123
    assert not extra


@pytest.mark.parametrize("path", ["/v1/completions", "/v1/chat/completions"])
@pytest.mark.parametrize("kind", ["generator", "static_lora"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("abort", [False, True, "early"])
def test_http_restores_before_completion_and_on_abort(  # noqa: C901 -- lifecycle cases
    tmp_path, stream, abort, kind, monkeypatch, path
):
    bundle = tmp_path / "bundle"
    write_bundle(bundle, kind)
    monkeypatch.setenv(
        "SPECULATORS_SERVER_CONFIG",
        json.dumps({"static_adapter_policy": "merged_once"}),
    )
    model = draft()
    base = model.model.layers[0].self_attn.o_proj.weight.detach().clone()
    controller = DraftAdapterController(bundle, model)
    original = model.model.layers[0].self_attn.o_proj.weight.detach().clone()
    if kind == "static_lora":
        assert not torch.equal(original, base)
        controller.profile()
        torch.testing.assert_close(
            model.model.layers[0].self_attn.o_proj.weight, original, rtol=0, atol=0
        )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(drafter=SimpleNamespace(draft_adapter=controller))
    )
    events = []

    class Client:
        async def collective_rpc(self, method, args=()):
            # The real vLLM API rejects callable objects during serialization.
            assert isinstance(method, str)
            json.dumps([method, args])
            events.append("cleanup")
            return [getattr(AdapterWorkerExtension, method)(worker, *args)]

    async def app(scope, receive, send):
        request_id = dict(scope["headers"])[b"x-request-id"].decode()
        response_id = (
            "chatcmpl-" if path.endswith("chat/completions") else "cmpl-"
        ) + request_id
        internal_id = (
            response_id
            + ("" if path.endswith("chat/completions") else "-0")
            + "-deadbeef"
        )
        controller.prepare(internal_id, 3, torch.arange(3), torch.randn(3, 12))
        if abort == "early":
            raise asyncio.CancelledError
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [
                    (
                        b"content-type",
                        b"text/event-stream" if stream else b"application/json",
                    )
                ],
            }
        )
        payload = json.dumps({"id": response_id, "choices": []}).encode()
        if stream:
            payload = b"data: " + payload + b"\n\n"
        await send(
            {
                "type": "http.response.body",
                "body": payload,
                "more_body": stream or abort,
            }
        )
        if abort:
            raise asyncio.CancelledError
        if stream:
            await send(
                {
                    "type": "http.response.body",
                    "body": b"data: [DONE]\n\n",
                    "more_body": True,
                }
            )
            await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def send(message):
        if message["type"] == "http.response.body" and (
            b"[DONE]" in message.get("body", b"") or not message.get("more_body", False)
        ):
            assert controller.active is None
        events.append("send")

    scope = {
        "type": "http",
        "path": path,
        "app": SimpleNamespace(state=SimpleNamespace(engine_client=Client())),
    }
    if abort:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(AdapterMiddleware(app)(scope, None, send))
    else:
        asyncio.run(AdapterMiddleware(app)(scope, None, send))
    assert controller.active is None
    torch.testing.assert_close(
        model.model.layers[0].self_attn.o_proj.weight, original, rtol=0, atol=0
    )
    assert controller.totals["completed_requests"] == 1
    if kind == "static_lora":
        assert controller.totals["merge_seconds"] == 0
        assert controller.totals["restore_seconds"] == 0
    assert events.count("cleanup") == 1


def test_http_correctness_records_reference_and_rejects_disagreement(
    frozen_validation, tmp_path, monkeypatch
):
    mismatch = False

    def request(url, body=None):
        if url.endswith("/models"):
            return {"data": [{"id": "speculators"}]}
        if url.endswith("/tokenize"):
            return {"tokens": [int(token) for token in body["prompt"].split()]}
        assert url.endswith("/completions")
        assert body["prompt"] == [1, 2]
        return {"choices": [{"token_ids": [9 if mismatch else 8]}]}

    monkeypatch.setattr(endpoint, "http_json", request)
    monkeypatch.setattr(endpoint, "server_status", lambda *_: None)
    reference = tmp_path / "reference.json"
    endpoint.check_endpoint("http://test", frozen_validation, reference, limit=1)
    result = endpoint.check_endpoint(
        "http://test",
        frozen_validation,
        tmp_path / "matched.json",
        reference=reference,
        limit=1,
    )
    assert result["status"] == "matched"
    mismatch = True
    with pytest.raises(ValueError, match="Output token disagreement"):
        endpoint.check_endpoint(
            "http://test",
            frozen_validation,
            tmp_path / "bad.json",
            reference=reference,
            limit=1,
        )
    assert json.loads((tmp_path / "bad.json").read_text())["status"] == "unverified"
