"""ASGI lifecycle cleanup and read-only diagnostics for independent HTTP clients."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from uuid import uuid4

import anyio

logger = logging.getLogger(__name__)


def worker_status(worker):
    runner = worker.model_runner
    drafter = getattr(runner, "drafter", None)
    result = {
        "runner_class": f"{type(runner).__module__}.{type(runner).__qualname__}",
        "use_v2_model_runner": getattr(worker, "use_v2_model_runner", False),
        "adapter_controller_present": getattr(drafter, "draft_adapter", None)
        is not None,
    }
    if os.environ.get("SPECULATORS_SERVER_CONFIG"):
        from speculators.bank.artifacts import file_digest  # noqa: PLC0415

        result["vllm_source_sha256"] = {
            name: file_digest(Path(module.__file__))
            for name in (
                "vllm.config.speculative",
                "vllm.v1.spec_decode.eagle",
                "vllm.v1.worker.gpu_model_runner",
            )
            if (module := sys.modules.get(name)) is not None
        }
    controller = getattr(
        getattr(worker.model_runner, "drafter", None), "draft_adapter", None
    )
    result.update(
        bundle_weights_sha256=controller.metadata["weights_sha256"]
        if controller
        else None,
        adapter_kind=controller.metadata["kind"] if controller else None,
        adapter_totals=dict(controller.totals) if controller else {},
        adapter_setup_merge_seconds=controller.setup_merge_seconds if controller else 0,
        active_request_id=controller.active if controller else None,
        active_request_ids=sorted(controller.active_requests) if controller else [],
    )
    return result


def finish_http_request(worker, response_id):
    controller = getattr(
        getattr(worker.model_runner, "drafter", None), "draft_adapter", None
    )
    if controller is None:
        return {}
    # The completions API appends '-0' for its single prompt. The engine then
    # appends its UUID suffix. Never finish a different queued HTTP request.
    for external in (response_id, response_id + "-0"):
        record = controller.finish_external(external)
        if record:
            return record
    return {}


class AdapterWorkerExtension:
    """Named GPU-worker RPC methods; no serialized Python callables required."""

    def speculators_status(self):
        return worker_status(self)

    def speculators_finish_http_request(self, response_id):
        return finish_http_request(self, response_id)


class AdapterMiddleware:
    """Restore on HTTP completion/disconnect without coupling clients to vLLM RPC."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):  # noqa: C901 -- streaming ASGI lifecycle
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        client = scope["app"].state.engine_client
        if scope["path"] == "/speculators/status":
            workers = await client.collective_rpc("speculators_status")
            payload = json.dumps(
                {
                    "configuration": json.loads(
                        os.environ["SPECULATORS_SERVER_CONFIG"]
                    ),
                    "workers": workers,
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": payload})
            return None
        prefixes = {"/v1/completions": "cmpl-", "/v1/chat/completions": "chatcmpl-"}
        if scope["path"] not in prefixes:
            return await self.app(scope, receive, send)
        # vLLM uses X-Request-Id as the completion response ID prefix. Assign a
        # fresh ID so cleanup also works before the first token and cannot target
        # a different queued request, even if clients reuse their own IDs.
        request_id = uuid4().hex
        scope = {
            **scope,
            "headers": [
                (key, value)
                for key, value in scope.get("headers", [])
                if key.lower() != b"x-request-id"
            ]
            + [(b"x-request-id", request_id.encode())],
        }
        response_id = prefixes[scope["path"]] + request_id
        streaming = False
        stream_tail = b""
        cleaned = False

        async def cleanup():
            nonlocal cleaned
            if response_id is None or cleaned:
                return
            # Starlette cancellation uses AnyIO scopes; also protect against
            # direct asyncio task cancellation while the worker restores weights.
            with anyio.CancelScope(shield=True):
                task = asyncio.create_task(
                    client.collective_rpc(
                        "speculators_finish_http_request", args=(response_id,)
                    )
                )
                try:
                    await asyncio.shield(task)
                    cleaned = True
                except asyncio.CancelledError:
                    await task
                    cleaned = True
                    raise

        async def capture(message):
            nonlocal streaming, stream_tail
            if message["type"] == "http.response.start":
                streaming = any(
                    key.lower() == b"content-type" and b"text/event-stream" in value
                    for key, value in message.get("headers", [])
                )
            if message["type"] == "http.response.body":
                chunk = stream_tail + message.get("body", b"")
                finished_stream = streaming and b"data: [DONE]" in chunk
                stream_tail = chunk[-len(b"data: [DONE]") :]
                if not message.get("more_body", False) or finished_stream:
                    # Restoration is included before the client sees completion.
                    await cleanup()
            await send(message)

        try:
            return await self.app(scope, receive, capture)
        finally:
            try:
                await cleanup()
            except Exception:
                logger.exception("HTTP adapter cleanup failed for %s", response_id)
                raise
