"""Exercise the real GuideLLM client against a CPU-only mock HTTP endpoint."""

# ruff: noqa: F811 -- imported shared pytest fixture

import importlib.util
import json
import os
import sys
import threading
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from speculators.bank.artifacts import digest, file_digest, write_json
from speculators.generator.serving.endpoint import check_endpoint
from speculators.generator.serving.guidellm import benchmark_endpoint, native_summary
from tests.unit.scripts.test_perf_utils import perf_utils  # noqa: F401


@pytest.fixture
def workload(tmp_path):
    tokenizer = Tokenizer(
        WordLevel(
            {"<unk>": 0, "one": 1, "two": 2, "three": 3, "four": 4}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    processor = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>")
    tokenizer_path = tmp_path / "tokenizer"
    processor.save_pretrained(tokenizer_path)
    rows = [
        {
            "row_id": f"math-00000/{i}",
            "subset": "math-00000",
            "category": "math",
            "prompt": "one two",
            "prompt_token_ids": [1, 2],
            "prompt_tokens_count": 2,
            "output_tokens_count": 4,
        }
        for i in range(2)
    ]
    data = tmp_path / "math.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    identity = {
        "version": 1,
        "models": {
            "target": {"id": "target", "revision": "a" * 40},
            "drafter": {"id": "draft", "revision": "b" * 40},
        },
        "subsets": ["math-00000"],
        "sources": [],
        "max_tokens": 4,
        "groups": {"math": {"file": data.name, "sha256": file_digest(data), "rows": 2}},
    }
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {**identity, "identity": digest(identity)})
    return manifest, tokenizer_path


@pytest.fixture
def http_server(request):  # noqa: C901 -- small mock protocol implementation
    counts = SimpleNamespace(
        completions=0, streaming=0, output_tokens=getattr(request, "param", 4)
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, body, status=200, content_type="application/json"):
            payload = json.dumps(body).encode() if isinstance(body, dict) else body
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/v1/models":
                self.reply({"data": [{"id": "speculators", "object": "model"}]})
            elif self.path == "/health":
                self.reply({"status": "ok"})
            elif self.path == "/metrics":
                n = counts.completions
                self.reply(
                    (
                        f"vllm:spec_decode_num_drafts_total {n}\n"
                        f"vllm:spec_decode_num_draft_tokens_total {3 * n}\n"
                        f"vllm:spec_decode_num_accepted_tokens_total {2 * n}\n"
                    ).encode(),
                    content_type="text/plain",
                )
            else:
                self.reply({"error": "not found"}, status=404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                assert body["prompt"] == "one two"
                assert body["add_special_tokens"] is False
                self.reply({"tokens": [1, 2]})
                return
            assert self.path == "/v1/completions"
            assert body["prompt"] in ("one two", [1, 2])
            assert body["temperature"] == 0
            assert body["add_special_tokens"] is False
            assert body["ignore_eos"] is False
            assert body["max_tokens"] == 4
            counts.completions += 1
            response_id = f"cmpl-{counts.completions}"
            if not body.get("stream"):
                self.reply(
                    {
                        "id": response_id,
                        "model": "speculators",
                        "choices": [
                            {
                                "index": 0,
                                "text": "three four three four"
                                if counts.output_tokens == 4
                                else "three four",
                                "finish_reason": "length"
                                if counts.output_tokens == 4
                                else "stop",
                                "token_ids": [3, 4, 3, 4][: counts.output_tokens],
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 2,
                            "completion_tokens": counts.output_tokens,
                            "total_tokens": 2 + counts.output_tokens,
                        },
                    }
                )
                return
            counts.streaming += 1
            chunks = []
            for text in ("three", " four", " three", " four")[: counts.output_tokens]:
                chunks.append(
                    {
                        "id": response_id,
                        "model": "speculators",
                        "object": "text_completion",
                        "choices": [{"index": 0, "text": text, "finish_reason": None}],
                    }
                )
            chunks.append(
                {
                    "id": response_id,
                    "model": "speculators",
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": counts.output_tokens,
                        "total_tokens": 2 + counts.output_tokens,
                    },
                }
            )
            payload = b"".join(
                b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks
            )
            self.reply(payload + b"data: [DONE]\n\n", content_type="text/event-stream")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", counts
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("http_server", [4, 2], indirect=True)
@pytest.mark.parametrize("allow_mismatch", [False, True])
def test_real_guidellm_against_independent_endpoint(  # noqa: PLR0917
    workload, http_server, tmp_path, perf_utils, monkeypatch, allow_mismatch
):
    if importlib.util.find_spec("guidellm") is None:
        pytest.skip(
            "Install scripts/evaluate/requirements.txt for GuideLLM integration"
        )
    manifest, tokenizer_path = workload
    target, counts = http_server
    reference = tmp_path / "check/reference.json"
    check_endpoint(target, manifest, reference)
    args = Namespace(
        target=target,
        validation_manifest=manifest,
        correctness_report=reference,
        subsets="all",
        max_requests=0,
        max_concurrency=1,
        mode="throughput",
        gen_kwargs="",
        output_dir=str(tmp_path / "performance"),
        warmups=1,
        seed=42,
        sweep_rate=2,
        server_provenance=None,
        tokenizer_path=tokenizer_path,
        allow_output_mismatch=False,
    )
    if allow_mismatch:
        failed = json.loads(reference.read_text())
        failed.update(status="unverified", mismatches=[next(iter(failed["rows"]))])
        write_json(reference, failed)
        with pytest.raises(ValueError, match="unverified"):
            benchmark_endpoint(args, perf_utils)
        args.allow_output_mismatch = True
    monkeypatch.setenv(
        "PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    )
    actual_run = perf_utils.subprocess.run

    def run(*args, **kwargs):
        kwargs.setdefault("timeout", 60)
        return actual_run(*args, **kwargs)

    monkeypatch.setattr(perf_utils.subprocess, "run", run)
    result = benchmark_endpoint(args, perf_utils)
    assert result["math"]["performance"][0]["successful_requests"] == 2
    assert result["math"]["acceptance"]["num_drafts"] == 2
    assert (
        result["math"]["performance"][0]["total_output_tokens"]
        == 2 * counts.output_tokens
    )
    assert counts.streaming == 2
    assert counts.completions == 5  # two correctness, one warmup, two measured.
    assert (tmp_path / "performance/perf_results.csv").is_file()
    assert (tmp_path / "performance/math/guidellm_command.json").is_file()
    identity = json.loads(
        (tmp_path / "performance/benchmark_identity.json").read_text()
    )
    assert identity["output_mismatch_override"] == allow_mismatch
    assert result["math"]["output_mismatch_override"] == allow_mismatch


def test_incomplete_guidellm_results_are_rejected(tmp_path):
    path = tmp_path / "results.json"
    write_json(
        path,
        {
            "benchmarks": [
                {
                    "metrics": {
                        "request_totals": {
                            "successful": 1,
                            "errored": 1,
                            "incomplete": 0,
                        }
                    }
                }
            ]
        },
    )
    with pytest.raises(ValueError, match="Incomplete or failed"):
        native_summary(path, 2)
