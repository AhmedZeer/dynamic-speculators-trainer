"""Own process groups so cleanup never targets unrelated vLLM services."""

import json
import os
import signal
import socket
import subprocess
import time
from contextlib import contextmanager, suppress
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from speculators.generator.serving.endpoint import validate_status


def ensure_port_free(port):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            pass
    except OSError:
        return
    raise RuntimeError(
        f"Port {port} is occupied; stop that service or choose another port"
    )


@contextmanager
def owned_process(command, log, *, cwd, env, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        process = subprocess.Popen(  # noqa: S603 -- explicit argument lists, no shell
            command,
            cwd=cwd,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            yield process
        finally:
            # Even if the parent exited, its workers may still own the process group.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=timeout)
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@contextmanager
def termination_cleanup():
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(*_):
        raise KeyboardInterrupt("Suite terminated")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def wait_ready(process, target, model, manifest, timeout):
    deadline = time.monotonic() + timeout
    last = "server not responding"
    expected = {
        "lora": "static_lora",
        "generator": "generator",
        "drafter": "eagle3_base",
    }
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {process.returncode}")
        try:
            root = target.removesuffix("/v1")
            with urlopen(root + "/health", timeout=2):  # noqa: S310 -- localhost only
                pass
            with urlopen(root + "/speculators/status", timeout=2) as response:  # noqa: S310
                status = json.load(response)
            validate_status(status, manifest)
            config = status["configuration"]
            if config["kind"] != expected[model["type"]]:
                raise ValueError("Connected server has the wrong model type")
            if model["type"] == "drafter":
                if config["models"]["drafter"] != {
                    "id": model["path"],
                    "revision": model["revision"],
                }:
                    raise ValueError("Connected drafter revision differs")
            else:
                if config["bundle_weights_sha256"] != model["weights_sha256"]:
                    raise ValueError("Connected adapter weights differ")
                if config["models"] != model["models"]:
                    raise ValueError("Connected bundle model revisions differ")
            return
        except (HTTPError, URLError, OSError) as exc:
            last = str(exc)
        time.sleep(0.5)
    raise TimeoutError(f"vLLM readiness timed out: {last}")
