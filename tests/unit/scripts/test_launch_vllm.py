import os
from argparse import Namespace

from scripts import launch_vllm
from scripts.launch_vllm import (
    DEFAULT_RENDERER_NUM_WORKERS,
    _preprocessing_workers,
    _set_render_thread_defaults,
    _with_render_defaults,
    render_throughput_defaults,
)
from speculators.data_generation.preprocessing import default_preprocessing_workers


def test_render_defaults_are_added():
    args = _with_render_defaults(["--port", "8000"])
    api_servers, renderer_workers = render_throughput_defaults()
    assert args[-6:] == [
        "--api-server-count",
        str(api_servers),
        "--renderer-num-workers",
        str(renderer_workers),
        "--port",
        "8000",
    ]


def test_explicit_flag_follows_default():
    args = _with_render_defaults(["--api-server-count", "1"])
    assert args[-2:] == ["--api-server-count", "1"]


def test_headless_does_not_get_api_server_defaults():
    args = _with_render_defaults(["--headless"])
    assert "--enable-scale-out" not in args
    assert "--api-server-count" not in args
    assert "--renderer-num-workers" not in args


def test_render_thread_defaults_are_bounded_and_overrideable(monkeypatch):
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    _set_render_thread_defaults()

    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert os.environ["RAYON_NUM_THREADS"] == "2"

    monkeypatch.setenv("RAYON_NUM_THREADS", "8")
    _set_render_thread_defaults()
    assert os.environ["RAYON_NUM_THREADS"] == "8"


def test_sizing_respects_the_combined_budget():
    assert default_preprocessing_workers(384) == 72
    assert _preprocessing_workers(384) == 72
    assert render_throughput_defaults(384) == (18, DEFAULT_RENDERER_NUM_WORKERS)


def test_sizing_scales_down_on_small_hosts():
    assert default_preprocessing_workers(16) == 3
    assert _preprocessing_workers(16) == 3
    assert render_throughput_defaults(16) == (1, DEFAULT_RENDERER_NUM_WORKERS)


def test_sizing_uses_one_combined_budget():
    preprocessing_workers = default_preprocessing_workers(160)
    api_servers, _ = render_throughput_defaults(160)

    assert preprocessing_workers == 30
    assert api_servers == 7
    assert preprocessing_workers * 3 + api_servers * 4 <= int(160 * 0.75)


def test_launch_inherits_output_and_preserves_statistics_logging(
    monkeypatch, tmp_path, capsys
):
    args = Namespace(
        subcommand="train",
        provenance_dir=str(tmp_path),
        model="target",
        no_hash_checkpoints=True,
        dry_run=False,
    )
    command = ["python", "-m", "vllm.entrypoints.cli.main", "serve", "target"]
    calls = []
    monkeypatch.setattr(launch_vllm, "parse_args", lambda: (args, []))
    monkeypatch.setattr(launch_vllm, "_build_train_cmd", lambda *_: command)
    monkeypatch.setattr(launch_vllm, "_save_vllm_provenance", lambda *a, **kw: None)
    monkeypatch.setattr(launch_vllm, "_set_render_thread_defaults", lambda: None)
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "INFO")
    stdout, stderr = os.fstat(1), os.fstat(2)

    def execvp(executable, argv):
        assert os.fstat(1) == stdout
        assert os.fstat(2) == stderr
        assert os.environ["VLLM_LOGGING_LEVEL"] == "INFO"
        calls.append((executable, argv))

    monkeypatch.setattr(launch_vllm.os, "execvp", execvp)
    launch_vllm.main()
    assert calls == [(command[0], command)]
    assert "--disable-log-stats" not in calls[0][1]
    assert "inherited stdout/stderr" in capsys.readouterr().out
