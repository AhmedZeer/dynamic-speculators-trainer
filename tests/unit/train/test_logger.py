import logging

import pytest

from speculators.train.logger import IsRank0Filter


def _record(**extra):
    record = logging.LogRecord(
        "speculators", logging.INFO, __file__, 0, "msg", None, None
    )
    for k, v in extra.items():
        setattr(record, k, v)
    return record


@pytest.fixture
def clean_rank_env(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_global_rank0_filter_passes_only_global_rank0(monkeypatch, clean_rank_env):
    # Multi-node: a non-zero global rank that happens to be local_rank 0
    # must still be filtered out (the bug this guards against).
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert IsRank0Filter().filter(_record()) is False

    monkeypatch.setenv("RANK", "0")
    assert IsRank0Filter().filter(_record()) is True


def test_override_bypasses_filter(clean_rank_env, monkeypatch):
    monkeypatch.setenv("RANK", "3")
    assert IsRank0Filter().filter(_record(override_rank0_filter=True)) is True


def test_plain_bank_console_skips_hparams_but_keeps_training_metrics(
    monkeypatch, capsys, clean_rank_env
):
    from speculators.train import logger as module  # noqa: PLC0415

    captured = {}
    monkeypatch.setattr(
        logging, "basicConfig", lambda **kwargs: captured.update(kwargs)
    )

    def rich_forbidden(*args, **kwargs):
        pytest.fail("Plain bank console must not instantiate RichHandler")

    monkeypatch.setattr(module, "RichHandler", rich_forbidden)
    module.setup_root_logger(use_rich=False)
    assert captured["force"] is True
    handler = captured["handlers"][0]
    delivered = []

    class Backend(logging.Handler):
        def emit(self, record):
            delivered.append(record.msg.copy())

    isolated = logging.Logger("test-metrics", level=logging.INFO)
    isolated.addHandler(Backend())
    isolated.addHandler(handler)
    isolated.info({"config": "saved"}, extra={"hparams": True})
    assert delivered == [{"config": "saved"}]
    assert not capsys.readouterr().err
    isolated.info({"train": {"loss": 0.25}})
    assert "train/loss=0.25" in capsys.readouterr().err
