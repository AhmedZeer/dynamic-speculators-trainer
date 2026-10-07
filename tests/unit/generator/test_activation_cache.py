from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import speculators.generator.activation_cache as module
from speculators.bank.transfer import BankFileTransfer
from speculators.generator.activation_cache import ActivationCache
from speculators.generator.config import ExperimentConfig
from speculators.generator.engine import set_learning_rate


def make_sources(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(3):
        save_file(
            {
                "hidden_states": torch.full((20, 4, 8), float(index)),
                "token_ids": torch.arange(20),
            },
            str(source / f"hs_{index}.safetensors"),
        )
    return BankFileTransfer(source)


def test_cache_reuses_local_data_and_evicts_lru(tmp_path):
    transfer = make_sources(tmp_path)
    size = (transfer.hidden_states_path / "hs_0.safetensors").stat().st_size
    cache = ActivationCache(transfer, tmp_path / "cache", 2 * size / 2**30, 0)
    first = cache.get_cached(0)
    cache.get_cached(1)
    cache.get_cached(0)  # Touch row 0, so row 1 is evicted next.
    cache.get_cached(2)
    assert (cache.root / "hs_0.safetensors").exists()
    assert not (cache.root / "hs_1.safetensors").exists()
    assert cache.hits == 1
    assert cache.misses == 3
    assert (
        sum(p.stat().st_size for p in cache.root.glob("*.safetensors"))
        <= cache.capacity
    )
    (transfer.hidden_states_path / "hs_0.safetensors").unlink()
    torch.testing.assert_close(
        cache.get_cached(0)["hidden_states"], first["hidden_states"]
    )


def test_concurrent_cache_instances_share_atomic_copies(tmp_path):
    transfer = make_sources(tmp_path)
    caches = [ActivationCache(transfer, tmp_path / "cache", 1, 0) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(lambda cache: cache.get_cached(0), caches))
    for value in values:
        torch.testing.assert_close(value["hidden_states"], values[0]["hidden_states"])
    assert sum(cache.misses for cache in caches) == 1
    assert sum(cache.hits for cache in caches) == 3
    assert not list(caches[0].root.glob("*.pending"))
    assert not list(caches[0].root.glob("*.reservation"))


def test_cache_space_reserve_bypass_and_interrupted_copy_cleanup(tmp_path, monkeypatch):
    transfer = make_sources(tmp_path)
    cache = ActivationCache(transfer, tmp_path / "cache", 1, 1)

    original_usage = module.shutil.disk_usage
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    assert cache.get_cached(0) is not None
    assert cache.bypasses == 1
    assert not list(cache.root.glob("*.safetensors"))
    monkeypatch.setattr(module.shutil, "disk_usage", original_usage)

    def interrupted(*args, **kwargs):
        raise OSError("interrupted copy")

    monkeypatch.setattr(module.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted copy"):
        cache.get_cached(0)
    assert not list(cache.root.glob("*.pending"))
    assert not list(cache.root.glob("*.reservation"))


def test_stale_reservations_are_recovered(tmp_path):
    transfer = make_sources(tmp_path)
    root = tmp_path / "cache"
    root.mkdir()
    (root / "hs_0.reservation").write_text("1000")
    (root / "hs_0.pending").write_bytes(b"unfinished")
    cache = ActivationCache(transfer, root, 1, 0)
    assert cache.get_cached(0) is not None
    assert not list(root.glob("*.pending"))
    assert not list(root.glob("*.reservation"))


def test_learning_rate_warmup_uses_global_update_counter(tmp_path):
    cfg = ExperimentConfig(
        bank_config=tmp_path / "bank.yaml", optimization={"warmup_updates": 50}
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    for step in [1, 25, 50, 70]:
        set_learning_rate(optimizer, cfg, step)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(
            cfg.optimization.lr * min(step / 50, 1)
        )
    cfg.optimization.warmup_updates = 0
    set_learning_rate(optimizer, cfg, 1)
    assert optimizer.param_groups[0]["lr"] == cfg.optimization.lr
