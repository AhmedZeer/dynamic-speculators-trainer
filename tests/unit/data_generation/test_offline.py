import pytest
import torch

from speculators.data_generation.offline import (
    check_hidden_states,
    get_existing_hidden_state_indices,
    get_indices_to_process,
)


def test_check_hidden_states_reports_nan_layer_slots():
    hidden_states = torch.zeros(3, 4, 2, dtype=torch.bfloat16)
    hidden_states[1, 2, 0] = torch.nan

    with pytest.raises(
        ValueError,
        match=r"shape=\(3, 4, 2\).*affected layer slots=\[2\]",
    ):
        check_hidden_states(
            {"token_ids": torch.tensor([1, 2, 3]), "hidden_states": hidden_states},
            [1, 2, 3],
        )


# ===== get_indices_to_process Tests =====


class TestGetIndicesToProcess:
    def test_single_node_no_max_samples(self):
        result = get_indices_to_process(10, None, [], world_size=1, rank=0)
        assert result == list(range(10))

    def test_single_node_with_max_samples(self):
        result = get_indices_to_process(10, 5, [], world_size=1, rank=0)
        assert result == [0, 1, 2, 3, 4]

    def test_single_node_max_samples_exceeds_num_samples(self):
        result = get_indices_to_process(5, 10, [], world_size=1, rank=0)
        assert result == list(range(5))

    def test_single_node_with_existing(self):
        result = get_indices_to_process(10, None, [2, 5, 7], world_size=1, rank=0)
        assert result == [0, 1, 3, 4, 6, 8, 9]

    def test_all_samples_already_processed(self):
        result = get_indices_to_process(5, None, list(range(5)), world_size=1, rank=0)
        assert result == []

    def test_multi_node_even_split(self):
        r0 = get_indices_to_process(10, None, [], world_size=2, rank=0)
        r1 = get_indices_to_process(10, None, [], world_size=2, rank=1)
        assert r0 == [0, 1, 2, 3, 4]
        assert r1 == [5, 6, 7, 8, 9]

    def test_multi_node_uneven_split(self):
        r0 = get_indices_to_process(10, None, [], world_size=3, rank=0)
        r1 = get_indices_to_process(10, None, [], world_size=3, rank=1)
        r2 = get_indices_to_process(10, None, [], world_size=3, rank=2)
        assert r0 == [0, 1, 2, 3]
        assert r1 == [4, 5, 6]
        assert r2 == [7, 8, 9]

    def test_multi_node_no_overlap_and_full_coverage(self):
        num_samples = 17
        world_size = 4
        all_indices = []
        for rank in range(world_size):
            chunk = get_indices_to_process(
                num_samples, None, [], world_size=world_size, rank=rank
            )
            all_indices.extend(chunk)
        assert sorted(all_indices) == list(range(num_samples))
        assert len(all_indices) == len(set(all_indices))

    def test_multi_node_with_max_samples(self):
        r0 = get_indices_to_process(100, 10, [], world_size=2, rank=0)
        r1 = get_indices_to_process(100, 10, [], world_size=2, rank=1)
        assert r0 == [0, 1, 2, 3, 4]
        assert r1 == [5, 6, 7, 8, 9]

    def test_multi_node_with_existing(self):
        result = get_indices_to_process(10, None, [1, 3], world_size=2, rank=0)
        assert result == [0, 2, 4]

    def test_multi_node_rank_fully_processed(self):
        result = get_indices_to_process(10, None, [0, 1, 2, 3, 4], world_size=2, rank=0)
        assert result == []

    def test_existing_exceeds_num_samples(self):
        result = get_indices_to_process(5, None, list(range(10)), world_size=1, rank=0)
        assert result == []


# ===== get_existing_hidden_state_indices Tests =====


class TestGetExistingHiddenStateIndices:
    def test_nonexistent_directory(self, tmp_path):
        result = get_existing_hidden_state_indices(tmp_path / "nonexistent")
        assert result == []

    def test_empty_directory(self, tmp_path):
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == []

    def test_finds_safetensor_files(self, tmp_path):
        (tmp_path / "hs_0.safetensors").touch()
        (tmp_path / "hs_3.safetensors").touch()
        (tmp_path / "hs_7.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0, 3, 7]

    def test_ignores_non_numeric_suffixes(self, tmp_path):
        (tmp_path / "hs_0.safetensors").touch()
        (tmp_path / "hs_abc.safetensors").touch()
        (tmp_path / "hs_.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0]

    def test_ignores_unrelated_files(self, tmp_path):
        (tmp_path / "hs_0.safetensors").touch()
        (tmp_path / "other_file.txt").touch()
        (tmp_path / "hs_1.pt").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0]

    def test_results_are_sorted(self, tmp_path):
        for i in [9, 2, 5, 0]:
            (tmp_path / f"hs_{i}.safetensors").touch()
        result = get_existing_hidden_state_indices(tmp_path)
        assert result == [0, 2, 5, 9]


@pytest.mark.parametrize("finite", [True, False])
def test_file_validation_reads_bounded_slices_and_detects_late_nan(  # noqa: C901
    tmp_path, monkeypatch, finite
):
    from safetensors.torch import save_file  # noqa: PLC0415

    from speculators.data_generation import offline  # noqa: PLC0415

    path = tmp_path / "states.safetensors"
    tokens = list(range(600))
    hidden = torch.zeros(600, 4, 3)
    if not finite:
        hidden[567, 2, 1] = torch.nan
    save_file({"token_ids": torch.tensor(tokens), "hidden_states": hidden}, str(path))
    opener = offline.safe_open
    slices = []

    class Slice:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def get_shape(self):
            return self.wrapped.get_shape()

        def get_dtype(self):
            return self.wrapped.get_dtype()

        def __getitem__(self, key):
            assert key.stop - key.start <= 256
            slices.append(key.start)
            return self.wrapped[key]

    class Reader:
        def __init__(self, *args, **kwargs):
            self.wrapped = opener(*args, **kwargs)

        def __enter__(self):
            self.wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self.wrapped.__exit__(*args)

        def keys(self):
            return self.wrapped.keys()

        def get_tensor(self, key):
            assert key == "token_ids", "Full hidden tensor must not be materialized"
            return self.wrapped.get_tensor(key)

        def get_slice(self, key):
            return Slice(self.wrapped.get_slice(key))

    monkeypatch.setattr(offline, "safe_open", Reader)
    if finite:
        offline.check_hidden_state_file(path, tokens, (600, 4, 3), "float32")
    else:
        with pytest.raises(ValueError, match="non-finite"):
            offline.check_hidden_state_file(path, tokens, (600, 4, 3), "float32")
    assert slices == [0, 256, 512]
