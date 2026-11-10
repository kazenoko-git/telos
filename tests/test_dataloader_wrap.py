import numpy as np
import pytest

from telos.training.dataloader import get_global_targets_contiguous


def test_batches_larger_than_dataset_repeat_without_dropping_rows():
    data = np.arange(12).reshape(3, 4)
    batch, next_row = get_global_targets_contiguous(data, 2, 11, 4)
    np.testing.assert_array_equal(batch, data[(2 + np.arange(11)) % 3])
    assert batch.shape == (11, 4)
    assert next_row == 1


def test_contiguous_batch_keeps_memmap_view():
    data = np.arange(40).reshape(10, 4)
    batch, next_row = get_global_targets_contiguous(data, 2, 3, 4)
    assert np.shares_memory(batch, data)
    assert next_row == 5


def test_empty_dataset_is_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        get_global_targets_contiguous(np.empty((0, 4)), 0, 2, 4)
