import pytest
from telos.posttrain.trainer import _cuda_device_ids


def test_cuda_device_selection():
    assert _cuda_device_ids('auto', 4) == [0, 1, 2, 3]
    assert _cuda_device_ids('2', 4) == [0, 1]
    for count, available in [('auto', 0), ('8', 4), ('0', 4)]:
        with pytest.raises(ValueError):
            _cuda_device_ids(count, available)
