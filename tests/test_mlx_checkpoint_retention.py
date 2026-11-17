"""Checkpoint serialization must not keep obsolete training tensors alive."""
import gc

import numpy as np
import pytest

mx=pytest.importorskip('mlx.core')
if not mx.metal.is_available():pytest.skip('Metal required',allow_module_level=True)

from telos.training.checkpoint import save_mlx_state,load_mlx_state


def test_saved_tensor_is_released_without_waiting_for_cyclic_gc(tmp_path):
    mx.synchronize();gc.collect()
    baseline=mx.get_active_memory()
    enabled=gc.isenabled();gc.disable()
    try:
        state={'model':{'weights':mx.ones((1024,1024),dtype=mx.float32)},
            'step':17,'nested':[True,None,{'value':3}]}
        mx.eval(state['model'])
        target=tmp_path/'checkpoint.npz'
        save_mlx_state(target,state)
        del state
        # Cached buffers are excluded from active memory; 4 MiB of obsolete
        # tensors must be released even if the reference cycle isn't collected.
        assert mx.get_active_memory()-baseline<4096
        restored=load_mlx_state(target)
        assert restored['step']==17 and restored['nested']==[True,None,{'value':3}]
        np.testing.assert_array_equal(np.asarray(restored['model']['weights']),np.ones((1024,1024),dtype=np.float32))
    finally:
        if enabled:gc.enable()
