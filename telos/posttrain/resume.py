"""Full post-training state, with one atomic checkpoint per worker."""
import hashlib
import json
import random
from pathlib import Path

from telos.training.checkpoint import atomic_write, host_rng_state, restore_host_rng


def rank_path(path, rank):
    path = Path(path)
    return path if rank == 0 else path.with_name(f'{path.stem}.rank{rank}{path.suffix}')


def _tuples(value):
    return tuple(_tuples(x) for x in value) if isinstance(value, (list, tuple)) else value


class PosttrainResume:
    def __init__(self, path, backend, records, tokenizer, settings, rank=0, world_size=1):
        self.backend, self.rank = backend, rank
        self.identity = {'backend': backend, 'rank': rank, 'world_size': world_size,
                         'settings': settings,
                         'data': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                         'tokenizer': hashlib.sha256(tokenizer.to_str().encode()).hexdigest()}
        self.saved = None
        self.step = 0
        if path:
            path = rank_path(path, rank)
            if backend == 'mlx':
                from telos.training.checkpoint import load_mlx_state
                self.saved = load_mlx_state(path)
            else:
                import torch
                self.saved = torch.load(path, map_location='cpu', weights_only=True)
            if self.saved.get('posttrain_resume') != 1 or self.saved['identity'] != self.identity:
                raise ValueError('Post-training resume configuration, topology, tokenizer or data mismatch')
            self.step = self.saved['step']

    def restore(self, model, optimizer, reference=None, randomizer=None):
        state = self.saved
        if state is None:
            return
        if self.backend == 'mlx':
            import mlx.core as mx
            from mlx.utils import tree_flatten
            model.load_weights(tree_flatten(state['model']), strict=True)
            optimizer.state = state['optimizer']
            if state['master'] is not None:
                optimizer.master_params = state['master']
            optimizer.init(getattr(optimizer, 'master_params', model.trainable_parameters()))
            if reference is not None:
                reference.load_weights(tree_flatten(state['reference']), strict=True)
            high, low = state['rng'][0].tolist()
            mx.random.seed((int(high) << 32) | int(low))
        else:
            import torch
            getattr(model, 'module', model).load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            if reference is not None:
                getattr(reference, 'module', reference).load_state_dict(state['reference'])
            torch.set_rng_state(state['rng'])
            if self.backend == 'cuda':
                torch.cuda.set_rng_state_all(state['cuda_rng'])
            if self.backend == 'xla':
                import torch_xla.core.xla_model as xm
                xm.set_rng_state(state['xla_rng'])
        restore_host_rng(state['host_rng'])
        if randomizer is not None:
            randomizer.setstate(_tuples(state['sampler']))

    def save(self, path, model, optimizer, step, reference=None, randomizer=None):
        state = {'posttrain_resume': 1, 'identity': self.identity, 'step': step,
                 'host_rng': host_rng_state(),
                 'sampler': randomizer.getstate() if randomizer is not None else None}
        path = rank_path(path, self.rank)
        if self.backend == 'mlx':
            import mlx.core as mx
            from telos.training.checkpoint import save_mlx_state
            state.update(model=model.parameters(), optimizer=optimizer.state,
                         master=getattr(optimizer, 'master_params', None),
                         reference=reference.parameters() if reference is not None else None,
                         rng=list(mx.random.state))
            save_mlx_state(path, state)
        else:
            import torch
            state.update(model=getattr(model, 'module', model).state_dict(), optimizer=optimizer.state_dict(),
                         reference=getattr(reference, 'module', reference).state_dict() if reference is not None else None,
                         rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if self.backend == 'cuda' else None)
            if self.backend == 'xla':
                import torch_xla.core.xla_model as xm
                xm.mark_step()
                state['xla_rng'] = xm.get_rng_state()
            def cpu(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu().clone()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)):
                    return type(value)(cpu(v) for v in value)
                return value
            state = cpu(state)
            atomic_write(path, lambda tmp: torch.save(state, tmp))
