"""Keep instruction-model end-of-turn markers when generating through Télos."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
from telos.eval.adapters.huggingface import HuggingFaceAdapter


@pytest.mark.parametrize(('configured', 'expected'), [([1, 106, 50], [1, 106, 50]), (None, 1)])
def test_generate_preserves_model_stop_tokens(configured, expected):
    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def __call__(self, prompt, **kwargs):
            return Inputs(input_ids=torch.tensor([[2, 7]]))

        def decode(self, ids, **kwargs):
            return 'Answer: C'

    seen = {}

    def generate(**kwargs):
        seen.update(kwargs)
        return torch.tensor([[2, 7, 8, 106]])

    adapter = HuggingFaceAdapter.__new__(HuggingFaceAdapter)
    adapter.tokenizer = Tokenizer()
    adapter.model = SimpleNamespace(device=torch.device('cpu'), generate=generate,
                                    generation_config=SimpleNamespace(eos_token_id=configured))
    assert adapter.generate('question') == 'Answer: C'
    assert seen['eos_token_id'] == expected
