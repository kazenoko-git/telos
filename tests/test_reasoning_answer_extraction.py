"""Regression cases where prose beginning with A-E is not an answer letter."""
import pytest

from telos.eval.reasoning_eval import extract_arc_choice


@pytest.mark.parametrize(('text', 'expected'), [
    ('The answer could be D.', 'D'),
    ('Answer: Because the day becomes shorter, C is correct.', 'C'),
    ('The correct option is definitely B.', 'B'),
    ('Answer: Aardvark', None),
    ('Answer: C', 'C'),
    ('**Answer:** (B)', 'B'),
    (r'The result is \boxed{D}', 'D'),
    ('Answer: 3', '3'),
])
def test_arc_answer_requires_a_complete_label(text, expected):
    assert extract_arc_choice(text) == expected
