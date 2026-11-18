import importlib
import json
from pathlib import Path

import numpy as np
import pytest

prep = importlib.import_module("telos.dataprep.prepare")


@pytest.mark.parametrize("directory", [False, True])
def test_plain_files_use_bounded_reads_and_preserve_unicode_tail(tmp_path, monkeypatch, directory):
    text = "hello 🌍 café " * 100 + "\n x"
    source = tmp_path / "large.txt"
    source.write_text(text, encoding="utf-8")
    original_open = open
    reads = []

    class GuardedFile:
        def __enter__(self):
            self.stream = original_open(source, encoding="utf-8")
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            assert 0 < size <= 64
            reads.append(size)
            return self.stream.read(size)

    def guarded_open(path, *args, **kwargs):
        return GuardedFile() if Path(path) == source else original_open(path, *args, **kwargs)

    monkeypatch.setattr(prep, "open", guarded_open, raising=False)
    chunks = list(prep.iterate_text_sources(tmp_path if directory else source, chunk_chars=64))
    assert "".join(chunks) == text
    assert all(len(chunk) <= 64 for chunk in chunks)
    assert len(reads) > 2


def test_license_stripping_only_applies_to_document_start(tmp_path):
    first = "# Copyright example\n" + "a = 12345\n" * 5
    later = "# Copyright should stay\n" + "b = 98765\n" * 5
    source = tmp_path / "code.py"
    source.write_text(first + later)
    chunks = list(prep.iterate_text_sources(source, chunk_chars=len(first)))
    assert "".join(chunks) == first.split("\n", 1)[1] + later


def test_jsonl_document_limit_does_not_count_chunks(tmp_path):
    source = tmp_path / "rows.jsonl"
    text = "sample content " * 20 + "x"
    source.write_text(json.dumps({"text": text}) + "\n" + json.dumps({"text": "other " * 20}))
    chunks = list(prep.iterate_text_sources(source, chunk_chars=64, limit_docs=1))
    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 64


class RecordingTokenizer:
    def __init__(self):
        self.batches = []

    def get_vocab_size(self):
        return 8192

    def encode_batch(self, texts):
        from types import SimpleNamespace
        self.batches.append(list(texts))
        return [SimpleNamespace(ids=[ord(c) for c in text]) for text in texts]


@pytest.mark.parametrize("batch_size,budget", [(1, 100), (2000, 100), (8, 32)])
@pytest.mark.parametrize("limit", [None, 0, 1, 83, 10000])
def test_tokenization_bounds_batches_and_writes_exact_prefix(tmp_path, monkeypatch, batch_size, budget, limit):
    text = "plain text without headers " * 30 + " x"
    source = tmp_path / "corpus.txt"
    source.write_text(text)
    tokenizer = RecordingTokenizer()
    loads = []

    def load(path):
        loads.append(path)
        return tokenizer

    monkeypatch.setattr(prep, "load_tokenizer", load)
    output = tmp_path / "tokens.bin"
    prep.prepare_dataset(source, output, batch_size=batch_size, chunk_chars=64,
                         max_batch_chars=budget, max_tokens=limit)
    expected = [ord(c) for c in text]
    if limit is not None:
        expected = expected[:limit]
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.uint16), expected)
    assert json.loads(Path(str(output) + ".json").read_text())["total_tokens"] == len(expected)
    assert len(loads) == 1
    for batch in tokenizer.batches:
        assert len(batch) <= batch_size
        assert sum(map(len, batch)) <= budget
        assert max(map(len, batch)) <= min(64, budget)
    if limit == 0:
        assert tokenizer.batches == []


@pytest.mark.parametrize("parameter", ["batch_size", "chunk_chars", "max_batch_chars"])
@pytest.mark.parametrize("value", [0, -1])
def test_invalid_memory_limits_do_not_truncate_output(tmp_path, parameter, value):
    output = tmp_path / "tokens.bin"
    output.write_bytes(b"keep")
    with pytest.raises(ValueError, match="must be positive"):
        prep.prepare_dataset(output_path=output, **{parameter: value})
    assert output.read_bytes() == b"keep"


def test_hf_records_are_chunked_before_tokenization(monkeypatch):
    import sys
    from types import SimpleNamespace
    text = "dataset content " * 30
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(
        load_dataset=lambda *args, **kwargs: iter([{"content": text}])))
    chunks = list(prep.iterate_text_sources(dataset_name="fake", chunk_chars=64))
    assert "".join(chunks) == text
    assert max(map(len, chunks)) <= 64


def test_real_tokenizer_output_matches_independent_chunk_encoding(tmp_path):
    from telos.data.tokenizer import load_tokenizer
    tokenizer_path = Path(prep.__file__).parents[1] / "assets" / "tokenizer_0.json"
    tokenizer = load_tokenizer(tokenizer_path)
    source = tmp_path / "real.txt"
    source.write_text("Unicode café 🌍 with whitespace\n" * 20 + "x", encoding="utf-8")
    chunks = list(prep.iterate_text_sources(source, chunk_chars=64))
    expected = [token for chunk in chunks for token in tokenizer.encode(chunk).ids]
    output = tmp_path / "real.bin"
    prep.prepare_dataset(source, output, tokenizer_path=tokenizer_path,
                         chunk_chars=64, max_batch_chars=128)
    np.testing.assert_array_equal(np.fromfile(output, dtype=np.uint16), expected)


def test_cli_forwards_memory_limits(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(prep, "prepare_dataset", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(sys, "argv", ["telos dataprep", "--chunk-chars", "64",
                                    "--max-batch-chars", "128", "--batch-size", "3"])
    prep.main()
    assert calls[0]["chunk_chars"] == 64
    assert calls[0]["max_batch_chars"] == 128
    assert calls[0]["batch_size"] == 3


def test_legacy_builder_forwards_memory_limits(monkeypatch):
    from scripts.shared import build_tokenized_corpus as builder
    calls = []
    monkeypatch.setattr(builder, "prepare_dataset", lambda **kwargs: calls.append(kwargs))
    builder.build_corpus(chunk_chars=64, max_batch_chars=128)
    assert calls[0]["batch_size"] == 8
    assert calls[0]["chunk_chars"] == 64
    assert calls[0]["max_batch_chars"] == 128
