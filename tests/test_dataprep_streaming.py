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
