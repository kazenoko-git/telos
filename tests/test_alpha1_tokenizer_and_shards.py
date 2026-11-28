"""Unit tests for the Telos Alpha 1 32k polyglot+reasoning tokenizer and uint16 shard packer."""

from pathlib import Path
import numpy as np

from scripts.build_alpha1_tokenizer_and_shards import (
    AUDIT_SNIPPETS,
    EOS_TOKEN_ID,
    SPECIAL_TOKENS,
    pack_shards,
    train_tokenizer,
)


def test_train_tokenizer_special_ids_single_digits_and_pack_shards(tmp_path: Path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Write a synthetic polyglot + math corpus covering all target languages
    for i in range(20):
        (corpus_dir / f"sample_{i}.txt").write_text(
            "\n".join(AUDIT_SNIPPETS.values()) + f"\nvalue = {i * 137 + 42}\n",
            encoding="utf-8",
        )

    tok_path = tmp_path / "tokenizer_alpha1_test.json"
    tok = train_tokenizer(
        tok_path,
        vocab_size=1024,
        min_frequency=1,
        target_mb=1.0,
        local_corpus=[str(corpus_dir)],
    )
    assert tok_path.exists()
    for expected_id, special in enumerate(SPECIAL_TOKENS):
        assert tok.token_to_id(special) == expected_id

    # Verify single-digit splitting for math arithmetic
    enc = tok.encode("5050")
    assert len(enc.ids) == 4

    # Verify parallel uint16 shard packing + sidecar metadata
    shard_dir = tmp_path / "shards"
    written = pack_shards(
        tok_path,
        shard_dir,
        num_shards=2,
        tokens_per_shard=512,
        workers=2,
        batch_docs=4,
        local_corpus=[str(corpus_dir)],
    )
    assert len(written) == 2
    for sp in written:
        arr = np.fromfile(sp, dtype=np.uint16)
        assert arr.size == 512
        assert EOS_TOKEN_ID in arr
        assert Path(str(sp) + ".json").exists()


def test_pack_shards_reuses_existing_hf_shards(tmp_path: Path, monkeypatch):
    import sys
    import types

    # Create a tiny valid tokenizer
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "sample.txt").write_text("\n".join(AUDIT_SNIPPETS.values()), encoding="utf-8")
    tok_path = tmp_path / "tok.json"
    train_tokenizer(tok_path, vocab_size=512, min_frequency=1, target_mb=0.1, local_corpus=[str(corpus_dir)])

    # Create fake remote HF dataset files
    remote_dir = tmp_path / "fake_hf" / "shards"
    remote_dir.mkdir(parents=True)
    for idx in range(3):
        np.arange(64, dtype=np.uint16).tofile(remote_dir / f"shard_{idx:04d}.bin")
        (remote_dir / f"shard_{idx:04d}.bin.json").write_text('{"dtype":"uint16"}', encoding="utf-8")

    class _FakeHfApi:
        def list_repo_files(self, repo_id: str, repo_type: str = "dataset"):
            return [
                f"shards/shard_{i:04d}.bin" for i in range(3)
            ] + [
                f"shards/shard_{i:04d}.bin.json" for i in range(3)
            ]

    def _fake_download(repo_id: str, filename: str, repo_type: str = "dataset", local_dir: str = "."):
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((tmp_path / "fake_hf" / filename).read_bytes())
        return str(dest)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.HfApi = _FakeHfApi
    fake_hub.hf_hub_download = _fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    out_dir = tmp_path / "reused_shards"
    # Note: local_corpus=None would fail offline if it tried to stream HF; succeeding proves reuse worked!
    reused = pack_shards(
        tok_path,
        out_dir,
        num_shards=2,
        tokens_per_shard=64,
        hf_dataset_repo="Kazenowoko/telos-alpha1-shards",
        delete_after_upload=False,
    )
    assert [p.name for p in reused] == ["shard_0000.bin", "shard_0001.bin"]
    for p in reused:
        assert p.parent == out_dir
        assert np.fromfile(p, dtype=np.uint16).size == 64
        assert (out_dir / f"{p.name}.json").exists()
