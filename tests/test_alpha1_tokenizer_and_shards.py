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
