"""
Custom BPE (byte-pair encoding) tokenizer for Télos.

Uses HF tokenizers with ByteLevel pre-tokenization to ensure:
- Indentation and whitespace are preserved.
- Custom vocab size (e.g. 4096 or 8192).
- Special tokens are assigned fixed IDs:
    [PAD] -> 0
    [MASK] -> 1
    [BOS] -> 2
    [EOS] -> 3
    [UNK] -> 4
"""

from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders, processors

SPECIAL_TOKENS = ["[PAD]", "[MASK]", "[BOS]", "[EOS]", "[UNK]"]
PAD_TOKEN_ID = 0
MASK_TOKEN_ID = 1
BOS_TOKEN_ID = 2
EOS_TOKEN_ID = 3
UNK_TOKEN_ID = 4


def train_bpe_tokenizer(
    file_paths: list[str],
    vocab_size: int = 8192,
    save_path: str = "configs/shared/tokenizer_0.json"
) -> Tokenizer:
    """Trains a ByteLevel BPE Tokenizer on Python source code files."""
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS
    )
    tokenizer.train(file_paths, trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer


def load_tokenizer(save_path: str = "configs/shared/tokenizer_0.json") -> Tokenizer:
    """Loads a previously trained Tokenizer from JSON file."""
    path = Path(save_path)
    if not path.exists():
        # Fallback search paths
        for alt in ["configs/shared/tokenizer_mac.json", "configs/tokenizer_0.json", "configs/tokenizer_mac.json"]:
            if Path(alt).exists():
                path = Path(alt)
                break
    assert path.exists(), f"Tokenizer file not found at {save_path}"
    return Tokenizer.from_file(str(path))
