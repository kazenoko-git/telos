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


def load_tokenizer(save_path: str | Path | None = None) -> Tokenizer:
    """Loads a previously trained Tokenizer from JSON file or bundled package assets."""
    target = None
    if save_path is not None:
        p = Path(save_path)
        if p.exists():
            target = p

    if target is None:
        # 1. Check bundled asset in telos.assets
        try:
            import importlib.resources as pkg_resources
            asset_ref = pkg_resources.files("telos.assets").joinpath("tokenizer_0.json")
            if asset_ref.is_file():
                target = Path(str(asset_ref))
        except Exception:
            pass

    if target is None:
        # 2. Fallback search paths in workspace repository
        for alt in [
            save_path,
            "configs/shared/tokenizer_0.json",
            "configs/shared/tokenizer_mac.json",
            "configs/tokenizer_0.json",
            "configs/tokenizer_mac.json",
        ]:
            if alt and Path(alt).exists():
                target = Path(alt)
                break

    if target is None or not target.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found. Checked requested path '{save_path}', bundled package assets "
            "('telos.assets/tokenizer_0.json'), and standard config locations."
        )

    return Tokenizer.from_file(str(target))
