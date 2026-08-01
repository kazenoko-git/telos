"""
custom BPE (byte-pair encoding) tokenizer

uses HF tokenizers with ByteLevel pre-tokenization to ensure:
- indentation and whitespace are preserved
- custom vocab size (especially considering that it is 4096 for test 1 and 8192 for final test)
- special tokens are assigned fixed IDs:
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
    vocab_size: int = 4096,
    save_path: str = "configs/tokenizer.json"
) -> Tokenizer:
    """trains a ByteLevel BPE Tokenizer on Python source code files.
    Args:
        file_paths: list of raw Python text file paths to train on.
        vocab_size: target vocabulary size (4,096 for Phase A, 8,192 for Phase B).
        save_path: output JSON destination path.
    Returns:
        tokenizer: trained Tokenizer instance.
    """
    # initialize ByteLevel BPE Model
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    # set Pre-tokenizer and Decoder to ByteLevel
    # byteLevel ensures all byte sequences (including indents & tabs) can be tokenized
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    # configure BPE Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS
    )
    # train Tokenizer on provided corpus files
    tokenizer.train(file_paths, trainer)
    # attach Post-Processor for BOS/EOS tokens if needed
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    # save Tokenizer JSON
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)
    return tokenizer
def load_tokenizer(save_path: str = "configs/tokenizer.json") -> Tokenizer:
    """loads a previously trained Tokenizer from JSON file."""
    assert Path(save_path).exists(), f"Tokenizer file not found at {save_path}"
    tokenizer = Tokenizer.from_file(save_path)
    return tokenizer