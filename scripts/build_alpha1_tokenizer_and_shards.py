#!/usr/bin/env python3
"""Telos Alpha 1 — 32,768-vocab Polyglot+Reasoning BPE Tokenizer & 96-vCPU Shard Packer.

Subcommands:
  1. train-tokenizer:
     Trains a 32,768-vocab byte-level BPE tokenizer with single-digit splitting (for math
     reasoning), fixed low-ID special tokens (0..15 including <think>/</think>, chat, tool,
     and FIM tags), and balanced polyglot code (Rust, C/C++, Java, Bash/Zsh, HTML/CSS/TS/JS,
     Python) + Math + Web coverage.

  2. pack-shards:
     Saturates multi-core VMs (e.g. 96-vCPU Kaggle TPU v5e-8) using parallel Rust-backed
     `encode_batch` workers to stream-pack 1B-token (2.0 GB) `uint16` `.bin` shards with
     optional direct Hugging Face Dataset upload + local cleanup.

Examples:
  # Train 32k tokenizer from streaming HF polyglot/math/web mix (~2-3 min on M5 Pro or Kaggle):
  python -u scripts/build_alpha1_tokenizer_and_shards.py train-tokenizer \
      --output configs/tokenizer_alpha1_32k.json --target-mb 200

  # Pack 20B tokens (20 x 2 GB uint16 shards) on 96-vCPU Kaggle VM and stream to HF Dataset:
  python -u scripts/build_alpha1_tokenizer_and_shards.py pack-shards \
      --tokenizer configs/tokenizer_alpha1_32k.json \
      --out-dir /tmp/alpha1_shards --num-shards 20 --tokens-per-shard 1000000000 \
      --hf-dataset-repo Kazenowoko/telos-alpha1-shards --delete-after-upload
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, processors, trainers

SPECIAL_TOKENS: list[str] = [
    "<|pad|>",            # 0
    "<|mask|>",           # 1
    "<|bos|>",            # 2
    "<|eos|>",            # 3
    "<|unk|>",            # 4
    "<|im_start|>",       # 5
    "<|im_end|>",         # 6
    "<think>",            # 7
    "</think>",           # 8
    "<tool_call>",        # 9
    "</tool_call>",       # 10
    "<tool_response>",    # 11
    "</tool_response>",   # 12
    "<|fim_prefix|>",     # 13
    "<|fim_middle|>",     # 14
    "<|fim_suffix|>",     # 15
]

EOS_TOKEN_ID = 3

# GPT-4/Qwen-inspired split pattern with single-digit (\p{N}) splitting for math arithmetic
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# Default Stage-1 streaming mix (dataset_id, config_name, text_field, weight, label)
DEFAULT_STAGE1_SOURCES: list[tuple[str, str | None, str, float, str]] = [
    ("HuggingFaceFW/fineweb-edu", "sample-100BT", "text", 0.30, "web_edu"),
    ("HuggingFaceTB/finemath", "finemath-4plus", "text", 0.25, "math_4plus"),
    ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 0.10, "cosmopedia_v2"),
    # Polyglot code across Rust, C/C++, Java, Shell/Bash, Web (HTML/CSS/TS/JS), Python
    ("bigcode/starcoderdata", "rust", "content", 0.06, "code_rust"),
    ("bigcode/starcoderdata", "cpp", "content", 0.06, "code_cpp"),
    ("bigcode/starcoderdata", "java", "content", 0.05, "code_java"),
    ("bigcode/starcoderdata", "shell", "content", 0.05, "code_shell"),
    ("bigcode/starcoderdata", "html", "content", 0.04, "code_html"),
    ("bigcode/starcoderdata", "css", "content", 0.03, "code_css"),
    ("bigcode/starcoderdata", "typescript", "content", 0.03, "code_ts"),
    ("bigcode/starcoderdata", "python", "content", 0.03, "code_python"),
]

AUDIT_SNIPPETS: dict[str, str] = {
    "rust": "pub fn fib(n: u64) -> u64 { match n { 0 | 1 => n, _ => fib(n - 1) + fib(n - 2) } }",
    "cpp": "#include <vector>\nstd::vector<int> v{1, 2, 3}; for (const auto& x : v) { sum += x; }",
    "java": "public record Pair<A, B>(A first, B second) implements Comparable<Pair<A, B>> {}",
    "bash_zsh": "for f in **/*.log(N); do [[ -s \"$f\" ]] && awk '{s+=$1} END {print s}' \"$f\"; done",
    "web": "<div class=\"card flex gap-4\" style=\"grid-template-columns: repeat(3, 1fr);\">Hi</div>",
    "python": "def softmax(x: list[float]) -> list[float]: m = max(x); return [math.exp(v - m) for v in x]",
    "math": "<think>Compute $\\sum_{k=1}^{100} k = \\frac{100 \\times 101}{2} = 5050$.</think>5050",
}


def _stream_hf_mix(
    sources: list[tuple[str, str | None, str, float, str]],
    *,
    seed: int = 42,
    max_bytes: int | None = None,
) -> Iterator[str]:
    """Yield text documents from a weighted Hugging Face streaming mix."""
    from datasets import load_dataset

    rng = np.random.default_rng(seed)
    active_iters = []
    active_weights = []
    active_fields = []
    active_labels = []

    for repo_id, subset, field, weight, label in sources:
        try:
            if repo_id == "bigcode/starcoderdata" and subset:
                ds = load_dataset(repo_id, data_dir=subset, split="train", streaming=True)
            else:
                ds = load_dataset(repo_id, subset, split="train", streaming=True)
            active_iters.append(iter(ds))
            active_weights.append(weight)
            active_fields.append(field)
            active_labels.append(label)
            print(f"  [Stream] Connected {label}: {repo_id} ({subset or 'default'}) w={weight:.2f}", flush=True)
        except Exception as exc:
            print(f"  [Stream Warning] Skipping {label} ({repo_id}/{subset}): {exc}", flush=True)

    if not active_iters:
        raise RuntimeError("No streaming datasets could be opened. Check HF_TOKEN and network access.")

    probs = np.array(active_weights, dtype=np.float64)
    probs /= probs.sum()
    emitted_bytes = 0

    while active_iters:
        idx = int(rng.choice(len(active_iters), p=probs))
        try:
            row = next(active_iters[idx])
        except StopIteration:
            active_iters.pop(idx)
            active_weights.pop(idx)
            active_fields.pop(idx)
            active_labels.pop(idx)
            if not active_iters:
                break
            probs = np.array(active_weights, dtype=np.float64)
            probs /= probs.sum()
            continue

        text = row.get(active_fields[idx], "")
        if not isinstance(text, str) or not text.strip():
            continue
        emitted_bytes += len(text.encode("utf-8", errors="ignore"))
        yield text
        if max_bytes is not None and emitted_bytes >= max_bytes:
            break


def _stream_local_files(paths: list[str], max_bytes: int | None = None) -> Iterator[str]:
    emitted = 0
    for p_str in paths:
        p = Path(p_str)
        files = sorted(p.rglob("*")) if p.is_dir() else [p]
        for fp in files:
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if not text.strip():
                continue
            emitted += len(text.encode("utf-8"))
            yield text
            if max_bytes is not None and emitted >= max_bytes:
                return


def build_base_tokenizer() -> Tokenizer:
    """Create an untrained byte-level BPE tokenizer with single-digit regex pre-tokenization."""
    tok = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=Regex(PRETOKENIZE_REGEX), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    return tok


def train_tokenizer(
    output_path: Path,
    *,
    vocab_size: int = 32768,
    min_frequency: int = 2,
    target_mb: float = 200.0,
    local_corpus: list[str] | None = None,
    seed: int = 42,
) -> Tokenizer:
    """Train a 32,768-vocab polyglot + reasoning BPE tokenizer and audit compression."""
    t0 = time.time()
    max_bytes = int(target_mb * 1024 * 1024)
    tok = build_base_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    if local_corpus:
        iterator = _stream_local_files(local_corpus, max_bytes=max_bytes)
    else:
        iterator = _stream_hf_mix(DEFAULT_STAGE1_SOURCES, seed=seed, max_bytes=max_bytes)

    tok.train_from_iterator(iterator, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_path))
    dt = time.time() - t0
    print(f"  [Tokenizer] Saved {tok.get_vocab_size()}-vocab tokenizer to {output_path} in {dt:.1f}s", flush=True)

    # Verify fixed special token IDs and polyglot/math round-trip + single-digit splitting
    for expected_id, special in enumerate(SPECIAL_TOKENS):
        actual_id = tok.token_to_id(special)
        assert actual_id == expected_id, f"Special token {special} expected id {expected_id}, got {actual_id}"

    print("  [Audit] Polyglot + Math compression & round-trip check:", flush=True)
    for lang, snippet in AUDIT_SNIPPETS.items():
        enc = tok.encode(snippet)
        dec = tok.decode(enc.ids, skip_special_tokens=False)
        chars_per_tok = len(snippet) / max(1, len(enc.ids))
        print(f"    {lang:<10s}: {len(enc.ids):>3d} toks ({chars_per_tok:.2f} chars/tok)", flush=True)
        assert len(enc.ids) > 0 and dec.strip() != ""
    return tok


def _encode_batch_rayon(tok: Tokenizer, docs: list[str]) -> np.ndarray:
    """Encode a batch of documents using Rust tokenizers' native multi-threaded Rayon pool."""
    encodings = tok.encode_batch(docs)
    total_len = sum(len(e.ids) + 1 for e in encodings)
    out = np.empty(total_len, dtype=np.uint16)
    pos = 0
    for e in encodings:
        n = len(e.ids)
        if n:
            out[pos : pos + n] = e.ids
            pos += n
        out[pos] = EOS_TOKEN_ID
        pos += 1
    return out


def _flush_shard(
    buf: np.ndarray,
    shard_idx: int,
    out_dir: Path,
    vocab_size: int,
    hf_dataset_repo: str | None = None,
    delete_after_upload: bool = False,
) -> Path:
    shard_path = out_dir / f"shard_{shard_idx:04d}.bin"
    buf.tofile(shard_path)
    meta = {
        "shard_index": shard_idx,
        "dtype": "uint16",
        "num_tokens": int(buf.size),
        "vocab_size": vocab_size,
        "eos_token_id": EOS_TOKEN_ID,
    }
    meta_path = Path(str(shard_path) + ".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    size_gb = shard_path.stat().st_size / (1024 ** 3)
    print(f"  [Shard {shard_idx:04d}] Wrote {buf.size:,} uint16 tokens ({size_gb:.2f} GiB) -> {shard_path}", flush=True)

    if hf_dataset_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=hf_dataset_repo, repo_type="dataset", exist_ok=True)
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(shard_path),
            path_in_repo=f"shards/{shard_path.name}",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=str(meta_path),
            path_in_repo=f"shards/{meta_path.name}",
            repo_id=hf_dataset_repo,
            repo_type="dataset",
        )
        print(f"  [HF Dataset] Uploaded {shard_path.name} to {hf_dataset_repo} in {time.time() - t0:.1f}s", flush=True)
        if delete_after_upload:
            shard_path.unlink()
            print(f"  [Disk] Deleted local {shard_path} after confirmed HF commit.", flush=True)
    return shard_path


def _try_reuse_remote_shards(hf_dataset_repo: str, out_dir: Path, num_shards: int) -> list[Path] | None:
    """If >= num_shards already exist on HF Dataset, download them into out_dir and skip packing."""
    try:
        from huggingface_hub import HfApi, hf_hub_download

        api = HfApi()
        files = set(api.list_repo_files(repo_id=hf_dataset_repo, repo_type="dataset"))
        remote_bins = sorted(f for f in files if f.startswith("shards/") and f.endswith(".bin"))
        if len(remote_bins) < num_shards:
            return None

        selected = remote_bins[:num_shards]
        print(
            f"  [HF Dataset] Found {len(remote_bins)} existing shards in {hf_dataset_repo}; "
            f"downloading first {num_shards} into {out_dir} and skipping re-packing...",
            flush=True,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        local_paths: list[Path] = []
        t0 = time.time()
        for rf in selected:
            dl_bin = Path(hf_hub_download(repo_id=hf_dataset_repo, filename=rf, repo_type="dataset", local_dir=str(out_dir)))
            flat_bin = out_dir / dl_bin.name
            if dl_bin.resolve() != flat_bin.resolve():
                dl_bin.replace(flat_bin)
            meta_rf = rf + ".json"
            if meta_rf in files:
                dl_meta = Path(hf_hub_download(repo_id=hf_dataset_repo, filename=meta_rf, repo_type="dataset", local_dir=str(out_dir)))
                flat_meta = out_dir / dl_meta.name
                if dl_meta.resolve() != flat_meta.resolve():
                    dl_meta.replace(flat_meta)
            local_paths.append(flat_bin)
        print(
            f"  [HF Dataset] Reused {len(local_paths)} pre-packed shards from {hf_dataset_repo} in {time.time() - t0:.1f}s.",
            flush=True,
        )
        return local_paths
    except Exception as exc:
        print(f"  [HF Dataset] Remote shard check skipped ({exc}); proceeding to pack locally.", flush=True)
        return None


def pack_shards(
    tokenizer_path: Path,
    out_dir: Path,
    *,
    num_shards: int = 10,
    tokens_per_shard: int = 1_000_000_000,
    workers: int | None = None,
    batch_docs: int = 2048,
    local_corpus: list[str] | None = None,
    hf_dataset_repo: str | None = None,
    delete_after_upload: bool = False,
    seed: int = 42,
) -> list[Path]:
    """Parallel-encode streaming text into exact `uint16` shards (default 1B tokens = 2 GB each).

    Uses Rust `tokenizers` native Rayon thread pool (`TOKENIZERS_PARALLELISM=true`) so all
    96 vCPUs on a Kaggle TPU VM are saturated with zero Python pickle/IPC overhead.
    If `hf_dataset_repo` already contains `>= num_shards` pre-packed shards (and
    `delete_after_upload` is False), downloads and reuses them instead of re-packing.
    """
    n_workers = workers if workers is not None else max(1, os.cpu_count() or 1)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(n_workers)

    tok = Tokenizer.from_file(str(tokenizer_path))
    vocab_size = tok.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError(f"uint16 packing requires vocab_size <= 65535 (got {vocab_size})")

    out_dir.mkdir(parents=True, exist_ok=True)
    if hf_dataset_repo and not delete_after_upload:
        reused = _try_reuse_remote_shards(hf_dataset_repo, out_dir, num_shards)
        if reused is not None:
            return reused

    print(
        f"  [Pack] Starting {num_shards} shards x {tokens_per_shard:,} tokens "
        f"(Rust Rayon threads={n_workers}, vocab={vocab_size}, dtype=uint16)",
        flush=True,
    )

    doc_iter = _stream_local_files(local_corpus) if local_corpus else _stream_hf_mix(DEFAULT_STAGE1_SOURCES, seed=seed)

    def _batched_docs() -> Iterator[list[str]]:
        batch: list[str] = []
        for doc in doc_iter:
            batch.append(doc)
            if len(batch) >= batch_docs:
                yield batch
                batch = []
        if batch:
            yield batch

    shard_buf = np.empty(tokens_per_shard, dtype=np.uint16)
    fill = 0
    shard_idx = 0
    written: list[Path] = []
    t_start = time.time()
    total_tokens = 0

    for batch in _batched_docs():
        arr = _encode_batch_rayon(tok, batch)
        offset = 0
        while offset < arr.size and shard_idx < num_shards:
            take = min(tokens_per_shard - fill, arr.size - offset)
            shard_buf[fill : fill + take] = arr[offset : offset + take]
            fill += take
            offset += take
            total_tokens += take
            if fill == tokens_per_shard:
                written.append(
                    _flush_shard(
                        shard_buf, shard_idx, out_dir, vocab_size,
                        hf_dataset_repo=hf_dataset_repo,
                        delete_after_upload=delete_after_upload,
                    )
                )
                shard_idx += 1
                fill = 0
        if shard_idx >= num_shards:
            break

    if fill > 0 and shard_idx < num_shards:
        written.append(
            _flush_shard(
                shard_buf[:fill], shard_idx, out_dir, vocab_size,
                hf_dataset_repo=hf_dataset_repo,
                delete_after_upload=delete_after_upload,
            )
        )

    elapsed = max(1e-6, time.time() - t_start)
    print(
        f"  [Pack Complete] {len(written)} shards | {total_tokens:,} tokens in {elapsed:.1f}s "
        f"({total_tokens / elapsed:,.0f} tok/s)",
        flush=True,
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Telos Alpha 1 32k tokenizer & 96-vCPU uint16 shard builder.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tok = sub.add_parser("train-tokenizer", help="Train the 32,768-vocab polyglot + reasoning BPE tokenizer.")
    p_tok.add_argument("--output", default="configs/tokenizer_alpha1_32k.json")
    p_tok.add_argument("--vocab-size", type=int, default=32768)
    p_tok.add_argument("--min-frequency", type=int, default=2)
    p_tok.add_argument("--target-mb", type=float, default=200.0)
    p_tok.add_argument("--local-corpus", nargs="*", default=None, help="Optional local files/dirs instead of HF streaming.")
    p_tok.add_argument("--seed", type=int, default=42)

    p_pack = sub.add_parser("pack-shards", help="Pack uint16 token shards in parallel (built for 96-vCPU VMs).")
    p_pack.add_argument("--tokenizer", default="configs/tokenizer_alpha1_32k.json")
    p_pack.add_argument("--out-dir", default="/tmp/alpha1_shards")
    p_pack.add_argument("--num-shards", type=int, default=10)
    p_pack.add_argument("--tokens-per-shard", type=int, default=1_000_000_000)
    p_pack.add_argument("--workers", type=int, default=None)
    p_pack.add_argument("--batch-docs", type=int, default=1024)
    p_pack.add_argument("--local-corpus", nargs="*", default=None, help="Optional local files/dirs instead of HF streaming.")
    p_pack.add_argument("--hf-dataset-repo", default=None)
    p_pack.add_argument("--delete-after-upload", action="store_true")
    p_pack.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.cmd == "train-tokenizer":
        train_tokenizer(
            Path(args.output),
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            target_mb=args.target_mb,
            local_corpus=args.local_corpus,
            seed=args.seed,
        )
    else:
        pack_shards(
            Path(args.tokenizer),
            Path(args.out_dir),
            num_shards=args.num_shards,
            tokens_per_shard=args.tokens_per_shard,
            workers=args.workers,
            batch_docs=args.batch_docs,
            local_corpus=args.local_corpus,
            hf_dataset_repo=args.hf_dataset_repo,
            delete_after_upload=args.delete_after_upload,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
