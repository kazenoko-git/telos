#!/usr/bin/env bash
# Fast local Apple Silicon (M5 Pro) uint16 shard packer for Telos Alpha 1.
# Streams Stage-1 polyglot code + math + web data, packs 1B-token (2 GB) uint16
# shards across all local CPU cores via Rust Rayon, uploads each shard to
# Hugging Face Dataset Kazenowoko/telos-alpha1-shards, and deletes local temp
# files after each confirmed commit (~2 GB peak local disk usage).
#
# Usage:
#   ./scripts/pack_shards_mac.sh           # packs 12 shards (12B tokens)
#   ./scripts/pack_shards_mac.sh 6         # packs 6 shards (6B tokens)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NUM_SHARDS="${1:-12}"
TOKENS_PER_SHARD="${TOKENS_PER_SHARD:-1000000000}"
HF_DATASET_REPO="${HF_DATASET_REPO:-Kazenowoko/telos-alpha1-shards}"
OUT_DIR="${OUT_DIR:-/tmp/alpha1_shards_mac}"
TOKENIZER="${TOKENIZER:-configs/tokenizer_alpha1_32k.json}"
WORKERS="${WORKERS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 14)}"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

if [[ -z "${HF_TOKEN:-}" && ! -f "$HOME/.cache/huggingface/token" ]]; then
  echo "Error: HF_TOKEN is not set and no cached HF login found." >&2
  echo "Run: export HF_TOKEN=hf_...   (or: .venv/bin/huggingface-cli login)" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export RAYON_NUM_THREADS="$WORKERS"

echo "========================================================================"
echo "  Telos Alpha 1 — Local Mac Shard Packer"
echo "  Cores (Rayon): $WORKERS | Shards: $NUM_SHARDS x $TOKENS_PER_SHARD tokens"
echo "  HF Dataset:    $HF_DATASET_REPO | Temp dir: $OUT_DIR"
echo "========================================================================"

# Ensure tokenizer is mirrored in the HF dataset repo first
"$PYTHON_BIN" -u -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('$HF_DATASET_REPO', repo_type='dataset', exist_ok=True)
api.upload_file(
    path_or_fileobj='$TOKENIZER',
    path_in_repo='tokenizer_alpha1_32k.json',
    repo_id='$HF_DATASET_REPO',
    repo_type='dataset',
)
print('  [HF Dataset] Uploaded tokenizer_alpha1_32k.json -> $HF_DATASET_REPO', flush=True)
"

exec "$PYTHON_BIN" -u scripts/build_alpha1_tokenizer_and_shards.py pack-shards \
  --tokenizer "$TOKENIZER" \
  --out-dir "$OUT_DIR" \
  --num-shards "$NUM_SHARDS" \
  --tokens-per-shard "$TOKENS_PER_SHARD" \
  --workers "$WORKERS" \
  --batch-docs 2048 \
  --hf-dataset-repo "$HF_DATASET_REPO" \
  --delete-after-upload
