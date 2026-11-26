"""Download pinned Hugging Face Safetensors checkpoints as local files."""

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

_INDEX = "model.safetensors.index.json"
_WEIGHTS = "model.safetensors"
_ROOT_FILES = {"README.md", "NOTICE", "LICENSE", "merges.txt", "vocab.txt"}


def _shards(repo_id, revision, files):
    if _WEIGHTS in files:
        return {_WEIGHTS}
    if _INDEX not in files:
        raise ValueError("Expected model.safetensors or model.safetensors.index.json")
    index_path = hf_hub_download(repo_id, _INDEX, revision=revision)
    index = json.loads(Path(index_path).read_text(encoding="utf-8"))
    weights = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Invalid Safetensors weight index")
    for name in weights.values():
        if (not isinstance(name, str) or not re.fullmatch(r"[\w./-]+\.safetensors", name)
                or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or name not in files):
            raise ValueError(f"Invalid or missing Safetensors shard: {name}")
    return set(weights.values())


def main():
    parser = argparse.ArgumentParser(
        prog="telos download", description="Download a Hugging Face Safetensors checkpoint without loading it.",
    )
    parser.add_argument("repo_id", help="Hugging Face model repository, such as owner/model")
    parser.add_argument("--revision", help="Branch, tag, or commit hash (default: repository default branch)")
    parser.add_argument("--cache-dir", type=Path, default=Path("~/.cache/telos/models"),
                        help="Local root for repository and revision directories")
    parser.add_argument("--metadata-only", action="store_true", help="Fetch config, tokenizer and index without weights")
    parser.add_argument("--dry-run", action="store_true", help="Show file count, size and destination without saving a checkpoint")
    args = parser.parse_args()

    try:
        info = HfApi().model_info(args.repo_id, revision=args.revision, files_metadata=True)
        repo_id, revision = info.id, info.sha
        if (not re.fullmatch(r"[\w.-]+(?:/[\w.-]+)?", repo_id or "")
                or {"..", "."}.intersection(repo_id.split("/"))
                or not re.fullmatch(r"[0-9a-f]{40}", revision or "")):
            raise ValueError("Hugging Face returned invalid repository or revision metadata")
        files = {entry.rfilename: entry.size for entry in info.siblings or ()}
        if "config.json" not in files:
            raise ValueError("Repository has no root config.json")
        shards = _shards(repo_id, revision, files)
        selected = {
            name for name in files
            if re.fullmatch(r"[\w.-]+", name)
            and (name in _ROOT_FILES or name.endswith((".json", ".jinja", ".model", ".tiktoken")))
        }
        if _WEIGHTS in shards:
            selected.discard(_INDEX)
        if not args.metadata_only:
            selected.update(shards)
        destination = args.cache_dir.expanduser().resolve() / repo_id / revision
        sizes = [files[name] for name in selected]
        size = f"{sum(sizes) / 10**9:.3f} GB" if all(size is not None for size in sizes) else "size unknown"
        print(f"{repo_id}@{revision} · {len(selected)} files · {size}", file=sys.stderr)
        if args.dry_run:
            print(destination)
            return
        path = Path(snapshot_download(repo_id, revision=revision, local_dir=destination,
                                      allow_patterns=sorted(selected)))
        for name in selected:
            file = path / name
            if not file.is_file() or (files[name] is not None and file.stat().st_size != files[name]):
                raise OSError(f"Incomplete checkpoint file: {file}")
        print(path)
    except (HfHubHTTPError, OSError, ValueError) as exc:
        parser.exit(2, f"telos download: {exc}\n")
