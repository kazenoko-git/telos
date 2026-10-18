#!/usr/bin/env python3
"""Fetch only MDLM-530M inference weights without optimizer states.

Downloads the contiguous model-weight byte range (2.39 GB) from Hugging Face,
extracts all 99 model tensors (wte, 16 blocks, ln_f, lm_head), converts to BF16,
and saves directly to safetensors. Skips the 4.78 GB AdamW optimizer history.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "scratch/matched_530m"
OUT_FILE = OUT_DIR / "mdlm_step57786.safetensors"

MDLM_REPO = "michaelchenkj/mdlm-dclm-530m-53b"
MDLM_REV = "fa87570ef97361f414e8ef073bec8c6359937f2d"
URL = f"https://huggingface.co/{MDLM_REPO}/resolve/{MDLM_REV}/checkpoints/step57786.pt"
TOTAL_ARCHIVE_SIZE = 7171946911
# First optimizer tensor (step57786/data/99) starts at byte 2,390,664,400
MODEL_BYTES_END = 2390664400


class RemoteZipReader(io.RawIOBase):
    """Seekable IO stream that fulfills read requests using HTTP Range queries."""

    def __init__(self, url: str, total_size: int):
        self.url = url
        self.size = total_size
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.pos}-{self.pos + size - 1}"},
        )
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


class ModelMetadataUnpickler(pickle.Unpickler):
    """Extract model parameter names, shapes, and storage keys without executing code."""

    def persistent_load(self, pid):
        return pid

    def find_class(self, module: str, name: str):
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, requires_grad, backward_hooks):
                return {
                    "storage_key": storage[2],
                    "numel": storage[4],
                    "size": list(size),
                    "stride": list(stride),
                }
            return rebuild
        return super().find_class(module, name)


def download_model_prefix(dest_path: Path) -> None:
    """Download bytes 0 to MODEL_BYTES_END in chunks with progress reporting."""
    if dest_path.exists() and dest_path.stat().st_size == MODEL_BYTES_END:
        print(f"Prefix binary already present ({MODEL_BYTES_END:,d} bytes).", flush=True)
        return

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")
    current_size = temp_path.stat().st_size if temp_path.exists() else 0

    req = urllib.request.Request(
        URL,
        headers={"Range": f"bytes={current_size}-{MODEL_BYTES_END - 1}"},
    )
    print(f"Downloading model slice: bytes {current_size:,d} to {MODEL_BYTES_END - 1:,d}...", flush=True)

    start_time = time.monotonic()
    downloaded = current_size
    chunk_size = 16 * 1024 * 1024  # 16 MB chunks

    with urllib.request.urlopen(req) as resp, temp_path.open("ab" if current_size > 0 else "wb") as f:
        while downloaded < MODEL_BYTES_END:
            chunk = resp.read(min(chunk_size, MODEL_BYTES_END - downloaded))
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            elapsed = max(0.001, time.monotonic() - start_time)
            rate_mb = (downloaded - current_size) / (1024 * 1024 * elapsed)
            percent = (downloaded / MODEL_BYTES_END) * 100
            print(
                f"\rProgress: {downloaded / 1e6:.1f} / {MODEL_BYTES_END / 1e6:.1f} MB ({percent:.1f}%) "
                f"at {rate_mb:.2f} MB/s",
                end="",
                flush=True,
            )

    print()
    temp_path.rename(dest_path)
    print(f"Finished downloading model prefix to {dest_path}.", flush=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        print(f"MDLM weights already exist at {OUT_FILE}")
        return

    print("Step 1: Reading remote zip central directory to extract tensor metadata...")
    rz = RemoteZipReader(URL, TOTAL_ARCHIVE_SIZE)
    zf = zipfile.ZipFile(rz)
    pkl_bytes = zf.read("step57786/data.pkl")

    meta = ModelMetadataUnpickler(io.BytesIO(pkl_bytes)).load()
    model_meta = meta["model"]
    print(f"Discovered {len(model_meta)} model tensors in checkpoint.")

    # Collect member offsets for local slicing
    storage_to_info = {}
    for k, v in model_meta.items():
        skey = v["storage_key"]
        info = zf.getinfo(f"step57786/data/{skey}")
        storage_to_info[skey] = info

    # Step 2: Download raw model slice (2.39 GB)
    prefix_bin = OUT_DIR / "mdlm_model_prefix.bin"
    download_model_prefix(prefix_bin)

    # Step 3: Extract tensors from prefix binary
    print("Step 3: Extracting tensors and converting to bfloat16 safetensors...")
    state_dict = {}
    with prefix_bin.open("rb") as f:
        for param_name, v in model_meta.items():
            skey = v["storage_key"]
            info = storage_to_info[skey]
            # In zip format: local file header is 30 bytes + name_len + extra_len
            f.seek(info.header_offset)
            magic = f.read(4)
            assert magic == b"PK\x03\x04", f"Invalid local zip magic at {info.header_offset}"
            f.seek(info.header_offset + 26)
            name_len = int.from_bytes(f.read(2), "little")
            extra_len = int.from_bytes(f.read(2), "little")
            data_offset = info.header_offset + 30 + name_len + extra_len

            f.seek(data_offset)
            raw_bytes = f.read(info.file_size)
            tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float32)
            tensor = tensor.view(v["size"]).to(torch.bfloat16)
            state_dict[param_name] = tensor

    print(f"Successfully reconstructed {len(state_dict)} tensors.")

    # Save to safetensors
    print(f"Saving BF16 safetensors to {OUT_FILE}...")
    save_file(state_dict, str(OUT_FILE))
    file_size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_FILE} ({file_size_mb:.2f} MB).")

    # Clean up large prefix binary
    if prefix_bin.exists():
        prefix_bin.unlink()
        print(f"Cleaned up temporary slice {prefix_bin}.")


if __name__ == "__main__":
    main()
