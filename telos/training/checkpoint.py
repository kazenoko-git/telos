"""Atomic checkpoint files and reproducibility metadata shared by trainers."""

import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np

FORMAT_VERSION = 1


def atomic_write(path, writer):
    """Publish a complete file, leaving the previous checkpoint intact on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    try:
        writer(temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def host_rng_state():
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
    }


def restore_host_rng(state):
    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (tuple, list)) else value
    random.setstate(tuples(state["python"]))
    name, keys, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, gaussian, cached))


def dataset_signature(matrix):
    """Hash all tokens in bounded chunks, including their dtype and shape."""
    digest = hashlib.sha256()
    for start in range(0, len(matrix), 4096):
        digest.update(np.ascontiguousarray(matrix[start:start + 4096]).tobytes())
    return {"sha256": digest.hexdigest(), "shape": list(matrix.shape), "dtype": str(matrix.dtype)}


def tokenizer_signature(cfg):
    from importlib.resources import files
    path = cfg.get("data", {}).get("tokenizer")
    if path is None:
        return hashlib.sha256(files("telos.assets").joinpath("tokenizer_0.json").read_bytes()).hexdigest()
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Resume metadata requires a local tokenizer file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_resume(saved, cfg, backend, dataset=None):
    if saved.get("resume_format") != FORMAT_VERSION:
        raise ValueError("Checkpoint has no complete training state; use --init-checkpoint for a weights-only restart")
    if saved["backend"] != backend:
        raise ValueError("Reliable resume requires the original training backend/device")
    for key in ("model", "training", "corosred", "crsr", "phase", "paradigm"):
        if saved["config"].get(key) != cfg.get(key):
            raise ValueError(f"Resume configuration mismatch: {key}")
    step = saved["global_step"]
    if not isinstance(step, int) or not 0 <= step <= int(cfg["training"]["max_steps"]):
        raise ValueError("Invalid checkpoint step")
    if saved["tokenizer"] != tokenizer_signature(cfg):
        raise ValueError("Tokenizer differs from the saved training run")
    if dataset is not None and saved["dataset"] != dataset:
        raise ValueError("Training data differs from the saved training run")


def save_mlx_state(path, state):
    """Store a nested state tree and JSON structure together in one atomic NPZ."""
    import mlx.core as mx
    arrays = {}
    def encode(value):
        if isinstance(value, mx.array):
            key = f"array_{len(arrays)}"
            arrays[key] = value
            return {"array": key}
        if isinstance(value, dict):
            return {"dict": [[key, encode(item)] for key, item in value.items()]}
        if isinstance(value, (list, tuple)):
            return {"list": [encode(item) for item in value]}
        return {"value": value}
    structure = encode(state)
    arrays["structure"] = mx.array(list(json.dumps(structure).encode()), dtype=mx.uint8)
    mx.eval(*arrays.values())
    atomic_write(path, lambda temporary: mx.savez(temporary, **arrays))


def load_mlx_state(path):
    import mlx.core as mx
    arrays = mx.load(str(path))
    structure = json.loads(bytes(arrays.pop("structure").tolist()).decode())
    def decode(node):
        if "array" in node:
            return arrays[node["array"]]
        if "dict" in node:
            return {key: decode(item) for key, item in node["dict"]}
        if "list" in node:
            return [decode(item) for item in node["list"]]
        return node["value"]
    return decode(structure)


def read_resume(path):
    path = Path(path)
    if path.suffix == ".npz":
        state = load_mlx_state(path)
    elif path.suffix == ".pt":
        import torch
        state = torch.load(path, map_location="cpu", weights_only=True)
    else:
        raise ValueError("Resume requires a full .pt or .resume.npz checkpoint, not exported weights")
    if state.get("resume_format") != FORMAT_VERSION:
        raise ValueError("Checkpoint has no complete training state; use --init-checkpoint for a weights-only restart")
    return state
