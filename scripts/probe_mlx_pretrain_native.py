"""Bounded native MLX-Pretrain workflow smoke using decoded local corpus text.

This checks the actual upstream trainer, tokenization, validation and checkpoint
path. It is intentionally separate from the matched backbone speed comparison.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    token_dir = scratch / "tokenizer"
    token_dir.mkdir()
    tokenizer_file = root / "telos/assets/tokenizer_0.json"
    shutil.copy2(tokenizer_file, token_dir / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    raw = np.memmap(args.data, mode="r", dtype=np.uint16)
    prefix = np.array(raw[:66 * 256])
    for name, rows in [("train", prefix[:64 * 256].reshape(64, 256)),
                       ("validation", prefix[64 * 256:].reshape(2, 256))]:
        with (scratch / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps({"text": tokenizer.decode(row.tolist())}) + "\n")
    config = {
        "name": "native-smoke", "overwrite": False,
        "data": {"input_file": str(scratch / "train.jsonl"),
                 "validation_file": str(scratch / "validation.jsonl"),
                 "tokenizer_path": str(token_dir),
                 "preprocessing": {"max_context_size": 256, "chunk_overlap": 0},
                 "tokenizer": {"normal_vocab_size": 8192,
                               "special_tokens": {"pad": "[PAD]", "bos": "[BOS]", "eos": "[EOS]"}}},
        "model": {"architecture": "llama",
                  "dimensions": {"hidden_size": 512, "intermediate_size": 1408, "num_layers": 14},
                  "attention": {"num_heads": 8, "num_kv_heads": 8, "head_dim": 64,
                                "max_position_embeddings": 256},
                  "normalization": {"rms_norm_eps": 1e-5},
                  "rope": {"theta": 10000, "traditional": True, "scaling": None},
                  "misc": {"attention_bias": False, "mlp_bias": False, "tie_word_embeddings": True}},
        "training": {"hyperparameters": {"batch_size": 8, "learning_rate": 4e-4,
                                            "weight_decay": 0.01, "iters": 8},
                     "scheduler": {"type": "cosine", "min_lr_ratio": 0.1},
                     "optimization": {"optimizer": "adamw", "weight_decay": 0.01,
                                      "betas": [0.9, 0.999], "eps": 1e-8}},
        "logging": {"log_dir": "logs", "checkpoint_dir": "checkpoints",
                    "steps": {"logging_interval": 1, "checkpoint_interval": 1000,
                              "validation_interval": 0},
                    "metrics": {"log_loss": True, "log_perplexity": True,
                                "log_tokens_per_second": True, "log_learning_rate": True,
                                "log_tokens_processed": True}},
        "system": {"seed": 42, "device": "gpu"}}
    config_file = args.output / "config.yaml"
    config_file.write_text(yaml.safe_dump(config, sort_keys=False))
    metadata = {"upstream_commit": subprocess.check_output(
        ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True).strip(),
        "upstream_train_sha256": hashlib.sha256((args.upstream / "train.py").read_bytes()).hexdigest(),
        "raw_corpus_prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "limitations": ["Native FP32 workflow smoke, not matched throughput",
                        "Decoded corpus is retokenized, padded and wrapped in BOS/EOS by upstream",
                        "No initialization equivalence or convergence claim"]}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    spec = importlib.util.spec_from_file_location("upstream_mlx_pretrain", args.upstream / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mx.set_memory_limit(int(3.2 * 1024**3))
    os.chdir(scratch)
    started = time.perf_counter()
    trainer = module.Trainer(str(config_file))
    initial = trainer.validate()
    trainer.train()
    mx.eval(trainer.model.parameters(), trainer.optimizer.state)
    final = trainer.validate()
    assert math.isfinite(initial) and math.isfinite(final)
    checkpoint_files = list(trainer.checkpoint_dir.glob("*"))
    assert any(f.name.endswith("_model.safetensors") for f in checkpoint_files)
    result = {"status": "complete", "native_updates": trainer.total_steps,
              "initial_validation_loss": initial, "final_validation_loss": final,
              "end_to_end_seconds": time.perf_counter() - started,
              "peak_mlx_bytes": mx.get_peak_memory(),
              "checkpoint_files": [{"name": f.name, "bytes": f.stat().st_size} for f in checkpoint_files]}
    shutil.copy2(trainer.log_file, args.output / "upstream_training.log")
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
