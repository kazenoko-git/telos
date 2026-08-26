
import json
from pathlib import Path
import mlx.core as mx


def save_corosred_checkpoint(model, cfg: dict, output_dir: str, step: int | None = None):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    weights_name = f"checkpoint_step_{step}.safetensors" if step is not None else "model.safetensors"
    model.save_weights(str(out_path / weights_name))

    with open(out_path / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"  [COROSred Hub] Saved checkpoint to {out_path / weights_name}")


def load_corosred_checkpoint(model, checkpoint_file: str):
    ckpt_path = Path(checkpoint_file)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    model.load_weights(str(ckpt_path), strict=False)
    print(f"  [COROSred Hub] Successfully loaded weights from {ckpt_path}")
    return model
