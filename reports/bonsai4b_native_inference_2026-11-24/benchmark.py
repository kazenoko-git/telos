"""Compare native Telos and LM Studio on the same local Bonsai weights."""

import argparse
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bonsai4b_inference_2026-11-24"))
from benchmark import BUDGET, CONTEXT, MIB, MODEL_DIR, PROMPT, WARMUP, lm_studio


def native():
    import mlx.core as mx

    from telos.inference import TextGenerator

    assert "mlx_lm" not in sys.modules
    started = time.perf_counter()
    generator = TextGenerator.from_checkpoint(MODEL_DIR, device="mlx", context_length=CONTEXT)
    mx.eval(generator.model.parameters())
    load_seconds = time.perf_counter() - started
    ids = generator.tokenizer.encode(PROMPT, add_special_tokens=False).ids
    assert len(ids) + BUDGET <= CONTEXT
    generator.generate(WARMUP, max_new_tokens=8)
    mx.synchronize()
    mx.reset_peak_memory()
    active = mx.get_active_memory()
    original = generator._mlx_tokens
    timestamps, tokens = [], []

    def tracked(*args):
        for token in original(*args):
            if token not in generator.stop_ids:
                timestamps.append(time.perf_counter() - started)
                tokens.append(token)
            yield token

    generator._mlx_tokens = tracked
    started = time.perf_counter()
    chunks = []
    first_text = None
    for chunk in generator.stream(PROMPT, max_new_tokens=BUDGET, temperature=0, top_k=0, top_p=1):
        if first_text is None:
            first_text = time.perf_counter() - started
        chunks.append(chunk)
    elapsed = time.perf_counter() - started
    assert "mlx_lm" not in sys.modules
    return {
        "engine": "telos_native", "mlx": mx.__version__, "prompt_ids": ids,
        "context": CONTEXT, "load_seconds": load_seconds, "generated_tokens": len(tokens),
        "token_ids": tokens, "token_arrival_seconds": timestamps,
        "client_ttft_seconds": timestamps[0], "first_text_seconds": first_text, "wall_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / MIB,
        "mlx_active_before_mib": active / MIB, "mlx_peak_mib": mx.get_peak_memory() / MIB,
        "text": "".join(chunks),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["telos", "lmstudio"])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = native() if args.engine == "telos" else lm_studio()
    with args.output.open("a") as out:
        print(json.dumps(result, ensure_ascii=False), file=out)


if __name__ == "__main__":
    main()
