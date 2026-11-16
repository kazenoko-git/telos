"""Small correctness gate for the residual/RMSNorm pre-training probe.

Run separately from timed validation jobs. This is not a speed benchmark.
"""
import json
import mlx.core as mx
from residual_norm import residual_norm


def reference(x, branch, weight):
    residual = x + branch
    return residual, mx.fast.rms_norm(residual, weight, 1e-5)


def main():
    if not mx.metal.is_available():
        raise RuntimeError("Metal is required to test the actual prototype")
    results = []
    for dtype in (mx.float32, mx.bfloat16):
        for shape in ((2, 7, 64), (2, 17, 512), (1, 9, 768)):
            mx.random.seed(129)
            primals = (mx.random.normal(shape).astype(dtype),
                       mx.random.normal(shape).astype(dtype),
                       mx.random.uniform(low=.5, high=1.5, shape=(shape[-1],)).astype(dtype))
            cotangents = tuple(mx.random.normal(shape).astype(dtype) for _ in range(2))
            expected, expected_gradients = mx.vjp(reference, primals, cotangents)
            actual, gradients = mx.vjp(residual_norm, primals, cotangents)
            mx.eval(expected, expected_gradients, actual, gradients)
            differences = []
            for index, (a, b) in enumerate(zip((*expected, *expected_gradients), (*actual, *gradients))):
                a, b = a.astype(mx.float32), b.astype(mx.float32)
                maximum = mx.max(mx.abs(a - b)).item()
                close = mx.allclose(a, b, rtol=2e-5 if dtype == mx.float32 else .009,
                                    atol=2e-6 if dtype == mx.float32 else .008).item()
                if not close:
                    raise AssertionError((str(dtype), shape, index, maximum))
                differences.append(maximum)
            results.append(dict(dtype=str(dtype), shape=shape, max_abs=differences))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
