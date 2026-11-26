"""Experimental single-token NVFP4 projection with the BF16 GEMV rounding order."""

import mlx.core as mx


_GEMV = mx.fast.metal_kernel(
    name="telos_inkling_nvfp4_gemv",
    input_names=["x", "packed", "scales", "global_scale"],
    output_names=["out"],
    header="""
constant float e2m1_magnitude[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};

inline float decode_e4m3(uint code) {
    uint exponent = (code >> 3) & 15;
    uint mantissa = code & 7;
    if ((code & 127u) == 127u) return as_type<float>(0x7fc00000u);
    float magnitude = exponent ? as_type<float>(((exponent + 120u) << 23) | (mantissa << 20))
                               : float(mantissa) * 0x1p-9f;
    return (code & 128u) ? -magnitude : magnitude;
}
""",
    source="""
uint index = thread_position_in_grid.x;
uint row = index / LANES;
uint lane = index % LANES;
if (row >= ROWS) return;

float outer_scale = global_scale / 2688.0f;
float sum = 0.0f;
for (uint k = lane * 4; k < WIDTH; k += LANES * 4) {
    uint codes = packed[row * (WIDTH / 8) + k / 8] >> ((k & 4u) * 4);
    float scale = decode_e4m3(scales[row * (WIDTH / 16) + k / 16]) * outer_scale;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < 4; ++i) {
        uint code = (codes >> (4 * i)) & 15u;
        float value = e2m1_magnitude[code & 7];
        if (code & 8u) value = -value;
        // Round each weight exactly where MLX would store the dense BF16 matrix.
        float weight = float(bfloat(scale * value));
        sum = fma(weight, float(x[k + i]), sum);
    }
}
// Match the lane reduction of the non-split dense MLX GEMV.
for (ushort delta = LANES / 2; delta; delta >>= 1) {
    sum += simd_shuffle_down(sum, delta);
}
if (!lane) out[row] = bfloat(sum);
""",
)


def nvfp4_gemv(x, packed, scales, global_scale):
    """Return the packed BF16 GEMV, or None outside matching dense GEMV shapes."""
    if x.ndim != 2 or x.shape[0] != 1 or x.dtype != mx.bfloat16:
        return None
    width = x.shape[1]
    rows = packed.shape[0]
    lanes = 4 if width <= 64 else 32
    # MLX uses a different accumulation when it splits K across SIMD groups.
    if rows < 4 or rows % 4 or width % (4 * lanes) or (width > 64 and width >= 16 * rows):
        return None

    threads = 128
    return _GEMV(
        inputs=[x, packed, scales, global_scale.reshape(())],
        template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((rows * lanes + threads - 1) // threads * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(1, rows)],
        output_dtypes=[mx.bfloat16],
    )[0]
