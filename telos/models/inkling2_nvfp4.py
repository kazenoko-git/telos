"""Experimental single-token NVFP4 projection with the BF16 GEMV rounding order."""

from functools import lru_cache

import mlx.core as mx

from .inkling2 import _gated_activation


_HEADER = """
constant float e2m1_magnitude[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};

inline float decode_e4m3(uint code) {
    uint exponent = (code >> 3) & 15;
    uint mantissa = code & 7;
    if ((code & 127u) == 127u) return as_type<float>(0x7fc00000u);
    float magnitude = exponent ? as_type<float>(((exponent + 120u) << 23) | (mantissa << 20))
                               : float(mantissa) * 0x1p-9f;
    return (code & 128u) ? -magnitude : magnitude;
}
"""

_BODY = """
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
        sum = fma(weight, float(x[input_start + k + i]), sum);
    }
}
// Match the lane reduction of the non-split dense MLX GEMV.
for (ushort delta = LANES / 2; delta; delta >>= 1) {
    sum += simd_shuffle_down(sum, delta);
}
if (!lane) out[output_start + row] = bfloat(sum);
"""

_GEMV = mx.fast.metal_kernel(
    name="telos_inkling_nvfp4_gemv",
    input_names=["x", "packed", "scales", "global_scale"],
    output_names=["out"],
    header=_HEADER,
    source="uint input_start = 0; uint output_start = 0;\n" + _BODY,
)


def _gemv_lanes(width, rows):
    lanes = 4 if width <= 64 else 32
    # MLX uses a different accumulation when it splits K across SIMD groups.
    if rows < 4 or rows % 4 or width % (4 * lanes) or (width > 64 and width >= 16 * rows):
        return None
    return lanes


def nvfp4_gemv(x, packed, scales, global_scale):
    """Return the packed BF16 GEMV, or None outside matching dense GEMV shapes."""
    if x.ndim != 2 or x.shape[0] != 1 or x.dtype != mx.bfloat16:
        return None
    width = x.shape[1]
    rows = packed.shape[0]
    lanes = _gemv_lanes(width, rows)
    if lanes is None:
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


@lru_cache(maxsize=16)
def _resident_gemv(experts, per_expert_input):
    # Bind the original buffers. Building a packed stack would copy every weight.
    names = ["x"]
    cases = []
    for i in range(experts):
        names.extend((f"packed{i}", f"scales{i}", f"global{i}"))
        cases.append(f"case {i}: packed = packed{i}; scales = scales{i}; global_scale = global{i}; break;")
    source = """
uint expert = thread_position_in_grid.y;
const device uint* packed = packed0;
const device uchar* scales = scales0;
float global_scale = global0;
switch (expert) {
""" + "\n".join(cases) + "\n}\n"
    source += f"uint input_start = {'expert * WIDTH' if per_expert_input else '0'};\n"
    source += "uint output_start = expert * ROWS;\n"
    return mx.fast.metal_kernel(
        name=f"telos_inkling_resident_nvfp4_{experts}_{int(per_expert_input)}",
        input_names=names, output_names=["out"], header=_HEADER, source=source + _BODY,
    )


def nvfp4_resident_gemv(x, tensors, name, *, per_expert_input=False):
    """Run a single launch for already resident NVFP4 expert buffers."""
    count = len(tensors)
    if not 2 <= count <= 8 or x.dtype != mx.bfloat16:
        return None
    width = x.shape[-1]
    rows = tensors[0][name + ".weight"].shape[0]
    lanes = _gemv_lanes(width, rows)
    if lanes is None:
        return None
    inputs = [x]
    for expert in tensors:
        inputs.extend((expert[name + ".weight"], expert[name + ".scales"], expert[name + ".global_scale"].reshape(())))
    threads = 128
    return _resident_gemv(count, per_expert_input)(
        inputs=inputs, template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((rows * lanes + threads - 1) // threads * threads, count, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(count, 1, rows)], output_dtypes=[mx.bfloat16],
    )[0]


def _resident_project(x, tensors, gate_softcap, up_softcap):
    fused = nvfp4_resident_gemv(x, tensors, "gate_up_proj")
    if fused is None:
        return None
    activated = _gated_activation(fused, gate_softcap, up_softcap)
    return nvfp4_resident_gemv(activated, tensors, "down_proj", per_expert_input=True)


@lru_cache(maxsize=16)
def _compiled_resident_project(gate_softcap, up_softcap):
    def project(x, tensors):
        return _resident_project(x, tensors, gate_softcap, up_softcap)

    return mx.compile(project)


def nvfp4_resident_project(x, tensors, gate_softcap, up_softcap, *, compiled=False):
    if compiled:
        return _compiled_resident_project(gate_softcap, up_softcap)(x, tensors)
    return _resident_project(x, tensors, gate_softcap, up_softcap)


_COMBINE = mx.fast.metal_kernel(
    name="telos_inkling_combine_resident_experts",
    input_names=["values", "weights", "slots"], output_names=["out"],
    source="""
#pragma clang fp contract(off)
uint column = thread_position_in_grid.x;
if (column >= WIDTH) return;
float sum = 0.0f;
#pragma clang loop unroll(full)
for (uint expert = 0; expert < COUNT; ++expert) {
    float product = float(values[expert * WIDTH + column]) * weights[slots[expert]];
    sum += product;
}
out[column] = sum;
""",
)


def combine_resident_experts(values, weights, slots):
    """Keep sorted expert addition and separate FP32 multiplication rounding."""
    count, _, width = values.shape
    return _COMBINE(
        inputs=[values, weights, slots], template=[("COUNT", count), ("WIDTH", width)],
        grid=((width + 63) // 64 * 64, 1, 1), threadgroup=(64, 1, 1),
        output_shapes=[(1, width)], output_dtypes=[mx.float32],
    )[0]
