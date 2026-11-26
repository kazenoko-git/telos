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

inline float dot4_nvfp4(uint codes, float scale, float x0, float x1, float x2, float x3, float sum) {
    uint c0 = codes & 15u;
    float v0 = e2m1_magnitude[c0 & 7u];
    if (c0 & 8u) v0 = -v0;
    sum = fma(float(bfloat(scale * v0)), x0, sum);

    uint c1 = (codes >> 4u) & 15u;
    float v1 = e2m1_magnitude[c1 & 7u];
    if (c1 & 8u) v1 = -v1;
    sum = fma(float(bfloat(scale * v1)), x1, sum);

    uint c2 = (codes >> 8u) & 15u;
    float v2 = e2m1_magnitude[c2 & 7u];
    if (c2 & 8u) v2 = -v2;
    sum = fma(float(bfloat(scale * v2)), x2, sum);

    uint c3 = (codes >> 12u) & 15u;
    float v3 = e2m1_magnitude[c3 & 7u];
    if (c3 & 8u) v3 = -v3;
    return fma(float(bfloat(scale * v3)), x3, sum);
}

inline float dot8_e2m1(uint word, bfloat4 lo, bfloat4 hi) {
    float acc = 0.0f;
    uint c0 = word & 15u; float v0 = e2m1_magnitude[c0 & 7u]; acc = fma((c0 & 8u) ? -v0 : v0, float(lo.x), acc);
    uint c1 = (word >> 4u) & 15u; float v1 = e2m1_magnitude[c1 & 7u]; acc = fma((c1 & 8u) ? -v1 : v1, float(lo.y), acc);
    uint c2 = (word >> 8u) & 15u; float v2 = e2m1_magnitude[c2 & 7u]; acc = fma((c2 & 8u) ? -v2 : v2, float(lo.z), acc);
    uint c3 = (word >> 12u) & 15u; float v3 = e2m1_magnitude[c3 & 7u]; acc = fma((c3 & 8u) ? -v3 : v3, float(lo.w), acc);
    uint c4 = (word >> 16u) & 15u; float v4 = e2m1_magnitude[c4 & 7u]; acc = fma((c4 & 8u) ? -v4 : v4, float(hi.x), acc);
    uint c5 = (word >> 20u) & 15u; float v5 = e2m1_magnitude[c5 & 7u]; acc = fma((c5 & 8u) ? -v5 : v5, float(hi.y), acc);
    uint c6 = (word >> 24u) & 15u; float v6 = e2m1_magnitude[c6 & 7u]; acc = fma((c6 & 8u) ? -v6 : v6, float(hi.z), acc);
    uint c7 = (word >> 28u) & 15u; float v7 = e2m1_magnitude[c7 & 7u]; acc = fma((c7 & 8u) ? -v7 : v7, float(hi.w), acc);
    return acc;
}
"""

_BODY_QUAD_FAST = """
uint index = thread_position_in_grid.x;
uint quad = index / LANES;
uint lane = index % LANES;
uint row0 = quad * 4u;
if (row0 >= ROWS) return;

const uint groups = WIDTH / 16u;
const device bfloat4* x4 = (const device bfloat4*)(x + input_start);
const device uint2* p0 = (const device uint2*)(packed + row0 * (WIDTH / 8u));
const device uint2* p1 = (const device uint2*)(packed + (row0 + 1u) * (WIDTH / 8u));
const device uint2* p2 = (const device uint2*)(packed + (row0 + 2u) * (WIDTH / 8u));
const device uint2* p3 = (const device uint2*)(packed + (row0 + 3u) * (WIDTH / 8u));
const device uchar* s0 = scales + row0 * groups;
const device uchar* s1 = scales + (row0 + 1u) * groups;
const device uchar* s2 = scales + (row0 + 2u) * groups;
const device uchar* s3 = scales + (row0 + 3u) * groups;

float sum0 = 0.0f;
float sum1 = 0.0f;
float sum2 = 0.0f;
float sum3 = 0.0f;
for (uint g = lane; g < groups; g += LANES) {
    const device bfloat4* xg = x4 + g * 4u;
    bfloat4 xa = xg[0];
    bfloat4 xb = xg[1];
    bfloat4 xc = xg[2];
    bfloat4 xd = xg[3];

    uint2 w0 = p0[g];
    sum0 = fma(decode_e4m3(s0[g]), dot8_e2m1(w0.x, xa, xb) + dot8_e2m1(w0.y, xc, xd), sum0);
    uint2 w1 = p1[g];
    sum1 = fma(decode_e4m3(s1[g]), dot8_e2m1(w1.x, xa, xb) + dot8_e2m1(w1.y, xc, xd), sum1);
    uint2 w2 = p2[g];
    sum2 = fma(decode_e4m3(s2[g]), dot8_e2m1(w2.x, xa, xb) + dot8_e2m1(w2.y, xc, xd), sum2);
    uint2 w3 = p3[g];
    sum3 = fma(decode_e4m3(s3[g]), dot8_e2m1(w3.x, xa, xb) + dot8_e2m1(w3.y, xc, xd), sum3);
}
for (ushort delta = LANES / 2u; delta; delta >>= 1u) {
    sum0 += simd_shuffle_down(sum0, delta);
    sum1 += simd_shuffle_down(sum1, delta);
    sum2 += simd_shuffle_down(sum2, delta);
    sum3 += simd_shuffle_down(sum3, delta);
}
float outer_scale = global_scale / 2688.0f;
sum0 *= outer_scale;
sum1 *= outer_scale;
sum2 *= outer_scale;
sum3 *= outer_scale;
"""

_STORE_QUAD = """
if (!lane) {
    out[output_start + row0] = bfloat(sum0);
    out[output_start + row0 + 1u] = bfloat(sum1);
    out[output_start + row0 + 2u] = bfloat(sum2);
    out[output_start + row0 + 3u] = bfloat(sum3);
}
"""

_STORE_QUAD_SWIGLU = """
if (!lane) {
    float g0 = sum0;
    float u0 = sum1;
    float act0 = (gate_cap * tanh(g0 / gate_cap) * (1.0f / (1.0f + exp(-g0)))) * (up_cap * tanh(u0 / up_cap));
    out[output_start + quad * 2u] = bfloat(act0);

    float g1 = sum2;
    float u1 = sum3;
    float act1 = (gate_cap * tanh(g1 / gate_cap) * (1.0f / (1.0f + exp(-g1)))) * (up_cap * tanh(u1 / up_cap));
    out[output_start + quad * 2u + 1u] = bfloat(act1);
}
"""

_BODY_PAIR = """
uint index = thread_position_in_grid.x;
uint pair = index / LANES;
uint lane = index % LANES;
uint row0 = pair * 2u;
if (row0 >= ROWS) return;
uint row1 = row0 + 1u;

float outer_scale = global_scale / 2688.0f;
const device bfloat4* x4 = (const device bfloat4*)(x + input_start);
const device uint* p0 = packed + row0 * (WIDTH / 8u);
const device uint* p1 = packed + row1 * (WIDTH / 8u);
const device uchar* s0 = scales + row0 * (WIDTH / 16u);
const device uchar* s1 = scales + row1 * (WIDTH / 16u);

float sum0 = 0.0f;
float sum1 = 0.0f;
for (uint k = lane * 4u; k < WIDTH; k += LANES * 4u) {
    bfloat4 xv = x4[k / 4u];
    float x0 = float(xv.x);
    float x1 = float(xv.y);
    float x2 = float(xv.z);
    float x3 = float(xv.w);
    uint shift = (k & 4u) * 4u;
    uint widx = k / 8u;
    uint sidx = k / 16u;
    sum0 = dot4_nvfp4(p0[widx] >> shift, decode_e4m3(s0[sidx]) * outer_scale, x0, x1, x2, x3, sum0);
    sum1 = dot4_nvfp4(p1[widx] >> shift, decode_e4m3(s1[sidx]) * outer_scale, x0, x1, x2, x3, sum1);
}
// Match the lane reduction of the non-split dense MLX GEMV.
for (ushort delta = LANES / 2u; delta; delta >>= 1u) {
    sum0 += simd_shuffle_down(sum0, delta);
    sum1 += simd_shuffle_down(sum1, delta);
}
"""

_STORE_PAIR = """
if (!lane) {
    out[output_start + row0] = bfloat(sum0);
    out[output_start + row1] = bfloat(sum1);
}
"""

_STORE_SWIGLU = """
if (!lane) {
    float g = float(bfloat(sum0));
    float u = float(bfloat(sum1));
    float sig = 1.0f / (1.0f + exp(-g));
    float act = (gate_cap * tanh(g / gate_cap) * sig) * (up_cap * tanh(u / up_cap));
    out[output_start + pair] = bfloat(act);
}
"""

_GEMV = mx.fast.metal_kernel(
    name="telos_inkling_nvfp4_gemv",
    input_names=["x", "packed", "scales", "global_scale"],
    output_names=["out"],
    header=_HEADER,
    source="uint input_start = 0; uint output_start = 0;\n" + _BODY_PAIR + _STORE_PAIR,
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
    pairs = rows // 2
    return _GEMV(
        inputs=[x, packed, scales, global_scale.reshape(())],
        template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((pairs * lanes + threads - 1) // threads * threads, 1, 1),
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
        input_names=names, output_names=["out"], header=_HEADER, source=source + _BODY_PAIR + _STORE_PAIR,
    )


@lru_cache(maxsize=16)
def _resident_gate_up_swiglu(experts):
    # Fuse interleaved (gate, up) rows with softcapped SwiGLU in registers.
    names = ["x", "gate_cap", "up_cap"]
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
    source += "uint input_start = 0;\n"
    source += "uint output_start = expert * (ROWS / 2u);\n"
    return mx.fast.metal_kernel(
        name=f"telos_inkling_resident_nvfp4_swiglu_{experts}",
        input_names=names, output_names=["out"], header=_HEADER, source=source + _BODY_PAIR + _STORE_SWIGLU,
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
    pairs = rows // 2
    return _resident_gemv(count, per_expert_input)(
        inputs=inputs, template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((pairs * lanes + threads - 1) // threads * threads, count, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(count, 1, rows)], output_dtypes=[mx.bfloat16],
    )[0]


def nvfp4_resident_gate_up_swiglu(x, tensors, gate_softcap, up_softcap):
    """Run fused gate_up NVFP4 GEMV + softcapped SwiGLU across resident experts."""
    count = len(tensors)
    if not 2 <= count <= 8 or x.dtype != mx.bfloat16:
        return None
    width = x.shape[-1]
    rows = tensors[0]["gate_up_proj.weight"].shape[0]
    lanes = _gemv_lanes(width, rows)
    if lanes is None:
        return None
    inputs = [x, mx.array(gate_softcap, dtype=mx.float32), mx.array(up_softcap, dtype=mx.float32)]
    for expert in tensors:
        inputs.extend((
            expert["gate_up_proj.weight"],
            expert["gate_up_proj.scales"],
            expert["gate_up_proj.global_scale"].reshape(()),
        ))
    threads = 128
    pairs = rows // 2
    return _resident_gate_up_swiglu(count)(
        inputs=inputs, template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((pairs * lanes + threads - 1) // threads * threads, count, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(count, 1, pairs)], output_dtypes=[mx.bfloat16],
    )[0]


@lru_cache(maxsize=16)
def _resident_fast_gemv(experts, per_expert_input):
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
        name=f"telos_inkling_resident_nvfp4_fast_{experts}_{int(per_expert_input)}",
        input_names=names, output_names=["out"], header=_HEADER, source=source + _BODY_QUAD_FAST + _STORE_QUAD,
    )


@lru_cache(maxsize=16)
def _resident_fast_gate_up_swiglu(experts):
    names = ["x", "gate_cap", "up_cap"]
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
    source += "uint input_start = 0;\n"
    source += "uint output_start = expert * (ROWS / 2u);\n"
    return mx.fast.metal_kernel(
        name=f"telos_inkling_resident_nvfp4_fast_swiglu_{experts}",
        input_names=names, output_names=["out"], header=_HEADER, source=source + _BODY_QUAD_FAST + _STORE_QUAD_SWIGLU,
    )


def nvfp4_resident_fast_gemv(x, tensors, name, *, per_expert_input=False):
    """Full-scale-group (16-element uint2, 4-row/SIMD) resident NVFP4 GEMV."""
    count = len(tensors)
    if not 2 <= count <= 8 or x.dtype != mx.bfloat16:
        return None
    width = x.shape[-1]
    rows = tensors[0][name + ".weight"].shape[0]
    if rows < 4 or rows % 4 or width % 16:
        return None
    lanes = 4 if width <= 64 else 32
    inputs = [x]
    for expert in tensors:
        inputs.extend((expert[name + ".weight"], expert[name + ".scales"], expert[name + ".global_scale"].reshape(())))
    threads = 128
    quads = rows // 4
    return _resident_fast_gemv(count, per_expert_input)(
        inputs=inputs, template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((quads * lanes + threads - 1) // threads * threads, count, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(count, 1, rows)], output_dtypes=[mx.bfloat16],
    )[0]


def nvfp4_resident_fast_gate_up_swiglu(x, tensors, gate_softcap, up_softcap):
    """Full-scale-group (16-element uint2, 4-row/SIMD) fused gate_up + softcapped SwiGLU."""
    count = len(tensors)
    if not 2 <= count <= 8 or x.dtype != mx.bfloat16:
        return None
    width = x.shape[-1]
    rows = tensors[0]["gate_up_proj.weight"].shape[0]
    if rows < 4 or rows % 4 or width % 16:
        return None
    lanes = 4 if width <= 64 else 32
    inputs = [x, mx.array(gate_softcap, dtype=mx.float32), mx.array(up_softcap, dtype=mx.float32)]
    for expert in tensors:
        inputs.extend((
            expert["gate_up_proj.weight"],
            expert["gate_up_proj.scales"],
            expert["gate_up_proj.global_scale"].reshape(()),
        ))
    threads = 128
    quads = rows // 4
    return _resident_fast_gate_up_swiglu(count)(
        inputs=inputs, template=[("WIDTH", width), ("ROWS", rows), ("LANES", lanes)],
        grid=((quads * lanes + threads - 1) // threads * threads, count, 1), threadgroup=(threads, 1, 1),
        output_shapes=[(count, 1, rows // 2)], output_dtypes=[mx.bfloat16],
    )[0]


def _resident_project(x, tensors, gate_softcap, up_softcap, *, fuse_swiglu=False, fast_kernel=False):
    if fast_kernel:
        activated = nvfp4_resident_fast_gate_up_swiglu(x, tensors, gate_softcap, up_softcap)
        if activated is None:
            return None
        return nvfp4_resident_fast_gemv(activated, tensors, "down_proj", per_expert_input=True)
    if fuse_swiglu:
        activated = nvfp4_resident_gate_up_swiglu(x, tensors, gate_softcap, up_softcap)
        if activated is None:
            return None
        return nvfp4_resident_gemv(activated, tensors, "down_proj", per_expert_input=True)
    fused = nvfp4_resident_gemv(x, tensors, "gate_up_proj")
    if fused is None:
        return None
    activated = _gated_activation(fused, gate_softcap, up_softcap)
    return nvfp4_resident_gemv(activated, tensors, "down_proj", per_expert_input=True)


@lru_cache(maxsize=16)
def _compiled_resident_project(gate_softcap, up_softcap, fuse_swiglu, fast_kernel):
    def project(x, tensors):
        return _resident_project(x, tensors, gate_softcap, up_softcap, fuse_swiglu=fuse_swiglu, fast_kernel=fast_kernel)

    return mx.compile(project)


def nvfp4_resident_project(x, tensors, gate_softcap, up_softcap, *, compiled=False, fuse_swiglu=False, fast_kernel=False):
    if compiled:
        return _compiled_resident_project(gate_softcap, up_softcap, fuse_swiglu, fast_kernel)(x, tensors)
    return _resident_project(x, tensors, gate_softcap, up_softcap, fuse_swiglu=fuse_swiglu, fast_kernel=fast_kernel)


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
