"""Metal mask construction for single-token Inkling attention."""

import mlx.core as mx


_DECODE_MASK = mx.fast.metal_kernel(
    name="telos_inkling_relative_decode_mask",
    input_names=["relative", "sink", "keys", "window"],
    output_names=["mask"],
    source="""
uint column = thread_position_in_grid.x;
uint head = thread_position_in_grid.y;
if (column > keys) return;

uint index = head * (keys + 1) + column;
if (column == keys) {
    mask[index] = sink[head];
    return;
}

uint distance = keys - 1 - column;
if (window != 0 && distance >= window) {
    mask[index] = T(-INFINITY);
} else if (distance < EXTENT) {
    mask[index] = relative[head * EXTENT + distance];
} else {
    mask[index] = T(0);
}
""",
)

_FIXED_MASK = mx.fast.metal_kernel(
    name="telos_inkling_fixed_decode_mask",
    input_names=["relative", "sink", "cursor"], output_names=["mask"],
    source="""
uint column = thread_position_in_grid.x;
uint head = thread_position_in_grid.y;
if (column >= CAPACITY) return;
uint index = head * CAPACITY + column;
uint pos = uint(cursor);
if (column == pos + 1u) {
    mask[index] = sink[head];
} else if (column > pos || (WINDOW != 0 && pos - column >= WINDOW)) {
    mask[index] = T(-INFINITY);
} else if (pos - column < EXTENT) {
    mask[index] = relative[head * EXTENT + pos - column];
} else {
    mask[index] = T(0);
}
""",
)


def fused_decode_mask(relative, sink, keys, window):
    heads, extent = relative.shape[-2:]
    group = 64
    return _DECODE_MASK(
        inputs=[relative, sink, mx.array(keys, dtype=mx.uint32), mx.array(window or 0, dtype=mx.uint32)],
        template=[("T", relative.dtype), ("EXTENT", extent)],
        grid=((keys + group) // group * group, heads, 1),
        threadgroup=(group, 1, 1),
        output_shapes=[(1, heads, 1, keys + 1)],
        output_dtypes=[relative.dtype],
    )[0]


def fixed_decode_mask(relative, sink, cursor, capacity, window):
    heads, extent = relative.shape[-2:]
    return _FIXED_MASK(
        inputs=[relative, sink, cursor],
        template=[("T", relative.dtype), ("EXTENT", extent), ("CAPACITY", capacity), ("WINDOW", window or 0)],
        grid=((capacity + 63) // 64 * 64, heads, 1), threadgroup=(64, 1, 1),
        output_shapes=[(1, heads, 1, capacity)], output_dtypes=[relative.dtype],
    )[0]
