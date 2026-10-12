#ifndef CROSSVMAF_SPEED_CHROMA_SOURCE_GLSL
#define CROSSVMAF_SPEED_CHROMA_SOURCE_GLSL

layout(set = 0, binding = 0, std430) readonly buffer InputPlane0 {
    uint words[];
} input_plane_0;
layout(set = 0, binding = 1, std430) readonly buffer InputPlane1 {
    uint words[];
} input_plane_1;
layout(set = 0, binding = 2, std430) readonly buffer InputPlane2 {
    uint words[];
} input_plane_2;
layout(set = 0, binding = 3, std430) readonly buffer InputPlane3 {
    uint words[];
} input_plane_3;
layout(set = 0, binding = 5, std430) buffer Status {
    uint flags[];
} status_buffer;

const uint sample_fractional_bits = 20u;

uint source_stride_bytes(uint stream) {
    if (stream == 0u) return parameters.source_stride_bytes_0;
    if (stream == 1u) return parameters.source_stride_bytes_1;
    if (stream == 2u) return parameters.source_stride_bytes_2;
    return parameters.source_stride_bytes_3;
}

uint load_byte(uint byte_offset) {
    const uint word_index = byte_offset >> 2u;
    const uint stream = gl_GlobalInvocationID.z;
    uint word;
    if (stream == 0u) {
        word = input_plane_0.words[word_index];
    } else if (stream == 1u) {
        word = input_plane_1.words[word_index];
    } else if (stream == 2u) {
        word = input_plane_2.words[word_index];
    } else {
        word = input_plane_3.words[word_index];
    }
    return (word >> ((byte_offset & 3u) * 8u)) & 0xffu;
}

uint load_sample(uint row, uint column) {
    const uint bytes_per_sample = parameters.bit_depth == 8u ? 1u : 2u;
    const uint offset = row * source_stride_bytes(gl_GlobalInvocationID.z) +
                        column * bytes_per_sample;
    const uint low = load_byte(offset);
    return bytes_per_sample == 1u
        ? low
        : low | (load_byte(offset + 1u) << 8u);
}

void validate_source_sample(uint row, uint column) {
    const uint sample_value = load_sample(row, column);
    const uint maximum = parameters.bit_depth == 16u
        ? 0xffffu
        : (1u << parameters.bit_depth) - 1u;
    if (sample_value > maximum) atomicOr(status_buffer.flags[0], 1u);
}

int load_normalized(uint row, uint column) {
    const uint sample_value = load_sample(row, column);
    const uint maximum = parameters.bit_depth == 16u
        ? 0xffffu
        : (1u << parameters.bit_depth) - 1u;
    if (sample_value > maximum) atomicOr(status_buffer.flags[0], 1u);
    const uint shift = sample_fractional_bits + 8u - parameters.bit_depth;
    return int(sample_value << shift) -
           int(128u << sample_fractional_bits);
}

uvec2 sample_q32(int sample_value) {
    return cv_u64_shift_left(
        cv_i64_from_i32(sample_value), 32u - sample_fractional_bits);
}

#endif
