#ifndef CROSSVMAF_Q32_TRANSCENDENTALS_GLSL
#define CROSSVMAF_Q32_TRANSCENDENTALS_GLSL

#include "glsl/binary32_quantize.glsl"

uvec2 cv_q32_log2_positive(uvec2 value_bits) {
    const int msb = cv_u64_msb(value_bits);
    const int exponent = msb - 32;
    uvec2 normalized = msb > 32
        ? cv_u64_shift_right(value_bits, uint(msb - 32))
        : cv_u64_shift_left(value_bits, uint(32 - msb));
    uint fraction = 0u;
    const uvec2 two = uvec2(0u, 2u);
    for (uint bit = 0u; bit < 32u; ++bit) {
        normalized = cv_q32_multiply_positive_normalized_rne(
            normalized, normalized);
        if (!cv_u64_less(normalized, two)) {
            normalized = cv_u64_shift_right(normalized, 1u);
            fraction |= 1u << (31u - bit);
        }
    }
    return cv_u64_add(uvec2(0u, uint(exponent)), uvec2(fraction, 0u));
}

bool cv_q32_log2(uvec2 value_bits, out uvec2 output_bits) {
    if (cv_i64_negative(value_bits) || cv_u64_is_zero(value_bits)) {
        output_bits = uvec2(0u);
        return false;
    }
    output_bits = cv_q32_log2_positive(value_bits);
    return true;
}

const uvec2 cv_exp2_fraction_constants[32] = uvec2[32](
    uvec2(0x6a09e668u, 0x00000001u), uvec2(0x306fe0a3u, 0x00000001u),
    uvec2(0x172b83c8u, 0x00000001u), uvec2(0x0b5586d0u, 0x00000001u),
    uvec2(0x059b0d31u, 0x00000001u), uvec2(0x02c9a3e7u, 0x00000001u),
    uvec2(0x0163daa0u, 0x00000001u), uvec2(0x00b1afa6u, 0x00000001u),
    uvec2(0x0058c86eu, 0x00000001u), uvec2(0x002c605eu, 0x00000001u),
    uvec2(0x00162f39u, 0x00000001u), uvec2(0x000b175fu, 0x00000001u),
    uvec2(0x00058ba0u, 0x00000001u), uvec2(0x0002c5ccu, 0x00000001u),
    uvec2(0x000162e5u, 0x00000001u), uvec2(0x0000b172u, 0x00000001u),
    uvec2(0x000058b9u, 0x00000001u), uvec2(0x00002c5du, 0x00000001u),
    uvec2(0x0000162eu, 0x00000001u), uvec2(0x00000b17u, 0x00000001u),
    uvec2(0x0000058cu, 0x00000001u), uvec2(0x000002c6u, 0x00000001u),
    uvec2(0x00000163u, 0x00000001u), uvec2(0x000000b1u, 0x00000001u),
    uvec2(0x00000059u, 0x00000001u), uvec2(0x0000002cu, 0x00000001u),
    uvec2(0x00000016u, 0x00000001u), uvec2(0x0000000bu, 0x00000001u),
    uvec2(0x00000006u, 0x00000001u), uvec2(0x00000003u, 0x00000001u),
    uvec2(0x00000001u, 0x00000001u), uvec2(0x00000001u, 0x00000001u));

uvec2 cv_q32_exp2(uvec2 exponent_bits) {
    const int integer_part = int(exponent_bits.y);
    const uint fraction = exponent_bits.x;
    uvec2 result = uvec2(0u, 1u);
    for (uint index = 0u; index < 32u; ++index) {
        const uint bit = 31u - index;
        if ((fraction & (1u << bit)) != 0u) {
            result = cv_q32_multiply_rne(
                result, cv_exp2_fraction_constants[index]);
        }
    }

    if (integer_part >= 0) {
        if (integer_part >= 31) return uvec2(0xffffffffu, 0x7fffffffu);
        const uvec2 shifted = cv_u64_shift_left(result, uint(integer_part));
        const uvec2 limit = uvec2(0xffffffffu, 0x7fffffffu);
        return cv_u64_less(limit, shifted) ? limit : shifted;
    }
    const uint shift = uint(-(integer_part + 1)) + 1u;
    return cv_u64_shift_right_rne(result, shift);
}

bool cv_q32_cbrt(uvec2 value_bits, out uvec2 output_bits) {
    if (cv_u64_is_zero(value_bits)) {
        output_bits = uvec2(0u);
        return true;
    }
    const bool negative = cv_i64_negative(value_bits);
    const uvec2 magnitude = cv_i64_magnitude(value_bits);
    const uvec2 logarithm = cv_q32_log2_positive(magnitude);
    uvec2 exponent;
    const bool divided = cv_q32_divide_rne(
        logarithm, uvec2(0u, 3u), exponent);
    if (!divided) {
        output_bits = uvec2(0u);
        return false;
    }
    const uvec2 root = cv_q32_exp2(exponent);
    output_bits = negative ? cv_u64_negate(root) : root;
    return true;
}


#endif
