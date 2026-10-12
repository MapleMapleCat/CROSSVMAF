#ifndef CROSSVMAF_BINARY32_QUANTIZE_GLSL
#define CROSSVMAF_BINARY32_QUANTIZE_GLSL

#include "glsl/q48_linear.glsl"

int cv_u64_msb(uvec2 value) {
    // GLSL 450 findMSB is a core integer instruction (OpFindUMsb) and does not
    // introduce any extension, native int64, or vendor dependency. The 64-bit
    // pattern is reduced to one or two 32-bit count-leading-ones operations.
    if (value.y != 0u) return findMSB(value.y) + 32;
    return value.x == 0u ? -1 : findMSB(value.x);
}

uvec2 cv_binary32_quantize(uvec2 value_bits) {
    if (cv_u64_is_zero(value_bits)) return value_bits;
    const bool negative = cv_i64_negative(value_bits);
    const uvec2 magnitude = cv_i64_magnitude(value_bits);
    const int highest_bit = cv_u64_msb(magnitude);
    if (highest_bit <= 23) return value_bits;
    const uint shift = uint(highest_bit - 23);
    const uvec2 rounded = cv_u64_shift_right_rne(magnitude, shift);
    const uvec2 restored = cv_u64_shift_left(rounded, shift);
    const bool overflow = !cv_u64_equal(
        cv_u64_shift_right(restored, shift), rounded);
    return cv_i64_apply_magnitude_saturating(restored, negative, overflow);
}

uvec2 cv_i64_half_difference_rne_binary32(uvec2 lhs, uvec2 rhs) {
    const uvec2 lhs_half = cv_i64_arithmetic_shift_right(lhs, 1u);
    const uvec2 rhs_half = cv_i64_arithmetic_shift_right(rhs, 1u);
    uvec2 result = cv_u64_subtract(lhs_half, rhs_half);
    const bool lhs_is_odd = (lhs.x & 1u) != 0u;
    const bool rhs_is_odd = (rhs.x & 1u) != 0u;
    const bool tie_rounds_away_from_result =
        lhs_is_odd != rhs_is_odd && (result.x & 1u) != 0u;
    if (tie_rounds_away_from_result) {
        const uvec2 adjustment = lhs_is_odd
            ? uvec2(1u, 0u)
            : uvec2(0xffffffffu);
        result = cv_q32_add_saturating(result, adjustment);
    }
    return cv_binary32_quantize(result);
}

const uvec4 cv_q48_norm_square_limit =
    uvec4(0x00000001u, 0x00000000u, 0xffffffffu, 0x3fffffffu);

void cv_q48_norm_accumulate(inout uvec4 accumulator, uvec2 value_bits) {
    const uvec2 magnitude = cv_i64_magnitude(value_bits);
    const uvec4 square = cv_u64_multiply(magnitude, magnitude);
    if (cv_u128_less(cv_q48_norm_square_limit, square)) {
        accumulator = cv_q48_norm_square_limit;
        return;
    }

    const uvec4 sum = cv_u128_add(accumulator, square);
    const bool overflow = cv_u128_less(sum, accumulator);
    accumulator = overflow || cv_u128_less(cv_q48_norm_square_limit, sum)
        ? cv_q48_norm_square_limit
        : sum;
}

uvec2 cv_q48_norm_finalize_binary32(uvec4 square_sum) {
    return cv_binary32_quantize(cv_u128_sqrt_rne(square_sum));
}

uvec2 cv_q48_norm2_binary32(uvec2 lhs_bits, uvec2 rhs_bits) {
    const uvec2 lhs_magnitude = cv_i64_magnitude(lhs_bits);
    const uvec2 rhs_magnitude = cv_i64_magnitude(rhs_bits);
    const bool lhs_is_smaller = cv_u64_less(lhs_magnitude, rhs_magnitude);
    const uvec2 maximum = lhs_is_smaller ? rhs_magnitude : lhs_magnitude;
    const uvec2 minimum = lhs_is_smaller ? lhs_magnitude : rhs_magnitude;
    if (cv_u64_is_zero(maximum)) return uvec2(0u);

    const uvec4 minimum_square = cv_u64_multiply(minimum, minimum);
    const uvec4 maximum_as_u128 = uvec4(maximum, uvec2(0u));
    if (!cv_u128_less(maximum_as_u128, minimum_square)) {
        return cv_binary32_quantize(
            cv_i64_apply_magnitude_saturating(maximum, false, false));
    }

    uvec4 square_sum = uvec4(0u);
    cv_q48_norm_accumulate(square_sum, maximum);
    if (!cv_u128_equal(square_sum, cv_q48_norm_square_limit)) {
        cv_q48_norm_accumulate(square_sum, minimum);
    }
    return cv_q48_norm_finalize_binary32(square_sum);
}

uvec2 cv_q32_binary32_add(uvec2 lhs, uvec2 rhs) {
    return cv_binary32_quantize(cv_q32_add_saturating(lhs, rhs));
}

uvec2 cv_q32_binary32_subtract(uvec2 lhs, uvec2 rhs) {
    return cv_binary32_quantize(cv_q32_subtract_saturating(lhs, rhs));
}

uvec2 cv_q32_binary32_multiply(uvec2 lhs, uvec2 rhs) {
    return cv_binary32_quantize(cv_q32_multiply_rne(lhs, rhs));
}

bool cv_q32_binary32_divide(uvec2 lhs, uvec2 rhs,
                            out uvec2 output_bits) {
    if (!cv_q32_divide_rne(lhs, rhs, output_bits)) return false;
    output_bits = cv_binary32_quantize(output_bits);
    return true;
}

bool cv_q32_binary32_sqrt(uvec2 value, out uvec2 output_bits) {
    if (!cv_q32_sqrt_rne(value, output_bits)) return false;
    output_bits = cv_binary32_quantize(output_bits);
    return true;
}

uvec2 cv_q48_binary32_add(uvec2 lhs, uvec2 rhs) {
    return cv_binary32_quantize(cv_q32_add_saturating(lhs, rhs));
}

uvec2 cv_q48_binary32_subtract(uvec2 lhs, uvec2 rhs) {
    return cv_binary32_quantize(cv_q32_subtract_saturating(lhs, rhs));
}

uvec2 cv_q48_binary32_multiply(uvec2 lhs, uvec2 rhs) {
    return cv_binary32_quantize(cv_q48_multiply_rne(lhs, rhs));
}

bool cv_q48_binary32_divide(uvec2 lhs, uvec2 rhs,
                            out uvec2 output_bits) {
    if (!cv_q48_divide_rne(lhs, rhs, output_bits)) return false;
    output_bits = cv_binary32_quantize(output_bits);
    return true;
}

bool cv_q48_binary32_divide_base16(
    uvec2 lhs, uvec2 rhs, out uvec2 output_bits) {
    if (!cv_q48_divide_rne_base16(lhs, rhs, output_bits)) return false;
    output_bits = cv_binary32_quantize(output_bits);
    return true;
}

bool cv_q48_binary32_sqrt(uvec2 value, out uvec2 output_bits) {
    if (!cv_q48_sqrt_rne(value, output_bits)) return false;
    output_bits = cv_binary32_quantize(output_bits);
    return true;
}


#endif
