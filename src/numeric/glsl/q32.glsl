#ifndef CROSSVMAF_Q32_CORE_GLSL
#define CROSSVMAF_Q32_CORE_GLSL

#include "glsl/integer.glsl"

uvec2 cv_q32_add_saturating(uvec2 lhs, uvec2 rhs) {
    const uvec2 result = cv_u64_add(lhs, rhs);
    const bool lhs_negative = cv_i64_negative(lhs);
    if (lhs_negative == cv_i64_negative(rhs) &&
        lhs_negative != cv_i64_negative(result)) {
        return lhs_negative ? uvec2(0u, 0x80000000u)
                            : uvec2(0xffffffffu, 0x7fffffffu);
    }
    return result;
}

uvec2 cv_q32_subtract_saturating(uvec2 lhs, uvec2 rhs) {
    const uvec2 minimum = uvec2(0u, 0x80000000u);
    if (cv_u64_equal(rhs, minimum)) {
        if (!cv_i64_negative(lhs)) return uvec2(0xffffffffu, 0x7fffffffu);
        return cv_u64_subtract(lhs, rhs);
    }
    return cv_q32_add_saturating(lhs, cv_u64_negate(rhs));
}

uvec2 cv_q32_average_rne(uvec2 lhs, uvec2 rhs) {
    const uvec2 differing = lhs ^ rhs;
    uvec2 quotient = cv_u64_add(
        lhs & rhs, cv_i64_arithmetic_shift_right(differing, 1u));
    if ((differing.x & 1u) != 0u && (quotient.x & 1u) != 0u) {
        quotient = cv_u64_add(quotient, uvec2(1u, 0u));
    }
    return quotient;
}

uvec2 cv_q32_half_rne(uvec2 value) {
    return cv_i64_shift_right_rne(value, 1u);
}

uvec2 cv_q32_multiply_rne(uvec2 lhs_bits, uvec2 rhs_bits) {
    const bool negative = cv_i64_negative(lhs_bits) != cv_i64_negative(rhs_bits);
    const uvec4 product = cv_u64_multiply(cv_i64_magnitude(lhs_bits),
                                          cv_i64_magnitude(rhs_bits));

    uvec2 quotient = uvec2(product.y, product.z);
    bool overflow = product.w != 0u;
    const bool round_up = product.x > 0x80000000u ||
        (product.x == 0x80000000u && (quotient.x & 1u) != 0u);
    if (round_up) {
        quotient.x += 1u;
        if (quotient.x == 0u) {
            quotient.y += 1u;
            if (quotient.y == 0u) {
                overflow = true;
            }
        }
    }

    return cv_i64_apply_magnitude_saturating(quotient, negative, overflow);
}

uvec2 cv_q32_multiply_positive_normalized_rne(
    uvec2 lhs_bits, uvec2 rhs_bits) {
    // Both operands have raw form 2^32 + fraction. Expanding that product
    // leaves only fraction*fraction as a non-aligned partial product, so the
    // exact Q32.32 quotient and remainder need one 32x32 multiply instead of
    // the four partial products used by the general 64x64 path.
    const uvec2 fraction_product = cv_u32_multiply(
        lhs_bits.x, rhs_bits.x);

    uint quotient_low = lhs_bits.x + rhs_bits.x;
    uint quotient_high = 1u + uint(quotient_low < lhs_bits.x);
    const uint previous_quotient_low = quotient_low;
    quotient_low += fraction_product.y;
    quotient_high += uint(quotient_low < previous_quotient_low);

    const bool round_up = fraction_product.x > 0x80000000u ||
        (fraction_product.x == 0x80000000u &&
         (quotient_low & 1u) != 0u);
    if (round_up) {
        quotient_low += 1u;
        if (quotient_low == 0u) quotient_high += 1u;
    }
    return uvec2(quotient_low, quotient_high);
}

bool cv_q32_divide_by_u32_prepared_rne(
    uvec2 value_bits, CvPreparedU32Divider divider,
    out uvec2 output_bits) {
    if (divider.denominator == 0u) {
        output_bits = uvec2(0u);
        return false;
    }
    const bool negative = cv_i64_negative(value_bits);
    const uvec2 quotient = cv_u64_divide_u32_prepared_rne(
        cv_i64_magnitude(value_bits), divider);
    output_bits = cv_i64_apply_magnitude_saturating(
        quotient, negative, false);
    return true;
}

bool cv_q32_divide_rne(uvec2 lhs_bits, uvec2 rhs_bits, out uvec2 output_bits) {
    const uvec2 denominator = cv_i64_magnitude(rhs_bits);
    if (cv_u64_is_zero(denominator)) {
        output_bits = uvec2(0u);
        return false;
    }

    const bool negative = cv_i64_negative(lhs_bits) != cv_i64_negative(rhs_bits);
    const uvec2 numerator_value = cv_i64_magnitude(lhs_bits);
    const uvec4 numerator = uvec4(0u, numerator_value.x, numerator_value.y, 0u);
    uvec2 remainder = uvec2(0u);
    uvec2 quotient = uvec2(0u);
    bool overflow = false;

    for (int bit = 95; bit >= 0; --bit) {
        const bool remainder_overflow = (remainder.y & 0x80000000u) != 0u;
        remainder = cv_u64_shift_left_one(remainder);
        const uint unsigned_bit = uint(bit);
        const uint limb = unsigned_bit >> 5u;
        const uint offset = unsigned_bit & 31u;
        remainder.x |= (numerator[limb] >> offset) & 1u;
        if (remainder_overflow || !cv_u64_less(remainder, denominator)) {
            remainder = cv_u64_subtract(remainder, denominator);
            if (bit >= 64) {
                overflow = true;
            } else if (bit >= 32) {
                quotient.y |= 1u << (uint(bit) - 32u);
            } else {
                quotient.x |= 1u << uint(bit);
            }
        }
    }

    const uvec2 complement = cv_u64_subtract(denominator, remainder);
    const bool round_up = cv_u64_less(complement, remainder) ||
        (cv_u64_equal(remainder, complement) && (quotient.x & 1u) != 0u);
    if (round_up) {
        quotient.x += 1u;
        if (quotient.x == 0u) {
            quotient.y += 1u;
            if (quotient.y == 0u) overflow = true;
        }
    }
    output_bits = cv_i64_apply_magnitude_saturating(quotient, negative, overflow);
    return true;
}

bool cv_q32_sqrt_rne(uvec2 value_bits, out uvec2 output_bits) {
    if (cv_i64_negative(value_bits)) {
        output_bits = uvec2(0u);
        return false;
    }

    const uvec4 target = uvec4(0u, value_bits.x, value_bits.y, 0u);
    output_bits = cv_u128_sqrt_rne(target);
    return true;
}


#endif
