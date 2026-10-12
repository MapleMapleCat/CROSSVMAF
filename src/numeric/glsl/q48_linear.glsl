#ifndef CROSSVMAF_Q48_LINEAR_GLSL
#define CROSSVMAF_Q48_LINEAR_GLSL

#include "glsl/q32.glsl"

uvec2 cv_q32_to_q48_saturating(uvec2 value_bits) {
    const bool negative = cv_i64_negative(value_bits);
    const uvec2 magnitude = cv_i64_magnitude(value_bits);
    const uvec2 shifted = cv_u64_shift_left(magnitude, 16u);
    const bool overflow = !cv_u64_equal(
        cv_u64_shift_right(shifted, 16u), magnitude);
    return cv_i64_apply_magnitude_saturating(shifted, negative, overflow);
}

uvec2 cv_q48_to_q32_rne(uvec2 value_bits) {
    const bool negative = cv_i64_negative(value_bits);
    const uvec2 magnitude = cv_i64_magnitude(value_bits);
    const uvec2 rounded = cv_u64_shift_right_rne(magnitude, 16u);
    return cv_i64_apply_magnitude_saturating(rounded, negative, false);
}

uvec2 cv_q48_multiply_rne(uvec2 lhs_bits, uvec2 rhs_bits) {
    const bool negative = cv_i64_negative(lhs_bits) != cv_i64_negative(rhs_bits);
    const uvec4 product = cv_u64_multiply(cv_i64_magnitude(lhs_bits),
                                          cv_i64_magnitude(rhs_bits));
    uvec2 quotient = uvec2(
        (product.y >> 16u) | (product.z << 16u),
        (product.z >> 16u) | (product.w << 16u));
    bool overflow = (product.w & 0xffff0000u) != 0u;
    const uvec2 remainder = uvec2(product.x, product.y & 0xffffu);
    const uvec2 halfway = uvec2(0u, 0x8000u);
    const bool round_up = cv_u64_less(halfway, remainder) ||
        (cv_u64_equal(remainder, halfway) && (quotient.x & 1u) != 0u);
    if (round_up) {
        quotient.x += 1u;
        if (quotient.x == 0u) {
            quotient.y += 1u;
            if (quotient.y == 0u) overflow = true;
        }
    }
    return cv_i64_apply_magnitude_saturating(quotient, negative, overflow);
}

bool cv_q48_divide_rne(uvec2 lhs_bits, uvec2 rhs_bits,
                       out uvec2 output_bits) {
    const uvec2 denominator = cv_i64_magnitude(rhs_bits);
    if (cv_u64_is_zero(denominator)) {
        output_bits = uvec2(0u);
        return false;
    }
    const bool negative = cv_i64_negative(lhs_bits) != cv_i64_negative(rhs_bits);
    const uvec2 value = cv_i64_magnitude(lhs_bits);
    const uvec4 numerator = uvec4(
        0u,
        value.x << 16u,
        (value.x >> 16u) | (value.y << 16u),
        value.y >> 16u);
    uvec2 remainder = uvec2(0u);
    uvec2 quotient = uvec2(0u);
    bool overflow = false;
    for (int bit = 111; bit >= 0; --bit) {
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
    output_bits = cv_i64_apply_magnitude_saturating(
        quotient, negative, overflow);
    return !overflow;
}

bool cv_u112_divide_u64_base16_floor(
    uvec4 numerator_bits, uvec2 denominator_bits,
    out uvec2 quotient_bits, out uvec2 remainder_bits,
    out bool overflow) {
    // Radix 2^16 keeps quotient estimates and digit products within uint
    // while preserving an exact floor quotient and remainder for final RNE.
    const uint digit_mask = 0xffffu;
    const uint digit_base = 0x10000u;
    uint dividend_digits[8];
    uint divisor_digits[4];
    uint quotient_digits[8];

    for (uint digit_index = 0u; digit_index < 8u; ++digit_index) {
        const uint word_index = digit_index >> 1u;
        const uint word_shift = (digit_index & 1u) * 16u;
        dividend_digits[digit_index] =
            (numerator_bits[word_index] >> word_shift) & digit_mask;
        quotient_digits[digit_index] = 0u;
    }
    for (uint digit_index = 0u; digit_index < 4u; ++digit_index) {
        const uint word_index = digit_index >> 1u;
        const uint word_shift = (digit_index & 1u) * 16u;
        divisor_digits[digit_index] =
            (denominator_bits[word_index] >> word_shift) & digit_mask;
    }

    uint divisor_length = 4u;
    while (divisor_length > 0u &&
           divisor_digits[divisor_length - 1u] == 0u) {
        --divisor_length;
    }
    if (divisor_length == 0u) {
        quotient_bits = uvec2(0u);
        remainder_bits = uvec2(0u);
        overflow = false;
        return false;
    }

    // Q16.48 expansion occupies at most seven base-2^16 digits.
    uint dividend_length = 7u;
    while (dividend_length > 0u &&
           dividend_digits[dividend_length - 1u] == 0u) {
        --dividend_length;
    }
    if (dividend_length == 0u) {
        quotient_bits = uvec2(0u);
        remainder_bits = uvec2(0u);
        overflow = false;
        return true;
    }
    if (dividend_length < divisor_length) {
        quotient_bits = uvec2(0u);
        remainder_bits = uvec2(
            dividend_digits[0] | (dividend_digits[1] << 16u),
            dividend_digits[2] | (dividend_digits[3] << 16u));
        overflow = false;
        return true;
    }

    const uint divisor_top_digit = divisor_digits[divisor_length - 1u];
    const uint normalization_shift =
        15u - uint(findMSB(divisor_top_digit));
    if (normalization_shift != 0u) {
        uint carry = 0u;
        for (uint digit_index = 0u; digit_index < divisor_length;
             ++digit_index) {
            const uint source_digit = divisor_digits[digit_index];
            divisor_digits[digit_index] =
                ((source_digit << normalization_shift) & digit_mask) | carry;
            carry = source_digit >> (16u - normalization_shift);
        }

        carry = 0u;
        for (uint digit_index = 0u; digit_index < dividend_length;
             ++digit_index) {
            const uint source_digit = dividend_digits[digit_index];
            dividend_digits[digit_index] =
                ((source_digit << normalization_shift) & digit_mask) | carry;
            carry = source_digit >> (16u - normalization_shift);
        }
        dividend_digits[dividend_length] = carry;
    } else {
        dividend_digits[dividend_length] = 0u;
    }

    const uint quotient_length = dividend_length - divisor_length + 1u;
    for (int quotient_index = int(quotient_length) - 1;
         quotient_index >= 0; --quotient_index) {
        const uint unsigned_quotient_index = uint(quotient_index);
        const uint normalized_divisor_top =
            divisor_digits[divisor_length - 1u];
        const uint dividend_top = dividend_digits[
            unsigned_quotient_index + divisor_length];
        const uint dividend_next = dividend_digits[
            unsigned_quotient_index + divisor_length - 1u];

        uint quotient_estimate;
        uint estimate_remainder;
        if (dividend_top == normalized_divisor_top) {
            quotient_estimate = digit_mask;
            estimate_remainder = dividend_next + normalized_divisor_top;
        } else {
            const uint estimate_numerator =
                dividend_top * digit_base + dividend_next;
            quotient_estimate =
                estimate_numerator / normalized_divisor_top;
            estimate_remainder = estimate_numerator -
                quotient_estimate * normalized_divisor_top;
        }

        if (divisor_length > 1u) {
            const uint divisor_next =
                divisor_digits[divisor_length - 2u];
            const uint comparison_digit = dividend_digits[
                unsigned_quotient_index + divisor_length - 2u];
            for (uint correction = 0u; correction < 2u; ++correction) {
                if (estimate_remainder >= digit_base) break;
                const uint estimate_product =
                    quotient_estimate * divisor_next;
                const uint estimate_limit =
                    estimate_remainder * digit_base + comparison_digit;
                if (estimate_product <= estimate_limit) break;
                --quotient_estimate;
                estimate_remainder += normalized_divisor_top;
            }
        }

        uint multiply_borrow = 0u;
        for (uint divisor_index = 0u; divisor_index < divisor_length;
             ++divisor_index) {
            const uint product =
                quotient_estimate * divisor_digits[divisor_index] +
                multiply_borrow;
            const uint product_digit = product & digit_mask;
            multiply_borrow = product >> 16u;

            const uint dividend_index =
                unsigned_quotient_index + divisor_index;
            const uint dividend_digit = dividend_digits[dividend_index];
            if (dividend_digit < product_digit) {
                dividend_digits[dividend_index] =
                    dividend_digit + digit_base - product_digit;
                ++multiply_borrow;
            } else {
                dividend_digits[dividend_index] =
                    dividend_digit - product_digit;
            }
        }

        const uint subtraction_top_index =
            unsigned_quotient_index + divisor_length;
        const uint subtraction_top = dividend_digits[subtraction_top_index];
        const bool estimate_was_too_large = subtraction_top < multiply_borrow;
        dividend_digits[subtraction_top_index] =
            (subtraction_top - multiply_borrow) & digit_mask;
        if (estimate_was_too_large) {
            --quotient_estimate;
            uint add_carry = 0u;
            for (uint divisor_index = 0u; divisor_index < divisor_length;
                 ++divisor_index) {
                const uint dividend_index =
                    unsigned_quotient_index + divisor_index;
                const uint sum = dividend_digits[dividend_index] +
                    divisor_digits[divisor_index] + add_carry;
                dividend_digits[dividend_index] = sum & digit_mask;
                add_carry = sum >> 16u;
            }
            dividend_digits[subtraction_top_index] =
                (dividend_digits[subtraction_top_index] + add_carry) &
                digit_mask;
        }
        quotient_digits[unsigned_quotient_index] = quotient_estimate;
    }

    uint remainder_digits[4];
    for (uint digit_index = 0u; digit_index < 4u; ++digit_index) {
        remainder_digits[digit_index] = 0u;
        if (digit_index >= divisor_length) continue;
        if (normalization_shift == 0u) {
            remainder_digits[digit_index] = dividend_digits[digit_index];
        } else {
            const uint next_digit = digit_index + 1u < divisor_length
                ? dividend_digits[digit_index + 1u]
                : 0u;
            remainder_digits[digit_index] =
                (dividend_digits[digit_index] >> normalization_shift) |
                ((next_digit << (16u - normalization_shift)) & digit_mask);
        }
    }

    quotient_bits = uvec2(
        quotient_digits[0] | (quotient_digits[1] << 16u),
        quotient_digits[2] | (quotient_digits[3] << 16u));
    remainder_bits = uvec2(
        remainder_digits[0] | (remainder_digits[1] << 16u),
        remainder_digits[2] | (remainder_digits[3] << 16u));
    overflow = false;
    for (uint digit_index = 4u; digit_index < quotient_length;
         ++digit_index) {
        overflow = overflow || quotient_digits[digit_index] != 0u;
    }
    return true;
}

bool cv_q48_divide_rne_base16(
    uvec2 lhs_bits, uvec2 rhs_bits, out uvec2 output_bits) {
    const uvec2 denominator = cv_i64_magnitude(rhs_bits);
    if (cv_u64_is_zero(denominator)) {
        output_bits = uvec2(0u);
        return false;
    }

    const bool negative = cv_i64_negative(lhs_bits) != cv_i64_negative(rhs_bits);
    const uvec2 value = cv_i64_magnitude(lhs_bits);
    const uvec4 numerator = uvec4(
        0u,
        value.x << 16u,
        (value.x >> 16u) | (value.y << 16u),
        value.y >> 16u);
    uvec2 quotient;
    uvec2 remainder;
    bool overflow;
    if (!cv_u112_divide_u64_base16_floor(
            numerator, denominator, quotient, remainder, overflow)) {
        output_bits = uvec2(0u);
        return false;
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
    output_bits = cv_i64_apply_magnitude_saturating(
        quotient, negative, overflow);
    return !overflow;
}

bool cv_q48_sqrt_rne(uvec2 value_bits, out uvec2 output_bits) {
    if (cv_i64_negative(value_bits)) {
        output_bits = uvec2(0u);
        return false;
    }
    const uvec4 target = uvec4(
        0u,
        value_bits.x << 16u,
        (value_bits.x >> 16u) | (value_bits.y << 16u),
        value_bits.y >> 16u);
    output_bits = cv_u128_sqrt_rne(target);
    return true;
}


#endif
