#ifndef CROSSVMAF_INTEGER_GLSL
#define CROSSVMAF_INTEGER_GLSL

// All multiword values are little-endian 32-bit limbs. uvec2 represents a
// 64-bit two's-complement bit pattern; uvec4 represents an unsigned 128-bit
// intermediate. No shader in CrossVMAF may enable native int64 arithmetic.

struct CvPreparedU32Divider {
    uint denominator;
    uint reciprocal_low;
    uint reciprocal_high;
    uint power_of_two_shift;
};

const uint cv_prepared_divider_non_power_of_two_shift = 0xffffffffu;

uvec2 cv_u64_add(uvec2 lhs, uvec2 rhs) {
    const uint lo = lhs.x + rhs.x;
    const uint carry = lo < lhs.x ? 1u : 0u;
    return uvec2(lo, lhs.y + rhs.y + carry);
}

uvec2 cv_u64_negate(uvec2 value) {
    const uint lo = ~value.x + 1u;
    const uint carry = lo == 0u ? 1u : 0u;
    return uvec2(lo, ~value.y + carry);
}

uvec2 cv_u64_subtract(uvec2 lhs, uvec2 rhs) {
    return cv_u64_add(lhs, cv_u64_negate(rhs));
}

bool cv_u64_less(uvec2 lhs, uvec2 rhs) {
    return lhs.y < rhs.y || (lhs.y == rhs.y && lhs.x < rhs.x);
}

bool cv_u64_equal(uvec2 lhs, uvec2 rhs) {
    return all(equal(lhs, rhs));
}

bool cv_u64_is_zero(uvec2 value) {
    return value.x == 0u && value.y == 0u;
}

uvec2 cv_u64_shift_left_one(uvec2 value) {
    return uvec2(value.x << 1u, (value.y << 1u) | (value.x >> 31u));
}

uvec2 cv_u64_shift_left(uvec2 value, uint shift) {
    if (shift == 0u) return value;
    if (shift >= 64u) return uvec2(0u);
    if (shift >= 32u) {
        return uvec2(0u, value.x << (shift - 32u));
    }
    return uvec2(value.x << shift,
                 (value.y << shift) | (value.x >> (32u - shift)));
}

uvec2 cv_u64_shift_right(uvec2 value, uint shift) {
    if (shift == 0u) return value;
    if (shift >= 64u) return uvec2(0u);
    if (shift >= 32u) {
        return uvec2(value.y >> (shift - 32u), 0u);
    }
    return uvec2((value.x >> shift) | (value.y << (32u - shift)),
                 value.y >> shift);
}

uvec2 cv_u64_shift_right_rne(uvec2 value, uint shift) {
    if (shift == 0u) return value;
    if (shift > 64u) return uvec2(0u);
    if (shift == 64u) {
        const uvec2 halfway = uvec2(0u, 0x80000000u);
        return cv_u64_less(halfway, value) ? uvec2(1u, 0u)
                                            : uvec2(0u);
    }
    uvec2 quotient = cv_u64_shift_right(value, shift);
    const uvec2 restored = cv_u64_shift_left(quotient, shift);
    const uvec2 remainder = cv_u64_subtract(value, restored);
    const uvec2 halfway = cv_u64_shift_left(
        uvec2(1u, 0u), shift - 1u);
    if (cv_u64_less(halfway, remainder) ||
        (cv_u64_equal(remainder, halfway) && (quotient.x & 1u) != 0u)) {
        quotient = cv_u64_add(quotient, uvec2(1u, 0u));
    }
    return quotient;
}

uvec2 cv_u64_divide_u32_rne(uvec2 numerator, uint denominator) {
    if (denominator == 0u) return uvec2(0xffffffffu);
    uvec2 quotient = uvec2(0u);
    uint remainder = 0u;
    for (int bit = 63; bit >= 0; --bit) {
        const bool remainder_overflow = (remainder & 0x80000000u) != 0u;
        remainder = (remainder << 1u) |
            (bit >= 32 ? ((numerator.y >> uint(bit - 32)) & 1u)
                       : ((numerator.x >> uint(bit)) & 1u));
        if (remainder_overflow || remainder >= denominator) {
            remainder -= denominator;
            if (bit >= 32) {
                quotient.y |= 1u << uint(bit - 32);
            } else {
                quotient.x |= 1u << uint(bit);
            }
        }
    }
    const uint complement = denominator - remainder;
    if (remainder > complement ||
        (remainder == complement && (quotient.x & 1u) != 0u)) {
        quotient = cv_u64_add(quotient, uvec2(1u, 0u));
    }
    return quotient;
}

uvec2 cv_u64_divide_u32_floor(uvec2 numerator, uint denominator,
                              out uint remainder) {
    if (denominator == 0u) {
        remainder = 0u;
        return uvec2(0xffffffffu);
    }
    uvec2 quotient = uvec2(0u);
    remainder = 0u;
    for (int bit = 63; bit >= 0; --bit) {
        const bool remainder_overflow = (remainder & 0x80000000u) != 0u;
        remainder = (remainder << 1u) |
            (bit >= 32 ? ((numerator.y >> uint(bit - 32)) & 1u)
                       : ((numerator.x >> uint(bit)) & 1u));
        if (remainder_overflow || remainder >= denominator) {
            remainder -= denominator;
            if (bit >= 32) {
                quotient.y |= 1u << uint(bit - 32);
            } else {
                quotient.x |= 1u << uint(bit);
            }
        }
    }
    return quotient;
}

bool cv_i64_less(uvec2 lhs, uvec2 rhs) {
    const uint lhs_high = lhs.y ^ 0x80000000u;
    const uint rhs_high = rhs.y ^ 0x80000000u;
    return lhs_high < rhs_high ||
        (lhs_high == rhs_high && lhs.x < rhs.x);
}

bool cv_i64_negative(uvec2 value) {
    return (value.y & 0x80000000u) != 0u;
}

uvec2 cv_i64_magnitude(uvec2 value) {
    return cv_i64_negative(value) ? cv_u64_negate(value) : value;
}

uvec2 cv_i64_from_i32(int value) {
    return uvec2(uint(value), value < 0 ? 0xffffffffu : 0u);
}

uvec2 cv_u32_multiply(uint lhs, uint rhs) {
    uint high;
    uint low;
    umulExtended(lhs, rhs, high, low);
    return uvec2(low, high);
}

uvec2 cv_i64_multiply_i32_u32(int lhs, uint rhs) {
    const bool negative = lhs < 0;
    const uint magnitude = negative ? uint(-(lhs + 1)) + 1u : uint(lhs);
    const uvec2 product = cv_u32_multiply(magnitude, rhs);
    return negative ? cv_u64_negate(product) : product;
}

uvec2 cv_i64_multiply_i32(int lhs, int rhs) {
    const bool negative = (lhs < 0) != (rhs < 0);
    const uint lhs_magnitude = lhs < 0
        ? uint(-(lhs + 1)) + 1u
        : uint(lhs);
    const uint rhs_magnitude = rhs < 0
        ? uint(-(rhs + 1)) + 1u
        : uint(rhs);
    const uvec2 product = cv_u32_multiply(lhs_magnitude, rhs_magnitude);
    return negative ? cv_u64_negate(product) : product;
}

uvec2 cv_i64_arithmetic_shift_right(uvec2 value, uint shift) {
    if (shift == 0u) return value;
    if (shift >= 64u) {
        return cv_i64_negative(value) ? uvec2(0xffffffffu) : uvec2(0u);
    }
    uvec2 result = cv_u64_shift_right(value, shift);
    if (cv_i64_negative(value)) {
        const uvec2 sign_mask = cv_u64_shift_left(uvec2(0xffffffffu), 64u - shift);
        result |= sign_mask;
    }
    return result;
}

uvec2 cv_i64_shift_right_rne(uvec2 value, uint shift) {
    if (shift == 0u) return value;
    const bool negative = cv_i64_negative(value);
    const uvec2 magnitude = cv_i64_magnitude(value);
    const uvec2 rounded = cv_u64_shift_right_rne(magnitude, shift);
    return negative ? cv_u64_negate(rounded) : rounded;
}

void cv_u128_add_word(inout uvec4 value, uint limb, uint word) {
    uint carry = word;
    for (uint index = limb; index < 4u && carry != 0u; ++index) {
        const uint previous = value[index];
        value[index] = previous + carry;
        carry = value[index] < previous ? 1u : 0u;
    }
}

uvec4 cv_u64_multiply(uvec2 lhs, uvec2 rhs) {
    const uvec2 p00 = cv_u32_multiply(lhs.x, rhs.x);
    const uvec2 p01 = cv_u32_multiply(lhs.x, rhs.y);
    const uvec2 p10 = cv_u32_multiply(lhs.y, rhs.x);
    const uvec2 p11 = cv_u32_multiply(lhs.y, rhs.y);

    uvec4 product = uvec4(0u);
    cv_u128_add_word(product, 0u, p00.x);
    cv_u128_add_word(product, 1u, p00.y);
    cv_u128_add_word(product, 1u, p01.x);
    cv_u128_add_word(product, 2u, p01.y);
    cv_u128_add_word(product, 1u, p10.x);
    cv_u128_add_word(product, 2u, p10.y);
    cv_u128_add_word(product, 2u, p11.x);
    cv_u128_add_word(product, 3u, p11.y);
    return product;
}

uvec2 cv_u64_multiply_u32_low(uvec2 value, uint multiplier) {
    const uvec2 low_product = cv_u32_multiply(value.x, multiplier);
    const uvec2 high_product = cv_u32_multiply(value.y, multiplier);
    return uvec2(low_product.x, low_product.y + high_product.x);
}

uvec2 cv_u64_divide_u32_prepared_floor(
    uvec2 numerator, CvPreparedU32Divider divider, out uint remainder) {
    if (divider.denominator == 0u) {
        remainder = 0u;
        return uvec2(0xffffffffu);
    }
    if (divider.power_of_two_shift !=
        cv_prepared_divider_non_power_of_two_shift) {
        remainder = numerator.x & (divider.denominator - 1u);
        return cv_u64_shift_right(
            numerator, divider.power_of_two_shift);
    }

    const uvec4 reciprocal_product = cv_u64_multiply(
        numerator,
        uvec2(divider.reciprocal_low, divider.reciprocal_high));
    uvec2 quotient = reciprocal_product.zw;
    const uvec2 quotient_product = cv_u64_multiply_u32_low(
        quotient, divider.denominator);
    uvec2 exact_remainder = cv_u64_subtract(
        numerator, quotient_product);

    // floor(2^64 / d) underestimates the exact quotient by at most one.
    // The high remainder limb is therefore either zero or one, and a single
    // exact correction completes the division.
    if (exact_remainder.y != 0u ||
        exact_remainder.x >= divider.denominator) {
        quotient = cv_u64_add(quotient, uvec2(1u, 0u));
        exact_remainder = cv_u64_subtract(
            exact_remainder, uvec2(divider.denominator, 0u));
    }
    remainder = exact_remainder.x;
    return quotient;
}

uvec2 cv_u64_divide_u32_prepared_rne(
    uvec2 numerator, CvPreparedU32Divider divider) {
    uint remainder;
    uvec2 quotient = cv_u64_divide_u32_prepared_floor(
        numerator, divider, remainder);
    if (divider.denominator == 0u) return quotient;

    const uint complement = divider.denominator - remainder;
    if (remainder > complement ||
        (remainder == complement && (quotient.x & 1u) != 0u)) {
        quotient = cv_u64_add(quotient, uvec2(1u, 0u));
    }
    return quotient;
}

uint cv_u64_divide_u32_prepared_limb(
    uint incoming_remainder, uint numerator_limb,
    CvPreparedU32Divider divider, out uint outgoing_remainder) {
    const uvec2 quotient = cv_u64_divide_u32_prepared_floor(
        uvec2(numerator_limb, incoming_remainder),
        divider, outgoing_remainder);
    return quotient.x;
}

uvec3 cv_u96_divide_u32_prepared_floor(
    uvec3 numerator, CvPreparedU32Divider divider, out uint remainder) {
    remainder = 0u;
    uvec3 quotient;
    uint next_remainder;
    quotient.z = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.z, divider, next_remainder);
    remainder = next_remainder;
    quotient.y = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.y, divider, next_remainder);
    remainder = next_remainder;
    quotient.x = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.x, divider, next_remainder);
    remainder = next_remainder;
    return quotient;
}

uvec4 cv_u128_divide_u32_prepared_floor(
    uvec4 numerator, CvPreparedU32Divider divider, out uint remainder) {
    remainder = 0u;
    uvec4 quotient;
    uint next_remainder;
    quotient.w = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.w, divider, next_remainder);
    remainder = next_remainder;
    quotient.z = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.z, divider, next_remainder);
    remainder = next_remainder;
    quotient.y = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.y, divider, next_remainder);
    remainder = next_remainder;
    quotient.x = cv_u64_divide_u32_prepared_limb(
        remainder, numerator.x, divider, next_remainder);
    remainder = next_remainder;
    return quotient;
}

bool cv_u128_less(uvec4 lhs, uvec4 rhs) {
    for (int index = 3; index >= 0; --index) {
        if (lhs[index] != rhs[index]) return lhs[index] < rhs[index];
    }
    return false;
}

bool cv_u128_equal(uvec4 lhs, uvec4 rhs) {
    return all(equal(lhs, rhs));
}

uint cv_u32_subtract_borrow(uint lhs, uint rhs, inout uint borrow) {
    const uint difference = lhs - rhs;
    const uint first_borrow = lhs < rhs ? 1u : 0u;
    const uint result = difference - borrow;
    const uint second_borrow = difference < borrow ? 1u : 0u;
    borrow = (first_borrow | second_borrow);
    return result;
}

uvec4 cv_u128_subtract(uvec4 lhs, uvec4 rhs) {
    uint borrow = 0u;
    uvec4 result;
    result.x = cv_u32_subtract_borrow(lhs.x, rhs.x, borrow);
    result.y = cv_u32_subtract_borrow(lhs.y, rhs.y, borrow);
    result.z = cv_u32_subtract_borrow(lhs.z, rhs.z, borrow);
    result.w = cv_u32_subtract_borrow(lhs.w, rhs.w, borrow);
    return result;
}

uvec4 cv_u128_shift_left_one(uvec4 value) {
    return uvec4(
        value.x << 1u,
        (value.y << 1u) | (value.x >> 31u),
        (value.z << 1u) | (value.y >> 31u),
        (value.w << 1u) | (value.z >> 31u));
}

uvec4 cv_u128_shift_left_two(uvec4 value) {
    return uvec4(
        value.x << 2u,
        (value.y << 2u) | (value.x >> 30u),
        (value.z << 2u) | (value.y >> 30u),
        (value.w << 2u) | (value.z >> 30u));
}

uvec2 cv_u128_sqrt_rne(uvec4 target) {
    uvec2 root = uvec2(0u);
    uvec4 remainder = uvec4(0u);
    for (int pair_index = 63; pair_index >= 0; --pair_index) {
        const uint bit_offset = uint(pair_index) * 2u;
        const uint limb = bit_offset >> 5u;
        const uint limb_offset = bit_offset & 31u;
        const uint next_pair =
            (target[limb] >> limb_offset) & 3u;

        remainder = cv_u128_shift_left_two(remainder);
        remainder.x |= next_pair;

        uvec4 trial = cv_u128_shift_left_two(
            uvec4(root, uvec2(0u)));
        trial.x |= 1u;
        root = cv_u64_shift_left_one(root);
        if (!cv_u128_less(remainder, trial)) {
            remainder = cv_u128_subtract(remainder, trial);
            root.x |= 1u;
        }
    }

    if (all(equal(root, uvec2(0xffffffffu)))) return root;

    uvec4 adjacent_square_distance = cv_u128_shift_left_one(
        uvec4(root, uvec2(0u)));
    adjacent_square_distance.x |= 1u;
    const uvec4 upper_distance = cv_u128_subtract(
        adjacent_square_distance, remainder);
    if (cv_u128_less(upper_distance, remainder) ||
        (cv_u128_equal(remainder, upper_distance) &&
         (root.x & 1u) != 0u)) {
        root = cv_u64_add(root, uvec2(1u, 0u));
    }
    return root;
}

uvec2 cv_i64_apply_magnitude_saturating(uvec2 magnitude, bool negative,
                                         bool overflow) {
    if (negative) {
        const uvec2 limit = uvec2(0u, 0x80000000u);
        if (overflow || cv_u64_less(limit, magnitude)) return limit;
        return cv_u64_negate(magnitude);
    }
    const uvec2 limit = uvec2(0xffffffffu, 0x7fffffffu);
    if (overflow || cv_u64_less(limit, magnitude)) return limit;
    return magnitude;
}

uvec2 cv_i64_scale_pow2_rne_saturating(uvec2 value, int exponent) {
    if (cv_u64_is_zero(value) || exponent == 0) return value;
    const bool negative = cv_i64_negative(value);
    const uvec2 magnitude = cv_i64_magnitude(value);
    if (exponent < 0) {
        const uint shift = uint(-(exponent + 1)) + 1u;
        return cv_i64_apply_magnitude_saturating(
            cv_u64_shift_right_rne(magnitude, shift), negative, false);
    }
    const uint shift = uint(exponent);
    const uvec2 shifted = cv_u64_shift_left(magnitude, shift);
    const bool overflow = shift >= 64u || !cv_u64_equal(
        cv_u64_shift_right(shifted, shift), magnitude);
    return cv_i64_apply_magnitude_saturating(
        shifted, negative, overflow);
}

uvec4 cv_u128_add(uvec4 lhs, uvec4 rhs) {
    uvec4 result;
    uint carry = 0u;
    for (uint index = 0u; index < 4u; ++index) {
        const uint partial = lhs[index] + rhs[index];
        const uint first_carry = partial < lhs[index] ? 1u : 0u;
        result[index] = partial + carry;
        const uint second_carry = result[index] < partial ? 1u : 0u;
        carry = first_carry | second_carry;
    }
    return result;
}

uvec4 cv_u128_negate(uvec4 value) {
    uvec4 result = ~value;
    return cv_u128_add(result, uvec4(1u, 0u, 0u, 0u));
}

bool cv_i128_negative(uvec4 value) {
    return (value.w & 0x80000000u) != 0u;
}

uvec4 cv_i128_magnitude(uvec4 value) {
    return cv_i128_negative(value) ? cv_u128_negate(value) : value;
}

uvec4 cv_i64_multiply_wide(uvec2 lhs, uvec2 rhs) {
    uvec4 product = cv_u64_multiply(
        cv_i64_magnitude(lhs), cv_i64_magnitude(rhs));
    if (cv_i64_negative(lhs) != cv_i64_negative(rhs)) {
        product = cv_u128_negate(product);
    }
    return product;
}

bool cv_i128_add_checked(
    uvec4 lhs, uvec4 rhs, out uvec4 output_value) {
    const bool lhs_negative = cv_i128_negative(lhs);
    const bool rhs_negative = cv_i128_negative(rhs);
    output_value = cv_u128_add(lhs, rhs);
    const bool overflow = lhs_negative == rhs_negative &&
        cv_i128_negative(output_value) != lhs_negative;
    return !overflow;
}

bool cv_i128_add_i64_checked(inout uvec4 accumulator, uvec2 value) {
    const uint sign_extension = cv_i64_negative(value) ? 0xffffffffu : 0u;
    const uvec4 addend = uvec4(
        value.x, value.y, sign_extension, sign_extension);
    uvec4 result;
    const bool succeeded = cv_i128_add_checked(
        accumulator, addend, result);
    accumulator = result;
    return succeeded;
}

bool cv_i128_divide_i64_by_u32_prepared_rne_checked(
    uvec4 numerator_bits, CvPreparedU32Divider divider,
    out uvec2 output_bits) {
    if (divider.denominator == 0u) {
        output_bits = uvec2(0u);
        return false;
    }

    const bool negative = cv_i128_negative(numerator_bits);
    const uvec4 numerator = cv_i128_magnitude(numerator_bits);
    uint remainder;
    const uvec4 wide_quotient = cv_u128_divide_u32_prepared_floor(
        numerator, divider, remainder);
    uvec2 quotient = wide_quotient.xy;
    bool overflow = wide_quotient.z != 0u || wide_quotient.w != 0u;

    const uint complement = divider.denominator - remainder;
    const bool round_up = remainder > complement ||
        (remainder == complement && (quotient.x & 1u) != 0u);
    if (round_up) {
        quotient.x += 1u;
        if (quotient.x == 0u) {
            quotient.y += 1u;
            if (quotient.y == 0u) overflow = true;
        }
    }

    const uvec2 positive_limit = uvec2(0xffffffffu, 0x7fffffffu);
    const uvec2 negative_limit = uvec2(0u, 0x80000000u);
    const uvec2 signed_limit = negative ? negative_limit : positive_limit;
    if (cv_u64_less(signed_limit, quotient)) overflow = true;
    output_bits = cv_i64_apply_magnitude_saturating(
        quotient, negative, overflow);
    return !overflow;
}

bool cv_i128_divide_i64_by_u32_rne_checked(
    uvec4 numerator_bits, uint divisor, out uvec2 output_bits) {
    if (divisor == 0u) {
        output_bits = uvec2(0u);
        return false;
    }
    const bool negative = cv_i128_negative(numerator_bits);
    const uvec4 numerator = cv_i128_magnitude(numerator_bits);
    const uvec2 denominator = uvec2(divisor, 0u);
    uvec2 remainder = uvec2(0u);
    uvec2 quotient = uvec2(0u);
    bool overflow = false;
    for (int bit = 127; bit >= 0; --bit) {
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

    const uvec2 positive_limit = uvec2(0xffffffffu, 0x7fffffffu);
    const uvec2 negative_limit = uvec2(0u, 0x80000000u);
    const uvec2 signed_limit = negative ? negative_limit : positive_limit;
    if (cv_u64_less(signed_limit, quotient)) overflow = true;
    output_bits = cv_i64_apply_magnitude_saturating(
        quotient, negative, overflow);
    return !overflow;
}

bool cv_i128_divide_q64_by_u32_prepared_rne(
    uvec4 numerator_bits, CvPreparedU32Divider divider,
    out uvec2 output_bits) {
    if (divider.denominator == 0u) {
        output_bits = uvec2(0u);
        return false;
    }

    const bool negative = cv_i128_negative(numerator_bits);
    const uvec4 numerator = cv_i128_magnitude(numerator_bits);
    uint high_remainder;
    const uvec3 wide_quotient = cv_u96_divide_u32_prepared_floor(
        numerator.yzw, divider, high_remainder);
    uvec2 quotient = wide_quotient.xy;
    bool overflow = wide_quotient.z != 0u;

    // Dividing a Q64-scaled integer by d is equivalent to dividing the upper
    // 96 bits by d. The untouched low limb remains the low half of the exact
    // remainder against the full denominator d * 2^32.
    const uvec2 remainder = uvec2(numerator.x, high_remainder);
    const uvec2 denominator = uvec2(0u, divider.denominator);
    const uvec2 complement = cv_u64_subtract(denominator, remainder);
    const bool round_up = cv_u64_less(complement, remainder) ||
        (cv_u64_equal(remainder, complement) &&
         (quotient.x & 1u) != 0u);
    if (round_up) {
        quotient.x += 1u;
        if (quotient.x == 0u) {
            quotient.y += 1u;
            if (quotient.y == 0u) overflow = true;
        }
    }
    output_bits = cv_i64_apply_magnitude_saturating(
        quotient, negative, overflow);
    return true;
}

bool cv_i128_divide_q64_by_u32_rne(uvec4 numerator_bits, uint divisor,
                                    out uvec2 output_bits) {
    if (divisor == 0u) {
        output_bits = uvec2(0u);
        return false;
    }
    const bool negative = cv_i128_negative(numerator_bits);
    const uvec4 numerator = cv_i128_magnitude(numerator_bits);
    const uvec2 denominator = uvec2(0u, divisor);
    uvec2 remainder = uvec2(0u);
    uvec2 quotient = uvec2(0u);
    bool overflow = false;
    for (int bit = 127; bit >= 0; --bit) {
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
    return true;
}


#endif
