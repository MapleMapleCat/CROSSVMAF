#pragma once
#include "numeric/uint128.hpp"
#include <cstdint>
#include <limits>
#include <bit>

namespace crossvmaf::numeric::detail {

[[nodiscard]] constexpr std::uint64_t divide_u64_rne(std::uint64_t numerator,
                                                     std::uint64_t denominator) noexcept {
    if (denominator == 0U) return 0U;
    std::uint64_t quotient = numerator / denominator;
    const std::uint64_t remainder = numerator % denominator;
    const std::uint64_t complement = denominator - remainder;
    if (remainder > complement || (remainder == complement && (quotient & 1U) != 0U)) {
        ++quotient;
    }
    return quotient;
}

[[nodiscard]] constexpr std::int64_t apply_sign_saturating(std::uint64_t value, bool negative,
                                                           bool overflow = false) noexcept {
    constexpr std::uint64_t positive_limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    constexpr std::uint64_t negative_limit = positive_limit + 1U;
    if (!negative) {
        return overflow || value > positive_limit ? std::numeric_limits<std::int64_t>::max()
                                                  : static_cast<std::int64_t>(value);
    }
    if (overflow || value >= negative_limit) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return -static_cast<std::int64_t>(value);
}

[[nodiscard]] constexpr std::uint64_t shift_right_rne(std::uint64_t value,
                                                      unsigned shift) noexcept {
    if (shift == 0U) return value;
    if (shift > 64U) return 0U;
    if (shift == 64U) {
        constexpr std::uint64_t halfway = std::uint64_t{1} << 63U;
        return value > halfway ? 1U : 0U;
    }
    std::uint64_t quotient = value >> shift;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    const std::uint64_t remainder = value & mask;
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (quotient & 1U) != 0U)) {
        ++quotient;
    }
    return quotient;
}

[[nodiscard]] inline std::uint64_t rounded_quotient(UInt128 numerator, std::uint64_t denominator,
                                                    bool *overflow) noexcept {
    std::uint64_t remainder = 0;
    std::uint64_t quotient = divide_u128_u64(numerator, denominator, &remainder, overflow);
    if (*overflow) return quotient;

    const std::uint64_t complement = denominator - remainder;
    const bool round_up =
        remainder > complement || (remainder == complement && (quotient & 1U) != 0U);
    if (round_up) {
        if (quotient == std::numeric_limits<std::uint64_t>::max()) {
            *overflow = true;
        } else {
            ++quotient;
        }
    }
    return quotient;
}

[[nodiscard]] constexpr UInt128 shifted_q32_numerator(std::uint64_t value) noexcept {
    return UInt128{value >> 32U, value << 32U};
}

} // namespace crossvmaf::numeric::detail

namespace crossvmaf::numeric {

inline constexpr std::uint32_t prepared_divider_non_power_of_two_shift =
    std::numeric_limits<std::uint32_t>::max();

// Layout is mirrored by CvPreparedU32Divider in integer.glsl so instances can
// be copied directly into push constants.
struct PreparedU32Divider final {
    std::uint32_t denominator{};
    std::uint32_t reciprocal_low{};
    std::uint32_t reciprocal_high{};
    std::uint32_t power_of_two_shift{prepared_divider_non_power_of_two_shift};
};

[[nodiscard]] constexpr PreparedU32Divider prepare_u32_divider(std::uint32_t denominator) noexcept {
    PreparedU32Divider divider{};
    divider.denominator = denominator;
    if (denominator == 0U) return divider;

    if (std::has_single_bit(denominator)) {
        divider.power_of_two_shift = std::countr_zero(denominator);
        return divider;
    }

    constexpr std::uint64_t maximum = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t reciprocal = maximum / denominator;
    if (maximum % denominator == denominator - 1U) {
        ++reciprocal;
    }
    divider.reciprocal_low = static_cast<std::uint32_t>(reciprocal);
    divider.reciprocal_high = static_cast<std::uint32_t>(reciprocal >> 32U);
    return divider;
}

static_assert(sizeof(PreparedU32Divider) == 4U * sizeof(std::uint32_t));

} // namespace crossvmaf::numeric
