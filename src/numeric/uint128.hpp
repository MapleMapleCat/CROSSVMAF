#pragma once
#include <cstdint>
#include <limits>

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#endif

namespace crossvmaf::numeric::detail {

struct UInt128 {
    std::uint64_t hi{};
    std::uint64_t lo{};
};

[[nodiscard]] constexpr std::uint64_t magnitude(std::int64_t value) noexcept {
    const auto bits = static_cast<std::uint64_t>(value);
    return value < 0 ? (~bits + 1U) : bits;
}

[[nodiscard]] constexpr UInt128 multiply_u64(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    const std::uint64_t lhs_lo = static_cast<std::uint32_t>(lhs);
    const std::uint64_t lhs_hi = lhs >> 32U;
    const std::uint64_t rhs_lo = static_cast<std::uint32_t>(rhs);
    const std::uint64_t rhs_hi = rhs >> 32U;
    const std::uint64_t p00 = lhs_lo * rhs_lo;
    const std::uint64_t p01 = lhs_lo * rhs_hi;
    const std::uint64_t p10 = lhs_hi * rhs_lo;
    const std::uint64_t p11 = lhs_hi * rhs_hi;
    const std::uint64_t middle =
        (p00 >> 32U) + static_cast<std::uint32_t>(p01) + static_cast<std::uint32_t>(p10);
    const std::uint64_t lo = (middle << 32U) | static_cast<std::uint32_t>(p00);
    const std::uint64_t hi = p11 + (p01 >> 32U) + (p10 >> 32U) + (middle >> 32U);
    return UInt128{hi, lo};
}

[[nodiscard]] constexpr UInt128 add(UInt128 lhs, UInt128 rhs) noexcept {
    const std::uint64_t lo = lhs.lo + rhs.lo;
    return UInt128{lhs.hi + rhs.hi + (lo < lhs.lo ? 1U : 0U), lo};
}

[[nodiscard]] constexpr UInt128 negate(UInt128 value) noexcept {
    const std::uint64_t lo = ~value.lo + 1U;
    return UInt128{~value.hi + (lo == 0U ? 1U : 0U), lo};
}

[[nodiscard]] constexpr int compare(UInt128 lhs, UInt128 rhs) noexcept {
    if (lhs.hi != rhs.hi) return lhs.hi < rhs.hi ? -1 : 1;
    if (lhs.lo != rhs.lo) return lhs.lo < rhs.lo ? -1 : 1;
    return 0;
}

[[nodiscard]] constexpr UInt128 subtract(UInt128 lhs, UInt128 rhs) noexcept {
    const std::uint64_t lo = lhs.lo - rhs.lo;
    const std::uint64_t borrow = lhs.lo < rhs.lo ? 1U : 0U;
    return UInt128{lhs.hi - rhs.hi - borrow, lo};
}

[[nodiscard]] inline std::uint64_t divide_u128_u64(UInt128 numerator, std::uint64_t denominator,
                                                   std::uint64_t *remainder,
                                                   bool *overflow) noexcept {
    if (numerator.hi >= denominator) {
        *overflow = true;
        *remainder = 0;
        return std::numeric_limits<std::uint64_t>::max();
    }
#if defined(_MSC_VER) && defined(_M_X64) && !defined(__clang__)
    *overflow = false;
    return _udiv128(numerator.hi, numerator.lo, denominator, remainder);
#elif defined(__clang__) && defined(_M_X64)
    // clang-cl has no _udiv128, and Windows does not provide __udivti3.
    // The guard above guarantees that the hardware quotient fits in 64 bits.
    std::uint64_t quotient;
    std::uint64_t division_remainder;
    __asm__("divq %[denominator]"
            : "=a"(quotient), "=d"(division_remainder)
            : "a"(numerator.lo), "d"(numerator.hi), [denominator] "r"(denominator)
            : "cc");
    *overflow = false;
    *remainder = division_remainder;
    return quotient;
#elif defined(__SIZEOF_INT128__)
    __extension__ using NativeUInt128 = unsigned __int128;
    const auto wide = (static_cast<NativeUInt128>(numerator.hi) << 64U) |
                      static_cast<NativeUInt128>(numerator.lo);
    *overflow = false;
    *remainder = static_cast<std::uint64_t>(wide % denominator);
    return static_cast<std::uint64_t>(wide / denominator);
#else
    *overflow = false;
    std::uint64_t quotient = 0;
    UInt128 rem{};
    for (int bit = 127; bit >= 0; --bit) {
        rem.hi = (rem.hi << 1U) | (rem.lo >> 63U);
        rem.lo <<= 1U;
        const std::uint64_t incoming =
            bit >= 64 ? ((numerator.hi >> static_cast<unsigned>(bit - 64)) & 1U)
                      : ((numerator.lo >> static_cast<unsigned>(bit)) & 1U);
        rem.lo |= incoming;
        if (rem.hi != 0 || rem.lo >= denominator) {
            rem = subtract(rem, UInt128{0, denominator});
            if (bit < 64) {
                quotient |= std::uint64_t{1} << static_cast<unsigned>(bit);
            } else {
                *overflow = true;
            }
        }
    }
    *remainder = rem.lo;
    return quotient;
#endif
}

} // namespace crossvmaf::numeric::detail
