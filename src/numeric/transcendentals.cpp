#include "numeric/q32.hpp"
#include "numeric/rounding.hpp"
#include "numeric/uint128.hpp"
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace crossvmaf::numeric {
namespace {

constexpr std::array<std::uint64_t, 32> exp2_fraction_constants{{
    0x000000016a09e668ULL, 0x00000001306fe0a3ULL, 0x00000001172b83c8ULL, 0x000000010b5586d0ULL,
    0x00000001059b0d31ULL, 0x0000000102c9a3e7ULL, 0x000000010163daa0ULL, 0x0000000100b1afa6ULL,
    0x000000010058c86eULL, 0x00000001002c605eULL, 0x0000000100162f39ULL, 0x00000001000b175fULL,
    0x0000000100058ba0ULL, 0x000000010002c5ccULL, 0x00000001000162e5ULL, 0x000000010000b172ULL,
    0x00000001000058b9ULL, 0x0000000100002c5dULL, 0x000000010000162eULL, 0x0000000100000b17ULL,
    0x000000010000058cULL, 0x00000001000002c6ULL, 0x0000000100000163ULL, 0x00000001000000b1ULL,
    0x0000000100000059ULL, 0x000000010000002cULL, 0x0000000100000016ULL, 0x000000010000000bULL,
    0x0000000100000006ULL, 0x0000000100000003ULL, 0x0000000100000001ULL, 0x0000000100000001ULL,
}};

[[nodiscard]] std::uint64_t integer_sqrt_rne(detail::UInt128 target) noexcept {
    std::uint64_t low = 0;
    std::uint64_t high = std::numeric_limits<std::uint64_t>::max();
    while (low < high) {
        const std::uint64_t midpoint = low + ((high - low) >> 1U) + 1U;
        if (detail::compare(detail::multiply_u64(midpoint, midpoint), target) <= 0) {
            low = midpoint;
        } else {
            high = midpoint - 1U;
        }
    }

    if (low == std::numeric_limits<std::uint64_t>::max()) return low;
    const detail::UInt128 lower_square = detail::multiply_u64(low, low);
    const detail::UInt128 upper_square = detail::multiply_u64(low + 1U, low + 1U);
    const detail::UInt128 lower_distance = detail::subtract(target, lower_square);
    const detail::UInt128 upper_distance = detail::subtract(upper_square, target);
    const int distance_compare = detail::compare(lower_distance, upper_distance);
    if (distance_compare > 0 || (distance_compare == 0 && (low & 1U) != 0U)) {
        ++low;
    }
    return low;
}

} // namespace

namespace detail {

std::int64_t log2_positive_raw(std::uint64_t raw) noexcept {
    const unsigned msb = std::bit_width(raw) - 1U;
    const std::int64_t exponent =
        static_cast<std::int64_t>(msb) - static_cast<std::int64_t>(Q32_32::fractional_bits);
    std::uint64_t normalized = msb > Q32_32::fractional_bits
                                   ? raw >> (msb - Q32_32::fractional_bits)
                                   : raw << (Q32_32::fractional_bits - msb);

    std::uint64_t fraction = 0;
    for (unsigned bit = 0; bit < Q32_32::fractional_bits; ++bit) {
        const Q32_32 squared =
            Q32_32::multiply(Q32_32::from_raw(static_cast<std::int64_t>(normalized)),
                             Q32_32::from_raw(static_cast<std::int64_t>(normalized)));
        normalized = static_cast<std::uint64_t>(squared.raw());
        if (normalized >= (std::uint64_t{2} << Q32_32::fractional_bits)) {
            normalized >>= 1U;
            fraction |= std::uint64_t{1} << (Q32_32::fractional_bits - 1U - bit);
        }
    }

    return exponent * Q32_32::one_raw + static_cast<std::int64_t>(fraction);
}

} // namespace detail

Result<Q32_32> Q32_32::log2(Q32_32 value) noexcept {
    if (value.raw_ <= 0) {
        return Result<Q32_32>::failure(
            {ErrorCode::invalid_argument, "fixed-point log2 requires a positive value"});
    }

    return Result<Q32_32>::success(
        from_raw(detail::log2_positive_raw(static_cast<std::uint64_t>(value.raw_))));
}

Q32_32 Q32_32::exp2(Q32_32 exponent) noexcept {
    std::int64_t integer_part = exponent.raw_ / one_raw;
    std::int64_t fractional_raw = exponent.raw_ % one_raw;
    if (fractional_raw < 0) {
        --integer_part;
        fractional_raw += one_raw;
    }

    Q32_32 result = from_raw(one_raw);
    const auto fraction = static_cast<std::uint32_t>(fractional_raw);
    for (std::size_t index = 0; index < exp2_fraction_constants.size(); ++index) {
        const unsigned bit = 31U - static_cast<unsigned>(index);
        if ((fraction & (std::uint32_t{1} << bit)) != 0U) {
            result = multiply(result,
                              from_raw(static_cast<std::int64_t>(exp2_fraction_constants[index])));
        }
    }

    const auto positive_raw = static_cast<std::uint64_t>(result.raw_);
    if (integer_part >= 0) {
        if (integer_part >= 63) {
            return from_raw(std::numeric_limits<std::int64_t>::max());
        }
        const auto shift = static_cast<unsigned>(integer_part);
        constexpr std::uint64_t limit =
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        if (positive_raw > (limit >> shift)) {
            return from_raw(std::numeric_limits<std::int64_t>::max());
        }
        return from_raw(static_cast<std::int64_t>(positive_raw << shift));
    }

    const auto shift = static_cast<std::uint64_t>(-integer_part);
    if (shift >= 64U) return from_raw(0);
    const auto shift_bits = static_cast<unsigned>(shift);
    std::uint64_t quotient = positive_raw >> shift_bits;
    if (shift_bits != 0U) {
        const std::uint64_t mask = (std::uint64_t{1} << shift_bits) - 1U;
        const std::uint64_t remainder = positive_raw & mask;
        const std::uint64_t halfway = std::uint64_t{1} << (shift_bits - 1U);
        if (remainder > halfway || (remainder == halfway && (quotient & 1U) != 0U)) {
            ++quotient;
        }
    }
    return from_raw(static_cast<std::int64_t>(quotient));
}

Result<Q32_32> Q32_32::sqrt(Q32_32 value) noexcept {
    if (value.raw_ < 0) {
        return Result<Q32_32>::failure(
            {ErrorCode::invalid_argument, "fixed-point square root of a negative value"});
    }
    const detail::UInt128 target =
        detail::shifted_q32_numerator(static_cast<std::uint64_t>(value.raw_));
    const std::uint64_t root = integer_sqrt_rne(target);
    return Result<Q32_32>::success(from_raw(detail::apply_sign_saturating(root, false)));
}

Result<Q32_32> Q32_32::cbrt(Q32_32 value) noexcept {
    if (value.raw_ == 0) {
        return Result<Q32_32>::success(Q32_32{});
    }

    const bool negative = value.raw_ < 0;
    const std::uint64_t absolute = detail::magnitude(value.raw_);
    const Q32_32 logarithm = from_raw(detail::log2_positive_raw(absolute));
    const auto exponent = divide(logarithm, from_integer(3));
    if (!exponent) return Result<Q32_32>::failure(exponent.error());
    const Q32_32 root = exp2(exponent.value());
    if (!negative) return Result<Q32_32>::success(root);
    return Result<Q32_32>::success(from_raw(-root.raw_));
}

} // namespace crossvmaf::numeric
