#include "numeric/q32.hpp"
#include "numeric/uint128.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace crossvmaf::numeric {

Result<Q32_32> Q32_32::parse_decimal(const std::string &text) noexcept {
    if (text.empty()) {
        return Result<Q32_32>::failure({ErrorCode::invalid_argument, "empty fixed-point decimal"});
    }

    std::size_t cursor = 0;
    bool negative = false;
    if (text[cursor] == '+' || text[cursor] == '-') {
        negative = text[cursor] == '-';
        ++cursor;
    }

    std::uint64_t numerator = 0;
    std::uint64_t denominator = 1;
    bool seen_digit = false;
    bool seen_dot = false;
    std::size_t fractional_digits = 0;
    for (; cursor < text.size(); ++cursor) {
        const char character = text[cursor];
        if (character == '.' && !seen_dot) {
            seen_dot = true;
            continue;
        }
        if (character < '0' || character > '9') {
            return Result<Q32_32>::failure(
                {ErrorCode::invalid_argument, "invalid fixed-point decimal"});
        }
        seen_digit = true;
        const unsigned digit = static_cast<unsigned>(character - '0');
        if (numerator > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
            return Result<Q32_32>::failure(
                {ErrorCode::invalid_argument, "fixed-point decimal is out of range"});
        }
        numerator = numerator * 10U + digit;
        if (seen_dot) {
            if (++fractional_digits > 9U) {
                return Result<Q32_32>::failure(
                    {ErrorCode::invalid_argument,
                     "fixed-point decimal supports at most nine fractional digits"});
            }
            denominator *= 10U;
        }
    }

    if (!seen_digit ||
        numerator > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return Result<Q32_32>::failure(
            {ErrorCode::invalid_argument, "fixed-point decimal is out of range"});
    }
    const auto signed_numerator =
        negative ? -static_cast<std::int64_t>(numerator) : static_cast<std::int64_t>(numerator);
    return from_ratio(signed_numerator, static_cast<std::int64_t>(denominator));
}

double Q32_32::to_double() const noexcept {
    return std::ldexp(static_cast<double>(raw_), -static_cast<int>(fractional_bits));
}

std::string Q32_32::to_string(unsigned decimal_places) const {
    decimal_places = std::min(decimal_places, 9U);
    const bool negative = raw_ < 0;
    const std::uint64_t absolute = detail::magnitude(raw_);
    std::uint64_t integer = absolute >> fractional_bits;
    const std::uint64_t fraction = absolute & 0xFFFF'FFFFULL;

    std::uint64_t scale = 1;
    for (unsigned i = 0; i < decimal_places; ++i)
        scale *= 10U;

    const std::uint64_t product = fraction * scale;
    std::uint64_t decimal = product >> fractional_bits;
    const std::uint64_t remainder = product & 0xFFFF'FFFFULL;
    if (remainder > 0x8000'0000ULL || (remainder == 0x8000'0000ULL && (decimal & 1U) != 0U)) {
        ++decimal;
        if (decimal == scale) {
            decimal = 0;
            ++integer;
        }
    }

    std::ostringstream stream;
    if (negative && absolute != 0) stream << '-';
    stream << integer;
    if (decimal_places != 0U) {
        stream << '.' << std::setw(static_cast<int>(decimal_places)) << std::setfill('0')
               << decimal;
    }
    return stream.str();
}

} // namespace crossvmaf::numeric
