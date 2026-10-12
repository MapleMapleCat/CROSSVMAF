#include "numeric/q32.hpp"
#include "numeric/rounding.hpp"
#include "numeric/uint128.hpp"
#include <cstdint>
#include <limits>
#include "crossvmaf/value.hpp"

namespace crossvmaf::numeric {

Result<Q32_32> Q32_32::from_ratio(std::int64_t numerator, std::int64_t denominator) noexcept {
    if (denominator == 0) {
        return Result<Q32_32>::failure(
            {ErrorCode::invalid_argument, "fixed-point division by zero"});
    }
    const bool negative = (numerator < 0) != (denominator < 0);
    bool overflow = false;
    const std::uint64_t quotient =
        detail::rounded_quotient(detail::shifted_q32_numerator(detail::magnitude(numerator)),
                                 detail::magnitude(denominator), &overflow);
    if (overflow) {
        return Result<Q32_32>::success(from_raw(negative
                                                    ? std::numeric_limits<std::int64_t>::min()
                                                    : std::numeric_limits<std::int64_t>::max()));
    }
    return Result<Q32_32>::success(from_raw(detail::apply_sign_saturating(quotient, negative)));
}

Q32_32 Q32_32::add_saturating(Q32_32 lhs, Q32_32 rhs) noexcept {
    if (rhs.raw_ > 0 && lhs.raw_ > std::numeric_limits<std::int64_t>::max() - rhs.raw_) {
        return from_raw(std::numeric_limits<std::int64_t>::max());
    }
    if (rhs.raw_ < 0 && lhs.raw_ < std::numeric_limits<std::int64_t>::min() - rhs.raw_) {
        return from_raw(std::numeric_limits<std::int64_t>::min());
    }
    return from_raw(lhs.raw_ + rhs.raw_);
}

Q32_32 Q32_32::subtract_saturating(Q32_32 lhs, Q32_32 rhs) noexcept {
    if (rhs.raw_ == std::numeric_limits<std::int64_t>::min()) {
        return lhs.raw_ >= 0 ? from_raw(std::numeric_limits<std::int64_t>::max())
                             : from_raw(lhs.raw_ - rhs.raw_);
    }
    return add_saturating(lhs, from_raw(-rhs.raw_));
}

Q32_32 Q32_32::average(Q32_32 lhs, Q32_32 rhs) noexcept {
    const std::int64_t lhs_half = lhs.raw_ / 2;
    const std::int64_t rhs_half = rhs.raw_ / 2;
    std::int64_t quotient = lhs_half + rhs_half;
    const std::int64_t remainder = lhs.raw_ % 2 + rhs.raw_ % 2;
    if (remainder == 2) {
        ++quotient;
    } else if (remainder == -2) {
        --quotient;
    } else if (remainder == 1 && quotient % 2 != 0) {
        ++quotient;
    } else if (remainder == -1 && quotient % 2 != 0) {
        --quotient;
    }
    return from_raw(quotient);
}

Q32_32 Q32_32::multiply(Q32_32 lhs, Q32_32 rhs) noexcept {
    const bool negative = (lhs.raw_ < 0) != (rhs.raw_ < 0);
    const detail::UInt128 product =
        detail::multiply_u64(detail::magnitude(lhs.raw_), detail::magnitude(rhs.raw_));
    if (product.hi > 0xFFFF'FFFFULL) {
        return from_raw(negative ? std::numeric_limits<std::int64_t>::min()
                                 : std::numeric_limits<std::int64_t>::max());
    }
    std::uint64_t quotient = (product.hi << 32U) | (product.lo >> 32U);
    const std::uint32_t remainder = static_cast<std::uint32_t>(product.lo);
    if (remainder > 0x8000'0000U || (remainder == 0x8000'0000U && (quotient & 1U) != 0U)) {
        if (quotient == std::numeric_limits<std::uint64_t>::max()) {
            return from_raw(negative ? std::numeric_limits<std::int64_t>::min()
                                     : std::numeric_limits<std::int64_t>::max());
        }
        ++quotient;
    }
    return from_raw(detail::apply_sign_saturating(quotient, negative));
}

Result<Q32_32> Q32_32::divide(Q32_32 lhs, Q32_32 rhs) noexcept {
    return from_ratio(lhs.raw_, rhs.raw_);
}

} // namespace crossvmaf::numeric

namespace crossvmaf {

double Q32Value::value() const noexcept {
    return numeric::Q32_32::from_raw(raw_q32_32).to_double();
}

std::string Q32Value::decimal(unsigned places) const {
    return numeric::Q32_32::from_raw(raw_q32_32).to_string(places);
}

} // namespace crossvmaf
