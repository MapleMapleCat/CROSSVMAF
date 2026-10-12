#pragma once
#include "crossvmaf/error.hpp"
#include <compare>
#include <cstdint>
#include <string>

namespace crossvmaf::numeric {

// Internal deterministic Q32.32 arithmetic used by models and providers.
class Q32_32 final {
  public:
    static constexpr unsigned fractional_bits = 32;
    static constexpr std::int64_t one_raw = std::int64_t{1} << fractional_bits;

    constexpr Q32_32() noexcept = default;

    [[nodiscard]] static constexpr Q32_32 from_raw(std::int64_t raw) noexcept {
        return Q32_32(raw);
    }

    [[nodiscard]] static constexpr Q32_32 from_integer(std::int32_t value) noexcept {
        return Q32_32(static_cast<std::int64_t>(value) * one_raw);
    }

    [[nodiscard]] static Result<Q32_32> from_ratio(std::int64_t numerator,
                                                   std::int64_t denominator) noexcept;
    [[nodiscard]] static Result<Q32_32> parse_decimal(const std::string &text) noexcept;

    [[nodiscard]] constexpr std::int64_t raw() const noexcept { return raw_; }
    [[nodiscard]] double to_double() const noexcept;
    [[nodiscard]] std::string to_string(unsigned decimal_places = 9) const;

    [[nodiscard]] static Q32_32 add_saturating(Q32_32 lhs, Q32_32 rhs) noexcept;
    [[nodiscard]] static Q32_32 subtract_saturating(Q32_32 lhs, Q32_32 rhs) noexcept;
    [[nodiscard]] static Q32_32 average(Q32_32 lhs, Q32_32 rhs) noexcept;
    [[nodiscard]] static Q32_32 multiply(Q32_32 lhs, Q32_32 rhs) noexcept;
    [[nodiscard]] static Result<Q32_32> divide(Q32_32 lhs, Q32_32 rhs) noexcept;
    [[nodiscard]] static Result<Q32_32> sqrt(Q32_32 value) noexcept;
    [[nodiscard]] static Result<Q32_32> cbrt(Q32_32 value) noexcept;
    [[nodiscard]] static Result<Q32_32> log2(Q32_32 value) noexcept;
    [[nodiscard]] static Q32_32 exp2(Q32_32 exponent) noexcept;

    friend constexpr auto operator<=>(Q32_32, Q32_32) noexcept = default;

  private:
    explicit constexpr Q32_32(std::int64_t raw) noexcept : raw_(raw) {}

    std::int64_t raw_{};
};

} // namespace crossvmaf::numeric
