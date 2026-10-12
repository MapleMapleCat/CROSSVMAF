#pragma once
#include "metrics/speed_chroma/speed_chroma.hpp"
#include "numeric/q32.hpp"
#include "numeric/rounding.hpp"
#include "numeric/uint128.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace crossvmaf::metrics::speed_chroma::reference::linear {

using SquareMatrix = std::array<numeric::Q32_32, elements_per_block * elements_per_block>;

inline constexpr std::size_t householder_reflector_count = elements_per_block - 2U;
inline constexpr std::size_t householder_tail_count =
    (elements_per_block - 1U) * (elements_per_block - 2U) / 2U;

namespace detail {

struct Q16_48 {
    std::int64_t raw{};
};

using LinearSquareMatrix = std::array<Q16_48, elements_per_block * elements_per_block>;
using LinearVector = std::array<Q16_48, elements_per_block>;
using LinearSubdiagonal = std::array<Q16_48, elements_per_block - 1U>;

using UInt128 = crossvmaf::numeric::detail::UInt128;
using crossvmaf::numeric::detail::apply_sign_saturating;
using crossvmaf::numeric::detail::compare;
using crossvmaf::numeric::detail::divide_u128_u64;
using crossvmaf::numeric::detail::magnitude;
using crossvmaf::numeric::detail::multiply_u64;
using crossvmaf::numeric::detail::rounded_quotient;
using crossvmaf::numeric::detail::shift_right_rne;
using crossvmaf::numeric::detail::subtract;

inline constexpr unsigned fractional_bits = 48U;
inline constexpr std::int64_t one_raw = std::int64_t{1} << fractional_bits;
inline constexpr std::uint64_t fractional_mask = (std::uint64_t{1} << fractional_bits) - 1U;
inline constexpr std::uint64_t fractional_half = std::uint64_t{1} << (fractional_bits - 1U);
inline constexpr int inverse_quadratic_guard_bits = 16;
inline constexpr std::uint64_t positive_raw_limit =
    static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
inline constexpr Q16_48 zero{};
inline constexpr Q16_48 one{one_raw};
inline constexpr Q16_48 half{one_raw / 2};

[[nodiscard]] constexpr std::size_t matrix_index(std::size_t row, std::size_t column) noexcept {
    return row * elements_per_block + column;
}

[[nodiscard]] constexpr std::size_t reflector_tail_offset(std::size_t reflector_index) noexcept {
    return reflector_index * (2U * elements_per_block - reflector_index - 3U) / 2U;
}

[[nodiscard]] inline bool scale_power_of_two_checked(Q16_48 value, int exponent,
                                                     Q16_48 &output) noexcept {
    if (value.raw == 0 || exponent == 0) {
        output = value;
        return true;
    }
    const bool negative = value.raw < 0;
    const std::uint64_t absolute = magnitude(value.raw);
    if (exponent < 0) {
        const unsigned shift = static_cast<unsigned>(-(exponent + 1)) + 1U;
        output = {apply_sign_saturating(shift_right_rne(absolute, shift), negative)};
        return true;
    }
    const unsigned shift = static_cast<unsigned>(exponent);
    const std::uint64_t signed_limit = negative ? positive_raw_limit + 1U : positive_raw_limit;
    if (shift >= 64U || absolute > (signed_limit >> shift)) return false;
    output = {apply_sign_saturating(absolute << shift, negative)};
    return true;
}

[[nodiscard]] inline Q16_48 scale_power_of_two(Q16_48 value, int exponent) noexcept {
    Q16_48 output{};
    if (scale_power_of_two_checked(value, exponent, output)) return output;
    return {value.raw < 0 ? std::numeric_limits<std::int64_t>::min()
                          : std::numeric_limits<std::int64_t>::max()};
}

template <typename Range> [[nodiscard]] int normalization_exponent(const Range &values) noexcept {
    std::uint64_t largest = 0U;
    for (const Q16_48 value : values) {
        largest = std::max(largest, magnitude(value.raw));
    }
    if (largest == 0U) return 0;
    const unsigned highest_bit = 63U - std::countl_zero(largest);
    return static_cast<int>(fractional_bits) - static_cast<int>(highest_bit);
}

[[nodiscard]] inline int normalization_exponent(Q16_48 value) noexcept {
    return normalization_exponent(std::array<Q16_48, 1U>{value});
}

[[nodiscard]] inline int
public_normalization_exponent_from_magnitude(std::uint64_t largest_magnitude) noexcept {
    if (largest_magnitude == 0U) return 0;
    const unsigned highest_bit = 63U - std::countl_zero(largest_magnitude);
    return static_cast<int>(numeric::Q32_32::fractional_bits) - static_cast<int>(highest_bit);
}

template <typename Range>
[[nodiscard]] int public_normalization_exponent(const Range &values) noexcept {
    std::uint64_t largest_magnitude = 0U;
    for (const numeric::Q32_32 value : values) {
        largest_magnitude = std::max(largest_magnitude, magnitude(value.raw()));
    }
    return public_normalization_exponent_from_magnitude(largest_magnitude);
}

[[nodiscard]] inline bool convert_public_with_exponent(numeric::Q32_32 value,
                                                       int normalization_exponent_value,
                                                       Q16_48 &output) noexcept {
    constexpr int fractional_difference =
        static_cast<int>(fractional_bits) - static_cast<int>(numeric::Q32_32::fractional_bits);
    if (normalization_exponent_value > std::numeric_limits<int>::max() - fractional_difference) {
        return false;
    }
    return scale_power_of_two_checked(Q16_48{value.raw()},
                                      normalization_exponent_value + fractional_difference, output);
}

[[nodiscard]] inline bool convert_to_public_scaled(Q16_48 value, int exponent,
                                                   numeric::Q32_32 &output) noexcept {
    constexpr int fractional_difference =
        static_cast<int>(fractional_bits) - static_cast<int>(numeric::Q32_32::fractional_bits);
    if (exponent < std::numeric_limits<int>::min() + fractional_difference) {
        return false;
    }
    Q16_48 public_raw{};
    if (!scale_power_of_two_checked(value, exponent - fractional_difference, public_raw)) {
        return false;
    }
    output = numeric::Q32_32::from_raw(public_raw.raw);
    return true;
}

[[nodiscard]] inline std::uint64_t integer_sqrt_rne(UInt128 target) noexcept {
    std::uint64_t low = 0U;
    std::uint64_t high = std::numeric_limits<std::uint64_t>::max();
    while (low < high) {
        const std::uint64_t midpoint = low + ((high - low) >> 1U) + 1U;
        if (compare(multiply_u64(midpoint, midpoint), target) <= 0) {
            low = midpoint;
        } else {
            high = midpoint - 1U;
        }
    }
    if (low == std::numeric_limits<std::uint64_t>::max()) return low;
    const UInt128 lower_square = multiply_u64(low, low);
    const UInt128 upper_square = multiply_u64(low + 1U, low + 1U);
    const UInt128 lower_distance = subtract(target, lower_square);
    const UInt128 upper_distance = subtract(upper_square, target);
    const int distance_compare = compare(lower_distance, upper_distance);
    if (distance_compare > 0 || (distance_compare == 0 && (low & 1U) != 0U)) {
        ++low;
    }
    return low;
}

[[nodiscard]] inline Q16_48 quantize_binary32(Q16_48 value) noexcept {
    if (value.raw == 0) return value;
    const bool negative = value.raw < 0;
    const std::uint64_t absolute = magnitude(value.raw);
    const unsigned highest_bit = 63U - std::countl_zero(absolute);
    if (highest_bit <= 23U) return value;
    const unsigned shift = highest_bit - 23U;
    std::uint64_t rounded = absolute >> shift;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    const std::uint64_t remainder = absolute & mask;
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    const bool overflow = rounded > (std::numeric_limits<std::uint64_t>::max() >> shift);
    const std::uint64_t raw =
        overflow ? std::numeric_limits<std::uint64_t>::max() : rounded << shift;
    return {apply_sign_saturating(raw, negative, overflow)};
}

[[nodiscard]] inline numeric::Q32_32 quantize_public_binary32(numeric::Q32_32 value) noexcept {
    if (value.raw() == 0) return value;
    const bool negative = value.raw() < 0;
    const std::uint64_t absolute = magnitude(value.raw());
    const unsigned highest_bit = 63U - std::countl_zero(absolute);
    if (highest_bit <= 23U) return value;
    const unsigned shift = highest_bit - 23U;
    std::uint64_t rounded = absolute >> shift;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    const std::uint64_t remainder = absolute & mask;
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    const bool overflow = rounded > (std::numeric_limits<std::uint64_t>::max() >> shift);
    const std::uint64_t raw =
        overflow ? std::numeric_limits<std::uint64_t>::max() : rounded << shift;
    return numeric::Q32_32::from_raw(apply_sign_saturating(raw, negative, overflow));
}

[[nodiscard]] inline Q16_48 add_saturating(Q16_48 lhs, Q16_48 rhs) noexcept {
    if (rhs.raw > 0 && lhs.raw > std::numeric_limits<std::int64_t>::max() - rhs.raw) {
        return {std::numeric_limits<std::int64_t>::max()};
    }
    if (rhs.raw < 0 && lhs.raw < std::numeric_limits<std::int64_t>::min() - rhs.raw) {
        return {std::numeric_limits<std::int64_t>::min()};
    }
    return {lhs.raw + rhs.raw};
}

[[nodiscard]] inline Q16_48 subtract_saturating(Q16_48 lhs, Q16_48 rhs) noexcept {
    if (rhs.raw == std::numeric_limits<std::int64_t>::min()) {
        return lhs.raw >= 0 ? Q16_48{std::numeric_limits<std::int64_t>::max()}
                            : Q16_48{lhs.raw - rhs.raw};
    }
    return add_saturating(lhs, Q16_48{-rhs.raw});
}

[[nodiscard]] inline bool subtract_raw_checked(std::int64_t lhs, std::int64_t rhs,
                                               std::int64_t &output) noexcept {
    if (rhs > 0 && lhs < std::numeric_limits<std::int64_t>::min() + rhs) {
        return false;
    }
    if (rhs < 0 && lhs > std::numeric_limits<std::int64_t>::max() + rhs) {
        return false;
    }
    output = lhs - rhs;
    return true;
}

[[nodiscard]] inline Q16_48 multiply(Q16_48 lhs, Q16_48 rhs) noexcept {
    const bool negative = (lhs.raw < 0) != (rhs.raw < 0);
    const UInt128 product = multiply_u64(magnitude(lhs.raw), magnitude(rhs.raw));
    constexpr std::uint64_t maximum_high_limb =
        std::numeric_limits<std::uint64_t>::max() >> (64U - fractional_bits);
    if (product.hi > maximum_high_limb) {
        return {negative ? std::numeric_limits<std::int64_t>::min()
                         : std::numeric_limits<std::int64_t>::max()};
    }
    std::uint64_t quotient =
        (product.hi << (64U - fractional_bits)) | (product.lo >> fractional_bits);
    const std::uint64_t remainder = product.lo & fractional_mask;
    bool overflow = false;
    if (remainder > fractional_half || (remainder == fractional_half && (quotient & 1U) != 0U)) {
        if (quotient == std::numeric_limits<std::uint64_t>::max()) {
            overflow = true;
        } else {
            ++quotient;
        }
    }
    return {apply_sign_saturating(quotient, negative, overflow)};
}

[[nodiscard]] inline bool divide(Q16_48 numerator, Q16_48 denominator, Q16_48 &output) noexcept {
    if (denominator.raw == 0) return false;
    const bool negative = (numerator.raw < 0) != (denominator.raw < 0);
    const std::uint64_t absolute = magnitude(numerator.raw);
    const UInt128 shifted{absolute >> (64U - fractional_bits), absolute << fractional_bits};
    bool overflow = false;
    const std::uint64_t quotient = rounded_quotient(shifted, magnitude(denominator.raw), &overflow);
    output = {apply_sign_saturating(quotient, negative, overflow)};
    return !overflow;
}

[[nodiscard]] inline Q16_48 add(Q16_48 lhs, Q16_48 rhs) noexcept {
    return quantize_binary32(add_saturating(lhs, rhs));
}

[[nodiscard]] inline Q16_48 subtract_value(Q16_48 lhs, Q16_48 rhs) noexcept {
    return quantize_binary32(subtract_saturating(lhs, rhs));
}

[[nodiscard]] inline Q16_48 multiply_value(Q16_48 lhs, Q16_48 rhs) noexcept {
    return quantize_binary32(multiply(lhs, rhs));
}

[[nodiscard]] inline bool divide_value(Q16_48 numerator, Q16_48 denominator,
                                       Q16_48 &output) noexcept {
    if (!divide(numerator, denominator, output)) return false;
    output = quantize_binary32(output);
    return true;
}

[[nodiscard]] inline Q16_48 negate(Q16_48 value) noexcept {
    return subtract_saturating(zero, value);
}

[[nodiscard]] bool factor_tridiagonal_values(const LinearVector &diagonal,
                                             const LinearSubdiagonal &subdiagonal,
                                             LinearVector &diagonal_pivots,
                                             LinearSubdiagonal &subdiagonal_multipliers) noexcept;

// Exposes the internal raw operation for integer golden tests. Inputs and the
// result use signed Q16.48 raw values.
[[nodiscard]] std::int64_t compute_q16_48_hypot_raw(std::int64_t lhs_raw,
                                                    std::int64_t rhs_raw) noexcept;

} // namespace detail

// The covariance is normalized by 2^matrix_exponent before factorization.
// Each reflector has an implicit leading one; only its strict tail and tau are
// retained. The tridiagonal values therefore represent 2^matrix_exponent C.
struct CompactTridiagonalFactorization {
    int matrix_exponent{};
    std::array<detail::Q16_48, elements_per_block> diagonal{};
    std::array<detail::Q16_48, elements_per_block - 1U> subdiagonal{};
    std::array<detail::Q16_48, householder_reflector_count> reflector_tau{};
    std::array<detail::Q16_48, householder_tail_count> reflector_tails{};
};

struct TridiagonalLdlFactorization {
    std::array<detail::Q16_48, elements_per_block> diagonal_pivots{};
    std::array<detail::Q16_48, elements_per_block - 1U> subdiagonal_multipliers{};
};

struct TridiagonalDiagnostics {
    std::array<numeric::Q32_32, elements_per_block> diagonal{};
    std::array<numeric::Q32_32, elements_per_block - 1U> subdiagonal{};
};

enum class FactorizationStatus : std::uint8_t {
    computed,
    arithmetic_failure,
};

struct FactorizationOutcome {
    FactorizationStatus status{FactorizationStatus::arithmetic_failure};

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return status == FactorizationStatus::computed;
    }
};

enum class RegularityStatus : std::uint8_t {
    regular,
    singular,
    arithmetic_failure,
};

struct RegularityOutcome {
    RegularityStatus status{RegularityStatus::arithmetic_failure};
    std::uint32_t negative_pivot_count{};
    std::uint32_t stabilized_pivot_count{};
};

enum class LdlStatus : std::uint8_t {
    computed,
    not_positive_definite,
    arithmetic_failure,
};

[[nodiscard]] FactorizationOutcome
factorize_covariance(const SquareMatrix &covariance, CompactTridiagonalFactorization &factorization,
                     TridiagonalDiagnostics *diagnostics = nullptr) noexcept;

[[nodiscard]] RegularityOutcome
classify_covariance_regularity(const CompactTridiagonalFactorization &factorization,
                               numeric::Q32_32 epsilon = numeric::Q32_32::from_raw(4295)) noexcept;

[[nodiscard]] LdlStatus
factor_positive_definite_tridiagonal(const CompactTridiagonalFactorization &factorization,
                                     TridiagonalLdlFactorization &ldl_factorization) noexcept;

[[nodiscard]] bool compute_inverse_quadratic_form_averages(
    const CompactTridiagonalFactorization &factorization,
    const TridiagonalLdlFactorization &ldl_factorization,
    std::span<const numeric::Q32_32> independent, std::uint32_t columns,
    std::span<numeric::Q32_32> quadratic_form_averages) noexcept;

[[nodiscard]] bool
compute_shifted_log_determinant(const CompactTridiagonalFactorization &factorization,
                                numeric::Q32_32 variance, numeric::Q32_32 sigma,
                                numeric::Q32_32 &log_determinant) noexcept;

} // namespace crossvmaf::metrics::speed_chroma::reference::linear

namespace crossvmaf::metrics::speed_chroma::reference {

constexpr unsigned sample_fractional_bits = 20U;

struct ChannelStatistics {
    std::vector<numeric::Q32_32> entropies;
    std::vector<numeric::Q32_32> variances;
    bool singular{};
};

struct BlockScoreDiagnostics {
    numeric::Q32_32 spatial_reference{};
    numeric::Q32_32 spatial_distorted{};
    numeric::Q32_32 absolute_difference{};
    bool suppressed{};
};

struct ChannelScoreDiagnostics {
    numeric::Q32_32 base_entropy{};
    std::vector<BlockScoreDiagnostics> blocks;
};

[[nodiscard]] Result<Dimensions> make_dimensions(std::uint32_t width, std::uint32_t height,
                                                 numeric::Q32_32 prescale) noexcept;

[[nodiscard]] Result<ChannelStatistics>
estimate_parameters(const Dimensions &dimensions, std::span<const std::int32_t> samples,
                    numeric::Q32_32 sigma_nn, ParameterDiagnostics *diagnostics = nullptr);

namespace detail {

[[nodiscard]] numeric::Q32_32 absolute(numeric::Q32_32 value) noexcept;
[[nodiscard]] numeric::Q32_32 quantize_binary32(numeric::Q32_32 value) noexcept;
[[nodiscard]] numeric::Q32_32 binary32_add(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept;
[[nodiscard]] numeric::Q32_32 binary32_multiply(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept;
[[nodiscard]] bool binary32_divide(numeric::Q32_32 numerator, numeric::Q32_32 denominator,
                                   numeric::Q32_32 &output) noexcept;
[[nodiscard]] numeric::Q32_32 from_u32(std::uint32_t value) noexcept;
[[nodiscard]] Result<numeric::Q32_32> log2_checked(numeric::Q32_32 value,
                                                   const char *label) noexcept;

[[nodiscard]] Result<void> estimate_parameters_with_workspace(
    const Dimensions &dimensions, std::span<const std::int32_t> samples, numeric::Q32_32 sigma_nn,
    ChannelStatistics &statistics, std::vector<numeric::Q32_32> &independent) noexcept;

} // namespace detail

[[nodiscard]] Result<numeric::Q32_32>
channel_score(const Dimensions &dimensions, const ChannelStatistics &reference,
              const ChannelStatistics &distorted, numeric::Q32_32 sigma_nn,
              numeric::Q32_32 nn_floor, std::uint32_t weight_variance_mode,
              ChannelScoreDiagnostics *diagnostics = nullptr);

} // namespace crossvmaf::metrics::speed_chroma::reference
