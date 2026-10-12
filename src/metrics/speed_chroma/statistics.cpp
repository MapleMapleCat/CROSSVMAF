#include "metrics/speed_chroma/reference_math.hpp"
#include "numeric/rounding.hpp"
#include "numeric/uint128.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace crossvmaf::metrics::speed_chroma::reference {

using detail::binary32_add;
using detail::binary32_divide;
using detail::from_u32;
using detail::log2_checked;
using detail::quantize_binary32;

namespace {

using Q = numeric::Q32_32;
using SquareMatrix = std::array<Q, elements_per_block * elements_per_block>;
using Vector25 = std::array<Q, elements_per_block>;

namespace base_numeric = crossvmaf::numeric::detail;
using base_numeric::add;
using base_numeric::apply_sign_saturating;
using base_numeric::divide_u128_u64;
using base_numeric::magnitude;
using base_numeric::multiply_u64;
using base_numeric::negate;
using base_numeric::UInt128;

constexpr Q statistics_zero{};
constexpr Q statistics_log2_two_pi_e = Q::from_raw(17'584'417'180LL);

class statistics_GuardedProductSum final {
  public:
    void add_product(Q lhs, Q rhs) noexcept {
        UInt128 product = multiply_u64(magnitude(lhs.raw()), magnitude(rhs.raw()));
        if ((lhs.raw() < 0) != (rhs.raw() < 0)) product = negate(product);
        value_ = add(value_, product);
    }

    // Accumulates a Q32 value exactly, as if multiplied by one, so that
    // divide_to_q32 yields the round-to-nearest-even mean. This matches the
    // Vulkan means shader's exact wide-integer accumulation bit for bit.
    void add_value(Q value) noexcept {
        add_product(value, Q::from_integer(1));
    }

    [[nodiscard]] Q divide_to_q32(std::uint32_t divisor) const noexcept {
        const bool negative = (value_.hi & (std::uint64_t{1} << 63U)) != 0U;
        const UInt128 absolute = negative ? negate(value_) : value_;
        const std::uint64_t denominator = static_cast<std::uint64_t>(divisor) << Q::fractional_bits;
        std::uint64_t remainder{};
        bool overflow{};
        std::uint64_t quotient = divide_u128_u64(absolute, denominator, &remainder, &overflow);
        if (!overflow) {
            const std::uint64_t complement = denominator - remainder;
            const bool round_up =
                remainder > complement || (remainder == complement && (quotient & 1U) != 0U);
            if (round_up) {
                if (quotient == std::numeric_limits<std::uint64_t>::max()) {
                    overflow = true;
                } else {
                    ++quotient;
                }
            }
        }
        return Q::from_raw(apply_sign_saturating(quotient, negative, overflow));
    }

  private:
    UInt128 value_{};
};

[[nodiscard]] constexpr std::size_t statistics_matrix_index(std::size_t row,
                                                            std::size_t column) noexcept {
    return row * elements_per_block + column;
}

[[nodiscard]] Q statistics_from_sample(std::int32_t value) noexcept {
    constexpr unsigned shift = Q::fractional_bits - sample_fractional_bits;
    constexpr std::int64_t scale = std::int64_t{1} << shift;
    static_assert(static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) >=
                  std::numeric_limits<std::int64_t>::min() / scale);
    static_assert(static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) <=
                  std::numeric_limits<std::int64_t>::max() / scale);
    return quantize_binary32(Q::from_raw(static_cast<std::int64_t>(value) * scale));
}

[[nodiscard]] Result<void>
statistics_fill_guaranteed_singular_statistics(const Dimensions &dimensions, Q sigma_nn,
                                               ChannelStatistics &statistics) noexcept {
    if (statistics.variances.size() != dimensions.block_count ||
        statistics.entropies.size() != dimensions.block_count) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Speed Chroma singular statistics workspace has invalid geometry"});
    }
    const Q rounded_sigma_nn = binary32_add(statistics_zero, sigma_nn);
    const auto log_value = log2_checked(rounded_sigma_nn, "singular entropy");
    if (!log_value) {
        return Result<void>::failure(log_value.error());
    }

    const Q entropy_term = Q::add_saturating(log_value.value(), statistics_log2_two_pi_e);
    Q entropy{};
    for (std::size_t index = 0; index < elements_per_block; ++index) {
        entropy = quantize_binary32(Q::add_saturating(entropy, entropy_term));
    }

    statistics.singular = true;
    std::fill(statistics.variances.begin(), statistics.variances.end(), statistics_zero);
    std::fill(statistics.entropies.begin(), statistics.entropies.end(), entropy);
    return Result<void>::success();
}

} // namespace

[[nodiscard]] Result<void>
estimate_parameters_impl(const Dimensions &dimensions, std::span<const std::int32_t> samples,
                         numeric::Q32_32 sigma_nn, ParameterDiagnostics *diagnostics,
                         ChannelStatistics &statistics, std::vector<Q> &independent) {
    const std::uint64_t operating_count =
        static_cast<std::uint64_t>(dimensions.operating_width) * dimensions.operating_height;
    if (dimensions.block_count == 0U || sigma_nn.raw() <= 0 || operating_count > samples.size()) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Speed Chroma parameter estimation received invalid geometry"});
    }
    const std::uint64_t expected_sample_count =
        static_cast<std::uint64_t>(dimensions.submatrix_width) * dimensions.submatrix_height;
    if (expected_sample_count == 0U ||
        expected_sample_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        dimensions.covariance_sample_count != expected_sample_count) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "Speed Chroma parameter estimation received "
                                      "inconsistent covariance geometry"});
    }
    if (diagnostics != nullptr) {
        *diagnostics = {};
        diagnostics->covariance_sample_count = dimensions.covariance_sample_count;
    }
    if (guarantees_singular_covariance(dimensions)) {
        if (diagnostics != nullptr) {
            diagnostics->status = EstimationStatus::guaranteed_singular;
        }
        return statistics_fill_guaranteed_singular_statistics(dimensions, sigma_nn, statistics);
    }
    const std::size_t rectangular_element_count =
        static_cast<std::size_t>(elements_per_block) * dimensions.block_count;
    if (independent.size() != rectangular_element_count ||
        statistics.entropies.size() != dimensions.block_count ||
        statistics.variances.size() != dimensions.block_count) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma parameter workspace has invalid geometry"});
    }
    const std::uint32_t sample_count = dimensions.covariance_sample_count;

    Vector25 means{};
    for (std::uint32_t start_row = 0; start_row < block_size; ++start_row) {
        for (std::uint32_t start_column = 0; start_column < block_size; ++start_column) {
            statistics_GuardedProductSum sum;
            for (std::uint32_t row = 0; row < dimensions.submatrix_height; ++row) {
                for (std::uint32_t column = 0; column < dimensions.submatrix_width; ++column) {
                    const std::size_t source =
                        static_cast<std::size_t>(start_row + row) * dimensions.operating_width +
                        start_column + column;
                    sum.add_value(statistics_from_sample(samples[source]));
                }
            }
            means[start_row * block_size + start_column] =
                quantize_binary32(sum.divide_to_q32(sample_count));
        }
    }

    SquareMatrix covariance{};
    for (std::size_t x_index = 0; x_index < elements_per_block; ++x_index) {
        const std::uint32_t x_row = static_cast<std::uint32_t>(x_index / block_size);
        const std::uint32_t x_column = static_cast<std::uint32_t>(x_index % block_size);
        for (std::size_t y_index = 0; y_index <= x_index; ++y_index) {
            const std::uint32_t y_row = static_cast<std::uint32_t>(y_index / block_size);
            const std::uint32_t y_column = static_cast<std::uint32_t>(y_index % block_size);
            statistics_GuardedProductSum sum;
            for (std::uint32_t row = 0; row < dimensions.submatrix_height; ++row) {
                for (std::uint32_t column = 0; column < dimensions.submatrix_width; ++column) {
                    const std::size_t x_source =
                        static_cast<std::size_t>(x_row + row) * dimensions.operating_width +
                        x_column + column;
                    const std::size_t y_source =
                        static_cast<std::size_t>(y_row + row) * dimensions.operating_width +
                        y_column + column;
                    const Q x = Q::subtract_saturating(statistics_from_sample(samples[x_source]),
                                                       means[x_index]);
                    const Q y = Q::subtract_saturating(statistics_from_sample(samples[y_source]),
                                                       means[y_index]);
                    sum.add_product(x, y);
                }
            }
            const Q value = quantize_binary32(sum.divide_to_q32(sample_count));
            covariance[statistics_matrix_index(x_index, y_index)] = value;
            covariance[statistics_matrix_index(y_index, x_index)] = value;
        }
    }

    linear::CompactTridiagonalFactorization factorization{};
    linear::TridiagonalDiagnostics linear_diagnostics{};
    const linear::FactorizationOutcome factorization_outcome = linear::factorize_covariance(
        covariance, factorization, diagnostics != nullptr ? &linear_diagnostics : nullptr);
    if (!factorization_outcome) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "Speed Chroma compact Householder factorization arithmetic failed"});
    }
    if (diagnostics != nullptr) {
        diagnostics->tridiagonal_diagonal.assign(linear_diagnostics.diagonal.begin(),
                                                 linear_diagnostics.diagonal.end());
        diagnostics->tridiagonal_subdiagonal.assign(linear_diagnostics.subdiagonal.begin(),
                                                    linear_diagnostics.subdiagonal.end());
        diagnostics->means.assign(means.begin(), means.end());
        diagnostics->covariance.assign(covariance.begin(), covariance.end());
    }

    std::fill(independent.begin(), independent.end(), statistics_zero);
    for (std::uint32_t start_row = 0; start_row < block_size; ++start_row) {
        for (std::uint32_t start_column = 0; start_column < block_size; ++start_column) {
            const std::size_t output_row =
                static_cast<std::size_t>(start_row) * block_size + start_column;
            for (std::uint32_t block_row = 0; block_row < dimensions.blocks_vertical; ++block_row) {
                for (std::uint32_t block_column = 0; block_column < dimensions.blocks_horizontal;
                     ++block_column) {
                    const std::uint32_t source_row = start_row + block_row * block_size;
                    const std::uint32_t source_column = start_column + block_column * block_size;
                    const std::size_t block =
                        static_cast<std::size_t>(block_row) * dimensions.blocks_horizontal +
                        block_column;
                    independent[output_row * dimensions.block_count + block] =
                        statistics_from_sample(samples[static_cast<std::size_t>(source_row) *
                                                           dimensions.operating_width +
                                                       source_column]);
                }
            }
        }
    }

    const linear::RegularityOutcome regularity =
        linear::classify_covariance_regularity(factorization);
    if (regularity.status == linear::RegularityStatus::arithmetic_failure) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Speed Chroma shifted Sturm LDL arithmetic failed"});
    }
    const bool singular = regularity.status == linear::RegularityStatus::singular;
    std::fill(statistics.variances.begin(), statistics.variances.end(), statistics_zero);
    if (!singular) {
        linear::TridiagonalLdlFactorization ldl_factorization{};
        const linear::LdlStatus ldl_status =
            linear::factor_positive_definite_tridiagonal(factorization, ldl_factorization);
        if (ldl_status != linear::LdlStatus::computed) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 ldl_status == linear::LdlStatus::arithmetic_failure
                     ? "Speed Chroma tridiagonal LDL arithmetic failed"
                     : "Speed Chroma regular covariance produced a non-positive LDL "
                       "pivot"});
        }
        if (!linear::compute_inverse_quadratic_form_averages(factorization, ldl_factorization,
                                                             independent, dimensions.block_count,
                                                             statistics.variances)) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "Speed Chroma inverse quadratic-form average arithmetic failed"});
        }
    }

    statistics.singular = singular;
    for (std::uint32_t block = 0; block < dimensions.block_count; ++block) {
        Q entropy{};
        if (!linear::compute_shifted_log_determinant(factorization, statistics.variances[block],
                                                     sigma_nn, entropy)) {
            return Result<void>::failure({ErrorCode::invalid_state,
                                          "Speed Chroma shifted tridiagonal "
                                          "log-determinant arithmetic failed"});
        }
        for (std::size_t index = 0; index < elements_per_block; ++index) {
            entropy = binary32_add(entropy, statistics_log2_two_pi_e);
        }
        statistics.entropies[block] = entropy;
    }
    return Result<void>::success();
}

Result<ChannelStatistics> estimate_parameters(const Dimensions &dimensions,
                                              std::span<const std::int32_t> samples,
                                              numeric::Q32_32 sigma_nn,
                                              ParameterDiagnostics *diagnostics) {
    const std::size_t rectangular_element_count =
        static_cast<std::size_t>(elements_per_block) * dimensions.block_count;
    ChannelStatistics statistics{};
    statistics.entropies.resize(dimensions.block_count);
    statistics.variances.resize(dimensions.block_count);
    std::vector<Q> independent(rectangular_element_count);
    const auto result = estimate_parameters_impl(dimensions, samples, sigma_nn, diagnostics,
                                                 statistics, independent);
    if (!result) {
        return Result<ChannelStatistics>::failure(result.error());
    }
    return Result<ChannelStatistics>::success(std::move(statistics));
}

Result<void> detail::estimate_parameters_with_workspace(
    const Dimensions &dimensions, std::span<const std::int32_t> samples, numeric::Q32_32 sigma_nn,
    ChannelStatistics &statistics, std::vector<numeric::Q32_32> &independent) noexcept {
    return estimate_parameters_impl(dimensions, samples, sigma_nn, nullptr, statistics,
                                    independent);
}

} // namespace crossvmaf::metrics::speed_chroma::reference

namespace crossvmaf::metrics::speed_chroma::reference {
namespace {

using Q = numeric::Q32_32;

constexpr Q channel_score_one = Q::from_raw(Q::one_raw);
constexpr Q channel_score_quarter = Q::from_raw(Q::one_raw / 4);
constexpr Q channel_score_three_quarters = Q::from_raw(3 * (Q::one_raw / 4));
constexpr Q channel_score_log2_two_pi_e = Q::from_raw(17'584'417'180LL);

using crossvmaf::numeric::detail::apply_sign_saturating;
using crossvmaf::numeric::detail::magnitude;

} // namespace

numeric::Q32_32 detail::absolute(numeric::Q32_32 value) noexcept {
    if (value.raw() >= 0) return value;
    if (value.raw() == std::numeric_limits<std::int64_t>::min()) {
        return Q::from_raw(std::numeric_limits<std::int64_t>::max());
    }
    return Q::from_raw(-value.raw());
}

numeric::Q32_32 detail::quantize_binary32(numeric::Q32_32 value) noexcept {
    if (value.raw() == 0) return value;
    const bool negative = value.raw() < 0;
    const std::uint64_t absolute_value = magnitude(value.raw());
    const unsigned highest_bit = 63U - std::countl_zero(absolute_value);
    if (highest_bit <= 23U) return value;
    const unsigned shift = highest_bit - 23U;
    std::uint64_t rounded = absolute_value >> shift;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    const std::uint64_t remainder = absolute_value & mask;
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    const bool overflow = rounded > (std::numeric_limits<std::uint64_t>::max() >> shift);
    const std::uint64_t raw =
        overflow ? std::numeric_limits<std::uint64_t>::max() : rounded << shift;
    return Q::from_raw(apply_sign_saturating(raw, negative, overflow));
}

numeric::Q32_32 detail::binary32_add(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept {
    return quantize_binary32(Q::add_saturating(lhs, rhs));
}

numeric::Q32_32 detail::binary32_multiply(numeric::Q32_32 lhs, numeric::Q32_32 rhs) noexcept {
    return quantize_binary32(Q::multiply(lhs, rhs));
}

bool detail::binary32_divide(numeric::Q32_32 numerator, numeric::Q32_32 denominator,
                             numeric::Q32_32 &output) noexcept {
    const auto result = Q::divide(numerator, denominator);
    if (!result) return false;
    output = quantize_binary32(result.value());
    return true;
}

numeric::Q32_32 detail::from_u32(std::uint32_t value) noexcept {
    return Q::from_raw(static_cast<std::int64_t>(value) * Q::one_raw);
}

Result<numeric::Q32_32> detail::log2_checked(numeric::Q32_32 value, const char *label) noexcept {
    const auto result = Q::log2(value);
    if (!result) {
        return Result<Q>::failure(
            {ErrorCode::invalid_state, std::string("Speed Chroma log2 domain error in ") + label});
    }
    return Result<Q>::success(result.value());
}

Result<numeric::Q32_32> channel_score(const Dimensions &dimensions,
                                      const ChannelStatistics &reference,
                                      const ChannelStatistics &distorted, numeric::Q32_32 sigma_nn,
                                      numeric::Q32_32 nn_floor, std::uint32_t weight_variance_mode,
                                      ChannelScoreDiagnostics *diagnostics) {
    using detail::absolute;
    using detail::binary32_add;
    using detail::binary32_divide;
    using detail::binary32_multiply;
    using detail::from_u32;
    using detail::log2_checked;
    using detail::quantize_binary32;

    if (dimensions.block_count == 0U || reference.entropies.size() != dimensions.block_count ||
        reference.variances.size() != dimensions.block_count ||
        distorted.entropies.size() != dimensions.block_count ||
        distorted.variances.size() != dimensions.block_count) {
        return Result<numeric::Q32_32>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma score received incompatible statistics"});
    }
    if (weight_variance_mode != 5U) {
        return Result<numeric::Q32_32>::failure(
            {ErrorCode::unsupported_model,
             "CrossVMAF 1.0.16 Speed Chroma requires variance mode 5"});
    }
    const Q base_argument = binary32_multiply(binary32_add(channel_score_one, nn_floor), sigma_nn);
    const auto base_log = log2_checked(base_argument, "base entropy");
    if (!base_log) return Result<numeric::Q32_32>::failure(base_log.error());
    const Q base_entropy = quantize_binary32(
        Q::multiply(from_u32(elements_per_block),
                    Q::add_saturating(base_log.value(), channel_score_log2_two_pi_e)));
    if (diagnostics != nullptr) {
        diagnostics->base_entropy = base_entropy;
        diagnostics->blocks.assign(dimensions.block_count, {});
    }

    Q score{};
    for (std::uint32_t block = 0; block < dimensions.block_count; ++block) {
        if (reference.entropies[block] < base_entropy &&
            distorted.entropies[block] < base_entropy) {
            if (diagnostics != nullptr) {
                diagnostics->blocks[block].suppressed = true;
            }
            continue;
        }
        if (reference.entropies[block] == distorted.entropies[block] &&
            reference.variances[block] == distorted.variances[block]) {
            continue;
        }
        const auto reference_log =
            log2_checked(binary32_add(channel_score_one, reference.variances[block]),
                         "reference variance weight");
        if (!reference_log) {
            return Result<numeric::Q32_32>::failure(reference_log.error());
        }
        const Q distorted_variance =
            reference.variances[block] == distorted.variances[block]
                ? reference.variances[block]
                : Q::add_saturating(
                      Q::multiply(channel_score_three_quarters, reference.variances[block]),
                      Q::multiply(channel_score_quarter, distorted.variances[block]));
        const auto distorted_log = log2_checked(
            Q::add_saturating(channel_score_one, distorted_variance), "distorted variance weight");
        if (!distorted_log) {
            return Result<numeric::Q32_32>::failure(distorted_log.error());
        }
        const Q spatial_reference =
            quantize_binary32(Q::multiply(reference.entropies[block], reference_log.value()));
        const Q spatial_distorted =
            quantize_binary32(Q::multiply(distorted.entropies[block], distorted_log.value()));
        const Q difference = absolute(
            quantize_binary32(Q::subtract_saturating(spatial_reference, spatial_distorted)));
        if (diagnostics != nullptr) {
            diagnostics->blocks[block] = {spatial_reference, spatial_distorted, difference, false};
        }
        score = binary32_add(score, difference);
    }
    Q average{};
    if (!binary32_divide(score, from_u32(dimensions.block_count), average)) {
        return Result<numeric::Q32_32>::failure(
            {ErrorCode::invalid_state, "Speed Chroma score pooling division failed"});
    }
    return Result<numeric::Q32_32>::success(average);
}

} // namespace crossvmaf::metrics::speed_chroma::reference
