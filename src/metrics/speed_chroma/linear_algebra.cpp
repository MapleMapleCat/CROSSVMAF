#include "metrics/speed_chroma/reference_math.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <array>

namespace crossvmaf::metrics::speed_chroma::reference::linear {
namespace {

using detail::add;
using detail::compare;
using detail::convert_public_with_exponent;
using detail::convert_to_public_scaled;
using detail::divide_value;
using detail::half;
using detail::integer_sqrt_rne;
using detail::LinearSquareMatrix;
using detail::LinearVector;
using detail::magnitude;
using detail::matrix_index;
using detail::multiply_u64;
using detail::multiply_value;
using detail::negate;
using detail::one;
using detail::positive_raw_limit;
using detail::public_normalization_exponent;
using detail::Q16_48;
using detail::quantize_binary32;
using detail::reflector_tail_offset;
using detail::subtract_value;
using detail::UInt128;
using detail::zero;

constexpr UInt128 householder_norm_square_limit =
    multiply_u64(positive_raw_limit, positive_raw_limit);

[[nodiscard]] constexpr UInt128 householder_add_norm_square_capped(UInt128 accumulator,
                                                                   UInt128 square) noexcept {
    const std::uint64_t low = accumulator.lo + square.lo;
    const UInt128 sum{
        accumulator.hi + square.hi + (low < accumulator.lo ? 1U : 0U),
        low,
    };
    const bool overflow = compare(sum, accumulator) < 0;
    return overflow || compare(sum, householder_norm_square_limit) > 0
               ? householder_norm_square_limit
               : sum;
}

[[nodiscard]] bool householder_column_tail_norm(const LinearSquareMatrix &matrix,
                                                std::size_t column, std::size_t first_tail_row,
                                                Q16_48 &output) noexcept {
    UInt128 square_sum{};
    for (std::size_t row = first_tail_row; row < elements_per_block; ++row) {
        const std::uint64_t absolute_raw = magnitude(matrix[matrix_index(row, column)].raw);
        square_sum = householder_add_norm_square_capped(square_sum,
                                                        multiply_u64(absolute_raw, absolute_raw));
    }
    output = quantize_binary32(Q16_48{static_cast<std::int64_t>(integer_sqrt_rne(square_sum))});
    return true;
}

[[nodiscard]] bool householder_pythagoras(Q16_48 lhs, Q16_48 rhs, Q16_48 &output) noexcept {
    const std::uint64_t lhs_magnitude = magnitude(lhs.raw);
    const std::uint64_t rhs_magnitude = magnitude(rhs.raw);
    const std::uint64_t maximum = std::max(lhs_magnitude, rhs_magnitude);
    const std::uint64_t minimum = std::min(lhs_magnitude, rhs_magnitude);
    if (maximum == 0U) {
        output = zero;
        return true;
    }

    const UInt128 minimum_square = multiply_u64(minimum, minimum);
    if (compare(minimum_square, UInt128{0U, maximum}) <= 0) {
        output = quantize_binary32(
            Q16_48{static_cast<std::int64_t>(std::min(maximum, positive_raw_limit))});
        return true;
    }

    UInt128 square_sum = multiply_u64(maximum, maximum);
    square_sum = compare(square_sum, householder_norm_square_limit) > 0
                     ? householder_norm_square_limit
                     : square_sum;
    square_sum = householder_add_norm_square_capped(square_sum, minimum_square);
    output = quantize_binary32(Q16_48{static_cast<std::int64_t>(integer_sqrt_rne(square_sum))});
    return true;
}

[[nodiscard]] bool householder_create_householder_reflector(LinearSquareMatrix &matrix,
                                                            std::size_t column,
                                                            Q16_48 &tau) noexcept {
    const std::size_t leading_row = column + 1U;
    Q16_48 tail_norm{};
    if (!householder_column_tail_norm(matrix, column, leading_row + 1U, tail_norm)) {
        return false;
    }
    if (tail_norm.raw == 0) {
        tau = zero;
        return true;
    }

    const Q16_48 alpha = matrix[matrix_index(leading_row, column)];
    Q16_48 hypotenuse{};
    if (!householder_pythagoras(alpha, tail_norm, hypotenuse)) return false;
    const Q16_48 beta = alpha.raw >= 0 ? negate(hypotenuse) : hypotenuse;
    if (!divide_value(subtract_value(beta, alpha), beta, tau)) return false;
    const Q16_48 tail_scale = subtract_value(alpha, beta);
    if (tail_scale.raw == 0) return false;

    matrix[matrix_index(leading_row, column)] = beta;
    for (std::size_t row = leading_row + 1U; row < elements_per_block; ++row) {
        Q16_48 quotient{};
        if (!divide_value(matrix[matrix_index(row, column)], tail_scale, quotient)) {
            return false;
        }
        matrix[matrix_index(row, column)] = quotient;
    }
    return true;
}

[[nodiscard]] bool
householder_reduce_to_compact_tridiagonal(LinearSquareMatrix &matrix,
                                          CompactTridiagonalFactorization &factorization) noexcept {
    LinearVector reflector{};
    LinearVector product{};
    for (std::size_t column = 0; column < householder_reflector_count; ++column) {
        Q16_48 tau{};
        if (!householder_create_householder_reflector(matrix, column, tau)) return false;
        factorization.reflector_tau[column] = tau;
        const std::size_t leading_row = column + 1U;
        const std::size_t tail_offset = reflector_tail_offset(column);
        for (std::size_t row = leading_row + 1U; row < elements_per_block; ++row) {
            const Q16_48 tail = matrix[matrix_index(row, column)];
            factorization.reflector_tails[tail_offset + row - leading_row - 1U] = tail;
        }
        if (tau.raw == 0) continue;

        reflector.fill(zero);
        reflector[leading_row] = one;
        for (std::size_t row = leading_row + 1U; row < elements_per_block; ++row) {
            reflector[row] = matrix[matrix_index(row, column)];
        }

        for (std::size_t row = leading_row; row < elements_per_block; ++row) {
            product[row] = zero;
            for (std::size_t inner = leading_row; inner < elements_per_block; ++inner) {
                const Q16_48 matrix_vector =
                    multiply_value(matrix[matrix_index(row, inner)], reflector[inner]);
                product[row] = add(product[row], multiply_value(tau, matrix_vector));
            }
        }

        Q16_48 dot_product{};
        for (std::size_t row = leading_row; row < elements_per_block; ++row) {
            dot_product = add(dot_product, multiply_value(product[row], reflector[row]));
        }
        const Q16_48 correction_scale =
            negate(multiply_value(multiply_value(half, tau), dot_product));
        for (std::size_t row = leading_row; row < elements_per_block; ++row) {
            product[row] = add(product[row], multiply_value(correction_scale, reflector[row]));
        }

        for (std::size_t row = leading_row; row < elements_per_block; ++row) {
            for (std::size_t inner = leading_row; inner < elements_per_block; ++inner) {
                const Q16_48 symmetric_correction =
                    add(multiply_value(product[row], reflector[inner]),
                        multiply_value(reflector[row], product[inner]));
                matrix[matrix_index(row, inner)] =
                    subtract_value(matrix[matrix_index(row, inner)], symmetric_correction);
            }
        }
    }

    for (std::size_t index = 0; index < elements_per_block; ++index) {
        factorization.diagonal[index] = matrix[matrix_index(index, index)];
    }
    for (std::size_t index = 0; index + 1U < elements_per_block; ++index) {
        factorization.subdiagonal[index] = matrix[matrix_index(index + 1U, index)];
    }
    return true;
}

} // namespace

FactorizationOutcome factorize_covariance(const SquareMatrix &covariance,
                                          CompactTridiagonalFactorization &factorization,
                                          TridiagonalDiagnostics *diagnostics) noexcept {
    factorization = {};
    LinearSquareMatrix matrix{};
    factorization.matrix_exponent = public_normalization_exponent(covariance);
    for (std::size_t index = 0U; index < covariance.size(); ++index) {
        if (!convert_public_with_exponent(covariance[index], factorization.matrix_exponent,
                                          matrix[index])) {
            return {FactorizationStatus::arithmetic_failure};
        }
    }
    if (!householder_reduce_to_compact_tridiagonal(matrix, factorization)) {
        return {FactorizationStatus::arithmetic_failure};
    }

    if (diagnostics != nullptr) {
        for (std::size_t index = 0U; index < factorization.diagonal.size(); ++index) {
            if (!convert_to_public_scaled(factorization.diagonal[index],
                                          -factorization.matrix_exponent,
                                          diagnostics->diagonal[index])) {
                return {FactorizationStatus::arithmetic_failure};
            }
        }
        for (std::size_t index = 0U; index < factorization.subdiagonal.size(); ++index) {
            if (!convert_to_public_scaled(factorization.subdiagonal[index],
                                          -factorization.matrix_exponent,
                                          diagnostics->subdiagonal[index])) {
                return {FactorizationStatus::arithmetic_failure};
            }
        }
    }
    return {FactorizationStatus::computed};
}

std::int64_t detail::compute_q16_48_hypot_raw(std::int64_t lhs_raw, std::int64_t rhs_raw) noexcept {
    Q16_48 output{};
    const bool valid = householder_pythagoras(Q16_48{lhs_raw}, Q16_48{rhs_raw}, output);
    return valid ? output.raw : 0;
}

} // namespace crossvmaf::metrics::speed_chroma::reference::linear

namespace crossvmaf::metrics::speed_chroma::reference::linear {

bool detail::factor_tridiagonal_values(const LinearVector &diagonal,
                                       const LinearSubdiagonal &subdiagonal,
                                       LinearVector &diagonal_pivots,
                                       LinearSubdiagonal &subdiagonal_multipliers) noexcept {
    diagonal_pivots[0] = diagonal[0];
    if (diagonal_pivots[0].raw <= 0) return false;
    for (std::size_t index = 0; index + 1U < elements_per_block; ++index) {
        Q16_48 multiplier{};
        if (!divide_value(subdiagonal[index], diagonal_pivots[index], multiplier)) {
            return false;
        }
        subdiagonal_multipliers[index] = multiplier;
        diagonal_pivots[index + 1U] =
            subtract_value(diagonal[index + 1U], multiply_value(multiplier, subdiagonal[index]));
        if (diagonal_pivots[index + 1U].raw <= 0) return false;
    }
    return true;
}

LdlStatus
factor_positive_definite_tridiagonal(const CompactTridiagonalFactorization &factorization,
                                     TridiagonalLdlFactorization &ldl_factorization) noexcept {
    using detail::divide_value;
    using detail::multiply_value;
    using detail::Q16_48;
    using detail::subtract_value;

    ldl_factorization = {};
    ldl_factorization.diagonal_pivots[0] = factorization.diagonal[0];
    if (ldl_factorization.diagonal_pivots[0].raw <= 0) {
        return LdlStatus::not_positive_definite;
    }
    for (std::size_t index = 0; index + 1U < elements_per_block; ++index) {
        Q16_48 multiplier{};
        if (!divide_value(factorization.subdiagonal[index],
                          ldl_factorization.diagonal_pivots[index], multiplier)) {
            return LdlStatus::arithmetic_failure;
        }
        ldl_factorization.subdiagonal_multipliers[index] = multiplier;
        ldl_factorization.diagonal_pivots[index + 1U] =
            subtract_value(factorization.diagonal[index + 1U],
                           multiply_value(multiplier, factorization.subdiagonal[index]));
        if (ldl_factorization.diagonal_pivots[index + 1U].raw <= 0) {
            return LdlStatus::not_positive_definite;
        }
    }
    return LdlStatus::computed;
}

} // namespace crossvmaf::metrics::speed_chroma::reference::linear

namespace crossvmaf::metrics::speed_chroma::reference::linear {
namespace {

using detail::add;
using detail::convert_public_with_exponent;
using detail::convert_to_public_scaled;
using detail::factor_tridiagonal_values;
using detail::LinearSubdiagonal;
using detail::LinearVector;
using detail::multiply_value;
using detail::normalization_exponent;
using detail::public_normalization_exponent;
using detail::Q16_48;
using detail::quantize_public_binary32;
using detail::scale_power_of_two;

[[nodiscard]] bool
log_determinant_logarithm_of_positive_value(Q16_48 value, numeric::Q32_32 &logarithm) noexcept {
    if (value.raw <= 0) return false;
    const int value_exponent = normalization_exponent(value);
    const Q16_48 normalized = scale_power_of_two(value, value_exponent);
    numeric::Q32_32 normalized_public{};
    if (!convert_to_public_scaled(normalized, 0, normalized_public)) {
        return false;
    }
    const auto normalized_logarithm = numeric::Q32_32::log2(normalized_public);
    if (!normalized_logarithm) return false;
    const std::int64_t exponent_raw =
        static_cast<std::int64_t>(value_exponent) * numeric::Q32_32::one_raw;
    logarithm = quantize_public_binary32(numeric::Q32_32::subtract_saturating(
        normalized_logarithm.value(), numeric::Q32_32::from_raw(exponent_raw)));
    return true;
}

} // namespace

bool compute_shifted_log_determinant(const CompactTridiagonalFactorization &factorization,
                                     numeric::Q32_32 variance, numeric::Q32_32 sigma,
                                     numeric::Q32_32 &log_determinant) noexcept {
    if (variance.raw() < 0 || sigma.raw() <= 0) return false;

    const int variance_exponent = public_normalization_exponent(std::array{variance});
    const int sigma_exponent = public_normalization_exponent(std::array{sigma});
    Q16_48 normalized_variance{};
    Q16_48 normalized_sigma{};
    if (!convert_public_with_exponent(variance, variance_exponent, normalized_variance) ||
        !convert_public_with_exponent(sigma, sigma_exponent, normalized_sigma)) {
        return false;
    }
    const int covariance_product_exponent = factorization.matrix_exponent + variance_exponent;
    const int shifted_matrix_exponent = variance.raw() == 0
                                            ? sigma_exponent
                                            : std::min(covariance_product_exponent, sigma_exponent);
    const Q16_48 aligned_sigma =
        scale_power_of_two(normalized_sigma, shifted_matrix_exponent - sigma_exponent);

    LinearVector shifted_diagonal{};
    LinearSubdiagonal shifted_subdiagonal{};
    for (std::size_t index = 0; index < elements_per_block; ++index) {
        const Q16_48 covariance_term =
            scale_power_of_two(multiply_value(normalized_variance, factorization.diagonal[index]),
                               shifted_matrix_exponent - covariance_product_exponent);
        shifted_diagonal[index] = add(aligned_sigma, covariance_term);
    }
    for (std::size_t index = 0; index + 1U < elements_per_block; ++index) {
        shifted_subdiagonal[index] = scale_power_of_two(
            multiply_value(normalized_variance, factorization.subdiagonal[index]),
            shifted_matrix_exponent - covariance_product_exponent);
    }

    LinearVector diagonal_pivots{};
    LinearSubdiagonal subdiagonal_multipliers{};
    if (!factor_tridiagonal_values(shifted_diagonal, shifted_subdiagonal, diagonal_pivots,
                                   subdiagonal_multipliers)) {
        return false;
    }

    numeric::Q32_32 logarithm_sum{};
    for (const Q16_48 pivot : diagonal_pivots) {
        numeric::Q32_32 pivot_logarithm{};
        if (!log_determinant_logarithm_of_positive_value(pivot, pivot_logarithm)) {
            return false;
        }
        logarithm_sum = quantize_public_binary32(
            numeric::Q32_32::add_saturating(logarithm_sum, pivot_logarithm));
    }
    const std::int64_t matrix_scaling_log_raw = static_cast<std::int64_t>(shifted_matrix_exponent) *
                                                static_cast<std::int64_t>(elements_per_block) *
                                                numeric::Q32_32::one_raw;
    log_determinant = quantize_public_binary32(numeric::Q32_32::subtract_saturating(
        logarithm_sum, numeric::Q32_32::from_raw(matrix_scaling_log_raw)));
    return true;
}

} // namespace crossvmaf::metrics::speed_chroma::reference::linear

namespace crossvmaf::metrics::speed_chroma::reference::linear {
namespace {

using detail::add;
using detail::convert_public_with_exponent;
using detail::convert_to_public_scaled;
using detail::divide_value;
using detail::inverse_quadratic_guard_bits;
using detail::LinearVector;
using detail::magnitude;
using detail::multiply_value;
using detail::one_raw;
using detail::public_normalization_exponent_from_magnitude;
using detail::Q16_48;
using detail::reflector_tail_offset;
using detail::subtract_value;

void quadratic_form_apply_transposed_householder_product(
    const CompactTridiagonalFactorization &factorization, LinearVector &values) noexcept {
    // C = Q T Q^T with Q = H0 H1 ...; ascending application computes Q^T x.
    for (std::size_t reflector_index = 0; reflector_index < householder_reflector_count;
         ++reflector_index) {
        const Q16_48 tau = factorization.reflector_tau[reflector_index];
        if (tau.raw == 0) continue;
        const std::size_t leading_row = reflector_index + 1U;
        const std::size_t tail_offset = reflector_tail_offset(reflector_index);
        Q16_48 dot_product = values[leading_row];
        for (std::size_t row = leading_row + 1U; row < elements_per_block; ++row) {
            const Q16_48 tail = factorization.reflector_tails[tail_offset + row - leading_row - 1U];
            dot_product = add(dot_product, multiply_value(tail, values[row]));
        }
        const Q16_48 scaled_dot = multiply_value(tau, dot_product);
        values[leading_row] = subtract_value(values[leading_row], scaled_dot);
        for (std::size_t row = leading_row + 1U; row < elements_per_block; ++row) {
            const Q16_48 tail = factorization.reflector_tails[tail_offset + row - leading_row - 1U];
            values[row] = subtract_value(values[row], multiply_value(tail, scaled_dot));
        }
    }
}

} // namespace

bool compute_inverse_quadratic_form_averages(
    const CompactTridiagonalFactorization &factorization,
    const TridiagonalLdlFactorization &ldl_factorization,
    std::span<const numeric::Q32_32> independent, std::uint32_t columns,
    std::span<numeric::Q32_32> quadratic_form_averages) noexcept {
    const std::size_t expected_element_count =
        static_cast<std::size_t>(elements_per_block) * columns;
    if (columns == 0U || independent.size() != expected_element_count ||
        quadratic_form_averages.size() != columns) {
        return false;
    }
    for (const Q16_48 pivot : ldl_factorization.diagonal_pivots) {
        if (pivot.raw <= 0) return false;
    }

    LinearVector transformed{};
    LinearVector forward_substitution{};
    for (std::uint32_t column = 0; column < columns; ++column) {
        std::uint64_t largest_right_hand_side_magnitude = 0U;
        for (std::size_t row = 0; row < elements_per_block; ++row) {
            largest_right_hand_side_magnitude =
                std::max(largest_right_hand_side_magnitude,
                         magnitude(independent[row * columns + column].raw()));
        }
        const int right_hand_side_exponent =
            public_normalization_exponent_from_magnitude(largest_right_hand_side_magnitude) -
            inverse_quadratic_guard_bits;
        for (std::size_t row = 0; row < elements_per_block; ++row) {
            if (!convert_public_with_exponent(independent[row * columns + column],
                                              right_hand_side_exponent, transformed[row])) {
                return false;
            }
        }
        quadratic_form_apply_transposed_householder_product(factorization, transformed);

        forward_substitution[0] = transformed[0];
        for (std::size_t row = 1; row < elements_per_block; ++row) {
            forward_substitution[row] =
                subtract_value(transformed[row],
                               multiply_value(ldl_factorization.subdiagonal_multipliers[row - 1U],
                                              forward_substitution[row - 1U]));
        }

        Q16_48 quadratic_form{};
        for (std::size_t row = 0; row < elements_per_block; ++row) {
            Q16_48 term{};
            if (!divide_value(multiply_value(forward_substitution[row], forward_substitution[row]),
                              ldl_factorization.diagonal_pivots[row], term)) {
                return false;
            }
            quadratic_form = add(quadratic_form, term);
        }
        Q16_48 quadratic_form_average{};
        if (!divide_value(quadratic_form,
                          Q16_48{static_cast<std::int64_t>(elements_per_block) * one_raw},
                          quadratic_form_average) ||
            !convert_to_public_scaled(quadratic_form_average,
                                      factorization.matrix_exponent - 2 * right_hand_side_exponent,
                                      quadratic_form_averages[column])) {
            return false;
        }
    }
    return true;
}

} // namespace crossvmaf::metrics::speed_chroma::reference::linear

namespace crossvmaf::metrics::speed_chroma::reference::linear {
namespace {

using detail::convert_public_with_exponent;
using detail::magnitude;
using detail::multiply_u64;
using detail::Q16_48;
using detail::subtract_raw_checked;
using detail::UInt128;

constexpr std::size_t regularity_characteristic_integer_limb_count = 40U;

struct regularity_WideUnsignedInteger {
    // A 25x25 determinant of signed 64-bit coefficients is below 2^1660 by
    // Hadamard's bound. Forty limbs leave ample room for every recurrence
    // intermediate while keeping allocation and failure behavior fixed.
    std::array<std::uint64_t, regularity_characteristic_integer_limb_count> limbs{};
    std::size_t used_limb_count{};
};

struct regularity_WideSignedInteger {
    regularity_WideUnsignedInteger magnitude{};
    int sign{};
};

void regularity_normalize_wide_unsigned(regularity_WideUnsignedInteger &value) noexcept {
    while (value.used_limb_count > 0U && value.limbs[value.used_limb_count - 1U] == 0U) {
        --value.used_limb_count;
    }
}

[[nodiscard]] regularity_WideSignedInteger
regularity_make_wide_signed(std::int64_t value) noexcept {
    regularity_WideSignedInteger output{};
    if (value == 0) return output;
    output.sign = value < 0 ? -1 : 1;
    output.magnitude.limbs[0U] = magnitude(value);
    output.magnitude.used_limb_count = 1U;
    return output;
}

[[nodiscard]] int
regularity_compare_wide_unsigned(const regularity_WideUnsignedInteger &lhs,
                                 const regularity_WideUnsignedInteger &rhs) noexcept {
    if (lhs.used_limb_count != rhs.used_limb_count) {
        return lhs.used_limb_count < rhs.used_limb_count ? -1 : 1;
    }
    for (std::size_t reverse_index = lhs.used_limb_count; reverse_index > 0U; --reverse_index) {
        const std::size_t index = reverse_index - 1U;
        if (lhs.limbs[index] != rhs.limbs[index]) {
            return lhs.limbs[index] < rhs.limbs[index] ? -1 : 1;
        }
    }
    return 0;
}

[[nodiscard]] bool regularity_add_wide_unsigned(const regularity_WideUnsignedInteger &lhs,
                                                const regularity_WideUnsignedInteger &rhs,
                                                regularity_WideUnsignedInteger &output) noexcept {
    regularity_WideUnsignedInteger sum{};
    const std::size_t limb_count = std::max(lhs.used_limb_count, rhs.used_limb_count);
    std::uint64_t carry = 0U;
    for (std::size_t index = 0U; index < limb_count; ++index) {
        const std::uint64_t lhs_limb = index < lhs.used_limb_count ? lhs.limbs[index] : 0U;
        const std::uint64_t rhs_limb = index < rhs.used_limb_count ? rhs.limbs[index] : 0U;
        const std::uint64_t partial = lhs_limb + rhs_limb;
        const bool partial_overflow = partial < lhs_limb;
        const std::uint64_t result_limb = partial + carry;
        const bool carry_overflow = result_limb < partial;
        sum.limbs[index] = result_limb;
        carry = partial_overflow || carry_overflow ? 1U : 0U;
    }
    sum.used_limb_count = limb_count;
    if (carry != 0U) {
        if (sum.used_limb_count == sum.limbs.size()) return false;
        sum.limbs[sum.used_limb_count++] = carry;
    }
    regularity_normalize_wide_unsigned(sum);
    output = sum;
    return true;
}

void regularity_subtract_wide_unsigned(const regularity_WideUnsignedInteger &lhs,
                                       const regularity_WideUnsignedInteger &rhs,
                                       regularity_WideUnsignedInteger &output) noexcept {
    regularity_WideUnsignedInteger difference{};
    std::uint64_t borrow = 0U;
    for (std::size_t index = 0U; index < lhs.used_limb_count; ++index) {
        const std::uint64_t rhs_limb = index < rhs.used_limb_count ? rhs.limbs[index] : 0U;
        const std::uint64_t partial = lhs.limbs[index] - rhs_limb;
        const bool rhs_borrow = lhs.limbs[index] < rhs_limb;
        const std::uint64_t result_limb = partial - borrow;
        const bool carry_borrow = partial < borrow;
        difference.limbs[index] = result_limb;
        borrow = rhs_borrow || carry_borrow ? 1U : 0U;
    }
    difference.used_limb_count = lhs.used_limb_count;
    regularity_normalize_wide_unsigned(difference);
    output = difference;
}

[[nodiscard]] bool
regularity_multiply_wide_unsigned_u64(const regularity_WideUnsignedInteger &value,
                                      std::uint64_t factor,
                                      regularity_WideUnsignedInteger &output) noexcept {
    if (value.used_limb_count == 0U || factor == 0U) {
        output = {};
        return true;
    }
    regularity_WideUnsignedInteger product_value{};
    std::uint64_t carry = 0U;
    for (std::size_t index = 0U; index < value.used_limb_count; ++index) {
        const UInt128 product = multiply_u64(value.limbs[index], factor);
        const std::uint64_t result_limb = product.lo + carry;
        const std::uint64_t carry_from_low = result_limb < product.lo ? 1U : 0U;
        product_value.limbs[index] = result_limb;
        carry = product.hi + carry_from_low;
    }
    product_value.used_limb_count = value.used_limb_count;
    if (carry != 0U) {
        if (product_value.used_limb_count == product_value.limbs.size()) {
            return false;
        }
        product_value.limbs[product_value.used_limb_count++] = carry;
    }
    output = product_value;
    return true;
}

[[nodiscard]] bool
regularity_shift_wide_unsigned_limbs(const regularity_WideUnsignedInteger &value,
                                     std::size_t limb_shift,
                                     regularity_WideUnsignedInteger &output) noexcept {
    if (value.used_limb_count == 0U) {
        output = {};
        return true;
    }
    if (value.used_limb_count + limb_shift > value.limbs.size()) return false;
    regularity_WideUnsignedInteger shifted{};
    for (std::size_t index = 0U; index < value.used_limb_count; ++index) {
        shifted.limbs[index + limb_shift] = value.limbs[index];
    }
    shifted.used_limb_count = value.used_limb_count + limb_shift;
    output = shifted;
    return true;
}

[[nodiscard]] bool
regularity_multiply_wide_unsigned_u128(const regularity_WideUnsignedInteger &value, UInt128 factor,
                                       regularity_WideUnsignedInteger &output) noexcept {
    regularity_WideUnsignedInteger low_product{};
    regularity_WideUnsignedInteger high_product{};
    regularity_WideUnsignedInteger shifted_high_product{};
    if (!regularity_multiply_wide_unsigned_u64(value, factor.lo, low_product) ||
        !regularity_multiply_wide_unsigned_u64(value, factor.hi, high_product) ||
        !regularity_shift_wide_unsigned_limbs(high_product, 1U, shifted_high_product)) {
        return false;
    }
    return regularity_add_wide_unsigned(low_product, shifted_high_product, output);
}

[[nodiscard]] bool regularity_multiply_wide_signed(const regularity_WideSignedInteger &value,
                                                   std::int64_t factor,
                                                   regularity_WideSignedInteger &output) noexcept {
    if (value.sign == 0 || factor == 0) {
        output = {};
        return true;
    }
    regularity_WideSignedInteger product{};
    if (!regularity_multiply_wide_unsigned_u64(value.magnitude, magnitude(factor),
                                               product.magnitude)) {
        return false;
    }
    product.sign = factor < 0 ? -value.sign : value.sign;
    output = product;
    return true;
}

[[nodiscard]] bool
regularity_multiply_wide_signed_unsigned128(const regularity_WideSignedInteger &value,
                                            UInt128 factor,
                                            regularity_WideSignedInteger &output) noexcept {
    if (value.sign == 0 || (factor.hi == 0U && factor.lo == 0U)) {
        output = {};
        return true;
    }
    regularity_WideSignedInteger product{};
    if (!regularity_multiply_wide_unsigned_u128(value.magnitude, factor, product.magnitude)) {
        return false;
    }
    product.sign = value.sign;
    output = product;
    return true;
}

[[nodiscard]] bool regularity_subtract_wide_signed(const regularity_WideSignedInteger &lhs,
                                                   const regularity_WideSignedInteger &rhs,
                                                   regularity_WideSignedInteger &output) noexcept {
    if (rhs.sign == 0) {
        output = lhs;
        return true;
    }
    if (lhs.sign == 0) {
        output = rhs;
        output.sign = -output.sign;
        return true;
    }
    regularity_WideSignedInteger difference{};
    if (lhs.sign != rhs.sign) {
        if (!regularity_add_wide_unsigned(lhs.magnitude, rhs.magnitude, difference.magnitude)) {
            return false;
        }
        difference.sign = lhs.sign;
        output = difference;
        return true;
    }
    const int magnitude_comparison = regularity_compare_wide_unsigned(lhs.magnitude, rhs.magnitude);
    if (magnitude_comparison == 0) {
        output = {};
        return true;
    }
    if (magnitude_comparison > 0) {
        regularity_subtract_wide_unsigned(lhs.magnitude, rhs.magnitude, difference.magnitude);
        difference.sign = lhs.sign;
    } else {
        regularity_subtract_wide_unsigned(rhs.magnitude, lhs.magnitude, difference.magnitude);
        difference.sign = -lhs.sign;
    }
    output = difference;
    return true;
}

} // namespace

RegularityOutcome
classify_covariance_regularity(const CompactTridiagonalFactorization &factorization,
                               numeric::Q32_32 epsilon) noexcept {
    if (epsilon.raw() < 0) {
        return {RegularityStatus::arithmetic_failure};
    }
    Q16_48 shifted_epsilon{};
    if (!convert_public_with_exponent(epsilon, factorization.matrix_exponent, shifted_epsilon)) {
        return {RegularityStatus::arithmetic_failure};
    }

    std::uint32_t sign_variation_count = 0U;
    std::uint32_t exact_zero_count = 0U;
    std::size_t block_start = 0U;
    while (block_start < elements_per_block) {
        std::size_t block_end = block_start;
        while (block_end + 1U < elements_per_block &&
               factorization.subdiagonal[block_end].raw != 0) {
            ++block_end;
        }

        regularity_WideSignedInteger polynomial_previous = regularity_make_wide_signed(1);
        regularity_WideSignedInteger polynomial_current{};
        int previous_nonzero_sign = 1;
        for (std::size_t index = block_start; index <= block_end; ++index) {
            std::int64_t shifted_diagonal_raw{};
            if (!subtract_raw_checked(factorization.diagonal[index].raw, shifted_epsilon.raw,
                                      shifted_diagonal_raw)) {
                return {
                    RegularityStatus::arithmetic_failure,
                    sign_variation_count,
                    exact_zero_count,
                };
            }

            regularity_WideSignedInteger polynomial_next{};
            if (index == block_start) {
                polynomial_next = regularity_make_wide_signed(shifted_diagonal_raw);
            } else {
                regularity_WideSignedInteger diagonal_product{};
                regularity_WideSignedInteger coupling_product{};
                const std::uint64_t coupling_magnitude =
                    magnitude(factorization.subdiagonal[index - 1U].raw);
                const UInt128 coupling_square =
                    multiply_u64(coupling_magnitude, coupling_magnitude);
                if (!regularity_multiply_wide_signed(polynomial_current, shifted_diagonal_raw,
                                                     diagonal_product) ||
                    !regularity_multiply_wide_signed_unsigned128(
                        polynomial_previous, coupling_square, coupling_product) ||
                    !regularity_subtract_wide_signed(diagonal_product, coupling_product,
                                                     polynomial_next)) {
                    return {
                        RegularityStatus::arithmetic_failure,
                        sign_variation_count,
                        exact_zero_count,
                    };
                }
                polynomial_previous = polynomial_current;
            }
            polynomial_current = polynomial_next;

            if (polynomial_current.sign == 0) {
                ++exact_zero_count;
                continue;
            }
            if (polynomial_current.sign != previous_nonzero_sign) {
                ++sign_variation_count;
            }
            previous_nonzero_sign = polynomial_current.sign;
        }
        block_start = block_end + 1U;
    }
    return {
        sign_variation_count == 0U ? RegularityStatus::regular : RegularityStatus::singular,
        sign_variation_count,
        exact_zero_count,
    };
}

} // namespace crossvmaf::metrics::speed_chroma::reference::linear
