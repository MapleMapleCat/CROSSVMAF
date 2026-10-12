#include "metrics/adm3/wavelet.hpp"
#include "numeric/rounding.hpp"
#include <bit>
#include <cstdint>
#include <limits>

namespace crossvmaf::metrics::adm3::reference {
namespace {

[[nodiscard]] int floor_divide_three(int value) noexcept {
    if (value >= 0) return value / 3;
    return -((-value + 2) / 3);
}

[[nodiscard]] std::uint64_t shift_right_rne(std::uint64_t value, unsigned shift) noexcept {
    if (shift >= 64U) return 0;
    return crossvmaf::numeric::detail::shift_right_rne(value, shift);
}

[[nodiscard]] numeric::Q32_32 scale_power_of_two(numeric::Q32_32 value, int exponent) noexcept {
    if (value.raw() <= 0) return numeric::Q32_32{};
    const auto raw = static_cast<std::uint64_t>(value.raw());
    if (exponent >= 0) {
        if (exponent >= 63 ||
            raw > (static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) >>
                   static_cast<unsigned>(exponent))) {
            return numeric::Q32_32::from_raw(std::numeric_limits<std::int64_t>::max());
        }
        return numeric::Q32_32::from_raw(
            static_cast<std::int64_t>(raw << static_cast<unsigned>(exponent)));
    }
    const unsigned shift = static_cast<unsigned>(-exponent);
    return numeric::Q32_32::from_raw(static_cast<std::int64_t>(shift_right_rne(raw, shift)));
}

[[nodiscard]] std::uint32_t border_start(std::uint32_t size) noexcept {
    if (size < 5U) return 0U;
    return (size - 5U) / 10U;
}

[[nodiscard]] numeric::Q32_32 add_three(numeric::Q32_32 first, numeric::Q32_32 second,
                                        numeric::Q32_32 third) noexcept {
    return numeric::Q32_32::add_saturating(numeric::Q32_32::add_saturating(first, second), third);
}

} // namespace

unsigned ceil_log2(std::uint64_t value) noexcept {
    if (value <= 1U) return 0U;
    return std::bit_width(value - 1U);
}

numeric::Q32_32 cuberoot_scaled_u64(std::uint64_t accumulator, int binary_shift) noexcept {
    if (accumulator == 0U) return numeric::Q32_32{};
    const int msb = static_cast<int>(std::bit_width(accumulator) - 1U);
    const int root_exponent = floor_divide_three(msb - binary_shift);
    const int normalization_shift = 32 - binary_shift - 3 * root_exponent;

    std::uint64_t normalized_raw{};
    if (normalization_shift >= 0) {
        const unsigned shift = static_cast<unsigned>(normalization_shift);
        normalized_raw = shift >= 64U ? 0U : accumulator << shift;
    } else {
        normalized_raw = shift_right_rne(accumulator, static_cast<unsigned>(-normalization_shift));
    }
    if (normalized_raw == 0U ||
        normalized_raw > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        return numeric::Q32_32{};
    }
    const auto normalized_root =
        numeric::Q32_32::cbrt(numeric::Q32_32::from_raw(static_cast<std::int64_t>(normalized_raw)));
    if (!normalized_root) return numeric::Q32_32{};
    return scale_power_of_two(normalized_root.value(), root_exponent);
}

ScaleValues finalize_scale(const ScaleAccumulators &accumulators, unsigned scale,
                           std::uint32_t width, std::uint32_t height,
                           const Config &config) noexcept {
    const std::uint32_t left = border_start(width);
    const std::uint32_t top = border_start(height);
    const std::uint32_t region_width = width - 2U * left;
    const std::uint32_t region_height = height - 2U * top;
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(region_width) * region_height;

    int denominator_shift{};
    std::array<int, 3> numerator_shift{};
    if (scale == 0U) {
        const int area_shift = static_cast<int>(ceil_log2(pixel_count)) - 20;
        denominator_shift = 18 - (area_shift > 0 ? area_shift : 0);
        const int horizontal_width_shift = static_cast<int>(ceil_log2(width)) - 4;
        const int diagonal_width_shift = static_cast<int>(ceil_log2(width)) - 3;
        const int height_shift = static_cast<int>(ceil_log2(height));
        numerator_shift = {
            52 - horizontal_width_shift - height_shift,
            52 - horizontal_width_shift - height_shift,
            57 - diagonal_width_shift - height_shift,
        };
    } else {
        constexpr std::array<int, 3> denominator_base{{32, 27, 23}};
        constexpr std::array<int, 3> numerator_base{{45, 39, 36}};
        const int width_shift = static_cast<int>(ceil_log2(region_width));
        const int height_shift = static_cast<int>(ceil_log2(region_height));
        denominator_shift = denominator_base[scale - 1U] - width_shift - height_shift;
        numerator_shift.fill(numerator_base[scale - 1U] - static_cast<int>(ceil_log2(width)) -
                             static_cast<int>(ceil_log2(height)));
    }

    numeric::Q32_32 pixel_count_q32{};
    if (pixel_count <= static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        pixel_count_q32 = numeric::Q32_32::from_integer(static_cast<std::int32_t>(pixel_count));
    }
    const numeric::Q32_32 noise_value =
        numeric::Q32_32::multiply(pixel_count_q32, config.noise_weight);
    const auto noise_root_result = numeric::Q32_32::cbrt(noise_value);
    const numeric::Q32_32 noise_root =
        noise_root_result ? noise_root_result.value() : numeric::Q32_32{};

    std::array<numeric::Q32_32, 3> denominator_roots{};
    std::array<numeric::Q32_32, 3> numerator_roots{};
    std::array<numeric::Q32_32, 3> aim_roots{};
    for (std::size_t band = 0; band < 3U; ++band) {
        const std::uint32_t factor_raw =
            band == 2U ? config.diagonal_csf_raw[scale] : config.horizontal_csf_raw[scale];
        denominator_roots[band] = numeric::Q32_32::add_saturating(
            numeric::Q32_32::multiply(
                cuberoot_scaled_u64(accumulators.denominator[band], denominator_shift),
                numeric::Q32_32::from_raw(static_cast<std::int64_t>(factor_raw))),
            noise_root);
        numerator_roots[band] = numeric::Q32_32::add_saturating(
            cuberoot_scaled_u64(accumulators.numerator[band], numerator_shift[band]), noise_root);
        aim_roots[band] = cuberoot_scaled_u64(accumulators.aim[band], numerator_shift[band]);
    }

    return {
        add_three(denominator_roots[0], denominator_roots[1], denominator_roots[2]),
        add_three(numerator_roots[0], numerator_roots[1], numerator_roots[2]),
        add_three(aim_roots[0], aim_roots[1], aim_roots[2]),
    };
}

Result<Q32Value> combine_scales(const std::array<ScaleValues, 4> &scales,
                                const Config &config) noexcept {
    numeric::Q32_32 denominator{};
    numeric::Q32_32 numerator{};
    numeric::Q32_32 aim{};
    for (const ScaleValues &scale : scales) {
        denominator = numeric::Q32_32::add_saturating(denominator, scale.denominator);
        numerator = numeric::Q32_32::add_saturating(numerator, scale.numerator);
        aim = numeric::Q32_32::add_saturating(aim, scale.aim);
    }

    if (denominator.raw() == 0) {
        return Result<Q32Value>::success(Q32Value{numeric::Q32_32::from_integer(1).raw()});
    }
    const auto dlm_score = numeric::Q32_32::divide(numerator, denominator);
    const auto aim_score = numeric::Q32_32::divide(aim, denominator);
    if (!dlm_score || !aim_score) {
        return Result<Q32Value>::failure({ErrorCode::invalid_state, "ADM3 final division failed"});
    }

    const numeric::Q32_32 one = numeric::Q32_32::from_integer(1);
    const numeric::Q32_32 dlm_component =
        numeric::Q32_32::multiply(dlm_score.value(), config.dlm_weight);
    const numeric::Q32_32 aim_component =
        numeric::Q32_32::multiply(numeric::Q32_32::subtract_saturating(one, aim_score.value()),
                                  numeric::Q32_32::subtract_saturating(one, config.dlm_weight));
    const numeric::Q32_32 combined = numeric::Q32_32::add_saturating(dlm_component, aim_component);
    const numeric::Q32_32 clipped = combined < config.minimum ? config.minimum : combined;
    return Result<Q32Value>::success(Q32Value{clipped.raw()});
}

} // namespace crossvmaf::metrics::adm3::reference
