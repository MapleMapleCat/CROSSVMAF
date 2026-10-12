#include "video/frame.hpp"
#include "metrics/adm3/reference.hpp"
#include "metrics/adm3/wavelet.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace crossvmaf::metrics::adm3::reference {
namespace {

constexpr std::array<std::int32_t, 4> filter_low{{15826, 27411, 7345, -4240}};
constexpr std::array<std::int32_t, 4> filter_high{{-4240, -7345, 27411, -15826}};
constexpr std::int32_t filter_low_sum = 46342;
constexpr std::uint32_t reciprocal_q30 = 1073741824U;

enum Band : std::size_t {
    horizontal = 0,
    vertical = 1,
    diagonal = 2,
};

[[nodiscard]] std::uint32_t magnitude(std::int32_t value) noexcept {
    const auto bits = static_cast<std::uint32_t>(value);
    return value < 0 ? (~bits + 1U) : bits;
}

[[nodiscard]] std::int64_t arithmetic_shift_right(std::int64_t value, unsigned shift) noexcept {
    if (shift == 0U) return value;
    if (value >= 0) return value >> shift;
    const std::uint64_t absolute = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    return -static_cast<std::int64_t>((absolute + mask) >> shift);
}

[[nodiscard]] std::int32_t wrap_i32(std::int64_t value) noexcept {
    return std::bit_cast<std::int32_t>(static_cast<std::uint32_t>(value));
}

[[nodiscard]] std::int32_t wrap_i16(std::int64_t value) noexcept {
    const std::uint32_t bits = static_cast<std::uint32_t>(value) & 0xffffU;
    return bits >= 0x8000U ? static_cast<std::int32_t>(bits) - 0x10000
                           : static_cast<std::int32_t>(bits);
}

[[nodiscard]] int dwt_mirror(int coordinate, int size) noexcept {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= size) return 2 * size - coordinate - 1;
    return coordinate;
}

[[nodiscard]] int threshold_mirror(int coordinate, int size) noexcept {
    if (size <= 1) return 0;
    if (coordinate < 0) return -coordinate;
    if (coordinate >= size) return size - 1;
    return coordinate;
}

[[nodiscard]] std::uint16_t load_sample(const PlaneView &plane, std::uint32_t row,
                                        std::uint32_t column, std::uint8_t bit_depth) noexcept {
    const std::byte *source = plane.data + static_cast<std::size_t>(row) * plane.stride_bytes;
    if (bit_depth == 8U) {
        return static_cast<std::uint16_t>(source[column]);
    }
    const std::size_t offset = static_cast<std::size_t>(column) * 2U;
    const std::uint16_t low = static_cast<std::uint16_t>(source[offset]);
    const std::uint16_t high = static_cast<std::uint16_t>(source[offset + 1U]);
    return static_cast<std::uint16_t>(low | (high << 8U));
}

template <typename Loader>
void dwt(Loader &&load, std::uint32_t width, std::uint32_t height, unsigned scale,
         std::uint8_t bit_depth, std::vector<std::int32_t> &low,
         std::array<std::vector<std::int32_t>, 3> &detail) noexcept {
    const std::uint32_t output_width = ceil_half(width);
    const std::uint32_t output_height = ceil_half(height);
    const std::array<unsigned, 3> vertical_shift{{0U, 16U, 16U}};
    const std::array<std::int64_t, 3> vertical_round{{0, 32768, 32768}};
    const std::array<unsigned, 3> horizontal_shift{{15U, 16U, 15U}};
    const std::array<std::int64_t, 3> horizontal_round{{16384, 32768, 16384}};

    for (std::uint32_t output_row = 0; output_row < output_height; ++output_row) {
        std::array<int, 4> source_rows{};
        for (int tap = 0; tap < 4; ++tap) {
            source_rows[static_cast<std::size_t>(tap)] =
                dwt_mirror(static_cast<int>(2U * output_row) - 1 + tap, static_cast<int>(height));
        }
        for (std::uint32_t output_column = 0; output_column < output_width; ++output_column) {
            std::array<int, 4> source_columns{};
            for (int tap = 0; tap < 4; ++tap) {
                source_columns[static_cast<std::size_t>(tap)] = dwt_mirror(
                    static_cast<int>(2U * output_column) - 1 + tap, static_cast<int>(width));
            }

            std::array<std::int32_t, 4> vertical_low{};
            std::array<std::int32_t, 4> vertical_high{};
            for (std::size_t column_tap = 0; column_tap < 4U; ++column_tap) {
                std::int64_t low_sum = 0;
                std::int64_t high_sum = 0;
                for (std::size_t row_tap = 0; row_tap < 4U; ++row_tap) {
                    const std::int32_t sample =
                        load(source_rows[row_tap], source_columns[column_tap]);
                    low_sum += static_cast<std::int64_t>(filter_low[row_tap]) * sample;
                    high_sum += static_cast<std::int64_t>(filter_high[row_tap]) * sample;
                }
                if (scale == 0U) {
                    const std::int64_t center = std::int64_t{1} << (bit_depth - 1U);
                    low_sum -= static_cast<std::int64_t>(filter_low_sum) * center;
                    vertical_low[column_tap] =
                        wrap_i16(arithmetic_shift_right(low_sum + center, bit_depth));
                    vertical_high[column_tap] =
                        wrap_i16(arithmetic_shift_right(high_sum + center, bit_depth));
                } else {
                    vertical_low[column_tap] = wrap_i32(arithmetic_shift_right(
                        low_sum + vertical_round[scale - 1U], vertical_shift[scale - 1U]));
                    vertical_high[column_tap] = wrap_i32(arithmetic_shift_right(
                        high_sum + vertical_round[scale - 1U], vertical_shift[scale - 1U]));
                }
            }

            std::int64_t low_low = 0;
            std::int64_t low_high = 0;
            std::int64_t high_low = 0;
            std::int64_t high_high = 0;
            for (std::size_t tap = 0; tap < 4U; ++tap) {
                low_low += static_cast<std::int64_t>(filter_low[tap]) * vertical_low[tap];
                low_high += static_cast<std::int64_t>(filter_high[tap]) * vertical_low[tap];
                high_low += static_cast<std::int64_t>(filter_low[tap]) * vertical_high[tap];
                high_high += static_cast<std::int64_t>(filter_high[tap]) * vertical_high[tap];
            }
            const unsigned shift = scale == 0U ? 16U : horizontal_shift[scale - 1U];
            const std::int64_t round = scale == 0U ? 32768 : horizontal_round[scale - 1U];
            const std::size_t destination =
                static_cast<std::size_t>(output_row) * output_width + output_column;
            const auto store = [scale](std::int64_t value) noexcept {
                return scale == 0U ? wrap_i16(value) : wrap_i32(value);
            };
            low[destination] = store(arithmetic_shift_right(low_low + round, shift));
            detail[vertical][destination] = store(arithmetic_shift_right(low_high + round, shift));
            detail[horizontal][destination] =
                store(arithmetic_shift_right(high_low + round, shift));
            detail[diagonal][destination] = store(arithmetic_shift_right(high_high + round, shift));
        }
    }
}

[[nodiscard]] std::int32_t ratio_q15(std::int32_t original, std::int32_t distorted,
                                     unsigned scale) noexcept {
    if (original == 0) return 32768;
    std::int64_t quotient{};
    if (scale == 0U) {
        const std::uint32_t absolute = magnitude(original);
        const std::int32_t reciprocal = static_cast<std::int32_t>(reciprocal_q30 / absolute);
        std::int64_t product = static_cast<std::int64_t>(reciprocal) * distorted;
        if (original < 0) product = -product;
        quotient = arithmetic_shift_right(product + 16384, 15U);
    } else {
        const std::uint32_t absolute = magnitude(original);
        int normalization_shift = 0;
        std::uint32_t denominator = absolute;
        if (absolute >= 32768U) {
            normalization_shift = 17 - static_cast<int>(std::countl_zero(absolute));
            denominator =
                (absolute + (std::uint32_t{1} << static_cast<unsigned>(normalization_shift - 1))) >>
                static_cast<unsigned>(normalization_shift);
        }
        const std::uint32_t reciprocal = reciprocal_q30 / denominator;
        std::int64_t product = static_cast<std::int64_t>(reciprocal) * distorted;
        if (original < 0) product = -product;
        quotient = arithmetic_shift_right(
            product + (std::int64_t{1} << static_cast<unsigned>(14 + normalization_shift)),
            static_cast<unsigned>(15 + normalization_shift));
    }
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(quotient, 0, 32768));
}

void decouple(const std::array<std::vector<std::int32_t>, 3> &reference,
              std::array<std::vector<std::int32_t>, 3> &restored,
              std::array<std::vector<std::int32_t>, 3> &additive, std::size_t coefficient_count,
              unsigned scale) noexcept {
    for (std::size_t band = 0; band < 3U; ++band) {
        for (std::size_t index = 0; index < coefficient_count; ++index) {
            const std::int32_t original = reference[band][index];
            const std::int32_t distorted = restored[band][index];
            const std::int32_t ratio = ratio_q15(original, distorted, scale);
            const std::int32_t value = wrap_i32(
                arithmetic_shift_right(static_cast<std::int64_t>(ratio) * original + 16384, 15U));
            restored[band][index] = value;
            additive[band][index] = wrap_i32(static_cast<std::int64_t>(distorted) - value);
        }
    }
}

[[nodiscard]] std::int32_t csf_scaled(std::int32_t value, std::size_t band, unsigned scale,
                                      const Config &config) noexcept {
    const std::uint32_t raw_factor =
        band == diagonal ? config.diagonal_csf_raw[scale] : config.horizontal_csf_raw[scale];
    if (scale == 0U) {
        const unsigned factor_bits = band == diagonal ? 23U : 21U;
        const std::uint32_t factor = raw_factor >> (32U - factor_bits);
        const unsigned shift = band == diagonal ? 17U : 15U;
        const std::int64_t round = band == diagonal ? 65535 : 16384;
        return wrap_i32(
            arithmetic_shift_right(static_cast<std::int64_t>(value) * factor + round, shift));
    }
    return wrap_i32(arithmetic_shift_right(
        static_cast<std::int64_t>(value) * raw_factor + (std::int64_t{1} << 27U), 28U));
}

[[nodiscard]] std::int32_t filtered_value(std::int32_t scaled, unsigned scale,
                                          bool center) noexcept {
    const std::uint32_t absolute = magnitude(scaled);
    if (scale == 0U) {
        const std::uint32_t factor = center ? 8738U : 4369U;
        return wrap_i32(
            arithmetic_shift_right(static_cast<std::int64_t>(factor) * absolute + 2048, 12U));
    }
    const std::uint32_t factor = center ? 286331153U : 143165577U;
    return wrap_i32(arithmetic_shift_right(
        static_cast<std::int64_t>(factor) * absolute - (std::int64_t{1} << 31U), 32U));
}

[[nodiscard]] std::int32_t masking_threshold(const std::array<std::vector<std::int32_t>, 3> &mask,
                                             std::uint32_t width, std::uint32_t height,
                                             std::uint32_t row, std::uint32_t column,
                                             unsigned scale, const Config &config) noexcept {
    std::int64_t threshold = 0;
    for (std::size_t band = 0; band < 3U; ++band) {
        for (int row_offset = -1; row_offset <= 1; ++row_offset) {
            const int source_row =
                threshold_mirror(static_cast<int>(row) + row_offset, static_cast<int>(height));
            for (int column_offset = -1; column_offset <= 1; ++column_offset) {
                const int source_column = threshold_mirror(static_cast<int>(column) + column_offset,
                                                           static_cast<int>(width));
                const std::size_t index = static_cast<std::size_t>(source_row) * width +
                                          static_cast<std::size_t>(source_column);
                const std::int32_t scaled = csf_scaled(mask[band][index], band, scale, config);
                threshold += filtered_value(scaled, scale, row_offset == 0 && column_offset == 0);
            }
        }
    }
    return wrap_i32(threshold);
}

[[nodiscard]] std::uint64_t cube_u32(std::uint32_t value) noexcept {
    return static_cast<std::uint64_t>(value) * value * value;
}

[[nodiscard]] std::uint64_t numerator_contribution(std::int32_t source, std::int32_t threshold,
                                                   std::size_t band, unsigned scale,
                                                   std::uint32_t width,
                                                   const Config &config) noexcept {
    std::int64_t x{};
    unsigned square_shift{};
    std::uint64_t square_round{};
    unsigned cube_shift{};
    std::uint64_t cube_round{};
    if (scale == 0U) {
        const std::uint32_t raw_factor =
            band == diagonal ? config.diagonal_csf_raw[scale] : config.horizontal_csf_raw[scale];
        const unsigned factor_bits = band == diagonal ? 23U : 21U;
        const std::uint32_t factor = raw_factor >> (32U - factor_bits);
        const std::int64_t weighted = static_cast<std::int64_t>(source) * factor;
        const std::uint64_t absolute = weighted < 0
                                           ? static_cast<std::uint64_t>(-(weighted + 1)) + 1U
                                           : static_cast<std::uint64_t>(weighted);
        const unsigned threshold_shift = band == diagonal ? 12U : 10U;
        x = static_cast<std::int64_t>(absolute) -
            (static_cast<std::int64_t>(threshold) << threshold_shift);
        square_shift = band == diagonal ? 30U : 29U;
        square_round = std::uint64_t{1} << (square_shift - 1U);
        cube_shift = ceil_log2(width) - (band == diagonal ? 3U : 4U);
        cube_round = std::uint64_t{1} << (cube_shift - 1U);
    } else {
        const std::int32_t scaled = csf_scaled(source, band, scale, config);
        x = static_cast<std::int64_t>(magnitude(scaled)) - threshold;
        square_shift = 30U;
        square_round = std::uint64_t{1} << 29U;
        cube_shift = ceil_log2(width);
        cube_round = std::uint64_t{1} << (cube_shift - 1U);
    }
    if (x <= 0) return 0U;
    const std::uint64_t unsigned_x = static_cast<std::uint64_t>(x);
    const std::uint64_t square = (unsigned_x * unsigned_x + square_round) >> square_shift;
    return (square * unsigned_x + cube_round) >> cube_shift;
}

[[nodiscard]] std::uint64_t denominator_contribution(std::int32_t source, unsigned scale,
                                                     std::uint32_t region_width) noexcept {
    const std::uint32_t absolute = magnitude(source);
    if (scale == 0U) return cube_u32(absolute);
    constexpr std::array<unsigned, 3> square_shift{{31U, 30U, 31U}};
    const unsigned shift = square_shift[scale - 1U];
    const std::uint64_t square =
        (static_cast<std::uint64_t>(absolute) * absolute + (std::uint64_t{1} << shift)) >> shift;
    const unsigned cube_shift = ceil_log2(region_width);
    const std::uint64_t cube_round = std::uint64_t{1} << (cube_shift - 1U);
    return (square * absolute + cube_round) >> cube_shift;
}

[[nodiscard]] ScaleAccumulators
accumulate_scale(const std::array<std::vector<std::int32_t>, 3> &reference,
                 const std::array<std::vector<std::int32_t>, 3> &restored,
                 const std::array<std::vector<std::int32_t>, 3> &additive, std::uint32_t width,
                 std::uint32_t height, unsigned scale, const Config &config) noexcept {
    const std::uint32_t left = width < 5U ? 0U : (width - 5U) / 10U;
    const std::uint32_t top = height < 5U ? 0U : (height - 5U) / 10U;
    const std::uint32_t right = width - left;
    const std::uint32_t bottom = height - top;
    const std::uint32_t region_width = right - left;
    const std::uint32_t region_height = bottom - top;

    const std::uint64_t area = static_cast<std::uint64_t>(region_width) * region_height;
    const unsigned denominator_row_shift =
        scale == 0U
            ? static_cast<unsigned>(std::max<int>(static_cast<int>(ceil_log2(area)) - 20, 0))
            : ceil_log2(region_height);
    const std::uint64_t denominator_round =
        denominator_row_shift == 0U ? 0U : std::uint64_t{1} << (denominator_row_shift - 1U);
    const unsigned numerator_row_shift = ceil_log2(height);
    const std::uint64_t numerator_round =
        numerator_row_shift == 0U ? 0U : std::uint64_t{1} << (numerator_row_shift - 1U);

    ScaleAccumulators total{};
    for (std::uint32_t row = top; row < bottom; ++row) {
        std::array<std::uint64_t, 3> denominator_row{};
        std::array<std::uint64_t, 3> numerator_row{};
        std::array<std::uint64_t, 3> aim_row{};
        std::array<std::uint64_t, 3> reference_row{};
        std::array<std::uint64_t, 3> restored_row{};
        std::array<std::uint64_t, 3> additive_row{};
        std::array<std::uint64_t, 3> reference_weighted_row{};
        std::array<std::uint64_t, 3> distorted_weighted_row{};
        for (std::uint32_t column = left; column < right; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * width + column;
            const std::uint32_t weight = static_cast<std::uint32_t>(index) + 1U;
            const std::int32_t dlm_threshold =
                masking_threshold(additive, width, height, row, column, scale, config);
            const std::int32_t aim_threshold =
                masking_threshold(restored, width, height, row, column, scale, config);
            for (std::size_t band = 0; band < 3U; ++band) {
                denominator_row[band] +=
                    denominator_contribution(reference[band][index], scale, region_width);
                numerator_row[band] += numerator_contribution(restored[band][index], dlm_threshold,
                                                              band, scale, width, config);
                aim_row[band] += numerator_contribution(additive[band][index], aim_threshold, band,
                                                        scale, width, config);
                reference_row[band] +=
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(reference[band][index]));
                restored_row[band] +=
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(restored[band][index]));
                additive_row[band] +=
                    static_cast<std::uint64_t>(static_cast<std::int64_t>(additive[band][index]));
                const std::uint32_t reference_bits =
                    static_cast<std::uint32_t>(reference[band][index]);
                const std::uint32_t distorted_bits =
                    static_cast<std::uint32_t>(restored[band][index]) +
                    static_cast<std::uint32_t>(additive[band][index]);
                reference_weighted_row[band] += static_cast<std::uint64_t>(reference_bits) * weight;
                distorted_weighted_row[band] += static_cast<std::uint64_t>(distorted_bits) * weight;
            }
        }
        for (std::size_t band = 0; band < 3U; ++band) {
            total.denominator[band] +=
                (denominator_row[band] + denominator_round) >> denominator_row_shift;
            total.numerator[band] += (numerator_row[band] + numerator_round) >> numerator_row_shift;
            total.aim[band] += (aim_row[band] + numerator_round) >> numerator_row_shift;
            total.reference_sum[band] += reference_row[band];
            total.restored_sum[band] += restored_row[band];
            total.additive_sum[band] += additive_row[band];
            total.reference_weighted_sum[band] += reference_weighted_row[band];
            total.distorted_weighted_sum[band] += distorted_weighted_row[band];
        }
    }
    return total;
}

} // namespace

Reference::Workspace::Workspace(std::size_t maximum_band_count) {
    for (auto &low_slot : reference_low)
        low_slot.resize(maximum_band_count);
    for (auto &low_slot : distorted_low)
        low_slot.resize(maximum_band_count);
    for (auto &detail_band : reference_detail) {
        detail_band.resize(maximum_band_count);
    }
    for (auto &detail_band : restored_detail) {
        detail_band.resize(maximum_band_count);
    }
    for (auto &detail_band : additive_detail) {
        detail_band.resize(maximum_band_count);
    }
}

Reference::Reference(FrameDescriptor descriptor, Config config)
    : descriptor_(descriptor), config_(config),
      workspace_(static_cast<std::size_t>(ceil_half(descriptor.width)) *
                 ceil_half(descriptor.height)) {
}

Result<Q32Value> Reference::compute(const FrameView &reference,
                                    const FrameView &distorted) noexcept {
    auto detailed = compute_detailed(reference, distorted);
    if (!detailed) return Result<Q32Value>::failure(detailed.error());
    return Result<Q32Value>::success(detailed.value().value);
}

Result<DetailedResult> Reference::compute_detailed(const FrameView &reference,
                                                   const FrameView &distorted) noexcept {
    const auto descriptor_validation = video::validate_frame_layout(descriptor_);
    if (!descriptor_validation) {
        return Result<DetailedResult>::failure(descriptor_validation.error());
    }
    const auto reference_validation = video::validate_plane_views(descriptor_, reference);
    if (!reference_validation) {
        return Result<DetailedResult>::failure(reference_validation.error());
    }
    const auto distorted_validation = video::validate_plane_views(descriptor_, distorted);
    if (!distorted_validation) {
        return Result<DetailedResult>::failure(distorted_validation.error());
    }
    if (reference.index != distorted.index) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_argument, "ADM3 reference and distorted frame indices must match"});
    }
    if (config_.enhancement_gain_limit.raw() != numeric::Q32_32::one_raw ||
        config_.csf_mode != 2U) {
        return Result<DetailedResult>::failure(
            {ErrorCode::unsupported_model,
             "CrossVMAF 1.0.16 ADM3 requires gain limit 1 and legacy blended CSF"});
    }

    std::uint32_t width = descriptor_.width;
    std::uint32_t height = descriptor_.height;
    std::array<ScaleAccumulators, 4> scale_accumulators{};
    std::array<ScaleValues, 4> scale_values{};
    std::size_t input_low_slot = 1U;

    for (unsigned scale = 0; scale < 4U; ++scale) {
        const std::size_t output_low_slot = 1U - input_low_slot;
        std::vector<std::int32_t> &reference_output_low = workspace_.reference_low[output_low_slot];
        std::vector<std::int32_t> &distorted_output_low = workspace_.distorted_low[output_low_slot];
        if (scale == 0U) {
            dwt(
                [&](int row, int column) noexcept {
                    return static_cast<std::int32_t>(
                        load_sample(reference.planes[0], static_cast<std::uint32_t>(row),
                                    static_cast<std::uint32_t>(column), descriptor_.bit_depth));
                },
                width, height, scale, descriptor_.bit_depth, reference_output_low,
                workspace_.reference_detail);
            dwt(
                [&](int row, int column) noexcept {
                    return static_cast<std::int32_t>(
                        load_sample(distorted.planes[0], static_cast<std::uint32_t>(row),
                                    static_cast<std::uint32_t>(column), descriptor_.bit_depth));
                },
                width, height, scale, descriptor_.bit_depth, distorted_output_low,
                workspace_.restored_detail);
        } else {
            const std::uint32_t input_width = width;
            const std::vector<std::int32_t> &reference_input_low =
                workspace_.reference_low[input_low_slot];
            const std::vector<std::int32_t> &distorted_input_low =
                workspace_.distorted_low[input_low_slot];
            dwt(
                [&](int row, int column) noexcept {
                    return reference_input_low[static_cast<std::size_t>(row) * input_width +
                                               static_cast<std::size_t>(column)];
                },
                width, height, scale, descriptor_.bit_depth, reference_output_low,
                workspace_.reference_detail);
            dwt(
                [&](int row, int column) noexcept {
                    return distorted_input_low[static_cast<std::size_t>(row) * input_width +
                                               static_cast<std::size_t>(column)];
                },
                width, height, scale, descriptor_.bit_depth, distorted_output_low,
                workspace_.restored_detail);
        }

        width = ceil_half(width);
        height = ceil_half(height);
        input_low_slot = output_low_slot;

        const std::size_t coefficient_count = static_cast<std::size_t>(width) * height;
        decouple(workspace_.reference_detail, workspace_.restored_detail,
                 workspace_.additive_detail, coefficient_count, scale);
        const ScaleAccumulators accumulators =
            accumulate_scale(workspace_.reference_detail, workspace_.restored_detail,
                             workspace_.additive_detail, width, height, scale, config_);
        scale_values[scale] = finalize_scale(accumulators, scale, width, height, config_);
        scale_accumulators[scale] = accumulators;
    }

    auto combined = combine_scales(scale_values, config_);
    if (!combined) return Result<DetailedResult>::failure(combined.error());
    return Result<DetailedResult>::success(
        DetailedResult{scale_accumulators, scale_values, combined.value()});
}

} // namespace crossvmaf::metrics::adm3::reference
