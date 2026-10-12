#include "video/frame.hpp"
#include "metrics/cambi/reference.hpp"
#include "metrics/cambi/c_values.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace crossvmaf::metrics::cambi::reference {
namespace {

constexpr std::uint16_t base_window_size = 65U;
constexpr std::uint16_t mask_filter_size = 7U;
constexpr std::uint16_t value_band_base = 67U;
constexpr std::uint16_t value_band_size = 497U;
constexpr std::uint16_t visibility_luma = 78U;
constexpr std::array<std::uint16_t, 4> tvi_thresholds{{182U, 309U, 436U, 563U}};
constexpr std::array<std::uint32_t, 4> contrast_weights{{1U, 2U, 3U, 4U}};

[[nodiscard]] std::uint16_t load_sample(const PlaneView &plane, std::uint32_t row,
                                        std::uint32_t column, std::uint8_t bit_depth) noexcept {
    const std::byte *source = plane.data + static_cast<std::size_t>(row) * plane.stride_bytes;
    if (bit_depth == 8U) {
        return static_cast<std::uint16_t>(source[column]);
    }
    const std::size_t offset = static_cast<std::size_t>(column) * 2U;
    const auto low = static_cast<std::uint16_t>(source[offset]);
    const auto high = static_cast<std::uint16_t>(source[offset + 1U]);
    return static_cast<std::uint16_t>(low | (high << 8U));
}

[[nodiscard]] Result<void> preprocess(const FrameDescriptor &descriptor, const PlaneView &plane,
                                      std::vector<std::uint16_t> &normalized,
                                      std::vector<std::uint16_t> &filtered) noexcept {
    const std::uint32_t maximum =
        descriptor.bit_depth == 16U ? 0xffffU : (std::uint32_t{1} << descriptor.bit_depth) - 1U;
    const unsigned right_shift = descriptor.bit_depth > 10U ? descriptor.bit_depth - 10U : 0U;
    const std::uint32_t rounding = right_shift == 0U ? 0U : std::uint32_t{1} << (right_shift - 1U);
    for (std::uint32_t row = 0; row < descriptor.height; ++row) {
        for (std::uint32_t column = 0; column < descriptor.width; ++column) {
            const std::uint32_t sample = load_sample(plane, row, column, descriptor.bit_depth);
            if (sample > maximum) {
                return Result<void>::failure(
                    {ErrorCode::invalid_argument,
                     "CAMBI input contains a sample above the declared bit depth"});
            }
            const std::uint32_t value = descriptor.bit_depth < 10U
                                            ? sample << (10U - descriptor.bit_depth)
                                            : (sample + rounding) >> right_shift;
            normalized[static_cast<std::size_t>(row) * descriptor.width + column] =
                static_cast<std::uint16_t>(value);
        }
    }

    if (descriptor.bit_depth >= 10U) return Result<void>::success();

    for (std::uint32_t row = 0; row < descriptor.height; ++row) {
        for (std::uint32_t column = 0; column < descriptor.width; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * descriptor.width + column;
            if (row + 1U < descriptor.height && column + 1U < descriptor.width) {
                filtered[index] =
                    static_cast<std::uint16_t>((normalized[index] + normalized[index + 1U] +
                                                normalized[index + descriptor.width] +
                                                normalized[index + descriptor.width + 1U]) >>
                                               2U);
            } else if (row + 1U < descriptor.height) {
                filtered[index] = static_cast<std::uint16_t>(
                    (normalized[index] + normalized[index + descriptor.width]) >> 1U);
            } else if (column + 1U < descriptor.width) {
                filtered[index] =
                    static_cast<std::uint16_t>((normalized[index] + normalized[index + 1U]) >> 1U);
            } else {
                filtered[index] = normalized[index];
            }
        }
    }
    return Result<void>::success();
}

[[nodiscard]] std::uint16_t mode3(std::uint16_t a, std::uint16_t b, std::uint16_t c) noexcept {
    if (a == b || a == c) return a;
    if (b == c) return b;
    return std::min({a, b, c});
}

void mode_filter(const std::vector<std::uint16_t> &image, std::uint32_t width, std::uint32_t height,
                 std::vector<std::uint16_t> &output) noexcept {
    const std::size_t pixel_count = static_cast<std::size_t>(width) * height;
    std::copy_n(image.begin(), pixel_count, output.begin());
    if (width < 3U || height < 3U) return;
    const auto horizontal = [&](std::uint32_t row, std::uint32_t column) noexcept {
        const std::size_t index = static_cast<std::size_t>(row) * width + column;
        if (column == 0U || column + 1U == width) return image[index];
        return mode3(image[index - 1U], image[index], image[index + 1U]);
    };
    for (std::uint32_t row = 1U; row + 1U < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            output[static_cast<std::size_t>(row) * width + column] =
                mode3(horizontal(row - 1U, column), horizontal(row, column),
                      horizontal(row + 1U, column));
        }
    }
}

void spatial_mask(const std::vector<std::uint16_t> &image, std::uint32_t width,
                  std::uint32_t height, std::vector<std::uint16_t> &derivative,
                  std::vector<std::uint16_t> &mask) noexcept {
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::size_t index = static_cast<std::size_t>(row) * width + column;
            const bool horizontal = column + 1U == width || image[index] == image[index + 1U];
            const bool vertical = row + 1U == height || image[index] == image[index + width];
            derivative[index] = horizontal && vertical ? 1U : 0U;
        }
    }

    const std::uint16_t threshold = spatial_mask_index(width, height);
    constexpr int radius = static_cast<int>(mask_filter_size >> 1U);
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            std::uint32_t count{};
            for (int row_offset = -radius; row_offset <= radius; ++row_offset) {
                const int source_row = static_cast<int>(row) + row_offset;
                if (source_row < 0 || source_row >= static_cast<int>(height)) {
                    continue;
                }
                for (int column_offset = -radius; column_offset <= radius; ++column_offset) {
                    const int source_column = static_cast<int>(column) + column_offset;
                    if (source_column < 0 || source_column >= static_cast<int>(width)) {
                        continue;
                    }
                    count += derivative[static_cast<std::size_t>(source_row) * width +
                                        static_cast<std::size_t>(source_column)];
                }
            }
            mask[static_cast<std::size_t>(row) * width + column] = count > threshold ? 1U : 0U;
        }
    }
}

void decimate(const std::vector<std::uint16_t> &input, std::uint32_t input_width,
              std::uint32_t input_height, std::vector<std::uint16_t> &output) noexcept {
    const std::uint32_t width = (input_width + 1U) >> 1U;
    const std::uint32_t height = (input_height + 1U) >> 1U;
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            output[static_cast<std::size_t>(row) * width + column] =
                input[static_cast<std::size_t>(row * 2U) * input_width + column * 2U];
        }
    }
}

void update_histogram(std::array<std::uint16_t, value_band_size> &histogram, std::uint16_t value,
                      int delta) noexcept {
    const std::uint16_t compact = static_cast<std::uint16_t>(value - value_band_base);
    if (compact >= value_band_size) return;
    if (delta > 0) {
        ++histogram[compact];
    } else {
        --histogram[compact];
    }
}

[[nodiscard]] CValue
calculate_pixel_c_value(const std::array<std::uint16_t, value_band_size> &histogram,
                        std::uint16_t image_value) noexcept {
    const int compact = static_cast<int>(image_value) - value_band_base;
    if (compact < 0 || compact >= static_cast<int>(value_band_size)) return 0U;
    const std::uint16_t center = histogram[static_cast<std::size_t>(compact)];
    CValue maximum{};
    const std::uint16_t adjusted_value = image_value + 4U;
    for (std::size_t diff = 0; diff < contrast_weights.size(); ++diff) {
        const int distance = static_cast<int>(diff) + 1;
        if (adjusted_value > tvi_thresholds[diff] || adjusted_value + distance <= visibility_luma) {
            continue;
        }
        const int upper = compact + distance;
        const int lower = compact - distance;
        const std::uint16_t upper_count = upper < static_cast<int>(value_band_size)
                                              ? histogram[static_cast<std::size_t>(upper)]
                                              : 0U;
        const std::uint16_t lower_count =
            lower >= 0 ? histogram[static_cast<std::size_t>(lower)] : 0U;
        maximum = std::max(maximum, c_value_from_counts(center, std::max(upper_count, lower_count),
                                                        contrast_weights[diff]));
    }
    return maximum;
}

void calculate_c_values(const std::vector<std::uint16_t> &image,
                        const std::vector<std::uint16_t> &mask, std::uint32_t width,
                        std::uint32_t height, std::uint16_t window_size,
                        std::vector<CValue> &values) noexcept {
    const int radius = static_cast<int>(window_size >> 1U);
    std::fill_n(values.begin(), static_cast<std::size_t>(width) * height, CValue{});
    for (std::uint32_t row = 0; row < height; ++row) {
        std::array<std::uint16_t, value_band_size> histogram{};
        const int top = std::max<int>(static_cast<int>(row) - radius, 0);
        const int bottom =
            std::min<int>(static_cast<int>(row) + radius + 1, static_cast<int>(height));
        const int initial_right = std::min<int>(radius + 1, static_cast<int>(width));
        for (int source_row = top; source_row < bottom; ++source_row) {
            for (int source_column = 0; source_column < initial_right; ++source_column) {
                const std::size_t index = static_cast<std::size_t>(source_row) * width +
                                          static_cast<std::size_t>(source_column);
                if (mask[index] != 0U) update_histogram(histogram, image[index], 1);
            }
        }
        for (std::uint32_t column = 0; column < width; ++column) {
            if (column != 0U) {
                const int remove_column = static_cast<int>(column) - radius - 1;
                const int add_column = static_cast<int>(column) + radius;
                for (int source_row = top; source_row < bottom; ++source_row) {
                    if (remove_column >= 0) {
                        const std::size_t index = static_cast<std::size_t>(source_row) * width +
                                                  static_cast<std::size_t>(remove_column);
                        if (mask[index] != 0U) {
                            update_histogram(histogram, image[index], -1);
                        }
                    }
                    if (add_column < static_cast<int>(width)) {
                        const std::size_t index = static_cast<std::size_t>(source_row) * width +
                                                  static_cast<std::size_t>(add_column);
                        if (mask[index] != 0U) {
                            update_histogram(histogram, image[index], 1);
                        }
                    }
                }
            }
            const std::size_t index = static_cast<std::size_t>(row) * width + column;
            if (mask[index] != 0U) {
                values[index] = calculate_pixel_c_value(histogram, image[index]);
            }
        }
    }
}

} // namespace

Reference::Workspace::Workspace(std::size_t pixel_count)
    : image_primary(pixel_count), image_secondary(pixel_count), mask_derivative(pixel_count),
      mask_primary(pixel_count), mask_secondary(pixel_count), c_values(pixel_count) {
}

Reference::Reference(FrameDescriptor descriptor, Config config,
                     DetailedDiagnosticsMode diagnostics_mode)
    : descriptor_(descriptor), config_(config), diagnostics_mode_(diagnostics_mode),
      workspace_(static_cast<std::size_t>(descriptor.width) * descriptor.height) {
}

Result<Q32Value> Reference::compute(const FrameView &distorted) {
    auto detailed = compute_detailed(distorted);
    if (!detailed) return Result<Q32Value>::failure(detailed.error());
    return Result<Q32Value>::success(detailed.value().value);
}

Result<DetailedResult> Reference::compute_detailed(const FrameView &distorted) {
    const auto descriptor_validation = video::validate_frame_layout(descriptor_);
    if (!descriptor_validation) {
        return Result<DetailedResult>::failure(descriptor_validation.error());
    }
    const auto frame_validation = video::validate_plane_views(descriptor_, distorted);
    if (!frame_validation) {
        return Result<DetailedResult>::failure(frame_validation.error());
    }
    const auto metric_limit = metrics::cambi::validate_input(descriptor_);
    if (!metric_limit) {
        return Result<DetailedResult>::failure(metric_limit.error());
    }

    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(descriptor_.width) * descriptor_.height;
    const bool high_res =
        config_.high_res_speedup == 1080U && pixel_count >= std::uint64_t{1920} * 1080U;
    const std::uint16_t window_size =
        adjusted_window_size(base_window_size, descriptor_.width, descriptor_.height, high_res);
    if (static_cast<std::uint32_t>(window_size) * window_size >= 4226U) {
        return Result<DetailedResult>::failure(
            {ErrorCode::unsupported_format,
             "CAMBI adjusted window exceeds the deterministic histogram limit"});
    }

    const auto preprocess_result = preprocess(descriptor_, distorted.planes[0],
                                              workspace_.image_primary, workspace_.image_secondary);
    if (!preprocess_result) {
        return Result<DetailedResult>::failure(preprocess_result.error());
    }
    std::vector<std::uint16_t> *image =
        descriptor_.bit_depth < 10U ? &workspace_.image_secondary : &workspace_.image_primary;
    std::vector<std::uint16_t> *image_output =
        descriptor_.bit_depth < 10U ? &workspace_.image_primary : &workspace_.image_secondary;
    spatial_mask(*image, descriptor_.width, descriptor_.height, workspace_.mask_derivative,
                 workspace_.mask_primary);
    std::vector<std::uint16_t> *mask = &workspace_.mask_primary;
    std::vector<std::uint16_t> *mask_output = &workspace_.mask_secondary;
    std::uint32_t width = descriptor_.width;
    std::uint32_t height = descriptor_.height;
    std::array<numeric::Q32_32, 5> scale_scores{};
    std::optional<DetailedDiagnostics> diagnostics;
    if (diagnostics_mode_ == DetailedDiagnosticsMode::c_value_maps) {
        diagnostics.emplace();
    }
    for (std::size_t scale = 0; scale < scale_scores.size(); ++scale) {
        if (scale != 0U || high_res) {
            decimate(*image, width, height, *image_output);
            std::swap(image, image_output);
            decimate(*mask, width, height, *mask_output);
            std::swap(mask, mask_output);
            width = (width + 1U) >> 1U;
            height = (height + 1U) >> 1U;
        }
        mode_filter(*image, width, height, *image_output);
        std::swap(image, image_output);
        calculate_c_values(*image, *mask, width, height, window_size, workspace_.c_values);
        if (diagnostics.has_value()) {
            CValueMap &c_value_map = diagnostics->c_value_maps[scale];
            c_value_map.width = width;
            c_value_map.height = height;
            c_value_map.values.assign(workspace_.c_values.begin(),
                                      workspace_.c_values.begin() +
                                          static_cast<std::size_t>(width) * height);
        }
        scale_scores[scale] = pool_topk_in_place(std::span<CValue>(
            workspace_.c_values.data(), static_cast<std::size_t>(width) * height));
    }

    numeric::Q32_32 combined = combine_scale_scores(scale_scores, window_size);
    if (combined > config_.maximum) combined = config_.maximum;
    return Result<DetailedResult>::success(DetailedResult{
        scale_scores, Q32Value{combined.raw()}, window_size, high_res, std::move(diagnostics)});
}

} // namespace crossvmaf::metrics::cambi::reference
