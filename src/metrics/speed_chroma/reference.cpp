#include "video/frame.hpp"
#include "metrics/speed_chroma/reference.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crossvmaf::metrics::speed_chroma::reference {
namespace {

using Q = numeric::Q32_32;

constexpr Q canonical_sigma_nn = Q::from_raw(816'043'776LL);
constexpr Q canonical_nn_floor = Q::from_raw(429'496'736LL);

constexpr std::array<std::uint32_t, 9> antialias_filter{{
    324'909'248U,
    425'645'824U,
    516'207'552U,
    579'548'672U,
    602'344'640U,
    579'548'672U,
    516'207'552U,
    425'645'824U,
    324'909'248U,
}};

constexpr std::array<std::uint32_t, 3> local_mean_filter{{
    714'590'272U,
    2'865'786'624U,
    714'590'272U,
}};

using AxisSample = detail::AxisSample;

[[nodiscard]] std::int64_t round_shift_rne(std::int64_t value, unsigned shift) noexcept {
    if (shift == 0U) return value;
    const bool negative = value < 0;
    const std::uint64_t magnitude = negative ? static_cast<std::uint64_t>(-(value + 1)) + 1U
                                             : static_cast<std::uint64_t>(value);
    std::uint64_t quotient = magnitude >> shift;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    const std::uint64_t remainder = magnitude & mask;
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (quotient & 1U) != 0U)) {
        ++quotient;
    }
    if (!negative) {
        return quotient > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
                   ? std::numeric_limits<std::int64_t>::max()
                   : static_cast<std::int64_t>(quotient);
    }
    const std::uint64_t negative_limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1U;
    if (quotient >= negative_limit) {
        return std::numeric_limits<std::int64_t>::min();
    }
    return -static_cast<std::int64_t>(quotient);
}

[[nodiscard]] std::int32_t clamp_i32(std::int64_t value) noexcept {
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(
        value, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()));
}

[[nodiscard]] constexpr std::uint64_t magnitude(std::int64_t value) noexcept {
    const auto bits = static_cast<std::uint64_t>(value);
    return value < 0 ? (~bits + 1U) : bits;
}

[[nodiscard]] Q quantize_binary32(Q value) noexcept {
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
    if (rounded > (static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) >> shift)) {
        return Q::from_raw(negative ? std::numeric_limits<std::int64_t>::min()
                                    : std::numeric_limits<std::int64_t>::max());
    }
    const std::uint64_t raw = rounded << shift;
    if (!negative) return Q::from_raw(static_cast<std::int64_t>(raw));
    const std::uint64_t negative_limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1U;
    return raw >= negative_limit ? Q::from_raw(std::numeric_limits<std::int64_t>::min())
                                 : Q::from_raw(-static_cast<std::int64_t>(raw));
}

[[nodiscard]] Q binary32_add(Q lhs, Q rhs) noexcept {
    return quantize_binary32(Q::add_saturating(lhs, rhs));
}

[[nodiscard]] Q binary32_subtract(Q lhs, Q rhs) noexcept {
    return quantize_binary32(Q::subtract_saturating(lhs, rhs));
}

[[nodiscard]] Q binary32_multiply(Q lhs, Q rhs) noexcept {
    return quantize_binary32(Q::multiply(lhs, rhs));
}

[[nodiscard]] Q sample_to_q32(std::int32_t value) noexcept {
    constexpr unsigned shift = Q::fractional_bits - sample_fractional_bits;
    constexpr std::int64_t scale = std::int64_t{1} << shift;
    static_assert(static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::min()) >=
                  std::numeric_limits<std::int64_t>::min() / scale);
    static_assert(static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) <=
                  std::numeric_limits<std::int64_t>::max() / scale);
    return quantize_binary32(Q::from_raw(static_cast<std::int64_t>(value) * scale));
}

[[nodiscard]] int mirror(int coordinate, int size) noexcept {
    if (coordinate < 0) return -coordinate;
    if (coordinate >= size) return 2 * size - coordinate - 2;
    return coordinate;
}

[[nodiscard]] std::uint16_t load_sample(const PlaneView &plane, std::uint32_t row,
                                        std::uint32_t column, std::uint8_t bit_depth) noexcept {
    const std::byte *source = plane.data + static_cast<std::size_t>(row) * plane.stride_bytes;
    if (bit_depth == 8U) {
        return std::to_integer<std::uint8_t>(source[column]);
    }
    const std::size_t offset = static_cast<std::size_t>(column) * 2U;
    const std::uint16_t low = std::to_integer<std::uint8_t>(source[offset]);
    const std::uint16_t high = std::to_integer<std::uint8_t>(source[offset + 1U]);
    return static_cast<std::uint16_t>(low | (high << 8U));
}

[[nodiscard]] Result<void> validate_plane_samples(const PlaneView &plane, std::uint32_t width,
                                                  std::uint32_t height,
                                                  std::uint8_t bit_depth) noexcept {
    const std::uint32_t maximum = bit_depth == 16U ? 0xffffU : (std::uint32_t{1} << bit_depth) - 1U;
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            if (load_sample(plane, row, column, bit_depth) > maximum) {
                return Result<void>::failure(
                    {ErrorCode::invalid_argument,
                     "Speed Chroma sample exceeds the declared bit depth"});
            }
        }
    }
    return Result<void>::success();
}

[[nodiscard]] Result<std::vector<std::int32_t>> load_normalized_plane(const PlaneView &plane,
                                                                      std::uint32_t width,
                                                                      std::uint32_t height,
                                                                      std::uint8_t bit_depth) {
    if (bit_depth < 8U || bit_depth > 16U) {
        return Result<std::vector<std::int32_t>>::failure(
            {ErrorCode::unsupported_format, "Speed Chroma normalization requires 8 to 16 bits"});
    }
    std::vector<std::int32_t> output(static_cast<std::size_t>(width) * height);
    const std::uint32_t maximum = (std::uint32_t{1} << bit_depth) - 1U;
    const unsigned normalization_shift = sample_fractional_bits + 8U - bit_depth;
    const std::int64_t normalization_scale =
        static_cast<std::int64_t>(std::uint64_t{1} << normalization_shift);
    constexpr std::int64_t center = 128LL * (std::int64_t{1} << sample_fractional_bits);
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::uint16_t sample = load_sample(plane, row, column, bit_depth);
            if (sample > maximum) {
                return Result<std::vector<std::int32_t>>::failure(
                    {ErrorCode::invalid_argument,
                     "Speed Chroma sample exceeds the declared bit depth"});
            }
            const std::int64_t normalized =
                static_cast<std::int64_t>(sample) * normalization_scale - center;
            output[static_cast<std::size_t>(row) * width + column] =
                static_cast<std::int32_t>(normalized);
        }
    }
    return Result<std::vector<std::int32_t>>::success(std::move(output));
}

[[nodiscard]] Result<std::vector<AxisSample>> make_axis_samples(std::uint32_t source_size,
                                                                std::uint32_t destination_size) {
    std::vector<AxisSample> samples(destination_size);
    const std::uint64_t denominator = static_cast<std::uint64_t>(destination_size) * 2U;
    for (std::uint32_t destination = 0; destination < destination_size; ++destination) {
        const std::uint64_t numerator =
            (static_cast<std::uint64_t>(destination) * 2U + 1U) * source_size - destination_size;
        const std::uint64_t first = numerator / denominator;
        const std::uint64_t second = (numerator + denominator - 1U) / denominator;
        const std::uint64_t fractional_numerator = numerator - first * denominator;
        if (first >= source_size || second >= source_size ||
            denominator > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return Result<std::vector<AxisSample>>::failure(
                {ErrorCode::unsupported_format,
                 "Speed Chroma bilinear coordinate is out of range"});
        }
        const auto fraction = Q::from_ratio(static_cast<std::int64_t>(fractional_numerator),
                                            static_cast<std::int64_t>(denominator));
        if (!fraction) {
            return Result<std::vector<AxisSample>>::failure(fraction.error());
        }
        samples[destination] = {
            static_cast<std::uint32_t>(first),
            static_cast<std::uint32_t>(second),
            fraction.value(),
        };
    }
    return Result<std::vector<AxisSample>>::success(std::move(samples));
}

[[nodiscard]] Result<std::vector<std::int32_t>> scale_bilinear(std::span<const std::int32_t> input,
                                                               std::uint32_t source_width,
                                                               std::uint32_t source_height,
                                                               std::uint32_t destination_width,
                                                               std::uint32_t destination_height) {
    const auto x_samples = make_axis_samples(source_width, destination_width);
    const auto y_samples = make_axis_samples(source_height, destination_height);
    if (!x_samples || !y_samples) {
        return Result<std::vector<std::int32_t>>::failure(!x_samples ? x_samples.error()
                                                                     : y_samples.error());
    }
    std::vector<std::int32_t> output(static_cast<std::size_t>(destination_width) *
                                     destination_height);
    const Q one = Q::from_integer(1);
    for (std::uint32_t row = 0; row < destination_height; ++row) {
        const AxisSample &y = y_samples.value()[row];
        const Q y_first = Q::subtract_saturating(one, y.fraction);
        for (std::uint32_t column = 0; column < destination_width; ++column) {
            const AxisSample &x = x_samples.value()[column];
            const Q x_first = Q::subtract_saturating(one, x.fraction);
            const std::array<Q, 4> weights{{
                Q::multiply(y_first, x_first),
                Q::multiply(y_first, x.fraction),
                Q::multiply(y.fraction, x_first),
                Q::multiply(y.fraction, x.fraction),
            }};
            const std::array<std::size_t, 4> sources{{
                static_cast<std::size_t>(y.first) * source_width + x.first,
                static_cast<std::size_t>(y.first) * source_width + x.second,
                static_cast<std::size_t>(y.second) * source_width + x.first,
                static_cast<std::size_t>(y.second) * source_width + x.second,
            }};
            std::int64_t accumulator = 0;
            for (std::size_t index = 0; index < weights.size(); ++index) {
                accumulator +=
                    static_cast<std::int64_t>(input[sources[index]]) * weights[index].raw();
            }
            output[static_cast<std::size_t>(row) * destination_width + column] =
                clamp_i32(round_shift_rne(accumulator, 32U));
        }
    }
    return Result<std::vector<std::int32_t>>::success(std::move(output));
}

[[nodiscard]] std::vector<std::int32_t> scale_nearest(std::span<const std::int32_t> input,
                                                      std::uint32_t source_width,
                                                      std::uint32_t source_height,
                                                      std::uint32_t destination_width,
                                                      std::uint32_t destination_height) {
    std::vector<std::int32_t> output(static_cast<std::size_t>(destination_width) *
                                     destination_height);
    for (std::uint32_t row = 0; row < destination_height; ++row) {
        const std::uint32_t source_row = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(row) * source_height) / destination_height);
        for (std::uint32_t column = 0; column < destination_width; ++column) {
            const std::uint32_t source_column = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(column) * source_width) / destination_width);
            output[static_cast<std::size_t>(row) * destination_width + column] =
                input[static_cast<std::size_t>(source_row) * source_width + source_column];
        }
    }
    return output;
}

template <std::size_t FilterSize, typename Loader>
[[nodiscard]] std::vector<Q>
filter_separable_q32(std::uint32_t width, std::uint32_t height,
                     const std::array<std::uint32_t, FilterSize> &filter, Loader &&load) {
    static_assert((FilterSize & 1U) == 1U);
    std::vector<Q> vertical(static_cast<std::size_t>(width) * height);
    std::vector<Q> output(vertical.size());
    constexpr int radius = static_cast<int>(FilterSize / 2U);
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            Q accumulator{};
            for (std::size_t tap = 0; tap < FilterSize; ++tap) {
                const int source_row =
                    mirror(static_cast<int>(row) - radius + static_cast<int>(tap),
                           static_cast<int>(height));
                accumulator = binary32_add(
                    accumulator,
                    binary32_multiply(Q::from_raw(static_cast<std::int64_t>(filter[tap])),
                                      load(static_cast<std::uint32_t>(source_row), column)));
            }
            vertical[static_cast<std::size_t>(row) * width + column] = accumulator;
        }
    }
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            Q accumulator{};
            for (std::size_t tap = 0; tap < FilterSize; ++tap) {
                const int source_column =
                    mirror(static_cast<int>(column) - radius + static_cast<int>(tap),
                           static_cast<int>(width));
                accumulator = binary32_add(
                    accumulator,
                    binary32_multiply(Q::from_raw(static_cast<std::int64_t>(filter[tap])),
                                      vertical[static_cast<std::size_t>(row) * width +
                                               static_cast<std::uint32_t>(source_column)]));
            }
            output[static_cast<std::size_t>(row) * width + column] = accumulator;
        }
    }
    return output;
}

[[nodiscard]] Result<std::vector<std::int32_t>>
preprocess_channel_impl(const PlaneView &plane, const Dimensions &dimensions, const Config &config,
                        std::uint8_t bit_depth) {
    auto loaded = load_normalized_plane(plane, dimensions.original_width,
                                        dimensions.original_height, bit_depth);
    if (!loaded) {
        return Result<std::vector<std::int32_t>>::failure(loaded.error());
    }
    std::vector<std::int32_t> scaled;
    if (dimensions.original_width == dimensions.scaled_width &&
        dimensions.original_height == dimensions.scaled_height) {
        scaled = std::move(loaded).value();
    } else if (config.prescale_method == PrescaleMethod::bilinear) {
        auto result =
            scale_bilinear(loaded.value(), dimensions.original_width, dimensions.original_height,
                           dimensions.scaled_width, dimensions.scaled_height);
        if (!result) {
            return Result<std::vector<std::int32_t>>::failure(result.error());
        }
        scaled = std::move(result).value();
    } else {
        scaled =
            scale_nearest(loaded.value(), dimensions.original_width, dimensions.original_height,
                          dimensions.scaled_width, dimensions.scaled_height);
    }

    const std::vector<Q> antialiased = filter_separable_q32(
        dimensions.scaled_width, dimensions.scaled_height, antialias_filter,
        [&](std::uint32_t row, std::uint32_t column) {
            return sample_to_q32(
                scaled[static_cast<std::size_t>(row) * dimensions.scaled_width + column]);
        });
    std::vector<Q> decimated(static_cast<std::size_t>(dimensions.operating_width) *
                             dimensions.operating_height);
    for (std::uint32_t row = 0; row < dimensions.operating_height; ++row) {
        for (std::uint32_t column = 0; column < dimensions.operating_width; ++column) {
            decimated[static_cast<std::size_t>(row) * dimensions.operating_width + column] =
                antialiased[static_cast<std::size_t>(row * 16U) * dimensions.scaled_width +
                            column * 16U];
        }
    }
    const std::vector<Q> local_mean = filter_separable_q32(
        dimensions.operating_width, dimensions.operating_height, local_mean_filter,
        [&](std::uint32_t row, std::uint32_t column) {
            return decimated[static_cast<std::size_t>(row) * dimensions.operating_width + column];
        });
    std::vector<std::int32_t> highpass(decimated.size());
    for (std::size_t index = 0; index < decimated.size(); ++index) {
        const Q difference = binary32_subtract(decimated[index], local_mean[index]);
        highpass[index] =
            clamp_i32(round_shift_rne(difference.raw(), 32U - sample_fractional_bits));
    }
    return Result<std::vector<std::int32_t>>::success(std::move(highpass));
}

struct PreprocessWorkspaceView {
    std::vector<std::int32_t> &normalized;
    std::vector<std::int32_t> &scaled;
    std::vector<Q> &filter_vertical;
    std::vector<Q> &filter_output;
    std::vector<Q> &decimated;
    std::vector<AxisSample> &horizontal_samples;
    std::vector<AxisSample> &vertical_samples;
};

struct ParameterWorkspaceView {
    std::vector<Q> &independent;
};

struct ChannelWorkspaceView {
    PreprocessWorkspaceView preprocess;
    std::vector<std::int32_t> &reference_highpass;
    std::vector<std::int32_t> &distorted_highpass;
    ParameterWorkspaceView parameters;
    ChannelStatistics &reference_statistics;
    ChannelStatistics &distorted_statistics;
};

[[nodiscard]] Result<void> make_axis_samples_in_place(std::uint32_t source_size,
                                                      std::uint32_t destination_size,
                                                      std::vector<AxisSample> &samples) noexcept {
    const std::uint64_t denominator = static_cast<std::uint64_t>(destination_size) * 2U;
    for (std::uint32_t destination = 0; destination < destination_size; ++destination) {
        const std::uint64_t numerator =
            (static_cast<std::uint64_t>(destination) * 2U + 1U) * source_size - destination_size;
        const std::uint64_t first = numerator / denominator;
        const std::uint64_t second = (numerator + denominator - 1U) / denominator;
        const std::uint64_t fractional_numerator = numerator - first * denominator;
        if (first >= source_size || second >= source_size ||
            denominator > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
            return Result<void>::failure({ErrorCode::unsupported_format,
                                          "Speed Chroma bilinear coordinate is out of range"});
        }
        const auto fraction = Q::from_ratio(static_cast<std::int64_t>(fractional_numerator),
                                            static_cast<std::int64_t>(denominator));
        if (!fraction) return Result<void>::failure(fraction.error());
        samples[destination] = {
            static_cast<std::uint32_t>(first),
            static_cast<std::uint32_t>(second),
            fraction.value(),
        };
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void>
load_normalized_plane_in_place(const PlaneView &plane, std::uint32_t width, std::uint32_t height,
                               std::uint8_t bit_depth, std::vector<std::int32_t> &output) noexcept {
    if (bit_depth < 8U || bit_depth > 16U) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format, "Speed Chroma normalization requires 8 to 16 bits"});
    }
    const std::uint32_t maximum = (std::uint32_t{1} << bit_depth) - 1U;
    const unsigned normalization_shift = sample_fractional_bits + 8U - bit_depth;
    const std::int64_t normalization_scale =
        static_cast<std::int64_t>(std::uint64_t{1} << normalization_shift);
    constexpr std::int64_t center = 128LL * (std::int64_t{1} << sample_fractional_bits);
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            const std::uint16_t sample = load_sample(plane, row, column, bit_depth);
            if (sample > maximum) {
                return Result<void>::failure(
                    {ErrorCode::invalid_argument,
                     "Speed Chroma sample exceeds the declared bit depth"});
            }
            const std::int64_t normalized =
                static_cast<std::int64_t>(sample) * normalization_scale - center;
            output[static_cast<std::size_t>(row) * width + column] =
                static_cast<std::int32_t>(normalized);
        }
    }
    return Result<void>::success();
}

void scale_bilinear_in_place(std::span<const std::int32_t> input, std::uint32_t source_width,
                             std::uint32_t destination_width, std::uint32_t destination_height,
                             const std::vector<AxisSample> &horizontal_samples,
                             const std::vector<AxisSample> &vertical_samples,
                             std::vector<std::int32_t> &output) noexcept {
    const Q one = Q::from_integer(1);
    for (std::uint32_t row = 0; row < destination_height; ++row) {
        const AxisSample &vertical_sample = vertical_samples[row];
        const Q vertical_first = Q::subtract_saturating(one, vertical_sample.fraction);
        for (std::uint32_t column = 0; column < destination_width; ++column) {
            const AxisSample &horizontal_sample = horizontal_samples[column];
            const Q horizontal_first = Q::subtract_saturating(one, horizontal_sample.fraction);
            const std::array<Q, 4> weights{{
                Q::multiply(vertical_first, horizontal_first),
                Q::multiply(vertical_first, horizontal_sample.fraction),
                Q::multiply(vertical_sample.fraction, horizontal_first),
                Q::multiply(vertical_sample.fraction, horizontal_sample.fraction),
            }};
            const std::array<std::size_t, 4> sources{{
                static_cast<std::size_t>(vertical_sample.first) * source_width +
                    horizontal_sample.first,
                static_cast<std::size_t>(vertical_sample.first) * source_width +
                    horizontal_sample.second,
                static_cast<std::size_t>(vertical_sample.second) * source_width +
                    horizontal_sample.first,
                static_cast<std::size_t>(vertical_sample.second) * source_width +
                    horizontal_sample.second,
            }};
            std::int64_t accumulator = 0;
            for (std::size_t index = 0; index < weights.size(); ++index) {
                accumulator +=
                    static_cast<std::int64_t>(input[sources[index]]) * weights[index].raw();
            }
            output[static_cast<std::size_t>(row) * destination_width + column] =
                clamp_i32(round_shift_rne(accumulator, 32U));
        }
    }
}

void scale_nearest_in_place(std::span<const std::int32_t> input, std::uint32_t source_width,
                            std::uint32_t source_height, std::uint32_t destination_width,
                            std::uint32_t destination_height,
                            std::vector<std::int32_t> &output) noexcept {
    for (std::uint32_t row = 0; row < destination_height; ++row) {
        const std::uint32_t source_row = static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(row) * source_height) / destination_height);
        for (std::uint32_t column = 0; column < destination_width; ++column) {
            const std::uint32_t source_column = static_cast<std::uint32_t>(
                (static_cast<std::uint64_t>(column) * source_width) / destination_width);
            output[static_cast<std::size_t>(row) * destination_width + column] =
                input[static_cast<std::size_t>(source_row) * source_width + source_column];
        }
    }
}

template <std::size_t FilterSize, typename Loader>
void filter_separable_q32_in_place(std::uint32_t width, std::uint32_t height,
                                   const std::array<std::uint32_t, FilterSize> &filter,
                                   Loader &&load, std::vector<Q> &vertical,
                                   std::vector<Q> &output) noexcept {
    static_assert((FilterSize & 1U) == 1U);
    constexpr int radius = static_cast<int>(FilterSize / 2U);
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            Q accumulator{};
            for (std::size_t tap = 0; tap < FilterSize; ++tap) {
                const int source_row =
                    mirror(static_cast<int>(row) - radius + static_cast<int>(tap),
                           static_cast<int>(height));
                accumulator = binary32_add(
                    accumulator,
                    binary32_multiply(Q::from_raw(static_cast<std::int64_t>(filter[tap])),
                                      load(static_cast<std::uint32_t>(source_row), column)));
            }
            vertical[static_cast<std::size_t>(row) * width + column] = accumulator;
        }
    }
    for (std::uint32_t row = 0; row < height; ++row) {
        for (std::uint32_t column = 0; column < width; ++column) {
            Q accumulator{};
            for (std::size_t tap = 0; tap < FilterSize; ++tap) {
                const int source_column =
                    mirror(static_cast<int>(column) - radius + static_cast<int>(tap),
                           static_cast<int>(width));
                accumulator = binary32_add(
                    accumulator,
                    binary32_multiply(Q::from_raw(static_cast<std::int64_t>(filter[tap])),
                                      vertical[static_cast<std::size_t>(row) * width +
                                               static_cast<std::uint32_t>(source_column)]));
            }
            output[static_cast<std::size_t>(row) * width + column] = accumulator;
        }
    }
}

[[nodiscard]] Result<void>
preprocess_channel_in_place(const PlaneView &plane, const Dimensions &dimensions,
                            const Config &config, std::uint8_t bit_depth,
                            PreprocessWorkspaceView workspace,
                            std::vector<std::int32_t> &highpass) noexcept {
    const auto loaded =
        load_normalized_plane_in_place(plane, dimensions.original_width, dimensions.original_height,
                                       bit_depth, workspace.normalized);
    if (!loaded) return loaded;

    if (dimensions.original_width == dimensions.scaled_width &&
        dimensions.original_height == dimensions.scaled_height) {
        std::copy_n(workspace.normalized.begin(),
                    static_cast<std::size_t>(dimensions.scaled_width) * dimensions.scaled_height,
                    workspace.scaled.begin());
    } else if (config.prescale_method == PrescaleMethod::bilinear) {
        scale_bilinear_in_place(workspace.normalized, dimensions.original_width,
                                dimensions.scaled_width, dimensions.scaled_height,
                                workspace.horizontal_samples, workspace.vertical_samples,
                                workspace.scaled);
    } else {
        scale_nearest_in_place(workspace.normalized, dimensions.original_width,
                               dimensions.original_height, dimensions.scaled_width,
                               dimensions.scaled_height, workspace.scaled);
    }

    filter_separable_q32_in_place(
        dimensions.scaled_width, dimensions.scaled_height, antialias_filter,
        [&](std::uint32_t row, std::uint32_t column) noexcept {
            return sample_to_q32(
                workspace.scaled[static_cast<std::size_t>(row) * dimensions.scaled_width + column]);
        },
        workspace.filter_vertical, workspace.filter_output);
    for (std::uint32_t row = 0; row < dimensions.operating_height; ++row) {
        for (std::uint32_t column = 0; column < dimensions.operating_width; ++column) {
            workspace
                .decimated[static_cast<std::size_t>(row) * dimensions.operating_width + column] =
                workspace
                    .filter_output[static_cast<std::size_t>(row * 16U) * dimensions.scaled_width +
                                   column * 16U];
        }
    }
    filter_separable_q32_in_place(
        dimensions.operating_width, dimensions.operating_height, local_mean_filter,
        [&](std::uint32_t row, std::uint32_t column) noexcept {
            return workspace
                .decimated[static_cast<std::size_t>(row) * dimensions.operating_width + column];
        },
        workspace.filter_vertical, workspace.filter_output);
    const std::size_t operating_count =
        static_cast<std::size_t>(dimensions.operating_width) * dimensions.operating_height;
    for (std::size_t index = 0; index < operating_count; ++index) {
        const Q difference =
            binary32_subtract(workspace.decimated[index], workspace.filter_output[index]);
        highpass[index] =
            clamp_i32(round_shift_rne(difference.raw(), 32U - sample_fractional_bits));
    }
    return Result<void>::success();
}

[[nodiscard]] Result<Q> divide_by_two(Q value) noexcept {
    return Q::divide(value, Q::from_integer(2));
}

[[nodiscard]] Q clip(Q value, Q maximum) noexcept {
    return value > maximum ? maximum : value;
}

struct ChannelResult {
    Q score{};
    bool singular{};
};

[[nodiscard]] Result<ChannelResult>
compute_channel(const PlaneView &reference, const PlaneView &distorted,
                const Dimensions &dimensions, const Config &config, std::uint8_t bit_depth,
                const char *channel_label, ChannelWorkspaceView workspace) noexcept {
    if (guarantees_singular_covariance(dimensions)) {
        const auto reference_validation = validate_plane_samples(
            reference, dimensions.original_width, dimensions.original_height, bit_depth);
        if (!reference_validation) {
            return Result<ChannelResult>::failure(reference_validation.error());
        }
        const auto distorted_validation = validate_plane_samples(
            distorted, dimensions.original_width, dimensions.original_height, bit_depth);
        if (!distorted_validation) {
            return Result<ChannelResult>::failure(distorted_validation.error());
        }
        return Result<ChannelResult>::success(ChannelResult{Q{}, true});
    }

    const auto reference_samples =
        preprocess_channel_in_place(reference, dimensions, config, bit_depth, workspace.preprocess,
                                    workspace.reference_highpass);
    const auto distorted_samples =
        preprocess_channel_in_place(distorted, dimensions, config, bit_depth, workspace.preprocess,
                                    workspace.distorted_highpass);
    if (!reference_samples || !distorted_samples) {
        return Result<ChannelResult>::failure(!reference_samples ? reference_samples.error()
                                                                 : distorted_samples.error());
    }
    const auto reference_statistics = detail::estimate_parameters_with_workspace(
        dimensions, workspace.reference_highpass, canonical_sigma_nn,
        workspace.reference_statistics, workspace.parameters.independent);
    const auto distorted_statistics = detail::estimate_parameters_with_workspace(
        dimensions, workspace.distorted_highpass, canonical_sigma_nn,
        workspace.distorted_statistics, workspace.parameters.independent);
    if (!reference_statistics || !distorted_statistics) {
        const bool reference_failed = !reference_statistics;
        const Error &source_error =
            reference_failed ? reference_statistics.error() : distorted_statistics.error();
        return Result<ChannelResult>::failure(
            {source_error.code, std::string("Speed Chroma CPU ") + channel_label + " " +
                                    (reference_failed ? "reference: " : "distorted: ") +
                                    source_error.message});
    }
    const bool reference_singular = workspace.reference_statistics.singular;
    const bool distorted_singular = workspace.distorted_statistics.singular;
    Q score{};
    if (reference_singular == distorted_singular) {
        const auto result = channel_score(dimensions, workspace.reference_statistics,
                                          workspace.distorted_statistics, canonical_sigma_nn,
                                          canonical_nn_floor, config.weight_variance_mode);
        if (!result) return Result<ChannelResult>::failure(result.error());
        score = result.value();
    }
    return Result<ChannelResult>::success(
        ChannelResult{score, reference_singular || distorted_singular});
}

} // namespace

namespace detail {

Result<std::vector<std::int32_t>> preprocess_channel(const PlaneView &plane,
                                                     const Dimensions &dimensions,
                                                     const Config &config, std::uint8_t bit_depth) {
    return preprocess_channel_impl(plane, dimensions, config, bit_depth);
}

} // namespace detail

Reference::Workspace::Workspace(const Dimensions &dimensions)
    : normalized(static_cast<std::size_t>(dimensions.original_width) * dimensions.original_height),
      scaled(static_cast<std::size_t>(dimensions.scaled_width) * dimensions.scaled_height),
      filter_vertical(std::max(
          static_cast<std::size_t>(dimensions.scaled_width) * dimensions.scaled_height,
          static_cast<std::size_t>(dimensions.operating_width) * dimensions.operating_height)),
      filter_output(filter_vertical.size()),
      decimated(static_cast<std::size_t>(dimensions.operating_width) * dimensions.operating_height),
      reference_highpass(decimated.size()), distorted_highpass(decimated.size()),
      horizontal_samples(dimensions.scaled_width), vertical_samples(dimensions.scaled_height),
      independent(static_cast<std::size_t>(elements_per_block) * dimensions.block_count) {
    reference_statistics.entropies.resize(dimensions.block_count);
    reference_statistics.variances.resize(dimensions.block_count);
    distorted_statistics.entropies.resize(dimensions.block_count);
    distorted_statistics.variances.resize(dimensions.block_count);
    if (dimensions.scaled_width != 0U && dimensions.scaled_height != 0U) {
        const auto horizontal_result = make_axis_samples_in_place(
            dimensions.original_width, dimensions.scaled_width, horizontal_samples);
        const auto vertical_result = make_axis_samples_in_place(
            dimensions.original_height, dimensions.scaled_height, vertical_samples);
        if (!horizontal_result || !vertical_result) {
            throw std::length_error("Speed Chroma axis workspace geometry is invalid");
        }
    }
}

Reference::Reference(FrameDescriptor descriptor, Config config)
    : descriptor_(descriptor), config_(config) {
    const auto dimensions = make_dimensions(plane_width(descriptor_, 1U),
                                            plane_height(descriptor_, 1U), config_.prescale);
    if (!dimensions) {
        throw std::length_error("Speed Chroma workspace dimensions are invalid");
    }
    dimensions_ = dimensions.value();
    workspace_ = Workspace(dimensions_);
}

Result<Q32Value> Reference::compute(const FrameView &reference,
                                    const FrameView &distorted) noexcept {
    auto detailed = compute_detailed(reference, distorted);
    if (!detailed) return Result<Q32Value>::failure(detailed.error());
    return Result<Q32Value>::success(detailed.value().uv);
}

Result<DetailedResult> Reference::compute_detailed(const FrameView &reference,
                                                   const FrameView &distorted) noexcept {
    const auto descriptor_validation = video::validate_frame_layout(descriptor_);
    if (!descriptor_validation) {
        return Result<DetailedResult>::failure(descriptor_validation.error());
    }
    const auto reference_validation = video::validate_plane_views(descriptor_, reference);
    const auto distorted_validation = video::validate_plane_views(descriptor_, distorted);
    if (!reference_validation || !distorted_validation) {
        return Result<DetailedResult>::failure(
            !reference_validation ? reference_validation.error() : distorted_validation.error());
    }
    if (reference.index != distorted.index) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_argument,
             "Speed Chroma reference and distorted frame indices must match"});
    }
    if (config_.kernel_scale != Q::from_integer(1) || config_.sigma_nn.raw() <= 0 ||
        config_.nn_floor.raw() < 0 || config_.weight_variance_mode != 5U ||
        config_.maximum.raw() < 0) {
        return Result<DetailedResult>::failure(
            {ErrorCode::unsupported_model,
             "unsupported Speed Chroma parameters for pinned VMAF 1.0.16 models"});
    }

    const std::uint32_t chroma_width = plane_width(descriptor_, 1U);
    const std::uint32_t chroma_height = plane_height(descriptor_, 1U);
    const auto dimensions = make_dimensions(chroma_width, chroma_height, config_.prescale);
    if (!dimensions) {
        return Result<DetailedResult>::failure(dimensions.error());
    }
    ChannelWorkspaceView channel_workspace{
        PreprocessWorkspaceView{
            workspace_.normalized,
            workspace_.scaled,
            workspace_.filter_vertical,
            workspace_.filter_output,
            workspace_.decimated,
            workspace_.horizontal_samples,
            workspace_.vertical_samples,
        },
        workspace_.reference_highpass,
        workspace_.distorted_highpass,
        ParameterWorkspaceView{
            workspace_.independent,
        },
        workspace_.reference_statistics,
        workspace_.distorted_statistics,
    };
    const auto u = compute_channel(reference.planes[1], distorted.planes[1], dimensions.value(),
                                   config_, descriptor_.bit_depth, "U", channel_workspace);
    const auto v = compute_channel(reference.planes[2], distorted.planes[2], dimensions.value(),
                                   config_, descriptor_.bit_depth, "V", channel_workspace);
    if (!u || !v) {
        return Result<DetailedResult>::failure(!u ? u.error() : v.error());
    }

    Q uv{};
    if (u.value().singular && !v.value().singular) {
        uv = v.value().score;
    } else if (v.value().singular && !u.value().singular) {
        uv = u.value().score;
    } else {
        const auto average = divide_by_two(Q::add_saturating(u.value().score, v.value().score));
        if (!average) return Result<DetailedResult>::failure(average.error());
        uv = average.value();
    }

    return Result<DetailedResult>::success(DetailedResult{
        Q32Value{clip(u.value().score, config_.maximum).raw()},
        Q32Value{clip(v.value().score, config_.maximum).raw()},
        Q32Value{clip(uv, config_.maximum).raw()},
        u.value().singular,
        v.value().singular,
        dimensions.value(),
    });
}

} // namespace crossvmaf::metrics::speed_chroma::reference
