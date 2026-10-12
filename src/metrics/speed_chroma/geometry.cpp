#include "metrics/speed_chroma/reference_math.hpp"
#include <cstdint>
#include <limits>

namespace crossvmaf::metrics::speed_chroma::reference {
namespace {

constexpr numeric::Q32_32 one = numeric::Q32_32::from_raw(numeric::Q32_32::one_raw);

} // namespace

Result<Dimensions> make_dimensions(std::uint32_t width, std::uint32_t height,
                                   numeric::Q32_32 prescale) noexcept {
    if (width == 0U || height == 0U || prescale.raw() <= 0 || prescale > one) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_model,
             "CrossVMAF 1.0.16 Speed Chroma requires 0 < prescale <= 1"});
    }
    const auto scaled_dimension = [prescale](std::uint32_t value) -> Result<std::uint32_t> {
        const std::uint64_t scale = static_cast<std::uint64_t>(prescale.raw());
        constexpr std::uint64_t rounding = std::uint64_t{1} << 31U;
        if (value > (std::numeric_limits<std::uint64_t>::max() - rounding) / scale) {
            return Result<std::uint32_t>::failure(
                {ErrorCode::unsupported_format, "Speed Chroma scaled dimension overflows"});
        }
        const std::uint64_t product = static_cast<std::uint64_t>(value) * scale + rounding;
        const std::uint64_t scaled = product >> 32U;
        if (scaled == 0U || scaled > std::numeric_limits<std::uint32_t>::max()) {
            return Result<std::uint32_t>::failure(
                {ErrorCode::unsupported_format, "Speed Chroma scaled dimension is out of range"});
        }
        return Result<std::uint32_t>::success(static_cast<std::uint32_t>(scaled));
    };

    const auto scaled_width = scaled_dimension(width);
    const auto scaled_height = scaled_dimension(height);
    if (!scaled_width || !scaled_height) {
        return Result<Dimensions>::failure(!scaled_width ? scaled_width.error()
                                                         : scaled_height.error());
    }
    Dimensions dimensions{};
    dimensions.original_width = width;
    dimensions.original_height = height;
    dimensions.scaled_width = scaled_width.value();
    dimensions.scaled_height = scaled_height.value();
    dimensions.operating_width = dimensions.scaled_width >> 4U;
    dimensions.operating_height = dimensions.scaled_height >> 4U;
    dimensions.truncated_width = (dimensions.operating_width / block_size) * block_size;
    dimensions.truncated_height = (dimensions.operating_height / block_size) * block_size;
    if (dimensions.truncated_width == 0U || dimensions.truncated_height == 0U) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma image is too small after prescale and 16x decimation"});
    }
    dimensions.blocks_horizontal = dimensions.truncated_width / block_size;
    dimensions.blocks_vertical = dimensions.truncated_height / block_size;
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(dimensions.blocks_horizontal) * dimensions.blocks_vertical;
    if (block_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma block count exceeds deterministic Q32 limits"});
    }
    dimensions.block_count = static_cast<std::uint32_t>(block_count);
    dimensions.submatrix_width = dimensions.truncated_width - block_size + 1U;
    dimensions.submatrix_height = dimensions.truncated_height - block_size + 1U;
    const std::uint64_t covariance_sample_count =
        static_cast<std::uint64_t>(dimensions.submatrix_width) * dimensions.submatrix_height;
    if (covariance_sample_count == 0U ||
        covariance_sample_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma covariance sample count exceeds Q32 limits"});
    }
    dimensions.covariance_sample_count = static_cast<std::uint32_t>(covariance_sample_count);
    return Result<Dimensions>::success(dimensions);
}

} // namespace crossvmaf::metrics::speed_chroma::reference
