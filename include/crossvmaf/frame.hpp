#pragma once

#include "crossvmaf/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>

namespace crossvmaf {

enum class PixelFormat {
    yuv420p,
    yuv422p,
    yuv444p,
};

struct FrameDescriptor {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint8_t bit_depth{};
    PixelFormat pixel_format{PixelFormat::yuv420p};
};

// Rational number used for frame rates and timestamp time bases. A rational
// with a non-positive numerator or denominator is invalid and means
// "unknown"; unknown values are never reported as zero rates.
struct Rational final {
    std::int64_t numerator{};
    std::int64_t denominator{};

    [[nodiscard]] constexpr bool valid() const noexcept {
        return numerator > 0 && denominator > 0;
    }

    [[nodiscard]] constexpr double value() const noexcept {
        return valid()
            ? static_cast<double>(numerator) / static_cast<double>(denominator)
            : 0.0;
    }
};

struct PlaneView {
    const std::byte* data{};
    std::size_t stride_bytes{};
};

struct FrameView {
    std::array<PlaneView, 3> planes{};
    std::uint64_t index{};
};

struct WritablePlaneView {
    std::byte* data{};
    std::size_t size_bytes{};
    std::size_t stride_bytes{};
};

struct WritableFrameView {
    std::array<WritablePlaneView, 3> planes{};
    std::uint64_t index{};
};

using FrameInputWriter = std::function<Result<void>(
    WritableFrameView& reference, WritableFrameView& distorted)>;

[[nodiscard]] Result<void> validate_frame_descriptor(const FrameDescriptor& descriptor) noexcept;
[[nodiscard]] Result<void> validate_frame_view(const FrameDescriptor& descriptor,
                                               const FrameView& frame) noexcept;
[[nodiscard]] std::uint32_t plane_width(const FrameDescriptor& descriptor,
                                        std::size_t plane) noexcept;
[[nodiscard]] std::uint32_t plane_height(const FrameDescriptor& descriptor,
                                         std::size_t plane) noexcept;

}  // namespace crossvmaf
