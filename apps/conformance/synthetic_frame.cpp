#include "synthetic_frame.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>

namespace crossvmaf::conformance {

FrameView OwnedFrame::view() const noexcept {
    FrameView frame{};
    frame.index = index;
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        frame.planes[plane] = {planes[plane].data(), strides[plane]};
    }
    return frame;
}

Result<void> OwnedFrame::write_raw(
    const std::filesystem::path& path,
    const FrameDescriptor& descriptor) const {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return Result<void>::failure(
            {ErrorCode::io_error, "failed to create synthetic YUV file"});
    }
    const std::size_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    for (std::size_t plane = 0; plane < planes.size(); ++plane) {
        const std::size_t row_bytes =
            static_cast<std::size_t>(plane_width(descriptor, plane)) *
            bytes_per_sample;
        const std::uint32_t height = plane_height(descriptor, plane);
        for (std::uint32_t row = 0; row < height; ++row) {
            const char* source = reinterpret_cast<const char*>(
                planes[plane].data() + static_cast<std::size_t>(row) *
                                           strides[plane]);
            output.write(source, static_cast<std::streamsize>(row_bytes));
        }
    }
    if (!output) {
        return Result<void>::failure(
            {ErrorCode::io_error, "failed to write synthetic YUV file"});
    }
    return Result<void>::success();
}

OwnedFrame make_speed_chroma_frame(const FrameDescriptor& descriptor,
                                   bool distorted) {
    OwnedFrame frame{};
    const std::size_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    const std::uint32_t maximum = descriptor.bit_depth == 16U
        ? 0xffffU
        : (std::uint32_t{1} << descriptor.bit_depth) - 1U;
    for (std::size_t plane = 0; plane < frame.planes.size(); ++plane) {
        const std::uint32_t width = plane_width(descriptor, plane);
        const std::uint32_t height = plane_height(descriptor, plane);
        frame.strides[plane] = static_cast<std::size_t>(width) * bytes_per_sample;
        frame.planes[plane].resize(frame.strides[plane] * height, std::byte{0});
        for (std::uint32_t row = 0; row < height; ++row) {
            for (std::uint32_t column = 0; column < width; ++column) {
                std::uint32_t sample =
                    row * row * 17U + column * column * 29U +
                    row * column * 43U + static_cast<std::uint32_t>(plane) * 101U;
                if (distorted && plane != 0U &&
                    ((row / 7U + column / 9U + plane) % 3U) == 0U) {
                    sample += maximum / 9U;
                }
                sample &= maximum;
                const std::size_t offset =
                    static_cast<std::size_t>(row) * frame.strides[plane] +
                    static_cast<std::size_t>(column) * bytes_per_sample;
                frame.planes[plane][offset] =
                    static_cast<std::byte>(sample & 0xffU);
                if (bytes_per_sample == 2U) {
                    frame.planes[plane][offset + 1U] =
                        static_cast<std::byte>((sample >> 8U) & 0xffU);
                }
            }
        }
    }
    return frame;
}

}  // namespace crossvmaf::conformance
