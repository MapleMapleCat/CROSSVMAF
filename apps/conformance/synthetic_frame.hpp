#pragma once

#include "crossvmaf/frame.hpp"
#include "crossvmaf/error.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace crossvmaf::conformance {

struct OwnedFrame {
    std::array<std::vector<std::byte>, 3> planes;
    std::array<std::size_t, 3> strides{};
    std::uint64_t index{};

    [[nodiscard]] FrameView view() const noexcept;
    [[nodiscard]] Result<void> write_raw(
        const std::filesystem::path& path,
        const FrameDescriptor& descriptor) const;
};

[[nodiscard]] OwnedFrame make_speed_chroma_frame(
    const FrameDescriptor& descriptor, bool distorted);

}  // namespace crossvmaf::conformance
