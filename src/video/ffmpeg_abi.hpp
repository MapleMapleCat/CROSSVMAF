#pragma once

#include "crossvmaf/error.hpp"
#include <cstdint>

namespace crossvmaf::video {

struct FfmpegVersions final {
    std::uint32_t avutil{};
    std::uint32_t avcodec{};
    std::uint32_t avformat{};
};

// The private bridge accesses native structures from these exact headers.
// Widening this compatibility policy requires a separate ABI review.
[[nodiscard]] Result<void> validate_ffmpeg_versions(FfmpegVersions expected,
                                                    FfmpegVersions runtime);
[[nodiscard]] Result<void> require_ffmpeg_abi();

} // namespace crossvmaf::video
