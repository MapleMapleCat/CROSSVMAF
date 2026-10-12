#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace crossvmaf::studio {

struct LiveProgress final {
    std::vector<double> scores;
    std::optional<std::uint64_t> frame_count_hint;
    std::uint64_t decoded_frame_count{};
    double elapsed_seconds{};
    double frames_per_second{};
};

} // namespace crossvmaf::studio
