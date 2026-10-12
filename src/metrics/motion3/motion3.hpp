#pragma once
#include "crossvmaf/value.hpp"
#include <optional>
#include <vector>
#include "numeric/q32.hpp"
#include <array>
#include <cstdint>

namespace crossvmaf::metrics::motion3 {

struct Config {
    numeric::Q32_32 maximum{numeric::Q32_32::from_integer(18)};
    bool five_frame_window{};
    bool moving_average{};
};

struct MotionScore {
    std::uint64_t index{};
    Q32Value value{};
};

} // namespace crossvmaf::metrics::motion3

namespace crossvmaf::metrics::motion3::reference {
using metrics::motion3::Config;
using metrics::motion3::MotionScore;
} // namespace crossvmaf::metrics::motion3::reference

namespace crossvmaf::metrics::motion3::vulkan {
using metrics::motion3::Config;
using metrics::motion3::MotionScore;
} // namespace crossvmaf::metrics::motion3::vulkan
