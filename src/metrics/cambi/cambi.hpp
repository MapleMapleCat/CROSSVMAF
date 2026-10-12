#pragma once
#include "crossvmaf/value.hpp"
#include "crossvmaf/frame.hpp"
#include <optional>
#include <vector>
#include "numeric/q32.hpp"
#include <array>
#include <cstdint>

namespace crossvmaf::metrics::cambi {

struct Config {
    std::uint32_t high_res_speedup{};
    numeric::Q32_32 visibility_luminance_threshold{};
    numeric::Q32_32 maximum{};
};

enum class DetailedDiagnosticsMode : std::uint8_t {
    disabled,
    c_value_maps,
};

struct CValueMap {
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint32_t> values;
};

struct DetailedDiagnostics {
    std::array<CValueMap, 5> c_value_maps;
};

struct DetailedResult {
    std::array<numeric::Q32_32, 5> scales{};
    Q32Value value{};
    std::uint16_t window_size{};
    bool high_res_speedup{};
    std::optional<DetailedDiagnostics> diagnostics;
};

} // namespace crossvmaf::metrics::cambi

namespace crossvmaf::metrics::cambi::reference {
using metrics::cambi::Config;
using metrics::cambi::CValueMap;
using metrics::cambi::DetailedDiagnostics;
using metrics::cambi::DetailedDiagnosticsMode;
using metrics::cambi::DetailedResult;
} // namespace crossvmaf::metrics::cambi::reference

namespace crossvmaf::metrics::cambi::vulkan {
using metrics::cambi::Config;
using metrics::cambi::CValueMap;
using metrics::cambi::DetailedDiagnostics;
using metrics::cambi::DetailedDiagnosticsMode;
using metrics::cambi::DetailedResult;
} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi {
[[nodiscard]] inline Result<void> validate_input(const FrameDescriptor &descriptor) {
    if (descriptor.width < 216U && descriptor.height < 216U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "CAMBI requires at least one frame dimension to be 216 pixels"});
    }
    return Result<void>::success();
}
} // namespace crossvmaf::metrics::cambi
