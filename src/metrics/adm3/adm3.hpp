#pragma once
#include "crossvmaf/value.hpp"
#include <optional>
#include <vector>
#include "numeric/q32.hpp"
#include <array>
#include <cstdint>

namespace crossvmaf::metrics::adm3 {

struct Config {
    numeric::Q32_32 dlm_weight{};
    numeric::Q32_32 enhancement_gain_limit{};
    numeric::Q32_32 noise_weight{};
    numeric::Q32_32 minimum{};
    std::uint32_t csf_mode{};
    std::array<std::uint32_t, 4> horizontal_csf_raw{};
    std::array<std::uint32_t, 4> diagonal_csf_raw{};
};

struct ScaleAccumulators {
    std::array<std::uint64_t, 3> denominator{};
    std::array<std::uint64_t, 3> numerator{};
    std::array<std::uint64_t, 3> aim{};
    std::array<std::uint64_t, 3> reference_sum{};
    std::array<std::uint64_t, 3> restored_sum{};
    std::array<std::uint64_t, 3> additive_sum{};
    std::array<std::uint64_t, 3> reference_weighted_sum{};
    std::array<std::uint64_t, 3> distorted_weighted_sum{};
};

struct ScaleValues {
    numeric::Q32_32 denominator{};
    numeric::Q32_32 numerator{};
    numeric::Q32_32 aim{};
};

struct DetailedResult {
    std::array<ScaleAccumulators, 4> accumulators{};
    std::array<ScaleValues, 4> scales{};
    Q32Value value{};
};

} // namespace crossvmaf::metrics::adm3

namespace crossvmaf::metrics::adm3::reference {
using metrics::adm3::Config;
using metrics::adm3::DetailedResult;
using metrics::adm3::ScaleAccumulators;
using metrics::adm3::ScaleValues;
} // namespace crossvmaf::metrics::adm3::reference

namespace crossvmaf::metrics::adm3::vulkan {
using metrics::adm3::Config;
using metrics::adm3::DetailedResult;
using metrics::adm3::ScaleAccumulators;
using metrics::adm3::ScaleValues;
} // namespace crossvmaf::metrics::adm3::vulkan
