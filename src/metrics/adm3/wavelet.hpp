#pragma once
#include "metrics/adm3/adm3.hpp"
#include "crossvmaf/value.hpp"
#include "numeric/q32.hpp"
#include <array>
#include <cstdint>

namespace crossvmaf::metrics::adm3::reference {

[[nodiscard]] constexpr std::uint32_t ceil_half(std::uint32_t value) noexcept {
    return value / 2U + value % 2U;
}

[[nodiscard]] unsigned ceil_log2(std::uint64_t value) noexcept;
[[nodiscard]] numeric::Q32_32 cuberoot_scaled_u64(std::uint64_t accumulator,
                                                  int binary_shift) noexcept;
[[nodiscard]] ScaleValues finalize_scale(const ScaleAccumulators &accumulators, unsigned scale,
                                         std::uint32_t width, std::uint32_t height,
                                         const Config &config) noexcept;
[[nodiscard]] Result<Q32Value> combine_scales(const std::array<ScaleValues, 4> &scales,
                                              const Config &config) noexcept;

} // namespace crossvmaf::metrics::adm3::reference
