#pragma once
#include "numeric/q32.hpp"
#include <array>
#include <cstdint>
#include <span>

namespace crossvmaf::metrics::cambi::reference {

constexpr unsigned c_value_fractional_bits = 18U;
using CValue = std::uint32_t;

[[nodiscard]] std::uint16_t adjusted_window_size(std::uint16_t base_window_size,
                                                 std::uint32_t width, std::uint32_t height,
                                                 bool high_res_speedup) noexcept;
[[nodiscard]] std::uint16_t spatial_mask_index(std::uint32_t width, std::uint32_t height,
                                               std::uint16_t filter_size = 7U) noexcept;
[[nodiscard]] CValue c_value_from_counts(std::uint16_t center_count, std::uint16_t neighbor_count,
                                         std::uint32_t contrast_weight) noexcept;
[[nodiscard]] numeric::Q32_32 pool_topk_in_place(std::span<CValue> values) noexcept;
[[nodiscard]] numeric::Q32_32 combine_scale_scores(const std::array<numeric::Q32_32, 5> &scores,
                                                   std::uint16_t window_size) noexcept;

} // namespace crossvmaf::metrics::cambi::reference
