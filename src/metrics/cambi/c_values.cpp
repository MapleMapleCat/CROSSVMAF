#include "metrics/cambi/c_values.hpp"
#include "numeric/rounding.hpp"
#include "numeric/uint128.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace crossvmaf::metrics::cambi::reference {
namespace {

namespace base_numeric = crossvmaf::numeric::detail;
using base_numeric::divide_u64_rne;
using base_numeric::UInt128;

[[nodiscard]] std::uint64_t divide_u128_u64_rne(UInt128 numerator,
                                                std::uint64_t denominator) noexcept {
    if (denominator == 0U || numerator.hi >= denominator) {
        return std::numeric_limits<std::uint64_t>::max();
    }
    bool overflow = false;
    const std::uint64_t quotient =
        base_numeric::rounded_quotient(numerator, denominator, &overflow);
    return overflow ? std::numeric_limits<std::uint64_t>::max() : quotient;
}

[[nodiscard]] constexpr UInt128 shift_left_14(std::uint64_t value) noexcept {
    return {value >> 50U, value << 14U};
}

} // namespace

std::uint16_t adjusted_window_size(std::uint16_t base_window_size, std::uint32_t width,
                                   std::uint32_t height, bool high_res_speedup) noexcept {
    std::uint64_t adjusted = static_cast<std::uint64_t>(base_window_size) * (width + height);
    adjusted = (adjusted / 375U) >> 4U;
    if (high_res_speedup) adjusted = (adjusted + 1U) >> 1U;
    adjusted |= 1U;
    return static_cast<std::uint16_t>(
        std::min<std::uint64_t>(adjusted, std::numeric_limits<std::uint16_t>::max()));
}

std::uint16_t spatial_mask_index(std::uint32_t width, std::uint32_t height,
                                 std::uint16_t filter_size) noexcept {
    const std::uint32_t shifted_area = (width >> 6U) * (height >> 6U);
    const int logarithm =
        shifted_area == 0U ? 0 : static_cast<int>(std::bit_width(shifted_area - 1U));
    const int value = static_cast<int>(filter_size) * filter_size + 3 * (logarithm - 11) - 1;
    return static_cast<std::uint16_t>(std::max(value, 0) >> 1);
}

CValue c_value_from_counts(std::uint16_t center_count, std::uint16_t neighbor_count,
                           std::uint32_t contrast_weight) noexcept {
    const std::uint64_t denominator = static_cast<std::uint64_t>(center_count) + neighbor_count;
    if (denominator == 0U) return 0U;
    const std::uint64_t numerator =
        static_cast<std::uint64_t>(contrast_weight) * center_count * neighbor_count;
    const std::uint64_t scaled = numerator << c_value_fractional_bits;
    return static_cast<CValue>(divide_u64_rne(scaled, denominator));
}

numeric::Q32_32 pool_topk_in_place(std::span<CValue> values) noexcept {
    if (values.empty()) return numeric::Q32_32{};
    const std::size_t count = values.size();
    const std::size_t top_count = std::clamp<std::size_t>((count * 3U) / 5U, 1U, count);
    std::sort(values.begin(), values.end(), std::greater<>{});
    std::uint64_t sum{};
    for (std::size_t index = 0; index < top_count; ++index) {
        sum += values[index];
    }
    const std::uint64_t raw =
        divide_u128_u64_rne(shift_left_14(sum), static_cast<std::uint64_t>(top_count));
    const std::uint64_t maximum =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    return numeric::Q32_32::from_raw(static_cast<std::int64_t>(std::min(raw, maximum)));
}

numeric::Q32_32 combine_scale_scores(const std::array<numeric::Q32_32, 5> &scores,
                                     std::uint16_t window_size) noexcept {
    constexpr std::array<std::uint32_t, 5> weights{{16U, 8U, 4U, 2U, 1U}};
    std::uint64_t weighted{};
    for (std::size_t scale = 0; scale < scores.size(); ++scale) {
        const std::int64_t raw = scores[scale].raw();
        if (raw > 0) {
            weighted += static_cast<std::uint64_t>(raw) * weights[scale];
        }
    }
    const std::uint64_t pixels = static_cast<std::uint64_t>(window_size) * window_size;
    const std::uint64_t raw = divide_u64_rne(weighted, pixels);
    return numeric::Q32_32::from_raw(static_cast<std::int64_t>(std::min<std::uint64_t>(
        raw, static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))));
}

} // namespace crossvmaf::metrics::cambi::reference
