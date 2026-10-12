#pragma once

#include "crossvmaf/error.hpp"

#include <array>
#include <string_view>

namespace crossvmaf {

enum class ModelVariant {
    v1_0_16_1d5h_2160,
    v1_0_16_3d0h_2160,
    v1_0_16_3d0h,
    v1_0_16_5d0h,
    v1_0_16_hfr_1d5h_2160,
    v1_0_16_hfr_3d0h_2160,
    v1_0_16_hfr_3d0h,
    v1_0_16_hfr_5d0h,
};

inline constexpr ModelVariant default_model = ModelVariant::v1_0_16_3d0h;

struct FrameDescriptor;
struct FrameRateInfo;

inline constexpr std::string_view automatic_model_policy_version{"2.0"};

// Select from the decoded evaluation dimensions, without resizing. UHD-sized
// frames use 2160p/1.5H; smaller frames use the 1080p/3H baseline. Intermediate
// resolutions are an application fallback, not an upstream calibrated preset.
// Known constant rates >= 50 fps use HFR; unknown/VFR rates retain standard.
[[nodiscard]] ModelVariant select_model(
    const FrameDescriptor& frame, const FrameRateInfo& frame_rate) noexcept;

[[nodiscard]] std::string_view to_string(ModelVariant model) noexcept;
[[nodiscard]] Result<ModelVariant> parse_model(std::string_view name) noexcept;
[[nodiscard]] bool is_hfr(ModelVariant model) noexcept;
[[nodiscard]] std::array<ModelVariant, 8> all_models() noexcept;

}  // namespace crossvmaf
