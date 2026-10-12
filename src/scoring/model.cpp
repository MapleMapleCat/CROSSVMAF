#include "crossvmaf/model.hpp"
#include "crossvmaf/video.hpp"
#include <algorithm>
#include <array>
#include <string_view>
#include "scoring/model.hpp"
#include "numeric/q32.hpp"

namespace crossvmaf {
namespace {

struct ModelEntry {
    ModelVariant model;
    std::string_view name;
    bool hfr;
};

constexpr std::array<ModelEntry, 8> model_entries{{
    {ModelVariant::v1_0_16_1d5h_2160, "vmaf_v1.0.16_1d5h_2160", false},
    {ModelVariant::v1_0_16_3d0h_2160, "vmaf_v1.0.16_3d0h_2160", false},
    {ModelVariant::v1_0_16_3d0h, "vmaf_v1.0.16_3d0h", false},
    {ModelVariant::v1_0_16_5d0h, "vmaf_v1.0.16_5d0h", false},
    {ModelVariant::v1_0_16_hfr_1d5h_2160, "vmaf_v1.0.16_hfr_1d5h_2160", true},
    {ModelVariant::v1_0_16_hfr_3d0h_2160, "vmaf_v1.0.16_hfr_3d0h_2160", true},
    {ModelVariant::v1_0_16_hfr_3d0h, "vmaf_v1.0.16_hfr_3d0h", true},
    {ModelVariant::v1_0_16_hfr_5d0h, "vmaf_v1.0.16_hfr_5d0h", true},
}};

} // namespace

ModelVariant select_model(const FrameDescriptor& frame, const FrameRateInfo& frame_rate) noexcept {
    const bool uhd = std::max(frame.width, frame.height) >= 3840U ||
        std::min(frame.width, frame.height) >= 2160U;
    const bool hfr = frame_rate.mode == FrameRateMode::constant &&
        frame_rate.average.valid() && frame_rate.average.value() >= 50.0;
    if (uhd) return hfr ? ModelVariant::v1_0_16_hfr_1d5h_2160
                        : ModelVariant::v1_0_16_1d5h_2160;
    return hfr ? ModelVariant::v1_0_16_hfr_3d0h : ModelVariant::v1_0_16_3d0h;
}

std::string_view to_string(ModelVariant model) noexcept {
    for (const auto &entry : model_entries) {
        if (entry.model == model) return entry.name;
    }
    return "unknown";
}

Result<ModelVariant> parse_model(std::string_view name) noexcept {
    for (const auto &entry : model_entries) {
        if (entry.name == name) return Result<ModelVariant>::success(entry.model);
    }
    return Result<ModelVariant>::failure(
        {ErrorCode::unsupported_model, "unsupported CrossVMAF model"});
}

bool is_hfr(ModelVariant model) noexcept {
    for (const auto &entry : model_entries) {
        if (entry.model == model) return entry.hfr;
    }
    return false;
}

std::array<ModelVariant, 8> all_models() noexcept {
    std::array<ModelVariant, 8> models{};
    for (std::size_t index = 0; index < model_entries.size(); ++index) {
        models[index] = model_entries[index].model;
    }
    return models;
}

} // namespace crossvmaf

namespace crossvmaf::scoring {

Result<ModelSnapshot> make_model_snapshot(ModelVariant model_variant) noexcept {
    const scoring::PackedModel *const packed_model = scoring::find_packed_model(model_variant);
    if (packed_model == nullptr) {
        return Result<ModelSnapshot>::failure(
            {ErrorCode::unsupported_model,
             "model variant is not present in the pinned model pack"});
    }

    ModelSnapshot model_snapshot{};
    model_snapshot.cambi = {
        packed_model->cambi.high_res_speedup,
        numeric::Q32_32::from_raw(packed_model->cambi.visibility_luminance_threshold_raw),
        numeric::Q32_32::from_raw(packed_model->cambi.maximum_raw),
    };
    model_snapshot.speed_chroma = {
        numeric::Q32_32::from_raw(packed_model->speed_chroma.kernel_scale_raw),
        numeric::Q32_32::from_raw(packed_model->speed_chroma.prescale_raw),
        packed_model->speed_chroma.prescale_method == scoring::ScalingMethod::bilinear
            ? metrics::speed_chroma::PrescaleMethod::bilinear
            : metrics::speed_chroma::PrescaleMethod::nearest,
        numeric::Q32_32::from_raw(packed_model->speed_chroma.sigma_nn_raw),
        numeric::Q32_32::from_raw(packed_model->speed_chroma.nn_floor_raw),
        packed_model->speed_chroma.weight_variance_mode,
        numeric::Q32_32::from_raw(packed_model->speed_chroma.maximum_raw),
    };
    model_snapshot.adm3 = {
        numeric::Q32_32::from_raw(packed_model->adm3.dlm_weight_raw),
        numeric::Q32_32::from_raw(packed_model->adm3.enhancement_gain_limit_raw),
        numeric::Q32_32::from_raw(packed_model->adm3.noise_weight_raw),
        numeric::Q32_32::from_raw(packed_model->adm3.minimum_raw),
        packed_model->adm3.csf_mode,
        packed_model->adm3.horizontal_csf_raw,
        packed_model->adm3.diagonal_csf_raw,
    };
    model_snapshot.motion3 = {
        numeric::Q32_32::from_raw(packed_model->motion3.maximum_raw),
        packed_model->motion3.five_frame_window,
        packed_model->motion3.moving_average,
    };
    return Result<ModelSnapshot>::success(model_snapshot);
}

} // namespace crossvmaf::scoring
