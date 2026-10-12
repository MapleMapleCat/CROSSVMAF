#include "evaluation/scene_policy.hpp"

namespace crossvmaf::studio {

SceneId model_scene(ModelVariant model) noexcept {
    switch (model) {
    case ModelVariant::v1_0_16_1d5h_2160:
    case ModelVariant::v1_0_16_hfr_1d5h_2160:
        return SceneId::tv_4k_close;
    case ModelVariant::v1_0_16_3d0h_2160:
    case ModelVariant::v1_0_16_hfr_3d0h_2160:
        return SceneId::tv_4k_standard;
    case ModelVariant::v1_0_16_5d0h:
    case ModelVariant::v1_0_16_hfr_5d0h:
        return SceneId::cinema_far;
    default:
        return SceneId::tv_standard;
    }
}

ResolvedEvaluationConfig resolve_evaluation_config(const VideoMetadata* reference_metadata) {
    ResolvedEvaluationConfig resolved;
    resolved.policy_version = std::string(automatic_model_policy_version);
    if (!reference_metadata || !reference_metadata->descriptor) {
        resolved.decision_reason = "awaiting_decoded_resolution";
        return resolved;
    }
    resolved.model = select_model(*reference_metadata->descriptor, reference_metadata->frame_rate);
    resolved.scene = model_scene(resolved.model);
    resolved.decision_reason = resolved.scene == SceneId::tv_4k_close ? "auto_2160_1d5h" : "auto_1080_3d0h";
    resolved.decision_reason += is_hfr(resolved.model) ? "_hfr" : "_standard";
    if (reference_metadata->frame_rate.mode != FrameRateMode::constant ||
        !reference_metadata->frame_rate.average.valid()) {
        resolved.decision_reason += "_unknown_or_variable_frame_rate";
    }
    return resolved;
}

}  // namespace crossvmaf::studio
