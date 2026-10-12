#pragma once

#include "types.hpp"

#include <string>

namespace crossvmaf::studio {

struct ResolvedEvaluationConfig final {
    ModelVariant model{default_model};
    SceneId scene{SceneId::tv_standard};
    std::string decision_reason;
    std::string policy_version;
};

// Scene ids remain part of historical reports; new evaluations derive them
// from the automatic model instead of accepting a user-selected scene.
[[nodiscard]] SceneId model_scene(ModelVariant model) noexcept;
// Probe-time preview. The stream resolves again using decoded dimensions.
[[nodiscard]] ResolvedEvaluationConfig resolve_evaluation_config(
    const VideoMetadata* reference_metadata);

}  // namespace crossvmaf::studio
