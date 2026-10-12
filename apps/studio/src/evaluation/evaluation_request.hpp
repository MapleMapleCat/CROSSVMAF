#pragma once

#include <crossvmaf/crossvmaf.hpp>

namespace crossvmaf::studio {

// Stable viewing-scene identifiers. Display names are localized by the UI;
// persisted data always stores the id, never a localized string.
enum class SceneId : std::uint8_t {
    tv_standard = 0,
    tv_4k_close,
    tv_4k_standard,
    cinema_far,
};

// Display name in the active UI language (i18n::tr).
[[nodiscard]] std::string scene_display_name(SceneId scene);

struct EvaluationRequest final {
    std::filesystem::path reference_path;
    std::filesystem::path distorted_path;
    SceneId scene{SceneId::tv_standard};
    DeviceSelector device;
    DecodeMode decode_mode{DecodeMode::automatic};
    std::size_t decoder_threads_per_source{};
    std::size_t decoded_frame_queue_capacity{4U};

    // Automatic selection preview at enqueue time; decoded dimensions confirm
    // the actual model at stream open, before the run and report are published.
    ModelVariant model{default_model};
    std::string model_decision_reason;
    std::string scene_policy_version;
    // File identity snapshots taken at enqueue time so a changed input file
    // can be detected before the run starts.
    FileIdentity reference_identity;
    FileIdentity distorted_identity;
    DecodeBackendPreference decode_backend{DecodeBackendPreference::automatic};
    std::string decode_device{"same"};
    DecodeTransferPolicy decode_transfer{DecodeTransferPolicy::automatic};

};

} // namespace crossvmaf::studio
