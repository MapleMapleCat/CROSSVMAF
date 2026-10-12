#pragma once

#include "i18n/language.hpp"
#include "types.hpp"

#include <optional>

namespace crossvmaf::studio {

struct StudioSettings final {
    int theme{};
    // Absent follows the system language; persisted as "system" or a BCP 47 tag.
    std::optional<i18n::Language> language;
    std::string device_uuid;
    int export_format{};
    DecodeMode decode_mode{DecodeMode::automatic};
    std::size_t decoder_threads_per_source{};
    std::size_t decoded_frame_queue_capacity{4U};
    QueueFailurePolicy failure_policy{QueueFailurePolicy::pause_on_failure};
    DecodeBackendPreference decode_backend{DecodeBackendPreference::automatic};
    std::string decode_device{"same"};
    DecodeTransferPolicy decode_transfer{DecodeTransferPolicy::automatic};

};

[[nodiscard]] Result<StudioSettings> load_settings(const std::filesystem::path& path);
[[nodiscard]] Result<void> save_settings(const std::filesystem::path& path,
                                         const StudioSettings& settings);

} // namespace crossvmaf::studio
