#pragma once

#include "crossvmaf/device.hpp"
#include "crossvmaf/decode.hpp"
#include "crossvmaf/model.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string_view>

namespace crossvmaf {

enum class DecodeMode {
    automatic,
    software,
    // Require hardware for both inputs; never reopen an input in software.
    hardware,
};

[[nodiscard]] constexpr std::string_view to_string(DecodeMode mode) noexcept {
    switch (mode) {
    case DecodeMode::automatic: return "auto";
    case DecodeMode::software: return "software";
    case DecodeMode::hardware: return "hardware";
    }
    return "unknown";
}

struct StreamOptions final {
    std::filesystem::path reference_path;
    std::filesystem::path distorted_path;
    // Empty selects automatically after the decoded dimensions are known.
    std::optional<ModelVariant> model;
    DeviceSelector device;
    DecodeMode decode_mode{DecodeMode::automatic};

    std::size_t decoded_frame_queue_capacity{4U};
    // Zero enables automatic selection. The two software AV1 decoders share
    // one host CPU budget; native FFmpeg codecs retain their own policy.
    // Hardware backends may ignore this value or use it for bitstream work.
    std::size_t decoder_threads_per_source{};
    DecodeBackendPreference decode_backend{DecodeBackendPreference::automatic};
    std::string decode_device{"same"};
    DecodeTransferPolicy decode_transfer{DecodeTransferPolicy::automatic};
};

}  // namespace crossvmaf
