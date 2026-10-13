#pragma once

#include "crossvmaf/error.hpp"
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace crossvmaf {
enum class StreamDecodeBackend : std::uint8_t { software, vulkan_video, qsv, nvdec, vaapi, amf };
enum class DecodeBackendPreference : std::uint8_t { automatic, vulkan_video, qsv, nvdec, vaapi, amf };
enum class DecodeTransferPolicy : std::uint8_t { automatic, gpu_only, host };
enum class DecodeTransferPath : std::uint8_t { host, native, mapped, gpu_copy };
enum class DecodeStorage : std::uint8_t { host, vulkan, qsv_surface, cuda, vaapi, amf_surface };

[[nodiscard]] constexpr std::string_view to_string(StreamDecodeBackend value) noexcept {
    switch (value) {
    case StreamDecodeBackend::software:
        return "software";
    case StreamDecodeBackend::vulkan_video:
        return "vulkan_video";
    case StreamDecodeBackend::qsv:
        return "qsv";
    case StreamDecodeBackend::nvdec:
        return "nvdec";
    case StreamDecodeBackend::vaapi:
        return "vaapi";
    case StreamDecodeBackend::amf:
        return "amf";
    }
    return "unknown";
}
[[nodiscard]] constexpr std::string_view to_string(DecodeBackendPreference value) noexcept {
    switch (value) {
    case DecodeBackendPreference::automatic:
        return "auto";
    case DecodeBackendPreference::vulkan_video:
        return "vulkan";
    case DecodeBackendPreference::qsv:
        return "qsv";
    case DecodeBackendPreference::nvdec:
        return "nvdec";
    case DecodeBackendPreference::vaapi:
        return "vaapi";
    case DecodeBackendPreference::amf:
        return "amf";
    }
    return "unknown";
}
[[nodiscard]] constexpr std::string_view to_string(DecodeTransferPolicy value) noexcept {
    switch (value) {
    case DecodeTransferPolicy::automatic:
        return "auto";
    case DecodeTransferPolicy::gpu_only:
        return "gpu";
    case DecodeTransferPolicy::host:
        return "host";
    }
    return "unknown";
}
[[nodiscard]] constexpr std::string_view to_string(DecodeTransferPath value) noexcept {
    switch (value) {
    case DecodeTransferPath::host:
        return "host";
    case DecodeTransferPath::native:
        return "native";
    case DecodeTransferPath::mapped:
        return "mapped";
    case DecodeTransferPath::gpu_copy:
        return "gpu_copy";
    }
    return "unknown";
}
[[nodiscard]] constexpr std::string_view to_string(DecodeStorage value) noexcept {
    switch (value) {
    case DecodeStorage::host:
        return "host";
    case DecodeStorage::vulkan:
        return "vulkan";
    case DecodeStorage::qsv_surface:
        return "qsv_surface";
    case DecodeStorage::cuda:
        return "cuda";
    case DecodeStorage::vaapi:
        return "vaapi";
    case DecodeStorage::amf_surface:
        return "amf_surface";
    }
    return "unknown";
}
[[nodiscard]] constexpr std::optional<DecodeBackendPreference>
parse_decode_backend_preference(std::string_view value) noexcept {
    for (auto backend : {DecodeBackendPreference::automatic, DecodeBackendPreference::vulkan_video,
                         DecodeBackendPreference::qsv, DecodeBackendPreference::nvdec,
                         DecodeBackendPreference::vaapi, DecodeBackendPreference::amf}) {
        if (to_string(backend) == value)
            return backend;
    }
    if (value == "vulkan_video")
        return DecodeBackendPreference::vulkan_video;
    return std::nullopt;
}
[[nodiscard]] constexpr std::optional<DecodeTransferPolicy>
parse_decode_transfer(std::string_view value) noexcept {
    for (auto policy : {DecodeTransferPolicy::automatic, DecodeTransferPolicy::gpu_only,
                        DecodeTransferPolicy::host}) {
        if (to_string(policy) == value)
            return policy;
    }
    return std::nullopt;
}
struct DecodeBackendInfo final {
    StreamDecodeBackend backend{StreamDecodeBackend::software};
    bool built{};
    bool platform_supported{};
    std::string status;
};
struct DecodeDeviceInfo final {
    StreamDecodeBackend backend{StreamDecodeBackend::software};
    std::string id;
    std::string name;
    std::string physical_id;
    // Backend-specific input, not a stable device identity.
    std::string native_selector;
};
struct DecodeDiagnostics final {
    DecodeStorage native_storage{DecodeStorage::host};
    DecodeTransferPath transfer_path{DecodeTransferPath::host};
    std::string device_id;
    std::vector<std::string> attempts;
    std::uint64_t host_transfer_bytes{};
    std::uint64_t gpu_copy_bytes{};
    std::uint64_t transfer_nanoseconds{};
};
[[nodiscard]] std::vector<DecodeBackendInfo> enumerate_decode_backends();
[[nodiscard]] Result<std::vector<DecodeDeviceInfo>>
enumerate_decode_devices(DecodeBackendPreference preference = DecodeBackendPreference::automatic);
} // namespace crossvmaf
