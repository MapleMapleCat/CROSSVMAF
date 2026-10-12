#include "video/backend_selection.hpp"
#include "video/backends/vulkan.hpp"
#include "video/decode_backend.hpp"

#include <string>
#include <string_view>

namespace crossvmaf::video {
Result<void> validate_decode_selection(const StreamOptions &options) {
    if (to_string(options.decode_backend) == "unknown" ||
        to_string(options.decode_transfer) == "unknown" || options.decode_device.empty())
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "invalid decode backend, device or transfer policy"});
    if (options.decode_mode == DecodeMode::software &&
        (options.decode_backend != DecodeBackendPreference::automatic ||
         options.decode_transfer == DecodeTransferPolicy::gpu_only ||
         options.decode_device != "same"))
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "software decode conflicts with an explicit hardware "
                                          "backend/device or GPU-only transfer"});
    return Result<void>::success();
}
DecodeCandidatePlan plan_decode_candidates(const StreamOptions &options,
                                            const DeviceInfo &compute_device) {
    DecodeCandidatePlan plan;
    if (options.decode_mode == DecodeMode::software) {
        plan.candidates.push_back({StreamDecodeBackend::software, DecodeTransferPolicy::host});
        return plan;
    }
    if (options.decode_backend != DecodeBackendPreference::automatic) {
        plan.candidates.push_back({requested_backend(options.decode_backend), options.decode_transfer});
        return plan;
    }

    const auto add_candidate = [&](StreamDecodeBackend backend, DecodeTransferPolicy transfer) {
        if (!backend_built(backend)) {
            plan.skipped.push_back(std::string(to_string(backend)) + ": not built");
            return;
        }
        const std::string_view selector = options.decode_device;
        const bool selects_compute_device = backend == StreamDecodeBackend::vulkan_video &&
            !compute_device.uuid.empty() &&
            (selector == compute_device.uuid || selector == "vulkan:" + compute_device.uuid);
        const bool selects_backend_device = backend != StreamDecodeBackend::vulkan_video &&
            selector.starts_with(std::string(to_string(backend)) + ":");
        if (selector != "same" && !selects_compute_device && !selects_backend_device) {
            plan.skipped.push_back(std::string(to_string(backend)) +
                                   ": decode device does not match this backend");
            return;
        }
        const bool missing_gpu_identity = backend == StreamDecodeBackend::nvdec
                                              ? compute_device.uuid.empty()
                                              : backend != StreamDecodeBackend::vulkan_video &&
                                                    compute_device.physical_id.empty();
        if (transfer == DecodeTransferPolicy::gpu_only && missing_gpu_identity) {
            plan.skipped.push_back(std::string(to_string(backend)) +
                                   ": no verified compute device identity for GPU transfer");
            return;
        }
        plan.candidates.push_back({backend, transfer});
    };

    const auto add_backend = [&](StreamDecodeBackend backend) {
        add_candidate(backend, options.decode_transfer == DecodeTransferPolicy::host
                                   ? DecodeTransferPolicy::host
                                   : DecodeTransferPolicy::gpu_only);
        // Hardware-only requests may retain the preferred decoder by downloading
        // frames when its GPU route is unavailable. GPU-only requests never do.
        if (backend != StreamDecodeBackend::vulkan_video &&
            options.decode_mode == DecodeMode::hardware &&
            options.decode_transfer == DecodeTransferPolicy::automatic) {
            add_candidate(backend, DecodeTransferPolicy::host);
        }
    };
    const auto add_vendor_backend = [&](StreamDecodeBackend backend, std::uint32_t vendor_id) {
        const std::string_view selector = options.decode_device;
        // An explicit backend device may be on a different adapter from compute.
        // Runtime identity checks still reject cross-device GPU-only transfers.
        if ((selector == "same" && compute_device.vendor_id == vendor_id) ||
            selector.starts_with(std::string(to_string(backend)) + ":")) {
            add_backend(backend);
        } else {
            plan.skipped.push_back(std::string(to_string(backend)) +
                                   ": device vendor or selector does not match this backend");
        }
    };

    add_vendor_backend(StreamDecodeBackend::qsv, 0x8086U);
    add_vendor_backend(StreamDecodeBackend::nvdec, 0x10deU);
#if defined(__linux__)
    add_backend(StreamDecodeBackend::vaapi);
#elif defined(_WIN32)
    if (allow_automatic_hardware_decode(compute_device)) {
        add_backend(StreamDecodeBackend::vulkan_video);
    } else {
        plan.skipped.push_back("vulkan_video: disabled by driver conformance policy");
    }
#endif
    if (options.decode_mode == DecodeMode::automatic &&
        options.decode_transfer != DecodeTransferPolicy::gpu_only) {
        plan.candidates.push_back({StreamDecodeBackend::software, DecodeTransferPolicy::host});
    }
    return plan;
}
} // namespace crossvmaf::video
