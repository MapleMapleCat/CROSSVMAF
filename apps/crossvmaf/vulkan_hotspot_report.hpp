#pragma once

#include <nlohmann/json_fwd.hpp>

namespace crossvmaf::evaluation::telemetry { struct VulkanHotspotSnapshot; }

namespace crossvmaf::cli {

// Pure serialization of a telemetry snapshot; no environment or file access.
[[nodiscard]] nlohmann::ordered_json make_vulkan_hotspot_report(
    const evaluation::telemetry::VulkanHotspotSnapshot& snapshot);

} // namespace crossvmaf::cli
