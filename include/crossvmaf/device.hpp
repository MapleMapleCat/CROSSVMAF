#pragma once

#include <cstdint>
#include <string>

namespace crossvmaf {

struct DeviceSelector {
    std::string uuid;
};

struct DeviceInfo {
    std::string name;
    std::string uuid;
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t api_version{};
    std::uint32_t driver_version{};
    bool discrete{};
    bool memory_budget_available{};
    bool host_visible_device_local{};
    std::uint64_t device_local_heap_bytes{};
    // Stable cross-API physical identity (LUID on Windows, DRM device on Linux).
    std::string physical_id;
};

}  // namespace crossvmaf
