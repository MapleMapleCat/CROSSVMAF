#pragma once

#include "crossvmaf/error.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <memory>
#include <optional>

namespace crossvmaf::video {

struct VulkanDecodeQueueSelection final {
    std::uint32_t family{};
    std::uint32_t index{};
    bool operator==(const VulkanDecodeQueueSelection&) const = default;
};

// Keep this owner alive until FFmpeg releases every context using the resolver.
// Recording is serialized per native device. An optional queue selection maps
// logical video queue zero to one native queue for this decoder context.
struct VulkanDecodeDispatch final {
    std::shared_ptr<void> lifetime;
    PFN_vkGetInstanceProcAddr get_instance_proc_address{};
};

[[nodiscard]] Result<VulkanDecodeDispatch>
make_vulkan_decode_dispatch(VkInstance instance, VkDevice device,
                           PFN_vkGetInstanceProcAddr get_instance_proc_address,
                           std::optional<VulkanDecodeQueueSelection> queue = std::nullopt);

} // namespace crossvmaf::video
