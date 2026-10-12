#pragma once
#include "gpu/memory.hpp"
#include "gpu/device.hpp"
#include "crossvmaf/error.hpp"
#include <volk.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>

namespace crossvmaf::gpu {

struct BufferCreateInfo {
    VkDeviceSize size{};
    VkBufferUsageFlags usage{};
    VkMemoryPropertyFlags required_memory{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    VkMemoryPropertyFlags preferred_memory{VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT};
    bool persistently_mapped{false};
};

[[nodiscard]] inline BufferCreateInfo
host_visible_storage(VkDeviceSize size, VkBufferUsageFlags extra_usage = 0) noexcept {
    BufferCreateInfo create_info{};
    create_info.size = size;
    create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra_usage;
    create_info.required_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    create_info.preferred_memory =
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    create_info.persistently_mapped = true;
    return create_info;
}

struct BufferMemoryMetadata {
    std::uint32_t memory_type_index{};
    std::uint32_t memory_heap_index{};
    VkMemoryPropertyFlags property_flags{};
    VkDeviceSize non_coherent_atom_size{1U};
    bool host_visible{};
    bool host_coherent{};
    bool device_local{};
};

struct BufferWritePhases {
    std::function<void()> before_mapped_copy;
    std::function<void()> after_mapped_copy;
    std::function<void()> before_flush;
    std::function<void()> after_flush;
};

class Buffer final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Buffer>>
    create(DeviceView device, const BufferCreateInfo &create_info);

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    ~Buffer();

    [[nodiscard]] VkBuffer handle() const noexcept { return buffer_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }
    [[nodiscard]] VkBufferUsageFlags usage() const noexcept { return usage_; }
    [[nodiscard]] bool host_coherent() const noexcept { return allocation_.host_coherent(); }
    [[nodiscard]] BufferMemoryMetadata memory_metadata() const noexcept;

    [[nodiscard]] Result<void> write(std::span<const std::byte> source, VkDeviceSize offset = 0);
    // Writes a contiguous payload, zeroes the remainder of logical_size, and
    // flushes the complete logical range exactly once. Padding is deliberately
    // bounded so accidental large clears cannot hide a caller size mismatch.
    [[nodiscard]] Result<void> write_with_zero_padding(std::span<const std::byte> payload,
                                                       VkDeviceSize logical_size,
                                                       VkDeviceSize offset = 0,
                                                       VkDeviceSize maximum_padding_bytes = 3U,
                                                       const BufferWritePhases &phases = {});
    [[nodiscard]] Result<void> read(std::span<std::byte> destination, VkDeviceSize offset = 0);
    [[nodiscard]] Result<std::span<const std::byte>> readable_span(VkDeviceSize size,
                                                                   VkDeviceSize offset = 0U);
    [[nodiscard]] Result<void> fill(std::byte value = std::byte{0});
    [[nodiscard]] Result<std::span<std::byte>> writable_span(VkDeviceSize size,
                                                             VkDeviceSize offset = 0U);
    [[nodiscard]] Result<void> commit_external_write(VkDeviceSize payload_size,
                                                     VkDeviceSize logical_size,
                                                     VkDeviceSize offset = 0U,
                                                     VkDeviceSize maximum_padding_bytes = 3U);
    [[nodiscard]] Result<std::uint64_t> sampled_hash(VkDeviceSize size, VkDeviceSize offset,
                                                     VkDeviceSize sampling_stride) const;

  private:
    explicit Buffer(DeviceView device) noexcept : device_(device) {}

    [[nodiscard]] Result<void> flush_range(VkDeviceSize offset, VkDeviceSize size);
    [[nodiscard]] Result<void> invalidate_range(VkDeviceSize offset, VkDeviceSize size);

    DeviceView device_;
    VkBuffer buffer_{VK_NULL_HANDLE};
    DeviceMemoryAllocation allocation_;
    VkDeviceSize size_{};
    VkBufferUsageFlags usage_{};
    void *mapped_{};
};

} // namespace crossvmaf::gpu
