#pragma once
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <array>
#include <cstddef>
#include <volk.h>
#include <algorithm>
#include "crossvmaf/error.hpp"
#include <memory>

namespace crossvmaf::gpu {

struct FreeRange {
    std::uint64_t offset{};
    std::uint64_t size{};

    [[nodiscard]] bool operator==(const FreeRange &) const noexcept = default;
};

class FreeRangeAllocator final {
  public:
    explicit FreeRangeAllocator(std::uint64_t capacity);

    [[nodiscard]] std::optional<FreeRange> allocate(std::uint64_t size, std::uint64_t alignment);
    [[nodiscard]] bool release(FreeRange range);

    [[nodiscard]] std::uint64_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::uint64_t free_size() const noexcept;
    [[nodiscard]] std::span<const FreeRange> free_ranges() const noexcept { return free_ranges_; }

  private:
    std::uint64_t capacity_{};
    std::vector<FreeRange> free_ranges_;
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

// Describes the caller-visible allocation and the potentially larger range
// that must remain unavailable to neighboring pooled allocations.
struct MemoryReservation {
    std::uint64_t logical_size{};
    std::uint64_t reserved_size{};
    std::uint64_t alignment{};
};

[[nodiscard]] std::optional<MemoryReservation>
make_memory_reservation(std::uint64_t logical_size, std::uint64_t required_alignment,
                        std::uint64_t isolation_granularity) noexcept;

struct MemoryBlockAllocationAttempts {
    std::array<std::uint64_t, 2U> sizes{};
    std::size_t count{};
};

// A growth-sized pooled block is attempted first. If it is larger than the
// current reservation, an exact-sized pooled block is the OOM fallback.
[[nodiscard]] std::optional<MemoryBlockAllocationAttempts>
make_pooled_block_allocation_attempts(std::uint64_t preferred_block_size,
                                      std::uint64_t reserved_request_size) noexcept;

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

// A small discrete-GPU BAR heap is not a general-purpose input bank. Only
// prefer direct mappings when every eligible heap spans the largest local
// heap (full-aperture discrete GPUs, or unified-memory devices).
[[nodiscard]] inline bool
supports_direct_mapped_input(const VkPhysicalDeviceMemoryProperties &properties) noexcept {
    VkDeviceSize largest_local_heap = 0U;
    for (std::uint32_t heap_index = 0U; heap_index < properties.memoryHeapCount; ++heap_index) {
        const VkMemoryHeap &heap = properties.memoryHeaps[heap_index];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0U) {
            largest_local_heap = std::max(largest_local_heap, heap.size);
        }
    }
    if (largest_local_heap == 0U) return false;

    constexpr VkMemoryPropertyFlags required_properties =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    bool has_direct_mapping = false;
    for (std::uint32_t type_index = 0U; type_index < properties.memoryTypeCount; ++type_index) {
        const VkMemoryType &memory_type = properties.memoryTypes[type_index];
        if ((memory_type.propertyFlags & required_properties) != required_properties) {
            continue;
        }
        if (memory_type.heapIndex >= properties.memoryHeapCount ||
            properties.memoryHeaps[memory_type.heapIndex].size < largest_local_heap) {
            return false;
        }
        has_direct_mapping = true;
    }
    return has_direct_mapping;
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

class DeviceMemoryPool;
class DeviceResourceLifetime;
class MemoryBlock;

enum class MemoryMappingPolicy : std::uint8_t {
    unmapped,
    persistently_mapped,
};

struct DeviceMemoryPoolStats {
    VkDeviceSize reserved_bytes{};
    // Caller-visible logical bytes, excluding allocator padding and atom
    // isolation reserved around non-coherent mapped allocations.
    VkDeviceSize used_bytes{};
    VkDeviceSize peak_used_bytes{};
    std::uint64_t block_count{};
    std::uint64_t live_allocation_count{};
    bool preferred_memory_fallback{};
    bool memory_budget_available{};
    VkDeviceSize device_local_heap_bytes{};
    VkDeviceSize device_local_heap_budget{};
    VkDeviceSize device_local_heap_usage{};
};

struct DeviceMemoryRequest {
    VkMemoryRequirements requirements{};
    VkMemoryPropertyFlags required_properties{};
    VkMemoryPropertyFlags preferred_properties{};
    MemoryMappingPolicy mapping_policy{MemoryMappingPolicy::unmapped};
    bool prefers_dedicated_allocation{};
    bool requires_dedicated_allocation{};
    VkBuffer dedicated_buffer{VK_NULL_HANDLE};
};

class DeviceMemoryAllocation final {
  public:
    DeviceMemoryAllocation() noexcept = default;
    DeviceMemoryAllocation(const DeviceMemoryAllocation &) = delete;
    DeviceMemoryAllocation &operator=(const DeviceMemoryAllocation &) = delete;
    DeviceMemoryAllocation(DeviceMemoryAllocation &&other) noexcept;
    DeviceMemoryAllocation &operator=(DeviceMemoryAllocation &&other) noexcept;
    ~DeviceMemoryAllocation();

    [[nodiscard]] explicit operator bool() const noexcept { return memory_block_ != nullptr; }
    [[nodiscard]] VkDeviceMemory memory() const noexcept;
    [[nodiscard]] VkDeviceSize offset() const noexcept { return offset_; }
    [[nodiscard]] VkDeviceSize size() const noexcept { return size_; }
    [[nodiscard]] void *mapped_data() const noexcept;
    [[nodiscard]] std::uint32_t memory_type_index() const noexcept;
    [[nodiscard]] std::uint32_t memory_heap_index() const noexcept;
    [[nodiscard]] VkMemoryPropertyFlags memory_property_flags() const noexcept;
    [[nodiscard]] VkDeviceSize non_coherent_atom_size() const noexcept;
    [[nodiscard]] bool host_visible() const noexcept;
    [[nodiscard]] bool host_coherent() const noexcept;
    [[nodiscard]] bool device_local() const noexcept;

    [[nodiscard]] Result<void> flush(VkDeviceSize relative_offset, VkDeviceSize size) const;
    [[nodiscard]] Result<void> invalidate(VkDeviceSize relative_offset, VkDeviceSize size) const;

  private:
    friend class DeviceMemoryPool;

    DeviceMemoryAllocation(DeviceMemoryPool &pool, MemoryBlock &memory_block, VkDeviceSize offset,
                           VkDeviceSize logical_size, VkDeviceSize reserved_size) noexcept;
    void reset() noexcept;

    DeviceMemoryPool *pool_{};
    MemoryBlock *memory_block_{};
    VkDeviceSize offset_{};
    VkDeviceSize size_{};
    VkDeviceSize reserved_size_{};
};

// Thread-confinement and lifetime contract: DeviceMemoryPool, Device, and all
// DeviceViews/resources derived from that Device are used on one thread. The
// pool is not thread-safe, DeviceView is non-owning, and every derived Vulkan
// resource must be destroyed before its owning Device.
class DeviceMemoryPool final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<DeviceMemoryPool>>
    create(VkPhysicalDevice physical_device, VkDevice device, const VolkDeviceTable &table,
           DeviceResourceLifetime &resource_lifetime, bool memory_budget_enabled = false);

    DeviceMemoryPool(const DeviceMemoryPool &) = delete;
    DeviceMemoryPool &operator=(const DeviceMemoryPool &) = delete;
    ~DeviceMemoryPool();

    [[nodiscard]] Result<DeviceMemoryAllocation> allocate(const DeviceMemoryRequest &request);
    [[nodiscard]] DeviceMemoryPoolStats statistics() const noexcept;
    [[nodiscard]] Result<void> ensure_allocation_budget(VkDeviceSize additional_bytes) const;

  private:
    friend class DeviceMemoryAllocation;
    struct Implementation;

    explicit DeviceMemoryPool(std::unique_ptr<Implementation> implementation) noexcept;
    void release(MemoryBlock &memory_block, VkDeviceSize offset, VkDeviceSize logical_size,
                 VkDeviceSize reserved_size) noexcept;
    [[nodiscard]] Result<void> synchronize_mapped_range(const MemoryBlock &memory_block,
                                                        VkDeviceSize allocation_offset,
                                                        VkDeviceSize allocation_size,
                                                        VkDeviceSize relative_offset,
                                                        VkDeviceSize size, bool invalidate) const;

    std::unique_ptr<Implementation> implementation_;
};

} // namespace crossvmaf::gpu
