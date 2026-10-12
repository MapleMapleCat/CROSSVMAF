#include "gpu/memory.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include "gpu/device.hpp"
#include "gpu/error.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace crossvmaf::gpu {
namespace {

[[nodiscard]] std::optional<std::uint64_t>
free_range_allocator_align_up(std::uint64_t value, std::uint64_t alignment) noexcept {
    if (alignment == 0U) return std::nullopt;
    const std::uint64_t remainder = value % alignment;
    if (remainder == 0U) return value;
    const std::uint64_t increment = alignment - remainder;
    if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
        return std::nullopt;
    }
    return value + increment;
}

[[nodiscard]] bool free_range_allocator_range_end(FreeRange range, std::uint64_t &end) noexcept {
    if (range.size > std::numeric_limits<std::uint64_t>::max() - range.offset) {
        return false;
    }
    end = range.offset + range.size;
    return true;
}

} // namespace

FreeRangeAllocator::FreeRangeAllocator(std::uint64_t capacity) : capacity_(capacity) {
    if (capacity != 0U) free_ranges_.push_back({0U, capacity});
}

std::optional<FreeRange> FreeRangeAllocator::allocate(std::uint64_t size, std::uint64_t alignment) {
    if (size == 0U || alignment == 0U) return std::nullopt;

    std::optional<std::size_t> best_range_index;
    std::uint64_t best_aligned_offset{};
    std::uint64_t best_range_size = std::numeric_limits<std::uint64_t>::max();

    for (std::size_t range_index = 0U; range_index < free_ranges_.size(); ++range_index) {
        const FreeRange candidate = free_ranges_[range_index];
        const auto aligned_offset = free_range_allocator_align_up(candidate.offset, alignment);
        if (!aligned_offset || *aligned_offset < candidate.offset) continue;

        const std::uint64_t prefix_size = *aligned_offset - candidate.offset;
        if (prefix_size > candidate.size || size > candidate.size - prefix_size) {
            continue;
        }

        const bool is_better_fit = candidate.size < best_range_size;
        const bool is_equal_fit_at_lower_offset =
            candidate.size == best_range_size &&
            (!best_range_index || *aligned_offset < best_aligned_offset);
        if (is_better_fit || is_equal_fit_at_lower_offset) {
            best_range_index = range_index;
            best_aligned_offset = *aligned_offset;
            best_range_size = candidate.size;
        }
    }

    if (!best_range_index) return std::nullopt;

    const FreeRange selected = free_ranges_[*best_range_index];
    const std::uint64_t prefix_size = best_aligned_offset - selected.offset;
    const std::uint64_t suffix_offset = best_aligned_offset + size;
    const std::uint64_t suffix_size = selected.size - prefix_size - size;

    auto selected_position = free_ranges_.begin() + static_cast<std::ptrdiff_t>(*best_range_index);
    selected_position = free_ranges_.erase(selected_position);
    if (suffix_size != 0U) {
        selected_position = free_ranges_.insert(selected_position, {suffix_offset, suffix_size});
    }
    if (prefix_size != 0U) {
        free_ranges_.insert(selected_position, {selected.offset, prefix_size});
    }
    return FreeRange{best_aligned_offset, size};
}

bool FreeRangeAllocator::release(FreeRange range) {
    std::uint64_t released_end{};
    if (range.size == 0U || !free_range_allocator_range_end(range, released_end) ||
        released_end > capacity_) {
        return false;
    }

    auto next = std::lower_bound(free_ranges_.begin(), free_ranges_.end(), range.offset,
                                 [](const FreeRange &free_range, std::uint64_t offset) {
                                     return free_range.offset < offset;
                                 });
    if (next != free_ranges_.end() && released_end > next->offset) return false;

    if (next != free_ranges_.begin()) {
        const auto previous = std::prev(next);
        std::uint64_t previous_end{};
        if (!free_range_allocator_range_end(*previous, previous_end) ||
            previous_end > range.offset) {
            return false;
        }
        if (previous_end == range.offset) {
            range.offset = previous->offset;
            range.size += previous->size;
            next = free_ranges_.erase(previous);
        }
    }

    if (next != free_ranges_.end()) {
        std::uint64_t coalesced_end{};
        if (!free_range_allocator_range_end(range, coalesced_end)) return false;
        if (coalesced_end == next->offset) {
            range.size += next->size;
            next = free_ranges_.erase(next);
        }
    }
    free_ranges_.insert(next, range);
    return true;
}

std::uint64_t FreeRangeAllocator::free_size() const noexcept {
    std::uint64_t total_free_size = 0U;
    for (const FreeRange range : free_ranges_)
        total_free_size += range.size;
    return total_free_size;
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

std::optional<MemoryReservation>
make_memory_reservation(std::uint64_t logical_size, std::uint64_t required_alignment,
                        std::uint64_t isolation_granularity) noexcept {
    if (logical_size == 0U || required_alignment == 0U || isolation_granularity == 0U) {
        return std::nullopt;
    }

    const std::uint64_t alignment_divisor = std::gcd(required_alignment, isolation_granularity);
    const std::uint64_t alignment_multiplier = isolation_granularity / alignment_divisor;
    if (required_alignment > std::numeric_limits<std::uint64_t>::max() / alignment_multiplier) {
        return std::nullopt;
    }
    const std::uint64_t combined_alignment = required_alignment * alignment_multiplier;

    const std::uint64_t remainder = logical_size % isolation_granularity;
    std::uint64_t reserved_size = logical_size;
    if (remainder != 0U) {
        const std::uint64_t increment = isolation_granularity - remainder;
        if (logical_size > std::numeric_limits<std::uint64_t>::max() - increment) {
            return std::nullopt;
        }
        reserved_size += increment;
    }

    return MemoryReservation{logical_size, reserved_size, combined_alignment};
}

std::optional<MemoryBlockAllocationAttempts>
make_pooled_block_allocation_attempts(std::uint64_t preferred_block_size,
                                      std::uint64_t reserved_request_size) noexcept {
    if (reserved_request_size == 0U || preferred_block_size < reserved_request_size) {
        return std::nullopt;
    }

    MemoryBlockAllocationAttempts attempts{};
    attempts.sizes[attempts.count++] = preferred_block_size;
    if (preferred_block_size != reserved_request_size) {
        attempts.sizes[attempts.count++] = reserved_request_size;
    }
    return attempts;
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {
namespace {

constexpr VkDeviceSize memory_minimum_pooled_block_size = 64U * 1024U;
constexpr VkDeviceSize memory_maximum_pooled_block_size = 64U * 1024U * 1024U;
constexpr VkDeviceSize memory_oversized_allocation_threshold =
    memory_maximum_pooled_block_size / 2U;

[[nodiscard]] bool memory_trace_memory_pool() noexcept {
    static const bool enabled = [] {
#if defined(_WIN32)
        char *value = nullptr;
        std::size_t length = 0U;
        const errno_t status = _dupenv_s(&value, &length, "CROSSVMAF_VULKAN_TRACE");
        const bool result = status == 0 && value != nullptr && std::string_view(value) != "0";
        std::free(value);
        return result;
#else
        const char *const value = std::getenv("CROSSVMAF_VULKAN_TRACE");
        return value != nullptr && std::string_view(value) != "0";
#endif
    }();
    return enabled;
}

void memory_trace_memory_type_fallback(VkMemoryPropertyFlags required_properties,
                                       VkMemoryPropertyFlags preferred_properties,
                                       VkMemoryPropertyFlags selected_properties,
                                       std::uint32_t memory_type_index) noexcept {
    if (!memory_trace_memory_pool()) return;
    std::fprintf(stderr,
                 "[crossvmaf-vulkan] memory_type_fallback required=0x%x "
                 "preferred=0x%x selected=0x%x type=%u\n",
                 static_cast<unsigned>(required_properties),
                 static_cast<unsigned>(preferred_properties),
                 static_cast<unsigned>(selected_properties), memory_type_index);
}

struct memory_MemoryTypeSelection {
    std::uint32_t index{};
    VkMemoryPropertyFlags properties{};
};

struct memory_BucketKey {
    std::uint32_t memory_type_index{};
    MemoryMappingPolicy mapping_policy{MemoryMappingPolicy::unmapped};

    [[nodiscard]] bool operator==(const memory_BucketKey &) const noexcept = default;
};

[[nodiscard]] std::optional<memory_MemoryTypeSelection>
memory_select_memory_type(const VkPhysicalDeviceMemoryProperties &memory_properties,
                          std::uint32_t allowed_memory_types,
                          VkMemoryPropertyFlags required_properties,
                          VkMemoryPropertyFlags preferred_properties) noexcept {
    std::optional<memory_MemoryTypeSelection> fallback;
    for (std::uint32_t memory_type_index = 0U;
         memory_type_index < memory_properties.memoryTypeCount; ++memory_type_index) {
        const bool memory_type_is_allowed =
            (allowed_memory_types & (std::uint32_t{1} << memory_type_index)) != 0U;
        if (!memory_type_is_allowed) continue;

        const VkMemoryPropertyFlags properties =
            memory_properties.memoryTypes[memory_type_index].propertyFlags;
        if ((properties & required_properties) != required_properties) continue;

        const memory_MemoryTypeSelection selection{memory_type_index, properties};
        if ((properties & preferred_properties) == preferred_properties) {
            return selection;
        }
        if (!fallback) fallback = selection;
    }
    return fallback;
}

struct memory_HeapBudgetSnapshot {
    VkDeviceSize budget{};
    VkDeviceSize usage{};
    bool available{};
};

[[nodiscard]] memory_HeapBudgetSnapshot
memory_query_heap_budget(VkPhysicalDevice physical_device, bool memory_budget_enabled,
                         std::uint32_t heap_index) noexcept {
    memory_HeapBudgetSnapshot snapshot{};
    if (!memory_budget_enabled || heap_index >= VK_MAX_MEMORY_HEAPS) {
        return snapshot;
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 memory_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    memory_properties.pNext = &budget_properties;
    vkGetPhysicalDeviceMemoryProperties2(physical_device, &memory_properties);
    snapshot.available = true;
    snapshot.budget = budget_properties.heapBudget[heap_index];
    snapshot.usage = budget_properties.heapUsage[heap_index];
    return snapshot;
}

[[nodiscard]] VkDeviceSize memory_round_up_to_power_of_two(VkDeviceSize value) noexcept {
    if (value <= 1U) return 1U;
    VkDeviceSize rounded_value = 1U;
    while (rounded_value < value &&
           rounded_value <= std::numeric_limits<VkDeviceSize>::max() / 2U) {
        rounded_value *= 2U;
    }
    return rounded_value < value ? std::numeric_limits<VkDeviceSize>::max() : rounded_value;
}

class memory_PendingMemoryResources final {
  public:
    memory_PendingMemoryResources(VkDevice device, const VolkDeviceTable &table,
                                  VkDeviceMemory &memory) noexcept
        : device_(device), table_(&table), memory_(&memory) {}

    memory_PendingMemoryResources(const memory_PendingMemoryResources &) = delete;
    memory_PendingMemoryResources &operator=(const memory_PendingMemoryResources &) = delete;

    ~memory_PendingMemoryResources() {
        if (released_ || *memory_ == VK_NULL_HANDLE) return;
        if (mapped_) table_->vkUnmapMemory(device_, *memory_);
        table_->vkFreeMemory(device_, *memory_, nullptr);
    }

    void mark_mapped() noexcept { mapped_ = true; }
    void release() noexcept { released_ = true; }

  private:
    VkDevice device_{VK_NULL_HANDLE};
    const VolkDeviceTable *table_{};
    VkDeviceMemory *memory_{};
    bool mapped_{};
    bool released_{};
};

} // namespace

class MemoryBlock final {
  public:
    MemoryBlock(VkDevice device, const VolkDeviceTable &table, VkDeviceMemory memory,
                VkDeviceSize size, memory_BucketKey bucket_key, std::uint32_t memory_heap_index,
                VkMemoryPropertyFlags memory_properties, VkDeviceSize non_coherent_atom_size,
                void *mapped_base, bool dedicated)
        : device_(device), table_(&table), memory_(memory), size_(size), bucket_key_(bucket_key),
          memory_heap_index_(memory_heap_index), memory_properties_(memory_properties),
          non_coherent_atom_size_(non_coherent_atom_size), mapped_base_(mapped_base),
          dedicated_(dedicated), free_ranges_(size) {}

    MemoryBlock(const MemoryBlock &) = delete;
    MemoryBlock &operator=(const MemoryBlock &) = delete;

    ~MemoryBlock() {
        if (mapped_base_ != nullptr) table_->vkUnmapMemory(device_, memory_);
        if (memory_ != VK_NULL_HANDLE) table_->vkFreeMemory(device_, memory_, nullptr);
    }

    VkDevice device_{VK_NULL_HANDLE};
    const VolkDeviceTable *table_{};
    VkDeviceMemory memory_{VK_NULL_HANDLE};
    VkDeviceSize size_{};
    memory_BucketKey bucket_key_{};
    std::uint32_t memory_heap_index_{};
    VkMemoryPropertyFlags memory_properties_{};
    VkDeviceSize non_coherent_atom_size_{1U};
    void *mapped_base_{};
    bool dedicated_{};
    FreeRangeAllocator free_ranges_;
};

struct DeviceMemoryPool::Implementation {
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    const VolkDeviceTable *table{};
    DeviceResourceLifetime *resource_lifetime{};
    VkPhysicalDeviceMemoryProperties memory_properties{};
    VkDeviceSize non_coherent_atom_size{1U};
    bool memory_budget_enabled{};
    std::uint32_t device_local_heap_index{VK_MAX_MEMORY_HEAPS};
    DeviceMemoryPoolStats statistics{};
    std::vector<std::unique_ptr<MemoryBlock>> memory_blocks;
};

namespace {

template <typename ImplementationType>
[[nodiscard]] Result<std::unique_ptr<MemoryBlock>>
memory_create_memory_block(ImplementationType &implementation, memory_BucketKey bucket_key,
                           VkMemoryPropertyFlags memory_properties, VkDeviceSize block_size,
                           bool dedicated, VkBuffer dedicated_buffer,
                           bool &memory_allocation_failed) {
    memory_allocation_failed = false;
    VkMemoryDedicatedAllocateInfo dedicated_allocation_info{};
    dedicated_allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated_allocation_info.buffer = dedicated_buffer;

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.pNext = dedicated ? &dedicated_allocation_info : nullptr;
    allocation_info.allocationSize = block_size;
    allocation_info.memoryTypeIndex = bucket_key.memory_type_index;

    VkDeviceMemory memory = VK_NULL_HANDLE;
    memory_PendingMemoryResources pending_resources{implementation.device, *implementation.table,
                                                    memory};
    VkResult result = implementation.table->vkAllocateMemory(implementation.device,
                                                             &allocation_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        memory_allocation_failed = true;
        return Result<std::unique_ptr<MemoryBlock>>::failure(
            make_error("vkAllocateMemory", result));
    }

    void *mapped_base = nullptr;
    if (bucket_key.mapping_policy == MemoryMappingPolicy::persistently_mapped) {
        result = implementation.table->vkMapMemory(implementation.device, memory, 0U, VK_WHOLE_SIZE,
                                                   0U, &mapped_base);
        if (result != VK_SUCCESS) {
            return Result<std::unique_ptr<MemoryBlock>>::failure(make_error("vkMapMemory", result));
        }
        pending_resources.mark_mapped();
    }

    auto memory_block = std::make_unique<MemoryBlock>(
        implementation.device, *implementation.table, memory, block_size, bucket_key,
        implementation.memory_properties.memoryTypes[bucket_key.memory_type_index].heapIndex,
        memory_properties, implementation.non_coherent_atom_size, mapped_base, dedicated);
    pending_resources.release();
    return Result<std::unique_ptr<MemoryBlock>>::success(std::move(memory_block));
}

template <typename ImplementationType>
[[nodiscard]] VkDeviceSize memory_choose_pooled_block_size(const ImplementationType &implementation,
                                                           memory_BucketKey bucket_key,
                                                           VkDeviceSize requested_size) noexcept {
    const VkDeviceSize doubled_request = requested_size > memory_maximum_pooled_block_size / 2U
                                             ? memory_maximum_pooled_block_size
                                             : requested_size * 2U;
    const VkDeviceSize rounded_request = memory_round_up_to_power_of_two(doubled_request);
    VkDeviceSize selected_size =
        std::max(memory_minimum_pooled_block_size,
                 std::min(rounded_request, memory_maximum_pooled_block_size));

    VkDeviceSize largest_existing_block = 0U;
    for (const auto &memory_block : implementation.memory_blocks) {
        if (!memory_block->dedicated_ && memory_block->bucket_key_ == bucket_key) {
            largest_existing_block = std::max(largest_existing_block, memory_block->size_);
        }
    }
    if (largest_existing_block != 0U) {
        const VkDeviceSize grown_size =
            largest_existing_block > memory_maximum_pooled_block_size / 2U
                ? memory_maximum_pooled_block_size
                : largest_existing_block * 2U;
        selected_size = std::max(selected_size, grown_size);
    }
    return selected_size;
}

} // namespace

DeviceMemoryAllocation::DeviceMemoryAllocation(DeviceMemoryPool &pool, MemoryBlock &memory_block,
                                               VkDeviceSize offset, VkDeviceSize logical_size,
                                               VkDeviceSize reserved_size) noexcept
    : pool_(&pool), memory_block_(&memory_block), offset_(offset), size_(logical_size),
      reserved_size_(reserved_size) {
}

DeviceMemoryAllocation::DeviceMemoryAllocation(DeviceMemoryAllocation &&other) noexcept
    : pool_(std::exchange(other.pool_, nullptr)),
      memory_block_(std::exchange(other.memory_block_, nullptr)),
      offset_(std::exchange(other.offset_, 0U)), size_(std::exchange(other.size_, 0U)),
      reserved_size_(std::exchange(other.reserved_size_, 0U)) {
}

DeviceMemoryAllocation &DeviceMemoryAllocation::operator=(DeviceMemoryAllocation &&other) noexcept {
    if (this == &other) return *this;
    reset();
    pool_ = std::exchange(other.pool_, nullptr);
    memory_block_ = std::exchange(other.memory_block_, nullptr);
    offset_ = std::exchange(other.offset_, 0U);
    size_ = std::exchange(other.size_, 0U);
    reserved_size_ = std::exchange(other.reserved_size_, 0U);
    return *this;
}

DeviceMemoryAllocation::~DeviceMemoryAllocation() {
    reset();
}

VkDeviceMemory DeviceMemoryAllocation::memory() const noexcept {
    return memory_block_ != nullptr ? memory_block_->memory_ : VK_NULL_HANDLE;
}

void *DeviceMemoryAllocation::mapped_data() const noexcept {
    if (memory_block_ == nullptr || memory_block_->mapped_base_ == nullptr) {
        return nullptr;
    }
    if constexpr (sizeof(std::size_t) < sizeof(VkDeviceSize)) {
        if (offset_ > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max())) {
            return nullptr;
        }
    }
    return static_cast<std::byte *>(memory_block_->mapped_base_) +
           static_cast<std::size_t>(offset_);
}

std::uint32_t DeviceMemoryAllocation::memory_type_index() const noexcept {
    return memory_block_ != nullptr ? memory_block_->bucket_key_.memory_type_index : 0U;
}

std::uint32_t DeviceMemoryAllocation::memory_heap_index() const noexcept {
    return memory_block_ != nullptr ? memory_block_->memory_heap_index_ : 0U;
}

VkMemoryPropertyFlags DeviceMemoryAllocation::memory_property_flags() const noexcept {
    return memory_block_ != nullptr ? memory_block_->memory_properties_ : 0U;
}

VkDeviceSize DeviceMemoryAllocation::non_coherent_atom_size() const noexcept {
    return memory_block_ != nullptr ? memory_block_->non_coherent_atom_size_ : 1U;
}

bool DeviceMemoryAllocation::host_visible() const noexcept {
    return (memory_property_flags() & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U;
}

bool DeviceMemoryAllocation::host_coherent() const noexcept {
    return memory_block_ != nullptr &&
           (memory_block_->memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
}

bool DeviceMemoryAllocation::device_local() const noexcept {
    return (memory_property_flags() & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0U;
}

Result<void> DeviceMemoryAllocation::flush(VkDeviceSize relative_offset, VkDeviceSize size) const {
    if (pool_ == nullptr || memory_block_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan memory allocation is empty"});
    }
    return pool_->synchronize_mapped_range(*memory_block_, offset_, size_, relative_offset, size,
                                           false);
}

Result<void> DeviceMemoryAllocation::invalidate(VkDeviceSize relative_offset,
                                                VkDeviceSize size) const {
    if (pool_ == nullptr || memory_block_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan memory allocation is empty"});
    }
    return pool_->synchronize_mapped_range(*memory_block_, offset_, size_, relative_offset, size,
                                           true);
}

void DeviceMemoryAllocation::reset() noexcept {
    DeviceMemoryPool *const pool = std::exchange(pool_, nullptr);
    MemoryBlock *const memory_block = std::exchange(memory_block_, nullptr);
    const VkDeviceSize offset = std::exchange(offset_, 0U);
    const VkDeviceSize logical_size = std::exchange(size_, 0U);
    const VkDeviceSize reserved_size = std::exchange(reserved_size_, 0U);
    if (pool != nullptr && memory_block != nullptr) {
        pool->release(*memory_block, offset, logical_size, reserved_size);
    }
}

DeviceMemoryPool::DeviceMemoryPool(std::unique_ptr<Implementation> implementation) noexcept
    : implementation_(std::move(implementation)) {
}

DeviceMemoryPool::~DeviceMemoryPool() {
    if (!implementation_->resource_lifetime->safe_to_destroy()) {
        static_cast<void>(implementation_.release());
        return;
    }
    assert(implementation_->statistics.live_allocation_count == 0U);
}

Result<std::unique_ptr<DeviceMemoryPool>>
DeviceMemoryPool::create(VkPhysicalDevice physical_device, VkDevice device,
                         const VolkDeviceTable &table, DeviceResourceLifetime &resource_lifetime,
                         bool memory_budget_enabled) {
    if (physical_device == VK_NULL_HANDLE || device == VK_NULL_HANDLE) {
        return Result<std::unique_ptr<DeviceMemoryPool>>::failure(
            {ErrorCode::invalid_argument, "device memory pool requires a complete Vulkan device"});
    }

    auto implementation = std::make_unique<Implementation>();
    implementation->physical_device = physical_device;
    implementation->device = device;
    implementation->table = &table;
    implementation->resource_lifetime = &resource_lifetime;
    implementation->memory_budget_enabled = memory_budget_enabled;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &implementation->memory_properties);
    implementation->statistics.memory_budget_available = memory_budget_enabled;
    for (std::uint32_t heap_index = 0U;
         heap_index < implementation->memory_properties.memoryHeapCount; ++heap_index) {
        const VkMemoryHeap &heap = implementation->memory_properties.memoryHeaps[heap_index];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0U) continue;
        if (implementation->device_local_heap_index == VK_MAX_MEMORY_HEAPS ||
            heap.size > implementation->statistics.device_local_heap_bytes) {
            implementation->device_local_heap_index = heap_index;
            implementation->statistics.device_local_heap_bytes = heap.size;
        }
    }
    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(physical_device, &device_properties);
    implementation->non_coherent_atom_size =
        std::max<VkDeviceSize>(1U, device_properties.limits.nonCoherentAtomSize);
    return Result<std::unique_ptr<DeviceMemoryPool>>::success(
        std::unique_ptr<DeviceMemoryPool>(new DeviceMemoryPool(std::move(implementation))));
}

Result<DeviceMemoryAllocation> DeviceMemoryPool::allocate(const DeviceMemoryRequest &request) {
    if (request.requirements.size == 0U || request.requirements.alignment == 0U ||
        request.requirements.memoryTypeBits == 0U) {
        return Result<DeviceMemoryAllocation>::failure(
            {ErrorCode::invalid_argument, "Vulkan memory request contains invalid requirements"});
    }

    const auto memory_type = memory_select_memory_type(
        implementation_->memory_properties, request.requirements.memoryTypeBits,
        request.required_properties, request.preferred_properties);
    if (!memory_type) {
        return Result<DeviceMemoryAllocation>::failure(
            {ErrorCode::backend_error,
             "no Vulkan memory type satisfies the allocation requirements"});
    }
    if ((memory_type->properties & request.preferred_properties) != request.preferred_properties) {
        if (!implementation_->statistics.preferred_memory_fallback) {
            implementation_->statistics.preferred_memory_fallback = true;
            memory_trace_memory_type_fallback(request.required_properties,
                                              request.preferred_properties, memory_type->properties,
                                              memory_type->index);
        }
    }
    if (request.mapping_policy == MemoryMappingPolicy::persistently_mapped &&
        (memory_type->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0U) {
        return Result<DeviceMemoryAllocation>::failure(
            {ErrorCode::invalid_argument,
             "persistently mapped Vulkan memory must be host visible"});
    }

    const memory_BucketKey bucket_key{memory_type->index, request.mapping_policy};
    const bool use_dedicated_allocation =
        request.requires_dedicated_allocation || request.prefers_dedicated_allocation ||
        request.requirements.size > memory_oversized_allocation_threshold;
    if (use_dedicated_allocation && request.dedicated_buffer == VK_NULL_HANDLE) {
        return Result<DeviceMemoryAllocation>::failure(
            {ErrorCode::invalid_argument,
             "dedicated Vulkan buffer allocation requires a buffer handle"});
    }

    const bool requires_atom_isolation =
        !use_dedicated_allocation &&
        request.mapping_policy == MemoryMappingPolicy::persistently_mapped &&
        (memory_type->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U;
    const VkDeviceSize isolation_granularity =
        requires_atom_isolation ? implementation_->non_coherent_atom_size : 1U;
    const auto reservation = make_memory_reservation(
        request.requirements.size, request.requirements.alignment, isolation_granularity);
    if (!reservation) {
        return Result<DeviceMemoryAllocation>::failure(
            {ErrorCode::invalid_argument, "Vulkan memory reservation alignment or size overflows"});
    }
    if (implementation_->statistics.used_bytes >
            std::numeric_limits<VkDeviceSize>::max() - reservation->logical_size ||
        implementation_->statistics.live_allocation_count ==
            std::numeric_limits<std::uint64_t>::max()) {
        return Result<DeviceMemoryAllocation>::failure(
            {ErrorCode::backend_error, "Vulkan memory pool statistics overflow"});
    }

    MemoryBlock *selected_block = nullptr;
    std::optional<FreeRange> selected_range;
    if (!use_dedicated_allocation) {
        for (const auto &memory_block : implementation_->memory_blocks) {
            if (memory_block->dedicated_ || memory_block->bucket_key_ != bucket_key) {
                continue;
            }
            auto allocation_range = memory_block->free_ranges_.allocate(reservation->reserved_size,
                                                                        reservation->alignment);
            if (allocation_range) {
                selected_block = memory_block.get();
                selected_range = allocation_range;
                break;
            }
        }
    }

    if (selected_block == nullptr) {
        if (implementation_->statistics.block_count == std::numeric_limits<std::uint64_t>::max()) {
            return Result<DeviceMemoryAllocation>::failure(
                {ErrorCode::backend_error, "Vulkan memory pool block count overflows"});
        }

        const VkDeviceSize preferred_block_size =
            use_dedicated_allocation ? request.requirements.size
                                     : memory_choose_pooled_block_size(*implementation_, bucket_key,
                                                                       reservation->reserved_size);
        MemoryBlockAllocationAttempts allocation_attempts{};
        if (use_dedicated_allocation) {
            allocation_attempts.sizes[allocation_attempts.count++] = preferred_block_size;
        } else {
            const auto pooled_attempts = make_pooled_block_allocation_attempts(
                preferred_block_size, reservation->reserved_size);
            if (!pooled_attempts) {
                return Result<DeviceMemoryAllocation>::failure(
                    {ErrorCode::backend_error,
                     "Vulkan pooled block policy produced an invalid size"});
            }
            allocation_attempts = *pooled_attempts;
        }

        std::unique_ptr<MemoryBlock> new_memory_block;
        Error allocation_error{ErrorCode::backend_error,
                               "Vulkan memory block allocation was not attempted"};
        VkDeviceSize allocated_block_size = 0U;
        for (std::size_t attempt_index = 0U; attempt_index < allocation_attempts.count;
             ++attempt_index) {
            const VkDeviceSize attempted_block_size = allocation_attempts.sizes[attempt_index];
            if (implementation_->statistics.reserved_bytes >
                std::numeric_limits<VkDeviceSize>::max() - attempted_block_size) {
                allocation_error = {ErrorCode::backend_error,
                                    "Vulkan memory pool reserved byte count overflows"};
                continue;
            }

            const std::uint32_t heap_index =
                implementation_->memory_properties.memoryTypes[memory_type->index].heapIndex;
            const memory_HeapBudgetSnapshot budget =
                memory_query_heap_budget(implementation_->physical_device,
                                         implementation_->memory_budget_enabled, heap_index);
            if (budget.available && budget.budget != 0U &&
                budget.usage > budget.budget - std::min(budget.budget, attempted_block_size)) {
                allocation_error = {ErrorCode::backend_error,
                                    "Vulkan memory budget cannot satisfy the allocation"};
                continue;
            }

            bool memory_allocation_failed = false;
            auto new_block_result = memory_create_memory_block(
                *implementation_, bucket_key, memory_type->properties, attempted_block_size,
                use_dedicated_allocation, request.dedicated_buffer, memory_allocation_failed);
            if (new_block_result) {
                new_memory_block = std::move(new_block_result).value();
                allocated_block_size = attempted_block_size;
                break;
            }
            allocation_error = new_block_result.error();
            if (!memory_allocation_failed) break;
        }
        if (new_memory_block == nullptr) {
            return Result<DeviceMemoryAllocation>::failure(std::move(allocation_error));
        }
        implementation_->memory_blocks.push_back(std::move(new_memory_block));
        selected_block = implementation_->memory_blocks.back().get();
        selected_range = selected_block->free_ranges_.allocate(reservation->reserved_size,
                                                               reservation->alignment);
        if (!selected_range) {
            implementation_->memory_blocks.pop_back();
            return Result<DeviceMemoryAllocation>::failure(
                {ErrorCode::backend_error,
                 "new Vulkan memory block could not satisfy its request"});
        }
        implementation_->statistics.reserved_bytes += allocated_block_size;
        ++implementation_->statistics.block_count;
    }

    implementation_->statistics.used_bytes += reservation->logical_size;
    implementation_->statistics.peak_used_bytes = std::max(
        implementation_->statistics.peak_used_bytes, implementation_->statistics.used_bytes);
    ++implementation_->statistics.live_allocation_count;
    return Result<DeviceMemoryAllocation>::success(
        DeviceMemoryAllocation(*this, *selected_block, selected_range->offset,
                               reservation->logical_size, reservation->reserved_size));
}

DeviceMemoryPoolStats DeviceMemoryPool::statistics() const noexcept {
    DeviceMemoryPoolStats snapshot = implementation_->statistics;
    snapshot.memory_budget_available = implementation_->memory_budget_enabled;
    snapshot.device_local_heap_bytes = implementation_->statistics.device_local_heap_bytes;
    if (implementation_->memory_budget_enabled &&
        implementation_->device_local_heap_index < VK_MAX_MEMORY_HEAPS) {
        const memory_HeapBudgetSnapshot budget = memory_query_heap_budget(
            implementation_->physical_device, implementation_->memory_budget_enabled,
            implementation_->device_local_heap_index);
        snapshot.device_local_heap_budget = budget.budget;
        snapshot.device_local_heap_usage = budget.usage;
    }
    return snapshot;
}

Result<void> DeviceMemoryPool::ensure_allocation_budget(VkDeviceSize additional_bytes) const {
    if (additional_bytes == 0U || !implementation_->memory_budget_enabled) {
        return Result<void>::success();
    }
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 memory_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    memory_properties.pNext = &budget_properties;
    vkGetPhysicalDeviceMemoryProperties2(implementation_->physical_device, &memory_properties);
    const std::uint32_t heap_count = implementation_->memory_properties.memoryHeapCount;
    for (std::uint32_t heap_index = 0U; heap_index < heap_count; ++heap_index) {
        const VkDeviceSize budget = budget_properties.heapBudget[heap_index];
        const VkDeviceSize usage = budget_properties.heapUsage[heap_index];
        if (budget == 0U) continue;
        if (usage <= budget && budget - usage >= additional_bytes) {
            return Result<void>::success();
        }
    }
    return Result<void>::failure(
        {ErrorCode::backend_error, "Vulkan memory budget cannot hold the requested allocation"});
}

void DeviceMemoryPool::release(MemoryBlock &memory_block, VkDeviceSize offset,
                               VkDeviceSize logical_size, VkDeviceSize reserved_size) noexcept {
    if (!implementation_->resource_lifetime->safe_to_destroy()) return;

    assert(implementation_->statistics.used_bytes >= logical_size);
    assert(implementation_->statistics.live_allocation_count != 0U);
    implementation_->statistics.used_bytes -= logical_size;
    --implementation_->statistics.live_allocation_count;

    if (!memory_block.dedicated_) {
        const bool released = memory_block.free_ranges_.release({offset, reserved_size});
        assert(released);
        static_cast<void>(released);
        return;
    }

    const auto memory_block_position =
        std::find_if(implementation_->memory_blocks.begin(), implementation_->memory_blocks.end(),
                     [&memory_block](const std::unique_ptr<MemoryBlock> &candidate) {
                         return candidate.get() == &memory_block;
                     });
    assert(memory_block_position != implementation_->memory_blocks.end());
    if (memory_block_position == implementation_->memory_blocks.end()) return;
    implementation_->statistics.reserved_bytes -= memory_block.size_;
    --implementation_->statistics.block_count;
    implementation_->memory_blocks.erase(memory_block_position);
}

Result<void> DeviceMemoryPool::synchronize_mapped_range(const MemoryBlock &memory_block,
                                                        VkDeviceSize allocation_offset,
                                                        VkDeviceSize allocation_size,
                                                        VkDeviceSize relative_offset,
                                                        VkDeviceSize size, bool invalidate) const {
    if (memory_block.mapped_base_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan memory allocation is not mapped"});
    }
    if (relative_offset > allocation_size || size > allocation_size - relative_offset) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "mapped Vulkan memory range exceeds the allocation"});
    }
    if (size == 0U ||
        (memory_block.memory_properties_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U) {
        return Result<void>::success();
    }

    const VkDeviceSize absolute_offset = allocation_offset + relative_offset;
    const VkDeviceSize absolute_end = absolute_offset + size;
    const VkDeviceSize atom_size = implementation_->non_coherent_atom_size;
    const VkDeviceSize aligned_offset = absolute_offset - (absolute_offset % atom_size);
    VkDeviceSize aligned_end = absolute_end;
    const VkDeviceSize end_remainder = aligned_end % atom_size;
    if (end_remainder != 0U) {
        const VkDeviceSize increment = atom_size - end_remainder;
        aligned_end = increment > memory_block.size_ - aligned_end ? memory_block.size_
                                                                   : aligned_end + increment;
    }

    VkMappedMemoryRange mapped_range{};
    mapped_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    mapped_range.memory = memory_block.memory_;
    mapped_range.offset = aligned_offset;
    mapped_range.size =
        aligned_end == memory_block.size_ ? VK_WHOLE_SIZE : aligned_end - aligned_offset;
    const VkResult result = invalidate ? implementation_->table->vkInvalidateMappedMemoryRanges(
                                             implementation_->device, 1U, &mapped_range)
                                       : implementation_->table->vkFlushMappedMemoryRanges(
                                             implementation_->device, 1U, &mapped_range);
    return check(invalidate ? "vkInvalidateMappedMemoryRanges" : "vkFlushMappedMemoryRanges",
                 result);
}

} // namespace crossvmaf::gpu
