#include "gpu/buffer.hpp"
#include "gpu/error.hpp"
#include <cstring>
#include <limits>
#include <utility>

namespace crossvmaf::gpu {
namespace {

[[nodiscard]] bool span_fits(VkDeviceSize buffer_size, VkDeviceSize offset,
                             std::size_t byte_count) noexcept {
    if (offset > buffer_size) return false;
    if constexpr (sizeof(std::size_t) > sizeof(VkDeviceSize)) {
        if (byte_count > static_cast<std::size_t>(std::numeric_limits<VkDeviceSize>::max())) {
            return false;
        }
    }
    if constexpr (sizeof(std::size_t) < sizeof(VkDeviceSize)) {
        if (offset > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max())) {
            return false;
        }
    }
    return static_cast<VkDeviceSize>(byte_count) <= buffer_size - offset;
}

} // namespace

Result<std::unique_ptr<Buffer>> Buffer::create(DeviceView device,
                                               const BufferCreateInfo &create_info) {
    if (!device.complete()) {
        return Result<std::unique_ptr<Buffer>>::failure(
            {ErrorCode::backend_error, "buffer creation received an incomplete Vulkan device"});
    }
    if (create_info.size == 0 || create_info.usage == 0) {
        return Result<std::unique_ptr<Buffer>>::failure(
            {ErrorCode::invalid_argument, "Vulkan buffer size and usage must both be non-zero"});
    }
    if (create_info.persistently_mapped &&
        (create_info.required_memory & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0U) {
        return Result<std::unique_ptr<Buffer>>::failure(
            {ErrorCode::invalid_argument,
             "persistently mapped buffers must require host-visible memory"});
    }

    auto buffer = std::unique_ptr<Buffer>(new Buffer(device));
    buffer->size_ = create_info.size;
    buffer->usage_ = create_info.usage;
    const VolkDeviceTable &table = *device.table;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = create_info.size;
    buffer_info.usage = create_info.usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult result = table.vkCreateBuffer(device.device, &buffer_info, nullptr, &buffer->buffer_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<Buffer>>::failure(make_error("vkCreateBuffer", result));
    }

    VkMemoryDedicatedRequirements dedicated_requirements{};
    dedicated_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
    VkMemoryRequirements2 requirements{};
    requirements.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
    requirements.pNext = &dedicated_requirements;
    VkBufferMemoryRequirementsInfo2 requirements_info{};
    requirements_info.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
    requirements_info.buffer = buffer->buffer_;
    table.vkGetBufferMemoryRequirements2(device.device, &requirements_info, &requirements);

    DeviceMemoryRequest memory_request{};
    memory_request.requirements = requirements.memoryRequirements;
    memory_request.required_properties = create_info.required_memory;
    memory_request.preferred_properties = create_info.preferred_memory;
    memory_request.mapping_policy = create_info.persistently_mapped
                                        ? MemoryMappingPolicy::persistently_mapped
                                        : MemoryMappingPolicy::unmapped;
    memory_request.prefers_dedicated_allocation =
        dedicated_requirements.prefersDedicatedAllocation == VK_TRUE;
    memory_request.requires_dedicated_allocation =
        dedicated_requirements.requiresDedicatedAllocation == VK_TRUE;
    memory_request.dedicated_buffer = buffer->buffer_;
    auto allocation = device.memory_pool->allocate(memory_request);
    if (!allocation) {
        return Result<std::unique_ptr<Buffer>>::failure(allocation.error());
    }
    buffer->allocation_ = std::move(allocation).value();

    result = table.vkBindBufferMemory(device.device, buffer->buffer_, buffer->allocation_.memory(),
                                      buffer->allocation_.offset());
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<Buffer>>::failure(make_error("vkBindBufferMemory", result));
    }

    if (create_info.persistently_mapped) {
        buffer->mapped_ = buffer->allocation_.mapped_data();
        if (buffer->mapped_ == nullptr) {
            return Result<std::unique_ptr<Buffer>>::failure(
                {ErrorCode::backend_error,
                 "persistently mapped Vulkan allocation has no mapped address"});
        }
    }
    return Result<std::unique_ptr<Buffer>>::success(std::move(buffer));
}

Result<std::uint64_t> Buffer::sampled_hash(VkDeviceSize size, VkDeviceSize offset,
                                           VkDeviceSize sampling_stride) const {
    if (mapped_ == nullptr || !host_coherent()) {
        return Result<std::uint64_t>::failure(
            {ErrorCode::invalid_state, "sampled buffer hashing requires coherent mapped memory"});
    }
    if (sampling_stride == 0U || offset > size_ || size > size_ - offset ||
        size > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max()) ||
        offset > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max()) ||
        sampling_stride > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max())) {
        return Result<std::uint64_t>::failure(
            {ErrorCode::invalid_argument, "sampled buffer hash range is invalid"});
    }

    constexpr std::uint64_t fnv_offset_basis = 14'695'981'039'346'656'037ULL;
    constexpr std::uint64_t fnv_prime = 1'099'511'628'211ULL;
    const auto *const bytes =
        static_cast<const std::byte *>(mapped_) + static_cast<std::size_t>(offset);
    const std::size_t native_size = static_cast<std::size_t>(size);
    const std::size_t native_stride = static_cast<std::size_t>(sampling_stride);
    std::uint64_t hash = fnv_offset_basis;
    for (std::size_t sample_offset = 0U; sample_offset < native_size;
         sample_offset += native_stride) {
        hash ^= static_cast<std::uint8_t>(bytes[sample_offset]);
        hash *= fnv_prime;
    }
    const std::size_t tail_start = native_size > 256U ? native_size - 256U : 0U;
    for (std::size_t tail_offset = tail_start; tail_offset < native_size; ++tail_offset) {
        hash ^= static_cast<std::uint8_t>(bytes[tail_offset]);
        hash *= fnv_prime;
    }
    return Result<std::uint64_t>::success(hash);
}

Buffer::~Buffer() {
    if (buffer_ != VK_NULL_HANDLE && device_.resource_lifetime->safe_to_destroy()) {
        device_.table->vkDestroyBuffer(device_.device, buffer_, nullptr);
    }
    buffer_ = VK_NULL_HANDLE;
    mapped_ = nullptr;
}

BufferMemoryMetadata Buffer::memory_metadata() const noexcept {
    return {
        allocation_.memory_type_index(),     allocation_.memory_heap_index(),
        allocation_.memory_property_flags(), allocation_.non_coherent_atom_size(),
        allocation_.host_visible(),          allocation_.host_coherent(),
        allocation_.device_local(),
    };
}

Result<void> Buffer::write(std::span<const std::byte> source, VkDeviceSize offset) {
    if (mapped_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan buffer is not host mapped"});
    }
    if (!span_fits(size_, offset, source.size())) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "buffer write exceeds the allocation"});
    }
    if (!source.empty()) {
        auto *destination = static_cast<std::byte *>(mapped_) + static_cast<std::size_t>(offset);
        std::memcpy(destination, source.data(), source.size());
    }
    return flush_range(offset, static_cast<VkDeviceSize>(source.size()));
}

Result<void> Buffer::write_with_zero_padding(std::span<const std::byte> payload,
                                             VkDeviceSize logical_size, VkDeviceSize offset,
                                             VkDeviceSize maximum_padding_bytes,
                                             const BufferWritePhases &phases) {
    if (mapped_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan buffer is not host mapped"});
    }
    if (payload.size() > logical_size || offset > size_ || logical_size > size_ - offset) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "zero-padded buffer write exceeds the logical destination range"});
    }
    const VkDeviceSize payload_size = static_cast<VkDeviceSize>(payload.size());
    const VkDeviceSize padding_size = logical_size - payload_size;
    if (padding_size > maximum_padding_bytes) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "zero-padded buffer write exceeds its padding limit"});
    }
    if constexpr (sizeof(std::size_t) < sizeof(VkDeviceSize)) {
        if (offset > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max()) ||
            logical_size > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max())) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "mapped Vulkan write exceeds the host address space"});
        }
    }

    if (phases.before_mapped_copy) phases.before_mapped_copy();
    auto *destination = static_cast<std::byte *>(mapped_) + static_cast<std::size_t>(offset);
    if (!payload.empty()) {
        std::memcpy(destination, payload.data(), payload.size());
    }
    if (padding_size != 0U) {
        std::memset(destination + payload.size(), 0, static_cast<std::size_t>(padding_size));
    }
    if (phases.after_mapped_copy) phases.after_mapped_copy();

    if (phases.before_flush) phases.before_flush();
    const Result<void> flushed = flush_range(offset, logical_size);
    if (phases.after_flush) phases.after_flush();
    return flushed;
}

Result<void> Buffer::read(std::span<std::byte> destination, VkDeviceSize offset) {
    if (mapped_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan buffer is not host mapped"});
    }
    if (!span_fits(size_, offset, destination.size())) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "buffer read exceeds the allocation"});
    }
    const auto invalidation =
        invalidate_range(offset, static_cast<VkDeviceSize>(destination.size()));
    if (!invalidation) return invalidation;
    if (!destination.empty()) {
        const auto *source =
            static_cast<const std::byte *>(mapped_) + static_cast<std::size_t>(offset);
        std::memcpy(destination.data(), source, destination.size());
    }
    return Result<void>::success();
}

Result<std::span<const std::byte>> Buffer::readable_span(VkDeviceSize size, VkDeviceSize offset) {
    if (mapped_ == nullptr) {
        return Result<std::span<const std::byte>>::failure(
            {ErrorCode::invalid_state, "Vulkan buffer is not host mapped"});
    }
    if (!span_fits(size_, offset, static_cast<std::size_t>(size))) {
        return Result<std::span<const std::byte>>::failure(
            {ErrorCode::invalid_argument, "buffer read exceeds the allocation"});
    }
    const auto invalidation = invalidate_range(offset, size);
    if (!invalidation) {
        return Result<std::span<const std::byte>>::failure(invalidation.error());
    }
    const auto *source = static_cast<const std::byte *>(mapped_) + static_cast<std::size_t>(offset);
    return Result<std::span<const std::byte>>::success(
        std::span<const std::byte>(source, static_cast<std::size_t>(size)));
}

Result<void> Buffer::fill(std::byte value) {
    if (mapped_ == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan buffer is not host mapped"});
    }
    if constexpr (sizeof(std::size_t) < sizeof(VkDeviceSize)) {
        if (size_ > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max())) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "mapped Vulkan buffer exceeds the host address space"});
        }
    }
    std::memset(mapped_, std::to_integer<int>(value), static_cast<std::size_t>(size_));
    return flush_range(0U, size_);
}

Result<std::span<std::byte>> Buffer::writable_span(VkDeviceSize size, VkDeviceSize offset) {
    if (mapped_ == nullptr) {
        return Result<std::span<std::byte>>::failure(
            {ErrorCode::invalid_state, "Vulkan buffer is not host mapped"});
    }
    if constexpr (sizeof(std::size_t) < sizeof(VkDeviceSize)) {
        if (size > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max()) ||
            offset > static_cast<VkDeviceSize>(std::numeric_limits<std::size_t>::max())) {
            return Result<std::span<std::byte>>::failure(
                {ErrorCode::invalid_state, "mapped Vulkan span exceeds the host address space"});
        }
    }
    const std::size_t byte_count = static_cast<std::size_t>(size);
    if (!span_fits(size_, offset, byte_count)) {
        return Result<std::span<std::byte>>::failure(
            {ErrorCode::invalid_argument, "writable Vulkan span exceeds the allocation"});
    }
    auto *data = static_cast<std::byte *>(mapped_) + static_cast<std::size_t>(offset);
    return Result<std::span<std::byte>>::success(std::span<std::byte>(data, byte_count));
}

Result<void> Buffer::commit_external_write(VkDeviceSize payload_size, VkDeviceSize logical_size,
                                           VkDeviceSize offset,
                                           VkDeviceSize maximum_padding_bytes) {
    if (mapped_ == nullptr || payload_size > logical_size || offset > size_ ||
        logical_size > size_ - offset) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "external Vulkan write exceeds the destination range"});
    }
    const VkDeviceSize padding_size = logical_size - payload_size;
    if (padding_size > maximum_padding_bytes) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "external Vulkan write exceeds its padding limit"});
    }
    if (padding_size != 0U) {
        auto *destination =
            static_cast<std::byte *>(mapped_) + static_cast<std::size_t>(offset + payload_size);
        std::memset(destination, 0, static_cast<std::size_t>(padding_size));
    }
    return flush_range(offset, logical_size);
}

Result<void> Buffer::flush_range(VkDeviceSize offset, VkDeviceSize size) {
    return allocation_.flush(offset, size);
}

Result<void> Buffer::invalidate_range(VkDeviceSize offset, VkDeviceSize size) {
    return allocation_.invalidate(offset, size);
}

} // namespace crossvmaf::gpu
