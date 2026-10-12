#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "evaluation/input_lease_pool.hpp"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

namespace crossvmaf::evaluation {
namespace {
// The logical record owns native frames. The physical input lease borrows
// them only until its final GPU consumer retires, then drops the AVFrame.
void release_hardware_frame(video::Frame *&frame) noexcept {
    if (frame != nullptr) *frame = {};
    frame = nullptr;
}
} // namespace
} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {
namespace {

inline constexpr std::size_t segmented_input_lease_pool_reference_u_buffer_index = 0U;
inline constexpr std::size_t segmented_input_lease_pool_reference_v_buffer_index = 1U;
inline constexpr std::size_t segmented_input_lease_pool_distorted_y_buffer_index = 2U;
inline constexpr std::size_t segmented_input_lease_pool_distorted_u_buffer_index = 3U;
inline constexpr std::size_t segmented_input_lease_pool_distorted_v_buffer_index = 4U;

inline constexpr std::array<std::size_t, 5U> segmented_input_lease_pool_spatial_plane_indices{
    1U, 2U, 0U, 1U, 2U};

inline constexpr std::size_t segmented_input_lease_pool_maximum_segment_count =
    std::min<std::size_t>(std::numeric_limits<std::uint32_t>::max(),
                          std::numeric_limits<std::size_t>::max() / input_leases_per_segment);
inline constexpr std::size_t segmented_input_lease_pool_maximum_lease_count =
    segmented_input_lease_pool_maximum_segment_count * input_leases_per_segment;
inline constexpr std::size_t segmented_input_lease_pool_initial_free_location_capacity = 512U;

static_assert(input_leases_per_segment == 2U);

template <typename FreeLocation>
void segmented_input_lease_pool_reserve_free_location_capacity(
    std::vector<FreeLocation> &free_locations, std::size_t required_capacity) {
    if (free_locations.capacity() >= required_capacity) return;

    const std::size_t current_capacity = free_locations.capacity();
    const std::size_t doubled_capacity =
        current_capacity > segmented_input_lease_pool_maximum_lease_count / 2U
            ? segmented_input_lease_pool_maximum_lease_count
            : current_capacity * 2U;
    const std::size_t target_capacity = std::max({
        required_capacity,
        doubled_capacity,
        segmented_input_lease_pool_initial_free_location_capacity,
    });
    free_locations.reserve(target_capacity);
}

[[nodiscard]] constexpr std::size_t
segmented_input_lease_pool_segment_count_for_leases(std::size_t lease_count) noexcept {
    return lease_count / input_leases_per_segment +
           (lease_count % input_leases_per_segment == 0U ? 0U : 1U);
}

[[nodiscard]] Result<VkDeviceSize>
segmented_input_lease_pool_checked_add(VkDeviceSize left, VkDeviceSize right, std::string message) {
    if (right > std::numeric_limits<VkDeviceSize>::max() - left) {
        return Result<VkDeviceSize>::failure({ErrorCode::unsupported_format, std::move(message)});
    }
    return Result<VkDeviceSize>::success(left + right);
}

[[nodiscard]] Result<VkDeviceSize>
segmented_input_lease_pool_checked_multiply(VkDeviceSize value, VkDeviceSize multiplier,
                                            std::string message) {
    if (multiplier != 0U && value > std::numeric_limits<VkDeviceSize>::max() / multiplier) {
        return Result<VkDeviceSize>::failure({ErrorCode::unsupported_format, std::move(message)});
    }
    return Result<VkDeviceSize>::success(value * multiplier);
}

[[nodiscard]] Result<VkDeviceSize> segmented_input_lease_pool_align_up(VkDeviceSize value,
                                                                       VkDeviceSize alignment) {
    if (alignment == 0U) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::backend_error, "Vulkan reported a zero storage buffer offset alignment"});
    }
    const VkDeviceSize remainder = value % alignment;
    if (remainder == 0U) {
        return Result<VkDeviceSize>::success(value);
    }
    return segmented_input_lease_pool_checked_add(
        value, alignment - remainder,
        "segmented input lease slot alignment overflows VkDeviceSize");
}

[[nodiscard]] Result<VkDeviceSize>
segmented_input_lease_pool_plane_buffer_size(const FrameDescriptor &descriptor,
                                             std::size_t plane_index) {
    const std::uint64_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(plane_width(descriptor, plane_index)) * bytes_per_sample;
    const std::uint64_t height = plane_height(descriptor, plane_index);
    if (height != 0U && row_bytes > std::numeric_limits<std::uint64_t>::max() / height) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             "segmented input lease plane byte size overflows uint64"});
    }
    const std::uint64_t active_bytes = row_bytes * height;
    if (active_bytes > std::numeric_limits<std::uint64_t>::max() - 3U) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             "segmented input lease plane byte size overflows uint64"});
    }
    return Result<VkDeviceSize>::success(static_cast<VkDeviceSize>((active_bytes + 3U) & ~3ULL));
}

[[nodiscard]] std::size_t
segmented_input_lease_pool_active_plane_size(const FrameDescriptor &descriptor,
                                             std::size_t plane_index) noexcept {
    const std::size_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    return static_cast<std::size_t>(plane_width(descriptor, plane_index)) *
           plane_height(descriptor, plane_index) * bytes_per_sample;
}

[[nodiscard]] Result<std::unique_ptr<gpu::Buffer>>
segmented_input_lease_pool_create_plane_segment_buffer(gpu::DeviceView device,
                                                       VkDeviceSize slot_stride,
                                                       VkMemoryPropertyFlags required_memory) {
    auto segment_size = segmented_input_lease_pool_checked_multiply(
        slot_stride, input_leases_per_segment,
        "segmented input lease buffer size overflows VkDeviceSize");
    if (!segment_size) {
        return Result<std::unique_ptr<gpu::Buffer>>::failure(segment_size.error());
    }
    gpu::BufferCreateInfo create_info =
        gpu::host_visible_storage(segment_size.value(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    create_info.required_memory = required_memory;
    create_info.preferred_memory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return gpu::Buffer::create(device, create_info);
}

[[nodiscard]] Result<WritablePlaneView>
segmented_input_lease_pool_make_writable_plane_view(gpu::Buffer &buffer,
                                                    const FrameDescriptor &descriptor,
                                                    std::size_t plane_index, VkDeviceSize offset) {
    const std::size_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    const std::size_t stride_bytes =
        static_cast<std::size_t>(plane_width(descriptor, plane_index)) * bytes_per_sample;
    const std::size_t active_size = stride_bytes * plane_height(descriptor, plane_index);
    auto writable = buffer.writable_span(active_size, offset);
    if (!writable) {
        return Result<WritablePlaneView>::failure(writable.error());
    }
    return Result<WritablePlaneView>::success({writable.value().data(), active_size, stride_bytes});
}

void segmented_input_lease_pool_configure_input_memory(gpu::TelemetryCollector *telemetry_collector,
                                                       const gpu::Buffer &buffer) noexcept {
    if (telemetry_collector == nullptr || !telemetry_collector->enabled()) {
        return;
    }
    const gpu::BufferMemoryMetadata metadata = buffer.memory_metadata();
    telemetry_collector->configure_input_buffer_memory({
        metadata.memory_type_index,
        metadata.memory_heap_index,
        metadata.property_flags,
        metadata.non_coherent_atom_size,
        metadata.host_visible,
        metadata.host_coherent,
        metadata.device_local,
    });
}

[[nodiscard]] Error segmented_input_lease_pool_invariant_error() {
    return {ErrorCode::invalid_state,
            "segmented input lease pool entered sticky invariant failure"};
}

} // namespace

struct SegmentedInputLeasePool::SpatialLease final {
    std::array<WritablePlaneView, 5U> writable_planes{};
    std::uint32_t segment_index{};
    std::uint32_t slot_index{};
    std::uint64_t frame_index{};
    std::uint64_t generation{};
    std::size_t pending_consumer_count{};
    std::chrono::steady_clock::time_point reserved_at{};
    video::Frame *hardware_reference{};
    video::Frame *hardware_distorted{};
    LeaseState state{LeaseState::free};
};

struct SegmentedInputLeasePool::ReferenceLumaLease final {
    WritablePlaneView writable_plane{};
    std::uint32_t segment_index{};
    std::uint32_t slot_index{};
    std::uint64_t frame_index{};
    std::uint64_t generation{};
    std::size_t pending_consumer_count{};
    bool future_motion_consumer_pending{};
    std::chrono::steady_clock::time_point reserved_at{};
    LeaseState state{LeaseState::free};
};

struct SegmentedInputLeasePool::SpatialSegment final {
    std::uint32_t segment_index{};
    std::array<std::unique_ptr<gpu::Buffer>, 5U> buffers;
    std::array<SpatialLease, input_leases_per_segment> leases;
};

struct SegmentedInputLeasePool::ReferenceLumaSegment final {
    std::uint32_t segment_index{};
    std::unique_ptr<gpu::Buffer> buffer;
    std::array<ReferenceLumaLease, input_leases_per_segment> leases;
};

SegmentedInputLeasePool::SegmentedInputLeasePool(
    gpu::DeviceView device, FrameDescriptor descriptor,
    gpu::TelemetryCollector *telemetry_collector) noexcept
    : device_(device), descriptor_(descriptor), telemetry_collector_(telemetry_collector),
      profiling_enabled_(telemetry_collector != nullptr && telemetry_collector->enabled()) {
}

SegmentedInputLeasePool::~SegmentedInputLeasePool() = default;

Result<std::unique_ptr<SegmentedInputLeasePool>> SegmentedInputLeasePool::create(
    gpu::DeviceView device, FrameDescriptor descriptor, std::size_t initial_spatial_lease_count,
    std::size_t reference_luma_history_distance, bool vulkan_image_input,
    gpu::TelemetryCollector *telemetry_collector) {
    if (!device.complete() || initial_spatial_lease_count == 0U ||
        initial_spatial_lease_count > std::numeric_limits<std::uint32_t>::max() ||
        reference_luma_history_distance >
            std::numeric_limits<std::size_t>::max() - initial_spatial_lease_count) {
        return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(
            {ErrorCode::backend_error,
             "segmented input lease pool received an incomplete device or invalid depth"});
    }
    const std::size_t initial_luma_lease_count =
        initial_spatial_lease_count + reference_luma_history_distance;
    if (segmented_input_lease_pool_segment_count_for_leases(initial_spatial_lease_count) >
            segmented_input_lease_pool_maximum_segment_count ||
        segmented_input_lease_pool_segment_count_for_leases(initial_luma_lease_count) >
            segmented_input_lease_pool_maximum_segment_count) {
        return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(
            {ErrorCode::invalid_argument,
             "segmented input lease pool exceeds its handle segment space"});
    }

    const auto descriptor_validation = validate_frame_descriptor(descriptor);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(
            descriptor_validation.error());
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    const VkDeviceSize slot_alignment = std::max<VkDeviceSize>({
        4U,
        properties.limits.minStorageBufferOffsetAlignment,
        properties.limits.nonCoherentAtomSize,
    });

    auto pool = std::unique_ptr<SegmentedInputLeasePool>(
        new SegmentedInputLeasePool(device, descriptor, telemetry_collector));
    pool->vulkan_image_input_ = vulkan_image_input;

    for (std::size_t plane_index = 0U; plane_index < pool->plane_logical_sizes_.size();
         ++plane_index) {
        auto logical_size = segmented_input_lease_pool_plane_buffer_size(descriptor, plane_index);
        if (!logical_size) {
            return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(logical_size.error());
        }
        if (logical_size.value() > properties.limits.maxStorageBufferRange) {
            return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(
                {ErrorCode::unsupported_format,
                 "segmented input lease plane exceeds maxStorageBufferRange"});
        }
        auto slot_stride =
            segmented_input_lease_pool_align_up(logical_size.value(), slot_alignment);
        if (!slot_stride) {
            return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(slot_stride.error());
        }
        auto segment_size_validation = segmented_input_lease_pool_checked_multiply(
            slot_stride.value(), input_leases_per_segment,
            "segmented input lease buffer size overflows VkDeviceSize");
        if (!segment_size_validation) {
            return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(
                segment_size_validation.error());
        }
        pool->plane_logical_sizes_[plane_index] = logical_size.value();
        pool->plane_slot_strides_[plane_index] = slot_stride.value();
    }
    pool->reference_luma_history_distance_ = reference_luma_history_distance;

    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(device.physical_device, &memory_properties);
    const bool try_direct_mapping =
        !vulkan_image_input && gpu::supports_direct_mapped_input(memory_properties);
    if (try_direct_mapping) {
        // Write once into the final GPU input bank instead of forcing a
        // cached host bank plus a second device-local staging ring.
        pool->required_input_memory_ =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }
    auto capacity_created =
        pool->ensure_capacity(initial_spatial_lease_count, initial_luma_lease_count);
    if (!capacity_created && try_direct_mapping &&
        capacity_created.error().code == ErrorCode::backend_error) {
        // No lease has escaped yet. Retry the complete bank with the original
        // host-cached policy, never a mixture hidden behind a direct-GPU path.
        auto fallback_pool = std::unique_ptr<SegmentedInputLeasePool>(
            new SegmentedInputLeasePool(device, descriptor, telemetry_collector));
        fallback_pool->plane_logical_sizes_ = pool->plane_logical_sizes_;
        fallback_pool->plane_slot_strides_ = pool->plane_slot_strides_;
        fallback_pool->reference_luma_history_distance_ = reference_luma_history_distance;
        pool = std::move(fallback_pool);
        capacity_created =
            pool->ensure_capacity(initial_spatial_lease_count, initial_luma_lease_count);
    }
    if (!capacity_created) {
        return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(capacity_created.error());
    }
    const SpatialSegment *const first_spatial_segment = pool->find_spatial_segment(0U);
    const ReferenceLumaSegment *const first_luma_segment = pool->find_reference_luma_segment(0U);
    if (first_spatial_segment == nullptr || first_luma_segment == nullptr) {
        return Result<std::unique_ptr<SegmentedInputLeasePool>>::failure(
            {ErrorCode::invalid_state, "segmented input lease pool created no initial buffers"});
    }
    if (vulkan_image_input) {
        pool->input_memory_is_device_local_ = false;
    } else {
        pool->input_memory_is_device_local_ =
            first_spatial_segment->buffers[0U]->memory_metadata().device_local &&
            first_luma_segment->buffer->memory_metadata().device_local;
    }
    return Result<std::unique_ptr<SegmentedInputLeasePool>>::success(std::move(pool));
}

Result<void> SegmentedInputLeasePool::grow_spatial_segment() {
    if (invariant_failed()) {
        return Result<void>::failure(segmented_input_lease_pool_invariant_error());
    }
    if (spatial_segments_.size() >= segmented_input_lease_pool_maximum_segment_count) {
        mark_invariant_failure();
        return Result<void>::failure(
            {ErrorCode::invalid_state, "segmented spatial lease handle space is exhausted"});
    }
    const std::size_t next_segment_count = spatial_segments_.size() + 1U;
    segmented_input_lease_pool_reserve_free_location_capacity(
        spatial_free_locations_, next_segment_count * input_leases_per_segment);

    auto segment = std::make_unique<SpatialSegment>();
    segment->segment_index = static_cast<std::uint32_t>(spatial_segments_.size());
    VkDeviceSize segment_logical_bytes = 0U;

    if (!vulkan_image_input_) {
        for (std::size_t buffer_index = 0U; buffer_index < segment->buffers.size();
             ++buffer_index) {
            const std::size_t plane_index =
                segmented_input_lease_pool_spatial_plane_indices[buffer_index];
            auto buffer = segmented_input_lease_pool_create_plane_segment_buffer(
                device_, plane_slot_strides_[plane_index], required_input_memory_);
            if (!buffer) return Result<void>::failure(buffer.error());
            segment->buffers[buffer_index] = std::move(buffer).value();
            auto added_bytes = segmented_input_lease_pool_checked_add(
                segment_logical_bytes, segment->buffers[buffer_index]->size(),
                "segmented spatial lease logical bytes overflow VkDeviceSize");
            if (!added_bytes) {
                return Result<void>::failure(added_bytes.error());
            }
            segment_logical_bytes = added_bytes.value();
            segmented_input_lease_pool_configure_input_memory(telemetry_collector_,
                                                              *segment->buffers[buffer_index]);

            for (std::size_t slot_index = 0U; slot_index < input_leases_per_segment; ++slot_index) {
                const VkDeviceSize slot_offset = plane_slot_strides_[plane_index] * slot_index;
                auto writable = segmented_input_lease_pool_make_writable_plane_view(
                    *segment->buffers[buffer_index], descriptor_, plane_index, slot_offset);
                if (!writable) {
                    return Result<void>::failure(writable.error());
                }
                segment->leases[slot_index].writable_planes[buffer_index] = writable.value();
            }
        }
    }
    for (std::size_t slot_index = 0U; slot_index < input_leases_per_segment; ++slot_index) {
        SpatialLease &lease = segment->leases[slot_index];
        lease.segment_index = segment->segment_index;
        lease.slot_index = static_cast<std::uint32_t>(slot_index);
    }

    if (segment_logical_bytes > std::numeric_limits<VkDeviceSize>::max() - logical_buffer_bytes_) {
        return Result<void>::failure({ErrorCode::unsupported_format,
                                      "segmented input lease logical byte accounting overflowed"});
    }

    SpatialSegment *const published_segment = segment.get();
    spatial_segments_.push_back(std::move(segment));
    logical_buffer_bytes_ += segment_logical_bytes;
    for (std::size_t slot_index = 0U; slot_index < input_leases_per_segment; ++slot_index) {
        push_spatial_free(
            {published_segment->segment_index, static_cast<std::uint32_t>(slot_index)});
    }
    if (invariant_failed()) {
        return Result<void>::failure(segmented_input_lease_pool_invariant_error());
    }
    spatial_depth_snapshot_.store(spatial_segments_.size() * input_leases_per_segment,
                                  std::memory_order_relaxed);
    return Result<void>::success();
}

Result<void> SegmentedInputLeasePool::grow_reference_luma_segment() {
    if (invariant_failed()) {
        return Result<void>::failure(segmented_input_lease_pool_invariant_error());
    }
    if (reference_luma_segments_.size() >= segmented_input_lease_pool_maximum_segment_count) {
        mark_invariant_failure();
        return Result<void>::failure(
            {ErrorCode::invalid_state, "segmented luma lease handle space is exhausted"});
    }
    const std::size_t next_segment_count = reference_luma_segments_.size() + 1U;
    segmented_input_lease_pool_reserve_free_location_capacity(
        reference_luma_free_locations_, next_segment_count * input_leases_per_segment);

    auto segment = std::make_unique<ReferenceLumaSegment>();
    segment->segment_index = static_cast<std::uint32_t>(reference_luma_segments_.size());
    VkDeviceSize segment_logical_bytes = 0U;
    if (!vulkan_image_input_) {
        auto buffer = segmented_input_lease_pool_create_plane_segment_buffer(
            device_, plane_slot_strides_[0U], required_input_memory_);
        if (!buffer) return Result<void>::failure(buffer.error());
        segment->buffer = std::move(buffer).value();
        segment_logical_bytes = segment->buffer->size();
        segmented_input_lease_pool_configure_input_memory(telemetry_collector_, *segment->buffer);
    }

    for (std::size_t slot_index = 0U; slot_index < input_leases_per_segment; ++slot_index) {
        ReferenceLumaLease &lease = segment->leases[slot_index];
        lease.segment_index = segment->segment_index;
        lease.slot_index = static_cast<std::uint32_t>(slot_index);
        if (!vulkan_image_input_) {
            const VkDeviceSize slot_offset = plane_slot_strides_[0U] * slot_index;
            auto writable = segmented_input_lease_pool_make_writable_plane_view(
                *segment->buffer, descriptor_, 0U, slot_offset);
            if (!writable) return Result<void>::failure(writable.error());
            lease.writable_plane = writable.value();
        }
    }

    if (segment_logical_bytes > std::numeric_limits<VkDeviceSize>::max() - logical_buffer_bytes_) {
        return Result<void>::failure({ErrorCode::unsupported_format,
                                      "segmented input lease logical byte accounting overflowed"});
    }

    ReferenceLumaSegment *const published_segment = segment.get();
    reference_luma_segments_.push_back(std::move(segment));
    logical_buffer_bytes_ += segment_logical_bytes;
    for (std::size_t slot_index = 0U; slot_index < input_leases_per_segment; ++slot_index) {
        push_reference_luma_free(
            {published_segment->segment_index, static_cast<std::uint32_t>(slot_index)});
    }
    if (invariant_failed()) {
        return Result<void>::failure(segmented_input_lease_pool_invariant_error());
    }
    reference_luma_depth_snapshot_.store(reference_luma_segments_.size() * input_leases_per_segment,
                                         std::memory_order_relaxed);
    return Result<void>::success();
}

Result<void> SegmentedInputLeasePool::ensure_capacity(std::size_t spatial_lease_count,
                                                      std::size_t reference_luma_lease_count) {
    const std::size_t required_spatial_segments =
        segmented_input_lease_pool_segment_count_for_leases(spatial_lease_count);
    const std::size_t required_luma_segments =
        segmented_input_lease_pool_segment_count_for_leases(reference_luma_lease_count);
    if (required_spatial_segments > segmented_input_lease_pool_maximum_segment_count ||
        required_luma_segments > segmented_input_lease_pool_maximum_segment_count) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "segmented input lease capacity exceeds its handle space"});
    }
    while (spatial_depth() < spatial_lease_count) {
        VkDeviceSize estimated_bytes = 0U;
        for (std::size_t buffer_index = 0U;
             buffer_index < segmented_input_lease_pool_spatial_plane_indices.size();
             ++buffer_index) {
            const std::size_t plane_index =
                segmented_input_lease_pool_spatial_plane_indices[buffer_index];
            estimated_bytes += plane_slot_strides_[plane_index] * input_leases_per_segment;
        }
        if (!vulkan_image_input_ && device_.memory_pool != nullptr) {
            const auto budget = device_.memory_pool->ensure_allocation_budget(estimated_bytes);
            if (!budget) return budget;
        }
        const auto grown = grow_spatial_segment();
        if (!grown) return grown;
    }
    while (reference_luma_depth() < reference_luma_lease_count) {
        const VkDeviceSize estimated_bytes = plane_slot_strides_[0U] * input_leases_per_segment;
        if (!vulkan_image_input_ && device_.memory_pool != nullptr) {
            const auto budget = device_.memory_pool->ensure_allocation_budget(estimated_bytes);
            if (!budget) return budget;
        }
        const auto grown = grow_reference_luma_segment();
        if (!grown) return grown;
    }
    return Result<void>::success();
}

Result<void> SegmentedInputLeasePool::ensure_pair_capacity(std::size_t pair_target) {
    if (pair_target == 0U) {
        return Result<void>::success();
    }
    if (pair_target > std::numeric_limits<std::size_t>::max() - reference_luma_history_distance_) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "segmented input lease pair target overflows luma capacity"});
    }
    return ensure_capacity(pair_target, pair_target + reference_luma_history_distance_);
}

SegmentedInputLeasePool::SpatialSegment *
SegmentedInputLeasePool::find_spatial_segment(std::uint32_t segment_index) const noexcept {
    if (segment_index >= spatial_segments_.size()) {
        return nullptr;
    }
    return spatial_segments_[segment_index].get();
}

SegmentedInputLeasePool::ReferenceLumaSegment *
SegmentedInputLeasePool::find_reference_luma_segment(std::uint32_t segment_index) const noexcept {
    if (segment_index >= reference_luma_segments_.size()) {
        return nullptr;
    }
    return reference_luma_segments_[segment_index].get();
}

SegmentedInputLeasePool::SpatialLease *SegmentedInputLeasePool::find_spatial_lease(
    const evaluation::FrameStorageHandle &handle) const noexcept {
    if (!handle || handle.slot_index >= input_leases_per_segment) {
        return nullptr;
    }
    SpatialSegment *const segment = find_spatial_segment(handle.segment_index);
    return segment == nullptr ? nullptr : &segment->leases[handle.slot_index];
}

SegmentedInputLeasePool::ReferenceLumaLease *SegmentedInputLeasePool::find_reference_luma_lease(
    const evaluation::FrameStorageHandle &handle) const noexcept {
    if (!handle || handle.slot_index >= input_leases_per_segment) {
        return nullptr;
    }
    ReferenceLumaSegment *const segment = find_reference_luma_segment(handle.segment_index);
    return segment == nullptr ? nullptr : &segment->leases[handle.slot_index];
}

std::optional<SegmentedInputLeasePool::FreeLocation>
SegmentedInputLeasePool::try_pop_spatial_free() noexcept {
    if (spatial_free_locations_.empty()) return std::nullopt;

    const FreeLocation location = spatial_free_locations_.back();
    spatial_free_locations_.pop_back();
    if (find_spatial_segment(location.segment_index) == nullptr ||
        location.slot_index >= input_leases_per_segment) {
        mark_invariant_failure();
        return std::nullopt;
    }
    return location;
}

std::optional<SegmentedInputLeasePool::FreeLocation>
SegmentedInputLeasePool::try_pop_reference_luma_free() noexcept {
    if (reference_luma_free_locations_.empty()) return std::nullopt;

    const FreeLocation location = reference_luma_free_locations_.back();
    reference_luma_free_locations_.pop_back();
    if (find_reference_luma_segment(location.segment_index) == nullptr ||
        location.slot_index >= input_leases_per_segment) {
        mark_invariant_failure();
        return std::nullopt;
    }
    return location;
}

void SegmentedInputLeasePool::push_spatial_free(FreeLocation location) noexcept {
    if (find_spatial_segment(location.segment_index) == nullptr ||
        location.slot_index >= input_leases_per_segment ||
        spatial_free_locations_.size() >= spatial_free_locations_.capacity()) {
        mark_invariant_failure();
        return;
    }
    spatial_free_locations_.push_back(location);
}

void SegmentedInputLeasePool::push_reference_luma_free(FreeLocation location) noexcept {
    if (find_reference_luma_segment(location.segment_index) == nullptr ||
        location.slot_index >= input_leases_per_segment ||
        reference_luma_free_locations_.size() >= reference_luma_free_locations_.capacity()) {
        mark_invariant_failure();
        return;
    }
    reference_luma_free_locations_.push_back(location);
}

Result<std::optional<SegmentedWritableInputLease>>
SegmentedInputLeasePool::try_reserve(std::uint64_t frame_index) {
    return reserve_impl(frame_index);
}

Result<std::optional<SegmentedWritableInputLease>>
SegmentedInputLeasePool::reserve_impl(std::uint64_t frame_index) {
    std::optional<FreeLocation> spatial_location;
    std::optional<FreeLocation> luma_location;
    if (invariant_failed()) {
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            segmented_input_lease_pool_invariant_error());
    }
    if (frame_index != next_reservation_index_) {
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            {ErrorCode::invalid_argument, "segmented input lease reservations must be contiguous "
                                          "and start at zero: requested=" +
                                              std::to_string(frame_index) + ", expected=" +
                                              std::to_string(next_reservation_index_)});
    }
    if (frame_index == std::numeric_limits<std::uint64_t>::max()) {
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            {ErrorCode::invalid_argument, "segmented input lease frame index space is exhausted"});
    }
    if (producer_cancelled_ || end_of_stream_sealed_) {
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            {ErrorCode::invalid_state, producer_cancelled_
                                           ? "segmented input lease producer was cancelled"
                                           : "segmented input lease pool reached end of stream"});
    }

    spatial_location = try_pop_spatial_free();
    luma_location = try_pop_reference_luma_free();
    if (!spatial_location.has_value() || !luma_location.has_value()) {
        if (spatial_location.has_value()) {
            push_spatial_free(*spatial_location);
        }
        if (luma_location.has_value()) {
            push_reference_luma_free(*luma_location);
        }
        if (invariant_failed()) {
            return Result<std::optional<SegmentedWritableInputLease>>::failure(
                segmented_input_lease_pool_invariant_error());
        }
        return Result<std::optional<SegmentedWritableInputLease>>::success(std::nullopt);
    }

    SpatialSegment *const spatial_segment = find_spatial_segment(spatial_location->segment_index);
    ReferenceLumaSegment *const luma_segment =
        find_reference_luma_segment(luma_location->segment_index);
    if (spatial_segment == nullptr || luma_segment == nullptr) {
        mark_invariant_failure();
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            segmented_input_lease_pool_invariant_error());
    }

    SpatialLease &spatial_lease = spatial_segment->leases[spatial_location->slot_index];
    ReferenceLumaLease &luma_lease = luma_segment->leases[luma_location->slot_index];

    SegmentedWritableInputLease reservation{};
    if (producer_cancelled_ || end_of_stream_sealed_ || invariant_failed()) {
        push_spatial_free(*spatial_location);
        push_reference_luma_free(*luma_location);
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            invariant_failed()
                ? segmented_input_lease_pool_invariant_error()
                : Error{ErrorCode::invalid_state,
                        producer_cancelled_ ? "segmented input lease producer was cancelled"
                                            : "segmented input lease pool reached end of stream"});
    }
    if (spatial_lease.state != LeaseState::free || luma_lease.state != LeaseState::free ||
        spatial_lease.generation == std::numeric_limits<std::uint64_t>::max() ||
        luma_lease.generation == std::numeric_limits<std::uint64_t>::max() ||
        luma_history_.contains(frame_index)) {
        mark_invariant_failure();
        return Result<std::optional<SegmentedWritableInputLease>>::failure(
            segmented_input_lease_pool_invariant_error());
    }

    ++spatial_lease.generation;
    spatial_lease.frame_index = frame_index;
    spatial_lease.pending_consumer_count = 0U;
    spatial_lease.reserved_at = std::chrono::steady_clock::now();
    release_hardware_frame(spatial_lease.hardware_reference);
    release_hardware_frame(spatial_lease.hardware_distorted);
    spatial_lease.state = LeaseState::reserved;

    ++luma_lease.generation;
    luma_lease.frame_index = frame_index;
    luma_lease.pending_consumer_count = 0U;
    luma_lease.future_motion_consumer_pending = false;
    luma_lease.reserved_at = spatial_lease.reserved_at;
    luma_lease.state = LeaseState::reserved;

    reservation.token = {
        frame_index,
        {spatial_location->segment_index, spatial_location->slot_index, spatial_lease.generation},
        {luma_location->segment_index, luma_location->slot_index, luma_lease.generation},
    };
    luma_history_.emplace(frame_index, reservation.token.reference_luma);
    reservation.reference.index = frame_index;
    reservation.distorted.index = frame_index;
    reservation.reference.planes[0U] = luma_lease.writable_plane;
    reservation.reference.planes[1U] =
        spatial_lease.writable_planes[segmented_input_lease_pool_reference_u_buffer_index];
    reservation.reference.planes[2U] =
        spatial_lease.writable_planes[segmented_input_lease_pool_reference_v_buffer_index];
    reservation.distorted.planes[0U] =
        spatial_lease.writable_planes[segmented_input_lease_pool_distorted_y_buffer_index];
    reservation.distorted.planes[1U] =
        spatial_lease.writable_planes[segmented_input_lease_pool_distorted_u_buffer_index];
    reservation.distorted.planes[2U] =
        spatial_lease.writable_planes[segmented_input_lease_pool_distorted_v_buffer_index];

    ++statistics_.reservation_count;
    ++statistics_.active_spatial_lease_count;
    ++statistics_.active_reference_luma_lease_count;
    statistics_.maximum_active_spatial_lease_count = std::max(
        statistics_.maximum_active_spatial_lease_count, statistics_.active_spatial_lease_count);
    statistics_.maximum_active_reference_luma_lease_count =
        std::max(statistics_.maximum_active_reference_luma_lease_count,
                 statistics_.active_reference_luma_lease_count);
    ++next_reservation_index_;
    return Result<std::optional<SegmentedWritableInputLease>>::success(
        std::optional<SegmentedWritableInputLease>{std::move(reservation)});
}

Result<void> SegmentedInputLeasePool::validate_token(const SegmentedInputLeaseToken &token,
                                                     LeaseState expected_state) const {
    if (!token) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "segmented input lease token is invalid"});
    }
    SpatialLease *const spatial_lease = find_spatial_lease(token.spatial);
    ReferenceLumaLease *const luma_lease = find_reference_luma_lease(token.reference_luma);
    if (spatial_lease == nullptr || luma_lease == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "segmented input lease token references an invalid segment or slot"});
    }
    const bool token_matches = spatial_lease->frame_index == token.frame_index &&
                               spatial_lease->generation == token.spatial.generation &&
                               luma_lease->frame_index == token.frame_index &&
                               luma_lease->generation == token.reference_luma.generation;
    if (!token_matches || spatial_lease->state != expected_state ||
        luma_lease->state != expected_state) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "segmented input lease token is stale or in the wrong lifecycle state"});
    }
    return Result<void>::success();
}

Result<void> SegmentedInputLeasePool::publish(const SegmentedInputLeaseToken &token) {
    if (invariant_failed())
        return Result<void>::failure(segmented_input_lease_pool_invariant_error());
    const auto validation = validate_token(token, LeaseState::reserved);
    if (!validation) return validation;
    if (producer_cancelled_ || end_of_stream_sealed_) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "segmented input lease cannot publish after cancellation or end of stream"});
    }
    find_spatial_lease(token.spatial)->state = LeaseState::published;
    find_reference_luma_lease(token.reference_luma)->state = LeaseState::published;
    return Result<void>::success();
}

Result<void> SegmentedInputLeasePool::store_hardware_decoded_frame(
    const SegmentedInputLeaseToken &token, evaluation::FramePairSide side, video::Frame &frame) {
    if (!vulkan_image_input_) {
        return Result<void>::failure({ErrorCode::invalid_state,
                                      "software input lease cannot retain a Vulkan decoded frame"});
    }
    const auto validation = validate_token(token, LeaseState::reserved);
    if (!validation) return validation;
    if (!frame.has_vulkan_storage() ||
        frame.hardware_surface_format() == video::HardwareSurfaceFormat::none) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "hardware input lease requires an NV12 or P010 Vulkan frame"});
    }

    SpatialLease *const spatial_lease = find_spatial_lease(token.spatial);
    video::Frame *&destination = side == evaluation::FramePairSide::reference
                                     ? spatial_lease->hardware_reference
                                     : spatial_lease->hardware_distorted;
    if (destination != nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "hardware input lease received the same frame side twice"});
    }
    destination = &frame;
    return Result<void>::success();
}

void SegmentedInputLeasePool::record_spatial_release(SpatialLease &lease) noexcept {
    const auto hold_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - lease.reserved_at);
    statistics_.spatial_hold_total += hold_duration;
    statistics_.spatial_hold_maximum = std::max(statistics_.spatial_hold_maximum, hold_duration);
    ++statistics_.spatial_release_count;
    if (statistics_.active_spatial_lease_count != 0U) {
        --statistics_.active_spatial_lease_count;
    }
}

void SegmentedInputLeasePool::record_luma_release(ReferenceLumaLease &lease) noexcept {
    const auto hold_duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - lease.reserved_at);
    statistics_.reference_luma_hold_total += hold_duration;
    statistics_.reference_luma_hold_maximum =
        std::max(statistics_.reference_luma_hold_maximum, hold_duration);
    ++statistics_.reference_luma_release_count;
    if (statistics_.active_reference_luma_lease_count != 0U) {
        --statistics_.active_reference_luma_lease_count;
    }
}

void SegmentedInputLeasePool::abandon(const SegmentedInputLeaseToken &token) noexcept {
    if (!token) return;
    SpatialLease *const spatial_lease = find_spatial_lease(token.spatial);
    ReferenceLumaLease *const luma_lease = find_reference_luma_lease(token.reference_luma);
    if (spatial_lease == nullptr || luma_lease == nullptr) return;
    const bool token_matches = spatial_lease->frame_index == token.frame_index &&
                               spatial_lease->generation == token.spatial.generation &&
                               luma_lease->frame_index == token.frame_index &&
                               luma_lease->generation == token.reference_luma.generation;
    const bool spatial_can_be_abandoned = spatial_lease->state == LeaseState::reserved ||
                                          spatial_lease->state == LeaseState::published;
    const bool luma_can_be_abandoned =
        luma_lease->state == LeaseState::reserved || luma_lease->state == LeaseState::published;
    if (!token_matches || !spatial_can_be_abandoned || !luma_can_be_abandoned) {
        return;
    }
    // Reservation is single-producer and strictly ordered. If the newest
    // reservation could not be published into the adaptive pipeline, its
    // frame index must remain retryable instead of leaving a permanent hole
    // in the contiguous reservation sequence. Older reservations can still be
    // abandoned during cancellation, but cannot rewind past newer ones.
    if (next_reservation_index_ != 0U && token.frame_index == next_reservation_index_ - 1U) {
        --next_reservation_index_;
    }
    record_spatial_release(*spatial_lease);
    record_luma_release(*luma_lease);
    release_hardware_frame(spatial_lease->hardware_reference);
    release_hardware_frame(spatial_lease->hardware_distorted);
    spatial_lease->state = LeaseState::free;
    spatial_lease->frame_index = 0U;
    spatial_lease->pending_consumer_count = 0U;
    luma_lease->state = LeaseState::free;
    luma_lease->frame_index = 0U;
    luma_lease->pending_consumer_count = 0U;
    luma_lease->future_motion_consumer_pending = false;
    luma_history_.erase(token.frame_index);
    push_spatial_free({token.spatial.segment_index, token.spatial.slot_index});
    push_reference_luma_free({token.reference_luma.segment_index, token.reference_luma.slot_index});
}

void SegmentedInputLeasePool::cancel_producers() noexcept {
    producer_cancelled_ = true;
}

Result<void> SegmentedInputLeasePool::commit_plane(gpu::Buffer &buffer, VkDeviceSize offset,
                                                   VkDeviceSize logical_size,
                                                   std::size_t plane_index) {
    return buffer.commit_external_write(
        segmented_input_lease_pool_active_plane_size(descriptor_, plane_index), logical_size,
        offset, 3U);
}

video::GpuPlaneView
SegmentedInputLeasePool::make_gpu_plane_view(const gpu::Buffer &buffer, VkDeviceSize offset,
                                             VkDeviceSize logical_size,
                                             std::size_t plane_index) const noexcept {
    const std::uint32_t bytes_per_sample = descriptor_.bit_depth == 8U ? 1U : 2U;
    return {
        &buffer,
        plane_width(descriptor_, plane_index),
        plane_height(descriptor_, plane_index),
        plane_width(descriptor_, plane_index) * bytes_per_sample,
        logical_size,
        offset,
    };
}

video::UploadedFramePair
SegmentedInputLeasePool::make_gpu_frame_pair(const SegmentedInputLeaseToken &token) const noexcept {
    const SpatialSegment &spatial_segment = *find_spatial_segment(token.spatial.segment_index);
    const std::size_t spatial_slot = token.spatial.slot_index;
    video::UploadedFramePair pair{};
    pair.reference.index = token.frame_index;
    pair.distorted.index = token.frame_index;
    const SpatialLease &spatial_lease = spatial_segment.leases[spatial_slot];
    if (vulkan_image_input_) {
        pair.hardware_reference =
            spatial_lease.hardware_reference ? &*spatial_lease.hardware_reference : nullptr;
        pair.hardware_distorted =
            spatial_lease.hardware_distorted ? &*spatial_lease.hardware_distorted : nullptr;
        return pair;
    }

    const ReferenceLumaSegment &luma_segment =
        *find_reference_luma_segment(token.reference_luma.segment_index);
    const std::size_t luma_slot = token.reference_luma.slot_index;
    pair.reference.planes[0U] = make_gpu_plane_view(
        *luma_segment.buffer, plane_slot_strides_[0U] * luma_slot, plane_logical_sizes_[0U], 0U);

    const auto make_spatial_plane = [&](std::size_t buffer_index) {
        const std::size_t plane_index =
            segmented_input_lease_pool_spatial_plane_indices[buffer_index];
        return make_gpu_plane_view(*spatial_segment.buffers[buffer_index],
                                   plane_slot_strides_[plane_index] * spatial_slot,
                                   plane_logical_sizes_[plane_index], plane_index);
    };
    pair.reference.planes[1U] =
        make_spatial_plane(segmented_input_lease_pool_reference_u_buffer_index);
    pair.reference.planes[2U] =
        make_spatial_plane(segmented_input_lease_pool_reference_v_buffer_index);
    pair.distorted.planes[0U] =
        make_spatial_plane(segmented_input_lease_pool_distorted_y_buffer_index);
    pair.distorted.planes[1U] =
        make_spatial_plane(segmented_input_lease_pool_distorted_u_buffer_index);
    pair.distorted.planes[2U] =
        make_spatial_plane(segmented_input_lease_pool_distorted_v_buffer_index);
    return pair;
}

Result<video::UploadedFramePair>
SegmentedInputLeasePool::commit_published(const SegmentedInputLeaseToken &token) {
    const auto validation = validate_token(token, LeaseState::published);
    if (!validation) {
        return Result<video::UploadedFramePair>::failure(validation.error());
    }
    find_spatial_lease(token.spatial)->state = LeaseState::acquired;
    ReferenceLumaLease *const luma_lease = find_reference_luma_lease(token.reference_luma);
    luma_lease->state = LeaseState::acquired;
    luma_lease->future_motion_consumer_pending = !end_of_stream_sealed_;

    SpatialLease *const spatial_lease = find_spatial_lease(token.spatial);
    if (vulkan_image_input_) {
        if (!(spatial_lease->hardware_reference != nullptr) ||
            !(spatial_lease->hardware_distorted != nullptr)) {
            spatial_lease->state = LeaseState::failed;
            luma_lease->state = LeaseState::failed;
            return Result<video::UploadedFramePair>::failure(
                {ErrorCode::invalid_state,
                 "published hardware input lease is missing a decoded frame"});
        }
        return Result<video::UploadedFramePair>::success(make_gpu_frame_pair(token));
    }

    SpatialSegment *const spatial_segment = find_spatial_segment(token.spatial.segment_index);
    ReferenceLumaSegment *const luma_segment =
        find_reference_luma_segment(token.reference_luma.segment_index);
    const std::size_t spatial_slot = token.spatial.slot_index;
    const std::size_t luma_slot = token.reference_luma.slot_index;
    auto committed = commit_plane(*luma_segment->buffer, plane_slot_strides_[0U] * luma_slot,
                                  plane_logical_sizes_[0U], 0U);
    for (std::size_t buffer_index = 0U; committed && buffer_index < spatial_segment->buffers.size();
         ++buffer_index) {
        const std::size_t plane_index =
            segmented_input_lease_pool_spatial_plane_indices[buffer_index];
        committed = commit_plane(*spatial_segment->buffers[buffer_index],
                                 plane_slot_strides_[plane_index] * spatial_slot,
                                 plane_logical_sizes_[plane_index], plane_index);
    }
    if (!committed) {
        find_spatial_lease(token.spatial)->state = LeaseState::failed;
        find_reference_luma_lease(token.reference_luma)->state = LeaseState::failed;
        return Result<video::UploadedFramePair>::failure(committed.error());
    }

    video::UploadedFramePair pair = make_gpu_frame_pair(token);
    if (profiling_enabled_) {
        std::uint64_t uploaded_bytes = 0U;
        for (std::size_t plane_index = 0U; plane_index < 3U; ++plane_index) {
            uploaded_bytes += pair.reference.planes[plane_index].size_bytes;
            uploaded_bytes += pair.distorted.planes[plane_index].size_bytes;
        }
        telemetry_collector_->record_upload_bytes(uploaded_bytes);
    }
    return Result<video::UploadedFramePair>::success(pair);
}

Result<SegmentedInputLeaseConsumer>
SegmentedInputLeasePool::register_consumer(SegmentedInputLeaseBank bank,
                                           const evaluation::FrameStorageHandle &handle) {
    if (bank == SegmentedInputLeaseBank::spatial) {
        SpatialLease *const lease = find_spatial_lease(handle);
        if (lease == nullptr || lease->generation != handle.generation ||
            lease->state != LeaseState::acquired) {
            return Result<SegmentedInputLeaseConsumer>::failure(
                {ErrorCode::invalid_state,
                 "spatial input consumer references a stale segmented lease"});
        }
        ++lease->pending_consumer_count;
    } else if (bank == SegmentedInputLeaseBank::reference_luma) {
        ReferenceLumaLease *const lease = find_reference_luma_lease(handle);
        if (lease == nullptr || lease->generation != handle.generation ||
            lease->state != LeaseState::acquired) {
            return Result<SegmentedInputLeaseConsumer>::failure(
                {ErrorCode::invalid_state,
                 "luma input consumer references a stale segmented lease"});
        }
        ++lease->pending_consumer_count;
    } else {
        return Result<SegmentedInputLeaseConsumer>::failure(
            {ErrorCode::invalid_argument,
             "segmented input consumer requires a concrete lease bank"});
    }
    return Result<SegmentedInputLeaseConsumer>::success({bank, handle});
}

Result<SegmentedInputLeaseConsumer>
SegmentedInputLeasePool::register_spatial_consumer(const SegmentedInputLeaseToken &token) {
    return register_consumer(SegmentedInputLeaseBank::spatial, token.spatial);
}

Result<std::vector<SegmentedInputLeaseConsumer>>
SegmentedInputLeasePool::register_consumers(const SegmentedInputLeaseToken &token, bool spatial,
                                            bool current_luma,
                                            std::optional<std::uint64_t> previous_luma_frame) {
    std::vector<SegmentedInputLeaseConsumer> consumers;
    consumers.reserve((spatial ? 1U : 0U) + (current_luma ? 1U : 0U) +
                      (previous_luma_frame.has_value() ? 1U : 0U));
    if (spatial) {
        auto consumer = register_consumer(SegmentedInputLeaseBank::spatial, token.spatial);
        if (!consumer) {
            return Result<std::vector<SegmentedInputLeaseConsumer>>::failure(consumer.error());
        }
        consumers.push_back(consumer.value());
    }
    if (current_luma) {
        auto consumer =
            register_consumer(SegmentedInputLeaseBank::reference_luma, token.reference_luma);
        if (!consumer) {
            return Result<std::vector<SegmentedInputLeaseConsumer>>::failure(consumer.error());
        }
        consumers.push_back(consumer.value());
    }
    if (previous_luma_frame.has_value()) {
        auto consumer = register_previous_luma_consumer(*previous_luma_frame);
        if (!consumer) {
            return Result<std::vector<SegmentedInputLeaseConsumer>>::failure(consumer.error());
        }
        consumers.push_back(consumer.value());
    }
    return Result<std::vector<SegmentedInputLeaseConsumer>>::success(std::move(consumers));
}

Result<SegmentedInputLeaseConsumer>
SegmentedInputLeasePool::register_previous_luma_consumer(std::uint64_t frame_index) {
    const auto history_entry = luma_history_.find(frame_index);
    if (history_entry == luma_history_.end()) {
        return Result<SegmentedInputLeaseConsumer>::failure(
            {ErrorCode::invalid_state,
             "previous-luma consumer references an unavailable segmented lease"});
    }
    ReferenceLumaLease *const lease = find_reference_luma_lease(history_entry->second);
    if (lease == nullptr || lease->frame_index != frame_index ||
        lease->generation != history_entry->second.generation ||
        lease->state != LeaseState::submitted || !lease->future_motion_consumer_pending) {
        return Result<SegmentedInputLeaseConsumer>::failure(
            {ErrorCode::invalid_state,
             "previous-luma consumer references an unavailable segmented lease"});
    }
    ++lease->pending_consumer_count;
    lease->future_motion_consumer_pending = false;
    return Result<SegmentedInputLeaseConsumer>::success({
        SegmentedInputLeaseBank::reference_luma,
        history_entry->second,
    });
}

Result<void> SegmentedInputLeasePool::finish_submission(const SegmentedInputLeaseToken &token) {
    const auto validation = validate_token(token, LeaseState::acquired);
    if (!validation) return validation;
    SpatialLease *const spatial_lease = find_spatial_lease(token.spatial);
    ReferenceLumaLease *const luma_lease = find_reference_luma_lease(token.reference_luma);
    if (spatial_lease->pending_consumer_count == 0U || luma_lease->pending_consumer_count == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "submitted segmented input lease has no registered GPU consumers"});
    }
    spatial_lease->state = LeaseState::submitted;
    luma_lease->state = LeaseState::submitted;
    return Result<void>::success();
}

Result<void>
SegmentedInputLeasePool::finish_staging_submission(const SegmentedInputLeaseToken &token) {
    const auto validation = validate_token(token, LeaseState::acquired);
    if (!validation) return validation;
    SpatialLease *const spatial_lease = find_spatial_lease(token.spatial);
    ReferenceLumaLease *const luma_lease = find_reference_luma_lease(token.reference_luma);
    if (spatial_lease->pending_consumer_count != 1U || luma_lease->pending_consumer_count != 1U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "staged input lease must have exactly one copy consumer per bank"});
    }
    spatial_lease->state = LeaseState::submitted;
    luma_lease->state = LeaseState::submitted;
    luma_lease->future_motion_consumer_pending = false;
    return Result<void>::success();
}

std::optional<evaluation::FrameStorageHandle>
SegmentedInputLeasePool::release_spatial_if_unused(SpatialLease &lease) noexcept {
    if (lease.state != LeaseState::submitted || lease.pending_consumer_count != 0U) {
        return std::nullopt;
    }
    record_spatial_release(lease);
    release_hardware_frame(lease.hardware_reference);
    release_hardware_frame(lease.hardware_distorted);
    lease.state = LeaseState::free;
    lease.frame_index = 0U;
    return evaluation::FrameStorageHandle{lease.segment_index, lease.slot_index, lease.generation};
}

std::optional<evaluation::FrameStorageHandle>
SegmentedInputLeasePool::release_luma_if_unused(ReferenceLumaLease &lease) noexcept {
    if (lease.state != LeaseState::submitted || lease.pending_consumer_count != 0U ||
        lease.future_motion_consumer_pending) {
        return std::nullopt;
    }
    const std::uint64_t released_frame_index = lease.frame_index;
    record_luma_release(lease);
    lease.state = LeaseState::free;
    lease.frame_index = 0U;
    const auto history_entry = luma_history_.find(released_frame_index);
    if (history_entry != luma_history_.end() &&
        history_entry->second.segment_index == lease.segment_index &&
        history_entry->second.slot_index == lease.slot_index &&
        history_entry->second.generation == lease.generation) {
        luma_history_.erase(history_entry);
    } else {
        mark_invariant_failure();
    }
    return evaluation::FrameStorageHandle{lease.segment_index, lease.slot_index, lease.generation};
}

Result<void> SegmentedInputLeasePool::retire_consumer(const SegmentedInputLeaseConsumer &consumer) {
    std::optional<evaluation::FrameStorageHandle> released_handle;
    if (!consumer) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "segmented input retirement requires a valid consumer"});
    }
    if (consumer.bank == SegmentedInputLeaseBank::spatial) {
        SpatialLease *const lease = find_spatial_lease(consumer.handle);
        if (lease == nullptr || lease->generation != consumer.handle.generation ||
            lease->state != LeaseState::submitted || lease->pending_consumer_count == 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "segmented input retirement references a stale spatial consumer"});
        }
        --lease->pending_consumer_count;
        released_handle = release_spatial_if_unused(*lease);
    } else if (consumer.bank == SegmentedInputLeaseBank::reference_luma) {
        ReferenceLumaLease *const lease = find_reference_luma_lease(consumer.handle);
        if (lease == nullptr || lease->generation != consumer.handle.generation ||
            lease->state != LeaseState::submitted || lease->pending_consumer_count == 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "segmented input retirement references a stale luma consumer"});
        }
        --lease->pending_consumer_count;
        released_handle = release_luma_if_unused(*lease);
    } else {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "segmented input retirement requires a concrete lease bank"});
    }
    if (released_handle.has_value()) {
        const FreeLocation location{released_handle->segment_index, released_handle->slot_index};
        if (consumer.bank == SegmentedInputLeaseBank::spatial) {
            push_spatial_free(location);
        } else {
            push_reference_luma_free(location);
        }
    }
    if (invariant_failed())
        return Result<void>::failure(segmented_input_lease_pool_invariant_error());
    return Result<void>::success();
}

Result<video::GpuPlaneView>
SegmentedInputLeasePool::reference_luma(std::uint64_t frame_index) const {
    const auto history_entry = luma_history_.find(frame_index);
    if (history_entry == luma_history_.end()) {
        return Result<video::GpuPlaneView>::failure(
            {ErrorCode::invalid_argument,
             "requested reference luma is outside segmented input history"});
    }
    const ReferenceLumaLease *const lease = find_reference_luma_lease(history_entry->second);
    const bool readable_state = lease != nullptr && (lease->state == LeaseState::acquired ||
                                                     lease->state == LeaseState::submitted);
    if (!readable_state || lease->frame_index != frame_index ||
        lease->generation != history_entry->second.generation) {
        return Result<video::GpuPlaneView>::failure(
            {ErrorCode::invalid_argument,
             "requested reference luma is outside segmented input history"});
    }
    const ReferenceLumaSegment *const segment = find_reference_luma_segment(lease->segment_index);
    const VkDeviceSize offset = plane_slot_strides_[0U] * lease->slot_index;
    return Result<video::GpuPlaneView>::success(
        make_gpu_plane_view(*segment->buffer, offset, plane_logical_sizes_[0U], 0U));
}

void SegmentedInputLeasePool::seal_end_of_stream() noexcept {
    end_of_stream_sealed_ = true;
    for (const auto &segment : reference_luma_segments_) {
        for (ReferenceLumaLease &lease : segment->leases) {
            lease.future_motion_consumer_pending = false;
            auto released = release_luma_if_unused(lease);
            if (released.has_value()) {
                push_reference_luma_free({released->segment_index, released->slot_index});
            }
        }
    }
}

std::size_t SegmentedInputLeasePool::spatial_depth() const noexcept {
    return spatial_depth_snapshot_.load(std::memory_order_relaxed);
}

std::size_t SegmentedInputLeasePool::reference_luma_depth() const noexcept {
    return reference_luma_depth_snapshot_.load(std::memory_order_relaxed);
}

VkDeviceSize SegmentedInputLeasePool::logical_buffer_bytes() const noexcept {
    return logical_buffer_bytes_;
}

SegmentedInputLeasePoolStatistics SegmentedInputLeasePool::statistics() const noexcept {
    return statistics_;
}

bool SegmentedInputLeasePool::input_memory_is_device_local() const noexcept {
    return input_memory_is_device_local_;
}

void SegmentedInputLeasePool::mark_invariant_failure() noexcept {
    invariant_failure_ = true;
}

bool SegmentedInputLeasePool::invariant_failed() const noexcept {
    return invariant_failure_;
}

} // namespace crossvmaf::evaluation
