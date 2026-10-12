#include "frame_uploader.hpp"

#include "gpu/timestamp.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>

namespace crossvmaf::conformance {
namespace {

[[nodiscard]] Result<VkDeviceSize> plane_buffer_size(
    const FrameDescriptor& descriptor, std::size_t plane) {
    const std::uint64_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(plane_width(descriptor, plane)) *
        bytes_per_sample;
    const std::uint64_t height = plane_height(descriptor, plane);
    if (row_bytes > std::numeric_limits<std::uint64_t>::max() / height) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format, "GPU plane byte size overflows uint64"});
    }
    const std::uint64_t active_bytes =
        row_bytes * height;
    if (active_bytes > std::numeric_limits<std::uint64_t>::max() - 3U) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format, "GPU plane byte size overflows uint64"});
    }
    return Result<VkDeviceSize>::success(
        static_cast<VkDeviceSize>((active_bytes + 3U) & ~3ULL));
}

[[nodiscard]] Result<std::unique_ptr<gpu::Buffer>> create_plane_buffer(
    gpu::DeviceView device, VkDeviceSize size) {
    return gpu::Buffer::create(device, gpu::host_visible_storage(size));
}

void configure_input_memory(
    gpu::TelemetryCollector* telemetry_collector, const gpu::Buffer& buffer) noexcept {
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

}  // namespace

FrameUploader::~FrameUploader() = default;

Result<std::unique_ptr<FrameUploader>> FrameUploader::create(
    gpu::DeviceView device, FrameDescriptor descriptor,
    std::size_t frame_slot_count,
    std::size_t reference_luma_history_distance,
    gpu::TelemetryCollector* telemetry_collector) {
    if (!device.complete() || frame_slot_count == 0U ||
        reference_luma_history_distance >
            std::numeric_limits<std::size_t>::max() - frame_slot_count) {
        return Result<std::unique_ptr<FrameUploader>>::failure(
            {ErrorCode::backend_error,
             "frame uploader received an incomplete device or invalid slot bank"});
    }
    const auto validation = validate_frame_descriptor(descriptor);
    if (!validation) {
        return Result<std::unique_ptr<FrameUploader>>::failure(validation.error());
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    std::array<VkDeviceSize, 3U> plane_sizes{};
    VkDeviceSize maximum_plane_size = 0;
    for (std::size_t plane = 0; plane < plane_sizes.size(); ++plane) {
        const auto size = plane_buffer_size(descriptor, plane);
        if (!size) {
            return Result<std::unique_ptr<FrameUploader>>::failure(size.error());
        }
        plane_sizes[plane] = size.value();
        maximum_plane_size = std::max(maximum_plane_size, size.value());
        if (size.value() > properties.limits.maxStorageBufferRange) {
            return Result<std::unique_ptr<FrameUploader>>::failure(
                {ErrorCode::unsupported_format,
                 "frame plane exceeds maxStorageBufferRange on the selected device"});
        }
    }
    if (maximum_plane_size > static_cast<VkDeviceSize>(
                                 std::numeric_limits<std::size_t>::max())) {
        return Result<std::unique_ptr<FrameUploader>>::failure(
            {ErrorCode::unsupported_format,
             "frame plane exceeds the host address space"});
    }

    const std::size_t luma_ring_size =
        frame_slot_count + reference_luma_history_distance;
    auto uploader = std::unique_ptr<FrameUploader>(new FrameUploader(
        device, descriptor, frame_slot_count, luma_ring_size,
        telemetry_collector));
    uploader->scratch_.resize(static_cast<std::size_t>(maximum_plane_size),
                              std::byte{0});
    uploader->reference_luma_ring_.resize(luma_ring_size);
    for (auto& luma : uploader->reference_luma_ring_) {
        auto created = create_plane_buffer(device, plane_sizes[0]);
        if (!created) {
            return Result<std::unique_ptr<FrameUploader>>::failure(created.error());
        }
        luma = std::move(created).value();
        configure_input_memory(telemetry_collector, *luma);
    }
    uploader->reference_chroma_slots_.resize(frame_slot_count);
    uploader->distorted_slots_.resize(frame_slot_count);
    for (std::size_t frame_slot = 0U; frame_slot < frame_slot_count;
         ++frame_slot) {
        for (std::size_t plane = 1; plane < 3U; ++plane) {
            auto created = create_plane_buffer(device, plane_sizes[plane]);
            if (!created) {
                return Result<std::unique_ptr<FrameUploader>>::failure(
                    created.error());
            }
            uploader->reference_chroma_slots_[frame_slot][plane - 1U] =
                std::move(created).value();
            configure_input_memory(
                telemetry_collector,
                *uploader->reference_chroma_slots_[frame_slot][plane - 1U]);
        }
        for (std::size_t plane = 0; plane < 3U; ++plane) {
            auto created = create_plane_buffer(device, plane_sizes[plane]);
            if (!created) {
                return Result<std::unique_ptr<FrameUploader>>::failure(
                    created.error());
            }
            uploader->distorted_slots_[frame_slot][plane] =
                std::move(created).value();
            configure_input_memory(
                telemetry_collector,
                *uploader->distorted_slots_[frame_slot][plane]);
        }
    }
    return Result<std::unique_ptr<FrameUploader>>::success(std::move(uploader));
}

Result<void> FrameUploader::upload_plane(gpu::Buffer& destination,
                                         const PlaneView& source,
                                         std::size_t plane) {
    const std::size_t bytes_per_sample = descriptor_.bit_depth == 8U ? 1U : 2U;
    const std::size_t row_bytes =
        static_cast<std::size_t>(plane_width(descriptor_, plane)) *
        bytes_per_sample;
    const std::uint32_t height = plane_height(descriptor_, plane);
    const std::size_t active_bytes = row_bytes * height;
    std::span<const std::byte> contiguous_payload;
    if (source.stride_bytes == row_bytes) {
        contiguous_payload = {source.data, active_bytes};
    } else {
        for (std::uint32_t row = 0; row < height; ++row) {
            const std::byte* const source_row = source.data +
                static_cast<std::size_t>(row) * source.stride_bytes;
            std::byte* const destination_row = scratch_.data() +
                static_cast<std::size_t>(row) * row_bytes;
            std::memcpy(destination_row, source_row, row_bytes);
        }
        contiguous_payload = {scratch_.data(), active_bytes};
    }

    return destination.write_with_zero_padding(
        contiguous_payload, destination.size(), 0U, 3U);
}

crossvmaf::video::GpuPlaneView FrameUploader::make_plane_view(const gpu::Buffer& buffer,
                                            std::size_t plane) const noexcept {
    const std::uint32_t bytes_per_sample = descriptor_.bit_depth == 8U ? 1U : 2U;
    return {
        &buffer,
        plane_width(descriptor_, plane),
        plane_height(descriptor_, plane),
        plane_width(descriptor_, plane) * bytes_per_sample,
        buffer.size(),
    };
}

Result<crossvmaf::video::UploadedFramePair> FrameUploader::upload(
    const FrameView& reference, const FrameView& distorted) {
    const auto reference_validation = validate_frame_view(descriptor_, reference);
    if (!reference_validation) {
        return Result<crossvmaf::video::UploadedFramePair>::failure(reference_validation.error());
    }
    const auto distorted_validation = validate_frame_view(descriptor_, distorted);
    if (!distorted_validation) {
        return Result<crossvmaf::video::UploadedFramePair>::failure(distorted_validation.error());
    }
    if (reference.index != distorted.index || reference.index != next_index_) {
        return Result<crossvmaf::video::UploadedFramePair>::failure(
            {ErrorCode::invalid_argument,
             "uploaded frame indices must match, be contiguous, and start at zero"});
    }

    const std::size_t luma_slot =
        static_cast<std::size_t>(next_index_ % luma_ring_size_);
    const std::size_t frame_slot =
        static_cast<std::size_t>(next_index_ % frame_slot_count_);
    auto transferred = upload_plane(
        *reference_luma_ring_[luma_slot], reference.planes[0], 0);
    if (!transferred) {
        return Result<crossvmaf::video::UploadedFramePair>::failure(transferred.error());
    }
    for (std::size_t plane = 1U; plane < 3U; ++plane) {
        transferred = upload_plane(
            *reference_chroma_slots_[frame_slot][plane - 1U],
            reference.planes[plane], plane);
        if (!transferred) {
            return Result<crossvmaf::video::UploadedFramePair>::failure(transferred.error());
        }
    }
    for (std::size_t plane = 0U; plane < 3U; ++plane) {
        transferred = upload_plane(
            *distorted_slots_[frame_slot][plane],
            distorted.planes[plane], plane);
        if (!transferred) {
            return Result<crossvmaf::video::UploadedFramePair>::failure(transferred.error());
        }
    }

    crossvmaf::video::UploadedFramePair pair{};
    pair.reference.index = next_index_;
    pair.distorted.index = next_index_;
    pair.reference.planes[0] =
        make_plane_view(*reference_luma_ring_[luma_slot], 0);
    for (std::size_t plane = 1; plane < 3U; ++plane) {
        pair.reference.planes[plane] =
            make_plane_view(
                *reference_chroma_slots_[frame_slot][plane - 1U], plane);
    }
    for (std::size_t plane = 0; plane < 3U; ++plane) {
        pair.distorted.planes[plane] =
            make_plane_view(*distorted_slots_[frame_slot][plane], plane);
    }
    if (telemetry_collector_ != nullptr && telemetry_collector_->enabled()) {
        std::uint64_t uploaded_bytes = 0U;
        for (std::size_t plane = 0U; plane < 3U; ++plane) {
            uploaded_bytes += pair.reference.planes[plane].size_bytes;
            uploaded_bytes += pair.distorted.planes[plane].size_bytes;
        }
        telemetry_collector_->record_upload_bytes(uploaded_bytes);
    }
    ++next_index_;
    return Result<crossvmaf::video::UploadedFramePair>::success(pair);
}

}  // namespace crossvmaf::conformance
