#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "evaluation/input_staging_ring.hpp"
#include "shaders/semiplanar_nv12_spv.hpp"
#include "shaders/semiplanar_p010_spv.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>

namespace crossvmaf::evaluation {
namespace {

inline constexpr std::array<std::size_t, 5U> input_staging_ring_spatial_plane_indices{1U, 2U, 0U,
                                                                                      1U, 2U};

[[nodiscard]] Result<VkDeviceSize>
input_staging_ring_checked_add(VkDeviceSize left, VkDeviceSize right, const char *message) {
    if (right > std::numeric_limits<VkDeviceSize>::max() - left) {
        return Result<VkDeviceSize>::failure({ErrorCode::unsupported_format, message});
    }
    return Result<VkDeviceSize>::success(left + right);
}

[[nodiscard]] Result<VkDeviceSize> input_staging_ring_checked_multiply(VkDeviceSize value,
                                                                       VkDeviceSize multiplier,
                                                                       const char *message) {
    if (multiplier != 0U && value > std::numeric_limits<VkDeviceSize>::max() / multiplier) {
        return Result<VkDeviceSize>::failure({ErrorCode::unsupported_format, message});
    }
    return Result<VkDeviceSize>::success(value * multiplier);
}

[[nodiscard]] Result<VkDeviceSize> input_staging_ring_align_up(VkDeviceSize value,
                                                               VkDeviceSize alignment) {
    if (alignment == 0U) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::backend_error, "Vulkan input staging alignment is zero"});
    }
    const VkDeviceSize remainder = value % alignment;
    if (remainder == 0U) return Result<VkDeviceSize>::success(value);
    return input_staging_ring_checked_add(value, alignment - remainder,
                                          "Vulkan input staging layout overflows VkDeviceSize");
}

struct input_staging_ring_PlaneLayout final {
    std::uint32_t stride_bytes{};
    VkDeviceSize size_bytes{};
};

[[nodiscard]] Result<input_staging_ring_PlaneLayout>
input_staging_ring_plane_layout(const FrameDescriptor &descriptor, std::size_t plane_index,
                                bool align_rows_for_gpu_writes) {
    const VkDeviceSize bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    auto row_bytes =
        input_staging_ring_checked_multiply(plane_width(descriptor, plane_index), bytes_per_sample,
                                            "Vulkan input staging row size overflows VkDeviceSize");
    if (!row_bytes) {
        return Result<input_staging_ring_PlaneLayout>::failure(row_bytes.error());
    }
    VkDeviceSize stride_bytes = row_bytes.value();
    if (align_rows_for_gpu_writes) {
        auto aligned_stride = input_staging_ring_align_up(stride_bytes, 4U);
        if (!aligned_stride) {
            return Result<input_staging_ring_PlaneLayout>::failure(aligned_stride.error());
        }
        stride_bytes = aligned_stride.value();
    }
    auto active_bytes = input_staging_ring_checked_multiply(
        stride_bytes, plane_height(descriptor, plane_index),
        "Vulkan input staging plane size overflows VkDeviceSize");
    if (!active_bytes) {
        return Result<input_staging_ring_PlaneLayout>::failure(active_bytes.error());
    }
    auto aligned_size = input_staging_ring_align_up(active_bytes.value(), 4U);
    if (!aligned_size) {
        return Result<input_staging_ring_PlaneLayout>::failure(aligned_size.error());
    }
    if (stride_bytes > std::numeric_limits<std::uint32_t>::max()) {
        return Result<input_staging_ring_PlaneLayout>::failure(
            {ErrorCode::unsupported_format, "Vulkan input staging stride exceeds uint32"});
    }
    return Result<input_staging_ring_PlaneLayout>::success(
        {static_cast<std::uint32_t>(stride_bytes), aligned_size.value()});
}

[[nodiscard]] Result<std::unique_ptr<gpu::Buffer>>
input_staging_ring_create_device_input_buffer(gpu::DeviceView device, VkDeviceSize size) {
    gpu::BufferCreateInfo create_info{};
    create_info.size = size;
    create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    create_info.required_memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    create_info.preferred_memory = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    return gpu::Buffer::create(device, create_info);
}

} // namespace

InputStagingRing::InputStagingRing(gpu::DeviceView device, FrameDescriptor descriptor,
                                   std::size_t luma_ring_size) noexcept
    : device_(device), descriptor_(descriptor), luma_ring_size_(luma_ring_size) {
}

InputStagingRing::~InputStagingRing() {
    if (!device_.resource_lifetime->safe_to_destroy()) return;
    for (FrameSlotResources &frame_slot : frame_slots_) {
        for (VkSemaphore &semaphore : frame_slot.feature_ready_semaphores) {
            if (semaphore == VK_NULL_HANDLE) continue;
            device_.table->vkDestroySemaphore(device_.device, semaphore, nullptr);
            semaphore = VK_NULL_HANDLE;
        }
    }
}

Result<std::unique_ptr<InputStagingRing>>
InputStagingRing::create(gpu::DeviceView device, FrameDescriptor descriptor,
                         std::size_t frame_slot_count, std::size_t reference_luma_history_distance,
                         bool vulkan_image_input, gpu::TelemetryCollector *telemetry_collector,
                         gpu::GpuOperationHandle telemetry_operation) {
    if (!device.complete() || frame_slot_count == 0U ||
        reference_luma_history_distance >
            std::numeric_limits<std::size_t>::max() - frame_slot_count) {
        return Result<std::unique_ptr<InputStagingRing>>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan input staging received an invalid device or ring depth"});
    }
    const auto descriptor_validation = validate_frame_descriptor(descriptor);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<InputStagingRing>>::failure(descriptor_validation.error());
    }

    const std::size_t luma_ring_size = frame_slot_count + reference_luma_history_distance;
    auto staging_ring =
        std::unique_ptr<InputStagingRing>(new InputStagingRing(device, descriptor, luma_ring_size));
    staging_ring->telemetry_collector_ = telemetry_collector;
    staging_ring->telemetry_operation_ = telemetry_operation;
    staging_ring->vulkan_image_input_ = vulkan_image_input;

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    const VkDeviceSize storage_alignment =
        std::max<VkDeviceSize>(4U, properties.limits.minStorageBufferOffsetAlignment);
    for (std::size_t plane_index = 0U; plane_index < staging_ring->plane_sizes_.size();
         ++plane_index) {
        auto layout =
            input_staging_ring_plane_layout(descriptor, plane_index, vulkan_image_input);
        if (!layout) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(layout.error());
        }
        if (layout.value().size_bytes > properties.limits.maxStorageBufferRange) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(
                {ErrorCode::unsupported_format,
                 "Vulkan staged input plane exceeds maxStorageBufferRange"});
        }
        staging_ring->plane_sizes_[plane_index] = layout.value().size_bytes;
        staging_ring->plane_strides_[plane_index] = layout.value().stride_bytes;
    }

    VkDeviceSize spatial_bytes = 0U;
    for (std::size_t buffer_index = 0U;
         buffer_index < input_staging_ring_spatial_plane_indices.size(); ++buffer_index) {
        auto aligned_offset = input_staging_ring_align_up(spatial_bytes, storage_alignment);
        if (!aligned_offset) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(aligned_offset.error());
        }
        staging_ring->spatial_offsets_[buffer_index] = aligned_offset.value();
        auto next_offset = input_staging_ring_checked_add(
            aligned_offset.value(),
            staging_ring->plane_sizes_[input_staging_ring_spatial_plane_indices[buffer_index]],
            "Vulkan spatial input staging layout overflows VkDeviceSize");
        if (!next_offset) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(next_offset.error());
        }
        spatial_bytes = next_offset.value();
    }
    auto spatial_slot_bytes = input_staging_ring_align_up(spatial_bytes, storage_alignment);
    if (!spatial_slot_bytes) {
        return Result<std::unique_ptr<InputStagingRing>>::failure(spatial_slot_bytes.error());
    }
    staging_ring->spatial_slot_bytes_ = spatial_slot_bytes.value();
    auto luma_slot_stride =
        input_staging_ring_align_up(staging_ring->plane_sizes_[0U], storage_alignment);
    if (!luma_slot_stride) {
        return Result<std::unique_ptr<InputStagingRing>>::failure(luma_slot_stride.error());
    }
    staging_ring->luma_slot_stride_ = luma_slot_stride.value();

    auto luma_buffer_size = input_staging_ring_checked_multiply(
        staging_ring->luma_slot_stride_, luma_ring_size,
        "Vulkan luma input staging ring overflows VkDeviceSize");
    if (!luma_buffer_size) {
        return Result<std::unique_ptr<InputStagingRing>>::failure(luma_buffer_size.error());
    }
    auto luma_buffer =
        input_staging_ring_create_device_input_buffer(device, luma_buffer_size.value());
    if (!luma_buffer) {
        return Result<std::unique_ptr<InputStagingRing>>::failure(luma_buffer.error());
    }
    staging_ring->luma_buffer_ = std::move(luma_buffer).value();
    staging_ring->logical_buffer_bytes_ = luma_buffer_size.value();

    staging_ring->frame_slots_.resize(frame_slot_count);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (FrameSlotResources &frame_slot : staging_ring->frame_slots_) {
        auto spatial_buffer = input_staging_ring_create_device_input_buffer(
            device, staging_ring->spatial_slot_bytes_);
        if (!spatial_buffer) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(spatial_buffer.error());
        }
        frame_slot.spatial_buffer = std::move(spatial_buffer).value();
        auto commands = gpu::CommandContext::create(device);
        if (!commands) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(commands.error());
        }
        frame_slot.commands = std::move(commands).value();
        auto gpu_timestamp = gpu::GpuTimestampPair::create(
            device, telemetry_collector, telemetry_operation,
            vulkan_image_input ? VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                   : VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (!gpu_timestamp) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(gpu_timestamp.error());
        }
        frame_slot.gpu_timestamp = std::move(gpu_timestamp).value();
        for (VkSemaphore &semaphore : frame_slot.feature_ready_semaphores) {
            const VkResult result = device.table->vkCreateSemaphore(device.device, &semaphore_info,
                                                                    nullptr, &semaphore);
            if (result != VK_SUCCESS) {
                return Result<std::unique_ptr<InputStagingRing>>::failure(
                    {result == VK_ERROR_DEVICE_LOST ? ErrorCode::device_lost
                                                    : ErrorCode::backend_error,
                     "vkCreateSemaphore failed for Vulkan input staging"});
            }
        }
        auto total_bytes = input_staging_ring_checked_add(
            staging_ring->logical_buffer_bytes_, staging_ring->spatial_slot_bytes_,
            "Vulkan input staging byte accounting overflowed");
        if (!total_bytes) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(total_bytes.error());
        }
        staging_ring->logical_buffer_bytes_ = total_bytes.value();
    }
    if (vulkan_image_input) {
        const std::span<const std::uint8_t> conversion_shader =
            descriptor.bit_depth == 8U
                ? std::span<const std::uint8_t>(gpu::generated::semiplanar_nv12_spv)
                : std::span<const std::uint8_t>(gpu::generated::semiplanar_p010_spv);
        auto converter = video::SemiplanarInputConverter::create(
            device, descriptor, frame_slot_count, conversion_shader);
        if (!converter) {
            return Result<std::unique_ptr<InputStagingRing>>::failure(converter.error());
        }
        staging_ring->semiplanar_converter_ = std::move(converter).value();
    }
    return Result<std::unique_ptr<InputStagingRing>>::success(std::move(staging_ring));
}

video::GpuPlaneView InputStagingRing::make_plane_view(const gpu::Buffer &buffer,
                                                      VkDeviceSize offset,
                                                      std::size_t plane_index) const noexcept {
    return {
        &buffer,
        plane_width(descriptor_, plane_index),
        plane_height(descriptor_, plane_index),
        plane_strides_[plane_index],
        plane_sizes_[plane_index],
        offset,
    };
}

Result<RecordedInputStagingCopy>
InputStagingRing::record_copy(std::size_t frame_slot_index,
                              const video::UploadedFramePair &source) {
    if (frame_slot_index >= frame_slots_.size() ||
        source.reference.index != source.distorted.index) {
        return Result<RecordedInputStagingCopy>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan input staging received an invalid slot or frame pair"});
    }
    FrameSlotResources &frame_slot = frame_slots_[frame_slot_index];
    auto command_buffer = frame_slot.commands->begin();
    if (!command_buffer) {
        return Result<RecordedInputStagingCopy>::failure(command_buffer.error());
    }
    gpu::CommandRecordingGuard recording_guard(*frame_slot.commands);
    auto timestamp_recording = frame_slot.gpu_timestamp->begin_recording(command_buffer.value());
    frame_slot.gpu_timestamp->write_start(command_buffer.value(), timestamp_recording);

    const std::uint64_t frame_index = source.reference.index;
    const std::size_t luma_slot = static_cast<std::size_t>(frame_index % luma_ring_size_);
    const VkDeviceSize luma_offset = luma_slot_stride_ * luma_slot;
    frame_slot.commands->copy_buffer(*source.reference.planes[0U].buffer, *luma_buffer_,
                                     source.reference.planes[0U].offset_bytes, luma_offset,
                                     plane_sizes_[0U]);

    const std::array<const video::GpuPlaneView *, 5U> spatial_sources{
        &source.reference.planes[1U], &source.reference.planes[2U], &source.distorted.planes[0U],
        &source.distorted.planes[1U], &source.distorted.planes[2U],
    };
    for (std::size_t buffer_index = 0U; buffer_index < spatial_sources.size(); ++buffer_index) {
        const video::GpuPlaneView &source_plane = *spatial_sources[buffer_index];
        frame_slot.commands->copy_buffer(
            *source_plane.buffer, *frame_slot.spatial_buffer, source_plane.offset_bytes,
            spatial_offsets_[buffer_index],
            plane_sizes_[input_staging_ring_spatial_plane_indices[buffer_index]]);
    }
    frame_slot.gpu_timestamp->write_end(command_buffer.value(), timestamp_recording);

    auto finished = frame_slot.commands->finish();
    if (!finished) {
        return Result<RecordedInputStagingCopy>::failure(finished.error());
    }
    recording_guard.release();
    frame_slot.pending_timestamp_recording = timestamp_recording;

    video::UploadedFramePair uploaded{};
    uploaded.reference.index = frame_index;
    uploaded.distorted.index = frame_index;
    uploaded.reference.planes[0U] = make_plane_view(*luma_buffer_, luma_offset, 0U);
    uploaded.reference.planes[1U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[0U], 1U);
    uploaded.reference.planes[2U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[1U], 2U);
    uploaded.distorted.planes[0U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[2U], 0U);
    uploaded.distorted.planes[1U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[3U], 1U);
    uploaded.distorted.planes[2U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[4U], 2U);
    return Result<RecordedInputStagingCopy>::success({uploaded, finished.value()});
}

Result<RecordedInputStagingCopy>
InputStagingRing::record_conversion(std::size_t frame_slot_index, std::uint64_t frame_index,
                                    const video::VulkanDecodedFrameAccess &reference,
                                    const video::VulkanDecodedFrameAccess &distorted) {
    if (frame_slot_index >= frame_slots_.size() || semiplanar_converter_ == nullptr ||
        !vulkan_image_input_) {
        return Result<RecordedInputStagingCopy>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan input conversion received an invalid slot or mode"});
    }
    FrameSlotResources &frame_slot = frame_slots_[frame_slot_index];
    auto command_buffer = frame_slot.commands->begin();
    if (!command_buffer) {
        return Result<RecordedInputStagingCopy>::failure(command_buffer.error());
    }
    gpu::CommandRecordingGuard recording_guard(*frame_slot.commands);
    auto timestamp_recording = frame_slot.gpu_timestamp->begin_recording(command_buffer.value());
    frame_slot.gpu_timestamp->write_start(command_buffer.value(), timestamp_recording);

    const std::size_t luma_slot = static_cast<std::size_t>(frame_index % luma_ring_size_);
    const VkDeviceSize luma_offset = luma_slot_stride_ * luma_slot;
    video::UploadedFramePair uploaded{};
    uploaded.reference.index = frame_index;
    uploaded.distorted.index = frame_index;
    uploaded.reference.planes[0U] = make_plane_view(*luma_buffer_, luma_offset, 0U);
    uploaded.reference.planes[1U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[0U], 1U);
    uploaded.reference.planes[2U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[1U], 2U);
    uploaded.distorted.planes[0U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[2U], 0U);
    uploaded.distorted.planes[1U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[3U], 1U);
    uploaded.distorted.planes[2U] =
        make_plane_view(*frame_slot.spatial_buffer, spatial_offsets_[4U], 2U);

    const auto conversion_recorded = semiplanar_converter_->record(
        frame_slot_index, command_buffer.value(), reference, distorted, uploaded);
    if (!conversion_recorded) {
        return Result<RecordedInputStagingCopy>::failure(conversion_recorded.error());
    }
    frame_slot.gpu_timestamp->write_end(command_buffer.value(), timestamp_recording);
    auto finished = frame_slot.commands->finish();
    if (!finished) {
        return Result<RecordedInputStagingCopy>::failure(finished.error());
    }
    recording_guard.release();
    frame_slot.pending_timestamp_recording = timestamp_recording;
    return Result<RecordedInputStagingCopy>::success({uploaded, finished.value()});
}

Result<void> InputStagingRing::mark_submitted(std::size_t frame_slot_index,
                                              gpu::SubmissionTicket ticket) {
    if (frame_slot_index >= frame_slots_.size()) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan input staging submission references an invalid slot"});
    }
    return frame_slots_[frame_slot_index].commands->mark_submitted(ticket);
}

Result<void> InputStagingRing::mark_retired(std::size_t frame_slot_index,
                                            gpu::SubmissionTicket ticket,
                                            std::chrono::nanoseconds submission_host_elapsed) {
    if (frame_slot_index >= frame_slots_.size()) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan input staging retirement references an invalid slot"});
    }
    FrameSlotResources &frame_slot = frame_slots_[frame_slot_index];
    const auto command_retired = frame_slot.commands->mark_retired(ticket);
    if (!command_retired) return command_retired;
    if (telemetry_collector_ != nullptr) {
        telemetry_collector_->record_gpu_submission_latency(telemetry_operation_,
                                                            submission_host_elapsed);
    }
    const auto timestamp_collected = frame_slot.gpu_timestamp->collect(
        frame_slot.pending_timestamp_recording, submission_host_elapsed);
    if (semiplanar_converter_ != nullptr) {
        semiplanar_converter_->retire_slot(frame_slot_index);
    }
    return timestamp_collected;
}

void InputStagingRing::cancel(std::size_t frame_slot_index) noexcept {
    if (frame_slot_index < frame_slots_.size()) {
        frame_slots_[frame_slot_index].commands->cancel();
        if (semiplanar_converter_ != nullptr) {
            semiplanar_converter_->retire_slot(frame_slot_index);
        }
    }
}

std::span<const VkSemaphore>
InputStagingRing::signal_semaphores(std::size_t frame_slot_index) const noexcept {
    if (frame_slot_index >= frame_slots_.size()) return {};
    return frame_slots_[frame_slot_index].feature_ready_semaphores;
}

VkSemaphore InputStagingRing::wait_semaphore(std::size_t frame_slot_index,
                                             std::size_t feature_lane_index) const noexcept {
    if (frame_slot_index >= frame_slots_.size() ||
        feature_lane_index >= input_staging_wait_semaphore_count) {
        return VK_NULL_HANDLE;
    }
    return frame_slots_[frame_slot_index].feature_ready_semaphores[feature_lane_index];
}

Result<video::GpuPlaneView> InputStagingRing::reference_luma(std::uint64_t frame_index) const {
    if (luma_buffer_ == nullptr || luma_ring_size_ == 0U) {
        return Result<video::GpuPlaneView>::failure(
            {ErrorCode::invalid_state, "Vulkan input staging luma ring is unavailable"});
    }
    const std::size_t luma_slot = static_cast<std::size_t>(frame_index % luma_ring_size_);
    return Result<video::GpuPlaneView>::success(
        make_plane_view(*luma_buffer_, luma_slot_stride_ * luma_slot, 0U));
}

VkDeviceSize InputStagingRing::logical_buffer_bytes() const noexcept {
    return logical_buffer_bytes_;
}

} // namespace crossvmaf::evaluation
