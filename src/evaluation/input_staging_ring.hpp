#pragma once
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/buffer.hpp"
#include "gpu/command.hpp"
#include "gpu/device.hpp"
#include "gpu/timestamp.hpp"
#include "gpu/submission.hpp"
#include "video/frame.hpp"
#include "video/vulkan_convert.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace crossvmaf::evaluation {

inline constexpr std::size_t input_staging_wait_semaphore_count = 3U;

struct RecordedInputStagingCopy final {
    video::UploadedFramePair uploaded;
    VkCommandBuffer command_buffer{VK_NULL_HANDLE};
};

// Owns device-local input slots independently of HOST_CACHED input leases.
// Spatial data follows FrameSlot lifetime. Reference luma uses a larger ring
// so reference-luma history remains valid after the source lease is released.
class InputStagingRing final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<InputStagingRing>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, std::size_t frame_slot_count,
           std::size_t reference_luma_history_distance, bool vulkan_image_input = false,
           gpu::TelemetryCollector *telemetry_collector = nullptr,
           gpu::GpuOperationHandle telemetry_operation = {});

    InputStagingRing(const InputStagingRing &) = delete;
    InputStagingRing &operator=(const InputStagingRing &) = delete;
    ~InputStagingRing();

    [[nodiscard]] Result<RecordedInputStagingCopy>
    record_copy(std::size_t frame_slot_index, const video::UploadedFramePair &source);
    [[nodiscard]] Result<RecordedInputStagingCopy>
    record_conversion(std::size_t frame_slot_index, std::uint64_t frame_index,
                      const video::VulkanDecodedFrameAccess &reference,
                      const video::VulkanDecodedFrameAccess &distorted);
    [[nodiscard]] Result<void> mark_submitted(std::size_t frame_slot_index,
                                              gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<void> mark_retired(std::size_t frame_slot_index,
                                            gpu::SubmissionTicket ticket,
                                            std::chrono::nanoseconds submission_host_elapsed);
    void cancel(std::size_t frame_slot_index) noexcept;

    [[nodiscard]] std::span<const VkSemaphore>
    signal_semaphores(std::size_t frame_slot_index) const noexcept;
    [[nodiscard]] VkSemaphore wait_semaphore(std::size_t frame_slot_index,
                                             std::size_t feature_lane_index) const noexcept;
    [[nodiscard]] Result<video::GpuPlaneView> reference_luma(std::uint64_t frame_index) const;
    [[nodiscard]] VkDeviceSize logical_buffer_bytes() const noexcept;

  private:
    struct FrameSlotResources final {
        std::unique_ptr<gpu::Buffer> spatial_buffer;
        std::unique_ptr<gpu::CommandContext> commands;
        std::unique_ptr<gpu::GpuTimestampPair> gpu_timestamp;
        gpu::GpuTimestampRecordingState pending_timestamp_recording;
        std::array<VkSemaphore, input_staging_wait_semaphore_count> feature_ready_semaphores{};
    };

    explicit InputStagingRing(gpu::DeviceView device, FrameDescriptor descriptor,
                              std::size_t luma_ring_size) noexcept;

    [[nodiscard]] video::GpuPlaneView make_plane_view(const gpu::Buffer &buffer,
                                                      VkDeviceSize offset,
                                                      std::size_t plane_index) const noexcept;

    gpu::DeviceView device_;
    FrameDescriptor descriptor_;
    gpu::TelemetryCollector *telemetry_collector_{};
    gpu::GpuOperationHandle telemetry_operation_{};
    std::size_t luma_ring_size_{};
    bool vulkan_image_input_{};
    std::array<VkDeviceSize, 3U> plane_sizes_{};
    std::array<std::uint32_t, 3U> plane_strides_{};
    std::array<VkDeviceSize, 5U> spatial_offsets_{};
    VkDeviceSize spatial_slot_bytes_{};
    VkDeviceSize luma_slot_stride_{};
    VkDeviceSize logical_buffer_bytes_{};
    std::vector<FrameSlotResources> frame_slots_;
    std::unique_ptr<gpu::Buffer> luma_buffer_;
    std::unique_ptr<video::SemiplanarInputConverter> semiplanar_converter_;
};

} // namespace crossvmaf::evaluation
#endif
