#pragma once
#include "gpu/submission.hpp"
#include "video/decoder.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/device.hpp"
#include "video/frame.hpp"
#include <volk.h>
#include <memory>
#include <vector>

namespace crossvmaf::video {

// Build and publish while both AVVkFrame locks are held. The caller must keep
// the owning AVFrames alive until the conversion fence retires, including on
// failure. Publish only after successful queue submission. GENERAL-only
// readers never extend the decoder's dependency chain.
class HardwareFrameSynchronization final {
  public:
    [[nodiscard]] static Result<HardwareFrameSynchronization>
    create(const video::VulkanDecodedFrameAccess &reference,
           const video::VulkanDecodedFrameAccess &distorted, bool unified_video_image_layouts,
           std::span<const VkSemaphore> binary_signals = {});

    // Returned spans remain valid only while this object stays alive and unmoved.
    [[nodiscard]] gpu::SubmissionSynchronization submission() const noexcept;
    void mark_submitted(video::VulkanDecodedFrameAccess &frame,
                        std::uint32_t compute_queue_family) const noexcept;

  private:
    [[nodiscard]] Result<void> append(const video::VulkanDecodedFrameAccess &frame);

    bool advance_decoder_timeline_{};
    std::array<VkSemaphore, 4U> wait_semaphores_{};
    std::array<VkPipelineStageFlags, 4U> wait_stages_{};
    std::array<std::uint64_t, 4U> wait_values_{};
    std::array<VkSemaphore, 7U> signal_semaphores_{};
    std::array<std::uint64_t, 7U> signal_values_{};
    std::size_t wait_count_{};
    std::size_t signal_count_{};
};

} // namespace crossvmaf::video

namespace crossvmaf::video {

// Converts FFmpeg Vulkan Video NV12/P010 images into the tightly defined
// planar little-endian buffers consumed by the feature pipelines. Each slot
// owns one descriptor set and four short-lived plane views.
class SemiplanarInputConverter final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<SemiplanarInputConverter>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, std::size_t frame_slot_count,
           std::span<const std::uint8_t> shader_spirv);

    SemiplanarInputConverter(const SemiplanarInputConverter &) = delete;
    SemiplanarInputConverter &operator=(const SemiplanarInputConverter &) = delete;
    ~SemiplanarInputConverter();

    [[nodiscard]] Result<void> record(std::size_t frame_slot_index, VkCommandBuffer command_buffer,
                                      const video::VulkanDecodedFrameAccess &reference,
                                      const video::VulkanDecodedFrameAccess &distorted,
                                      const UploadedFramePair &destination);
    void retire_slot(std::size_t frame_slot_index) noexcept;

  private:
    struct FrameSlotResources final {
        VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
        std::array<VkImageView, 4U> source_plane_views{};
    };

    explicit SemiplanarInputConverter(gpu::DeviceView device, FrameDescriptor descriptor) noexcept;

    [[nodiscard]] Result<void>
    create_source_views(FrameSlotResources &resources,
                        const video::VulkanDecodedFrameAccess &reference,
                        const video::VulkanDecodedFrameAccess &distorted);
    [[nodiscard]] Result<void> update_descriptor_set(FrameSlotResources &resources,
                                                     const UploadedFramePair &destination);
    void record_source_barriers(VkCommandBuffer command_buffer,
                                const video::VulkanDecodedFrameAccess &reference,
                                const video::VulkanDecodedFrameAccess &distorted) const noexcept;
    void destroy_source_views(FrameSlotResources &resources) noexcept;

    gpu::DeviceView device_;
    FrameDescriptor descriptor_;
    VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline pipeline_{VK_NULL_HANDLE};
    std::vector<FrameSlotResources> frame_slots_;
};

} // namespace crossvmaf::video
