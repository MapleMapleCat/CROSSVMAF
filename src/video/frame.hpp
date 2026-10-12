#pragma once
#include "crossvmaf/frame.hpp"
#include "crossvmaf/decode.hpp"
#include <vulkan/vulkan.h>
#include <chrono>
#include <memory>
#include <array>
#include <cstdint>

namespace crossvmaf::gpu {
class Buffer;
}

namespace crossvmaf::video {
class Frame;
}

namespace crossvmaf::video {

struct VideoTimestamp final {
    std::int64_t value{};
    std::int32_t time_base_numerator{};
    std::int32_t time_base_denominator{};
    bool valid{};
};

using VideoDecodeBackend = StreamDecodeBackend;

enum class HardwareSurfaceFormat : std::uint8_t {
    none,
    nv12,
    p010,
};

struct VulkanDecodedFrameAccess final {
    VkImage *images{};
    VkImageLayout *layouts{};
    VkAccessFlagBits2 *access_flags{};
    VkSemaphore *semaphores{};
    std::uint64_t *semaphore_values{};
    std::uint32_t *queue_families{};
    const VkFormat *image_formats{};
    std::size_t image_count{};
    HardwareSurfaceFormat surface_format{HardwareSurfaceFormat::none};
    std::uint32_t width{};
    std::uint32_t height{};
};

class LockedVulkanDecodedFrame final {
  public:
    LockedVulkanDecodedFrame() noexcept = default;
    ~LockedVulkanDecodedFrame();
    LockedVulkanDecodedFrame(LockedVulkanDecodedFrame &&other) noexcept;
    LockedVulkanDecodedFrame &operator=(LockedVulkanDecodedFrame &&other) noexcept;

    LockedVulkanDecodedFrame(const LockedVulkanDecodedFrame &) = delete;
    LockedVulkanDecodedFrame &operator=(const LockedVulkanDecodedFrame &) = delete;

    [[nodiscard]] VulkanDecodedFrameAccess &access() noexcept { return access_; }

    [[nodiscard]] const VulkanDecodedFrameAccess &access() const noexcept { return access_; }

  private:
    friend class Frame;

    using UnlockCallback = void (*)(void *frames_context, void *vulkan_frame) noexcept;

    LockedVulkanDecodedFrame(void *frames_context, void *vulkan_frame,
                             UnlockCallback unlock_callback,
                             VulkanDecodedFrameAccess access) noexcept;

    void unlock() noexcept;

    void *frames_context_{};
    void *vulkan_frame_{};
    UnlockCallback unlock_callback_{};
    VulkanDecodedFrameAccess access_{};
};

class Frame final {
  public:
    Frame() noexcept;
    ~Frame();
    Frame(Frame &&) noexcept;
    Frame &operator=(Frame &&) noexcept;

    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;

    [[nodiscard]] const FrameDescriptor &descriptor() const noexcept;
    [[nodiscard]] VideoTimestamp timestamp() const noexcept;
    [[nodiscard]] std::chrono::steady_clock::time_point decode_started_at() const noexcept;
    [[nodiscard]] std::chrono::steady_clock::time_point decode_completed_at() const noexcept;
    [[nodiscard]] VideoDecodeBackend decode_backend() const noexcept;
    [[nodiscard]] HardwareSurfaceFormat hardware_surface_format() const noexcept;
    [[nodiscard]] bool has_vulkan_storage() const noexcept;
    [[nodiscard]] Result<LockedVulkanDecodedFrame> lock_vulkan_frame() const;
    [[nodiscard]] Result<void> copy_to(const WritableFrameView &destination) const;

  private:
    friend class Decoder;

    class Impl;
    explicit Frame(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

struct GpuPlaneView {
    const gpu::Buffer *buffer{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t stride_bytes{};
    VkDeviceSize size_bytes{};
    VkDeviceSize offset_bytes{};
};

struct GpuFrameView {
    std::array<GpuPlaneView, 3U> planes{};
    std::uint64_t index{};
};

struct UploadedFramePair {
    GpuFrameView reference;
    GpuFrameView distorted;
    const video::Frame *hardware_reference{};
    const video::Frame *hardware_distorted{};

    [[nodiscard]] bool has_hardware_frames() const noexcept {
        return hardware_reference != nullptr && hardware_distorted != nullptr;
    }
};

[[nodiscard]] constexpr bool is_compatible_gpu_plane(const GpuPlaneView &plane,
                                                     std::uint32_t expected_width,
                                                     std::uint32_t expected_height,
                                                     std::uint8_t bit_depth) noexcept {
    const std::uint64_t bytes_per_sample = bit_depth == 8U ? 1U : 2U;
    const std::uint64_t minimum_stride =
        static_cast<std::uint64_t>(expected_width) * bytes_per_sample;
    const std::uint64_t required_size =
        static_cast<std::uint64_t>(plane.stride_bytes) * plane.height;
    return plane.buffer != nullptr && plane.width == expected_width &&
           plane.height == expected_height && plane.stride_bytes >= minimum_stride &&
           required_size <= plane.size_bytes;
}

} // namespace crossvmaf::video

namespace crossvmaf::video {
// Pixel-storage validation only. Metric-specific limits belong to metrics.
[[nodiscard]] Result<void> validate_frame_layout(const FrameDescriptor &descriptor) noexcept;
[[nodiscard]] Result<void> validate_plane_views(const FrameDescriptor &descriptor,
                                                const FrameView &frame) noexcept;
} // namespace crossvmaf::video
