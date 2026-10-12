#pragma once
#include "crossvmaf/error.hpp"
#include "video/decode_device.hpp"
#include "gpu/device.hpp"
#include "video/backends/vulkan_dispatch.hpp"
#include <vulkan/vulkan.h>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace crossvmaf::video {

using VulkanDecodeQueueFamily = gpu::NativeQueueFamily;

struct VulkanQueueSynchronization final {
    void *opaque{};
    void (*lock)(void *opaque, std::uint32_t queue_family, std::uint32_t queue_index) noexcept {};
    void (*unlock)(void *opaque, std::uint32_t queue_family, std::uint32_t queue_index) noexcept {};
};

using VulkanDecodeDeviceConfig = gpu::NativeDeviceCapabilities;

// Excludes a reproduced driver-specific pixel failure from automatic mode.
// Explicit hardware-only decoding still reaches the conformance tests.
[[nodiscard]] bool allow_automatic_hardware_decode(const DeviceInfo &device) noexcept;
[[nodiscard]] bool requires_serial_decode_commands(const DeviceInfo &device) noexcept;

// Owns a reference to FFmpeg's AV_HWDEVICE_TYPE_VULKAN wrapper. FFmpeg does not
// destroy the supplied device; its final-release state retains the runtime's
// actual device owner. Native FFmpeg
// declarations remain confined to the implementation file. Ordinary queues
// share the same host mutexes across FFmpeg workers and runtime submissions;
// internally synchronized queues need no application-side queue locks.
class VulkanDecodeDevice final : public DecodeDevice {
  public:
    [[nodiscard]] static Result<std::shared_ptr<VulkanDecodeDevice>>
    create(VulkanDecodeDeviceConfig config, bool serialize_decode_commands = false);

    VulkanDecodeDevice(const VulkanDecodeDevice &) = delete;
    VulkanDecodeDevice &operator=(const VulkanDecodeDevice &) = delete;
    ~VulkanDecodeDevice() override;

    // Returns a retained AVBufferRef as an opaque pointer. The caller must
    // transfer that reference to an AVCodecContext or release it with FFmpeg.
    [[nodiscard]] void *retain_native_device_context() const noexcept override;

    [[nodiscard]] StreamDecodeBackend backend() const noexcept override { return StreamDecodeBackend::vulkan_video; }
    // H.264 can keep each decoder on one queue while sharing the parent's
    // physical queue locks and command-recording guard. Other codecs retain
    // the ordinary device context.
    [[nodiscard]] Result<DecoderContext> retain_decoder_context(bool prefer_dedicated_queue) override;

    // Frame pools used only by compute need no automatically exported memory.
    // The child still retains this device and shares its exact queue locks.
    [[nodiscard]] Result<std::shared_ptr<VulkanDecodeDevice>> create_local_frame_context();

    // Non-owning callbacks, valid while this device is alive. Empty when
    // VK_KHR_internally_synchronized_queues supplies host synchronization.
    [[nodiscard]] VulkanQueueSynchronization queue_synchronization() const noexcept;

    // Decoded frames can be consumed without extending their timeline
    // semaphores, so the decoder does not need run-ahead to hide consumers.
    [[nodiscard]] bool unified_video_image_layouts() const noexcept override;

    [[nodiscard]] bool serializes_decode_commands() const noexcept override;

    // H.264 without a dedicated queue keeps the measured whole-call policy.
    [[nodiscard]] std::unique_lock<std::mutex> lock_codec_calls() const override;

  private:
    class QueueSynchronization;
    class NativeState;
    [[nodiscard]] static Result<std::shared_ptr<VulkanDecodeDevice>>
    create_context(VulkanDecodeDeviceConfig config, bool serialize_decode_commands,
                   VulkanDecodeDevice* queue_owner,
                   std::optional<VulkanDecodeQueueSelection> queue);
    VulkanDecodeDevice(void *native_device_context, NativeState *state) noexcept;

    void *native_device_context_{};
    NativeState *state_{};
};

[[nodiscard]] std::shared_ptr<VulkanDecodeDevice> prepare_decode_device(gpu::Device &device);

} // namespace crossvmaf::video
