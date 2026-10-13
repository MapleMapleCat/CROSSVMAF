#pragma once
#include <volk.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include "crossvmaf/device.hpp"
#include "crossvmaf/error.hpp"
#include <memory>
#include <vector>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <string_view>

namespace crossvmaf::gpu {

class DeviceMemoryPool;

struct DeviceQueueSynchronization final {
    void *opaque{};
    void (*lock)(void *opaque, std::uint32_t queue_family, std::uint32_t queue_index) noexcept {};
    void (*unlock)(void *opaque, std::uint32_t queue_family, std::uint32_t queue_index) noexcept {};

    [[nodiscard]] bool enabled() const noexcept { return lock != nullptr && unlock != nullptr; }
};

enum class DeviceResourceCleanupOutcome : std::uint8_t {
    safe_to_destroy,
    must_abandon_device_resources,
};

// Shared by every non-owning DeviceView derived from one Device. Once GPU
// quiescence cannot be proven, the state is permanent: device-child RAII
// wrappers must skip Vulkan destruction and DeviceCore must abandon the device.
class DeviceResourceLifetime final {
  public:
    using AbandonParentCallback = void (*)() noexcept;

    explicit DeviceResourceLifetime(
        AbandonParentCallback abandon_parent_callback = nullptr) noexcept
        : abandon_parent_callback_(abandon_parent_callback) {}

    [[nodiscard]] DeviceResourceCleanupOutcome cleanup_outcome() const noexcept {
        return cleanup_outcome_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool safe_to_destroy() const noexcept {
        return cleanup_outcome() == DeviceResourceCleanupOutcome::safe_to_destroy;
    }

    void mark_must_abandon_device_resources() noexcept {
        if (cleanup_outcome_.exchange(DeviceResourceCleanupOutcome::must_abandon_device_resources,
                                      std::memory_order_acq_rel) ==
            DeviceResourceCleanupOutcome::must_abandon_device_resources)
            return;
        if (abandon_parent_callback_ != nullptr) {
            abandon_parent_callback_();
        }
    }

  private:
    AbandonParentCallback abandon_parent_callback_{};
    std::atomic<DeviceResourceCleanupOutcome> cleanup_outcome_{
        DeviceResourceCleanupOutcome::safe_to_destroy};
};

// Upper bound on queues requested from the selected compute family. Feature
// lanes map onto queues by lane_index % lane_queue_count, so devices with one
// queue alias every lane onto queue index 0 without any semantic change.
inline constexpr std::uint32_t maximum_lane_queue_count = 4U;

// Non-owning compute access. Device must outlive every resource created from
// this view, even if FFmpeg retains DeviceCore longer: the memory pool, cache,
// and queue callback access still belong to the compute context's lifetime.
struct DeviceView {
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkQueue queue{VK_NULL_HANDLE};
    std::uint32_t queue_family{};
    // lane_queues[0] always mirrors queue. Remaining entries are valid only
    // for index < lane_queue_count.
    std::array<VkQueue, maximum_lane_queue_count> lane_queues{};
    std::array<std::uint32_t, maximum_lane_queue_count> lane_queue_indices{};
    std::uint32_t lane_queue_count{1U};
    const VolkDeviceTable *table{};
    VkPipelineCache pipeline_cache{VK_NULL_HANDLE};
    DeviceMemoryPool *memory_pool{};
    DeviceResourceLifetime *resource_lifetime{};
    bool shader_int16_enabled{};
    // False for upstream FFmpeg: driver support alone does not establish a
    // GENERAL-layout producer contract for wait-only consumers.
    bool unified_video_image_layouts{};
    DeviceQueueSynchronization queue_synchronization;

    [[nodiscard]] VkQueue queue_for_lane(std::size_t lane_index) const noexcept {
        if (lane_queue_count <= 1U) return queue;
        const VkQueue lane_queue = lane_queues[lane_index % lane_queue_count];
        return lane_queue != VK_NULL_HANDLE ? lane_queue : queue;
    }

    [[nodiscard]] std::uint32_t queue_index_for_lane(std::size_t lane_index) const noexcept {
        if (lane_queue_count <= 1U) return lane_queue_indices[0U];
        return lane_queue_indices[lane_index % lane_queue_count];
    }

    [[nodiscard]] bool complete() const noexcept {
        return physical_device != VK_NULL_HANDLE && device != VK_NULL_HANDLE &&
               queue != VK_NULL_HANDLE && table != nullptr && memory_pool != nullptr &&
               resource_lifetime != nullptr;
    }
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

struct DeviceCandidate {
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    DeviceInfo info;
    std::uint32_t compute_queue_family{};
    std::uint32_t compute_queue_count{1U};
    VkPhysicalDeviceType type{VK_PHYSICAL_DEVICE_TYPE_OTHER};
    bool supports_global_priority{};
    bool supports_memory_budget{};
    bool supports_vulkan_video{};
};

class Runtime final {
  public:
    [[nodiscard]] static Result<std::shared_ptr<Runtime>> acquire();
    static void abandon_process_runtime() noexcept;

    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;
    ~Runtime();

    [[nodiscard]] Result<std::vector<DeviceCandidate>> enumerate_candidates() const;
    [[nodiscard]] VkInstance instance() const noexcept { return instance_; }
    [[nodiscard]] std::uint32_t api_version() const noexcept { return api_version_; }

  private:
    Runtime(VkInstance instance, std::uint32_t api_version) noexcept;

    VkInstance instance_{VK_NULL_HANDLE};
    std::uint32_t api_version_{VK_API_VERSION_1_1};
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

struct NativeQueueFamily;

// Match FFmpeg's ReBAR heuristic for host image copies. UMA devices with fully
// host-visible device-local memory qualify without a PCIe BAR.
[[nodiscard]] bool has_full_host_visible_vram(
    const VkPhysicalDeviceMemoryProperties &memory_properties) noexcept;

struct VideoQueuePlan final {
    std::uint32_t runtime_queue_count{};
    VkDeviceQueueCreateFlags queue_create_flags{};
    std::vector<NativeQueueFamily> queue_families;
};

// Both sides use the same zero-based queues and image-sharing families.
// Synchronization is supplied either by the driver or FFmpeg's shared host
// mutexes, so a single compute queue is sufficient in either mode.
[[nodiscard]] std::optional<VideoQueuePlan>
plan_video_queues(std::uint32_t compute_queue_family, std::uint32_t available_compute_queues,
                  std::span<const NativeQueueFamily> video_queue_families,
                  bool internally_synchronized);

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

// Shared native-device ownership only. Compute allocations and FFmpeg contexts
// must not be owned here: they retain this core, never the other way around.
class DeviceCore final {
  public:
    DeviceCore(std::shared_ptr<Runtime> runtime, VkDevice device, VolkDeviceTable table) noexcept;
    ~DeviceCore();
    DeviceCore(const DeviceCore &) = delete;
    DeviceCore &operator=(const DeviceCore &) = delete;

    [[nodiscard]] VkDevice device() const noexcept { return device_; }
    [[nodiscard]] const VolkDeviceTable &table() const noexcept { return table_; }
    [[nodiscard]] DeviceResourceLifetime &resource_lifetime() noexcept {
        return resource_lifetime_;
    }

  private:
    std::shared_ptr<Runtime> runtime_;
    VkDevice device_{VK_NULL_HANDLE};
    VolkDeviceTable table_{};
    DeviceResourceLifetime resource_lifetime_{&Runtime::abandon_process_runtime};
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

[[nodiscard]] Result<void> run_device_self_test(DeviceView device);

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

class DeviceMemoryPool;
class DeviceCore;
class PersistentPipelineCache;

enum class VideoImageLayoutPolicy : std::uint8_t {
    automatic,
    // Keep the extension disabled on this device, including in FFmpeg.
    // Used to validate the compatibility path on unified-layout hardware.
    legacy,
};

struct NativeQueueFamily final {
    std::uint32_t index{};
    std::uint32_t queue_count{};
    VkQueueFlagBits flags{};
    VkVideoCodecOperationFlagsKHR video_codec_operations{};
};

struct NativeDeviceCapabilities final {
    // Must own the supplied VkDevice and retain its instance/loader. Kept by
    // the native the external queue consumer device, including references retained only by frames.
    std::shared_ptr<void> device_lifetime;
    PFN_vkGetInstanceProcAddr get_instance_proc_address{};
    VkInstance instance{VK_NULL_HANDLE};
    VkPhysicalDevice physical_device{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDeviceFeatures enabled_base_features{};
    VkPhysicalDeviceVulkan11Features enabled_vulkan_11_features{};
    VkPhysicalDeviceVulkan12Features enabled_vulkan_12_features{};
    VkPhysicalDeviceVulkan13Features enabled_vulkan_13_features{};
    // Optional frame-initialization capability, also published to FFmpeg.
    VkPhysicalDeviceHostImageCopyFeaturesEXT enabled_host_image_copy_features{};
#ifdef VK_KHR_video_decode_vp9
    VkPhysicalDeviceVideoDecodeVP9FeaturesKHR enabled_video_decode_vp9_features{};
#endif
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR enabled_unified_image_layouts_features{};
    VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR
        enabled_internally_synchronized_queues_features{};
    // Must match creation flags for every queue family shared with the external queue consumer.
    VkDeviceQueueCreateFlags queue_create_flags{};
    // A producer contract, not merely a driver capability. Kept false when
    // external video producers use decode layouts and require timeline feedback.
    bool unified_video_image_layouts{};
    std::vector<std::string> enabled_device_extensions;
    // Both the external queue consumer and the runtime use indices starting at zero. Ordinary
    // queues share the external queue consumer's default mutexes through the synchronization
    // binding.
    std::vector<NativeQueueFamily> queue_families;
};

struct DeviceRequirements final {
    bool video_decode{true};
    bool frame_interop{};
};

class Device final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Device>>
    create(const DeviceSelector &selector, std::string_view shader_manifest_sha256 = {},
           VideoImageLayoutPolicy video_image_layout_policy = VideoImageLayoutPolicy::automatic,
           DeviceRequirements requirements = {});

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    ~Device();

    [[nodiscard]] const DeviceInfo &device_info() const noexcept { return candidate_.info; }

    [[nodiscard]] DeviceView device_view() const noexcept;
    [[nodiscard]] bool video_decode_enabled() const noexcept { return video_decode_enabled_; }
    [[nodiscard]] const std::optional<NativeDeviceCapabilities> &
    native_capabilities() const noexcept {
        return native_capabilities_;
    }

    [[nodiscard]] const std::shared_ptr<void> &queue_synchronization_owner() const noexcept {
        return queue_synchronization_owner_;
    }

    // Install before exposing views to concurrent queue consumers. The owner
    // keeps callback storage alive; it must not retain this Device object.
    void bind_queue_synchronization(std::shared_ptr<void> owner,
                                    DeviceQueueSynchronization synchronization) noexcept {
        queue_synchronization_owner_ = std::move(owner);
        queue_synchronization_ = synchronization;
    }

  private:
    Device(DeviceCandidate candidate, std::shared_ptr<DeviceCore> device_core,
           std::array<VkQueue, maximum_lane_queue_count> lane_queues,
           std::array<std::uint32_t, maximum_lane_queue_count> lane_queue_indices,
           std::uint32_t lane_queue_count, bool shader_int16_enabled,
           bool unified_video_image_layouts) noexcept;

    DeviceCandidate candidate_;
    std::shared_ptr<DeviceCore> device_core_;
    std::array<VkQueue, maximum_lane_queue_count> lane_queues_{};
    std::array<std::uint32_t, maximum_lane_queue_count> lane_queue_indices_{};
    std::uint32_t lane_queue_count_{1U};
    bool shader_int16_enabled_{};
    bool unified_video_image_layouts_{};
    bool video_decode_enabled_{};
    std::unique_ptr<DeviceMemoryPool> memory_pool_;
    std::unique_ptr<PersistentPipelineCache> pipeline_cache_;
    std::optional<NativeDeviceCapabilities> native_capabilities_;
    std::shared_ptr<void> queue_synchronization_owner_;
    DeviceQueueSynchronization queue_synchronization_;
};

} // namespace crossvmaf::gpu
