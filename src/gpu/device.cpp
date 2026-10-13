#include "gpu/device.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "gpu/memory.hpp"
#include "gpu/pipeline.hpp"
#include <optional>
#include <span>
#include <string_view>

namespace crossvmaf::gpu {
namespace {

std::once_flag runtime_once;
std::atomic_bool process_runtime_abandoned{};
std::shared_ptr<Runtime> process_runtime;
Error runtime_error{ErrorCode::backend_error, "Vulkan runtime was not initialized"};

[[nodiscard]] std::string instance_result_message(const char *operation, VkResult result) {
    std::ostringstream stream;
    stream << operation << " failed with VkResult " << static_cast<int>(result);
    return stream.str();
}

[[nodiscard]] std::string instance_format_uuid(const std::uint8_t *uuid) {
    constexpr std::array<std::size_t, 4> separators{4U, 6U, 8U, 10U};
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (std::size_t index = 0; index < VK_UUID_SIZE; ++index) {
        if (std::find(separators.begin(), separators.end(), index) != separators.end()) {
            stream << '-';
        }
        stream << std::setw(2) << static_cast<unsigned>(uuid[index]);
    }
    return stream.str();
}

[[nodiscard]] int instance_device_priority(VkPhysicalDeviceType type) noexcept {
    if (type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) return 0;
    if (type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) return 1;
    return 2;
}

class instance_PendingRuntimeInitialization final {
  public:
    explicit instance_PendingRuntimeInitialization(VkInstance instance) noexcept
        : instance_(instance) {}

    instance_PendingRuntimeInitialization(const instance_PendingRuntimeInitialization &) = delete;
    instance_PendingRuntimeInitialization &
    operator=(const instance_PendingRuntimeInitialization &) = delete;

    ~instance_PendingRuntimeInitialization() {
        if (instance_ == VK_NULL_HANDLE) return;
        vkDestroyInstance(instance_, nullptr);
        volkFinalize();
    }

    void release() noexcept { instance_ = VK_NULL_HANDLE; }

  private:
    VkInstance instance_{VK_NULL_HANDLE};
};

[[nodiscard]] bool instance_supports_device_extension(VkPhysicalDevice physical_device,
                                                      const char *extension_name) {
    std::uint32_t count = 0;
    VkResult result =
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, nullptr);
    if (result != VK_SUCCESS || count == 0U) return false;

    std::vector<VkExtensionProperties> extensions(count);
    result =
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &count, extensions.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE) return false;
    extensions.resize(count);
    return std::any_of(extensions.begin(), extensions.end(),
                       [extension_name](const VkExtensionProperties &extension) {
                           return std::string_view(extension.extensionName) == extension_name;
                       });
}

struct instance_ComputeQueueSelection {
    std::uint32_t family{};
    std::uint32_t queue_count{1U};
};

[[nodiscard]] Result<instance_ComputeQueueSelection>
instance_select_compute_queue_family(VkPhysicalDevice physical_device) {
    std::uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, nullptr);
    if (count == 0) {
        return Result<instance_ComputeQueueSelection>::failure(
            {ErrorCode::no_compatible_device, "device exposes no queue families"});
    }

    std::vector<VkQueueFamilyProperties> properties(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &count, properties.data());
    std::uint32_t selected = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t selected_queue_count = 1U;
    bool selected_is_dedicated = false;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto flags = properties[index].queueFlags;
        if (properties[index].queueCount == 0 || (flags & VK_QUEUE_COMPUTE_BIT) == 0) {
            continue;
        }
        const bool dedicated = (flags & VK_QUEUE_GRAPHICS_BIT) == 0;
        if (selected == std::numeric_limits<std::uint32_t>::max() ||
            (dedicated && !selected_is_dedicated)) {
            selected = index;
            selected_queue_count = properties[index].queueCount;
            selected_is_dedicated = dedicated;
        }
    }

    if (selected == std::numeric_limits<std::uint32_t>::max()) {
        return Result<instance_ComputeQueueSelection>::failure(
            {ErrorCode::no_compatible_device, "device exposes no compute queue"});
    }
    return Result<instance_ComputeQueueSelection>::success(
        instance_ComputeQueueSelection{selected, selected_queue_count});
}

} // namespace

Runtime::Runtime(VkInstance instance, std::uint32_t api_version) noexcept
    : instance_(instance), api_version_(api_version) {
}

Runtime::~Runtime() {
    if (process_runtime_abandoned.load(std::memory_order_acquire)) {
        // A leaked VkDevice remains an instance child. Let process teardown
        // reclaim both instead of invoking an invalid parent destruction.
        instance_ = VK_NULL_HANDLE;
        return;
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    volkFinalize();
}

void Runtime::abandon_process_runtime() noexcept {
    process_runtime_abandoned.store(true, std::memory_order_release);
}

Result<std::shared_ptr<Runtime>> Runtime::acquire() {
    std::call_once(runtime_once, [] {
        const VkResult initialize_result = volkInitialize();
        if (initialize_result != VK_SUCCESS) {
            runtime_error = {
                ErrorCode::no_compatible_device,
                instance_result_message("loading the system Vulkan Loader", initialize_result)};
            return;
        }

        const std::uint32_t loader_version = volkGetInstanceVersion();
        if (loader_version < VK_API_VERSION_1_1) {
            runtime_error = {ErrorCode::no_compatible_device,
                             "the system Vulkan Loader does not support Vulkan 1.1"};
            volkFinalize();
            return;
        }

        const std::uint32_t requested_api_version = std::min(loader_version, VK_API_VERSION_1_3);
        const VkApplicationInfo application_info{
            VK_STRUCTURE_TYPE_APPLICATION_INFO,
            nullptr,
            "CrossVMAF",
            VK_MAKE_API_VERSION(0, 0, 1, 0),
            "CrossVMAF deterministic compute",
            VK_MAKE_API_VERSION(0, 0, 1, 0),
            requested_api_version,
        };
        const VkInstanceCreateInfo create_info{
            VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            nullptr,
            0,
            &application_info,
            0,
            nullptr,
            0,
            nullptr,
        };

        VkInstance instance = VK_NULL_HANDLE;
        const VkResult create_result = vkCreateInstance(&create_info, nullptr, &instance);
        if (create_result != VK_SUCCESS) {
            runtime_error = {ErrorCode::no_compatible_device,
                             instance_result_message("vkCreateInstance", create_result)};
            volkFinalize();
            return;
        }

        volkLoadInstanceOnly(instance);
        instance_PendingRuntimeInitialization pending_runtime(instance);
        auto runtime_owner = std::unique_ptr<Runtime>(new Runtime(instance, requested_api_version));
        pending_runtime.release();
        // The unique owner handles control-block allocation failure without
        // competing with the pending instance guard for native destruction.
        process_runtime = std::shared_ptr<Runtime>(std::move(runtime_owner));
    });

    if (!process_runtime) {
        return Result<std::shared_ptr<Runtime>>::failure(runtime_error);
    }
    return Result<std::shared_ptr<Runtime>>::success(process_runtime);
}

Result<std::vector<DeviceCandidate>> Runtime::enumerate_candidates() const {
    std::uint32_t count = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (result != VK_SUCCESS) {
        return Result<std::vector<DeviceCandidate>>::failure(
            {ErrorCode::backend_error,
             instance_result_message("vkEnumeratePhysicalDevices", result)});
    }

    std::vector<VkPhysicalDevice> physical_devices(count);
    if (count != 0) {
        result = vkEnumeratePhysicalDevices(instance_, &count, physical_devices.data());
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
            return Result<std::vector<DeviceCandidate>>::failure(
                {ErrorCode::backend_error,
                 instance_result_message("vkEnumeratePhysicalDevices", result)});
        }
        physical_devices.resize(count);
    }

    std::vector<DeviceCandidate> candidates;
    candidates.reserve(physical_devices.size());
    for (const VkPhysicalDevice physical_device : physical_devices) {
        VkPhysicalDeviceIDProperties id_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        properties.pNext = &id_properties;
        vkGetPhysicalDeviceProperties2(physical_device, &properties);

        const auto &base = properties.properties;
        if (base.apiVersion < VK_API_VERSION_1_1 ||
            (base.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU &&
             base.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ||
            base.limits.maxComputeWorkGroupInvocations < 256U ||
            base.limits.maxComputeWorkGroupSize[0] < 256U) {
            continue;
        }

        const auto queue_family = instance_select_compute_queue_family(physical_device);
        if (!queue_family) continue;

        DeviceInfo info;
        info.name = base.deviceName;
        info.uuid = instance_format_uuid(id_properties.deviceUUID);
#if defined(_WIN32)
        if (id_properties.deviceLUIDValid) {
            constexpr char hex[] = "0123456789abcdef";
            info.physical_id = "luid:";
            for (auto byte : id_properties.deviceLUID) { info.physical_id += hex[byte >> 4]; info.physical_id += hex[byte & 15]; }
        }
#elif defined(__linux__)
        VkPhysicalDeviceDrmPropertiesEXT drm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
        VkPhysicalDeviceProperties2 drm_properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drm};
        vkGetPhysicalDeviceProperties2(physical_device, &drm_properties);
        if (drm.hasRender) info.physical_id = "drm:" + std::to_string(drm.renderMajor) + ":" + std::to_string(drm.renderMinor);
#endif
        info.vendor_id = base.vendorID;
        info.device_id = base.deviceID;
        info.api_version = base.apiVersion;
        info.driver_version = base.driverVersion;
        info.discrete = base.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        VkDeviceSize device_local_heap_bytes = 0U;
        bool host_visible_device_local = false;
        for (std::uint32_t heap_index = 0U; heap_index < memory_properties.memoryHeapCount;
             ++heap_index) {
            if ((memory_properties.memoryHeaps[heap_index].flags &
                 VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0U) {
                device_local_heap_bytes = std::max(device_local_heap_bytes,
                                                   memory_properties.memoryHeaps[heap_index].size);
            }
        }
        for (std::uint32_t type_index = 0U; type_index < memory_properties.memoryTypeCount;
             ++type_index) {
            const VkMemoryPropertyFlags flags =
                memory_properties.memoryTypes[type_index].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0U &&
                (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U) {
                host_visible_device_local = true;
                break;
            }
        }
        const bool supports_memory_budget = instance_supports_device_extension(
            physical_device, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        info.memory_budget_available = supports_memory_budget;
        info.host_visible_device_local = host_visible_device_local;
        info.device_local_heap_bytes = static_cast<std::uint64_t>(device_local_heap_bytes);
        const bool supports_global_priority = instance_supports_device_extension(
            physical_device, VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
        const bool supports_vulkan_video =
            api_version_ >= VK_API_VERSION_1_3 && base.apiVersion >= VK_API_VERSION_1_3 &&
            instance_supports_device_extension(physical_device,
                                               VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) &&
            instance_supports_device_extension(physical_device,
                                               VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME) &&
            (instance_supports_device_extension(physical_device,
                                                VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME) ||
             instance_supports_device_extension(physical_device,
                                                VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME) ||
             instance_supports_device_extension(physical_device,
                                                VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME)
#ifdef VK_KHR_video_decode_vp9
             || instance_supports_device_extension(physical_device,
                                                   VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME)
#endif
            );
        candidates.push_back(DeviceCandidate{
            physical_device, std::move(info), queue_family.value().family,
            queue_family.value().queue_count, base.deviceType, supports_global_priority,
            supports_memory_budget, supports_vulkan_video});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const DeviceCandidate &lhs, const DeviceCandidate &rhs) {
                  return std::tuple{instance_device_priority(lhs.type), lhs.info.uuid} <
                         std::tuple{instance_device_priority(rhs.type), rhs.info.uuid};
              });
    candidates.erase(std::unique(candidates.begin(), candidates.end(),
                                 [](const DeviceCandidate &lhs, const DeviceCandidate &rhs) {
                                     return lhs.info.uuid == rhs.info.uuid;
                                 }),
                     candidates.end());
    return Result<std::vector<DeviceCandidate>>::success(std::move(candidates));
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

bool has_full_host_visible_vram(
    const VkPhysicalDeviceMemoryProperties &memory_properties) noexcept {
    VkDeviceSize max_vram = 0U;
    VkDeviceSize max_visible_vram = 0U;
    for (std::uint32_t index = 0U; index < memory_properties.memoryTypeCount; ++index) {
        const auto &type = memory_properties.memoryTypes[index];
        if ((type.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0U) continue;
        const auto size = memory_properties.memoryHeaps[type.heapIndex].size;
        max_vram = std::max(max_vram, size);
        if ((type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0U)
            max_visible_vram = std::max(max_visible_vram, size);
    }
    // Use the same 1 KiB tolerance as FFmpeg, but reject missing memory data.
    return max_vram != 0U && max_visible_vram != 0U && max_vram - max_visible_vram < 1024U;
}

std::optional<VideoQueuePlan>
plan_video_queues(std::uint32_t compute_queue_family, std::uint32_t available_compute_queues,
                  std::span<const NativeQueueFamily> video_queue_families,
                  bool internally_synchronized) {
    if (available_compute_queues == 0U || video_queue_families.empty()) {
        return std::nullopt;
    }

    VideoQueuePlan plan{};
    plan.runtime_queue_count = std::min(available_compute_queues, maximum_lane_queue_count);
    plan.queue_create_flags =
        internally_synchronized ? VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR : 0U;
    plan.queue_families.push_back(
        {compute_queue_family, plan.runtime_queue_count,
         static_cast<VkQueueFlagBits>(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT),
         VK_VIDEO_CODEC_OPERATION_NONE_KHR});

    for (const NativeQueueFamily &video_family : video_queue_families) {
        if (video_family.queue_count == 0U ||
            (video_family.flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) == 0U ||
            video_family.video_codec_operations == VK_VIDEO_CODEC_OPERATION_NONE_KHR) {
            return std::nullopt;
        }
        auto decoder_family = std::find_if(
            plan.queue_families.begin(), plan.queue_families.end(),
            [&](const NativeQueueFamily &family) { return family.index == video_family.index; });
        const std::uint32_t video_queue_count =
            std::min(video_family.queue_count, maximum_lane_queue_count);
        if (decoder_family == plan.queue_families.end()) {
            auto family = video_family;
            family.queue_count = video_queue_count;
            plan.queue_families.push_back(family);
        } else {
            // Merge codec capabilities without creating a family twice.
            decoder_family->queue_count = std::max(decoder_family->queue_count, video_queue_count);
            decoder_family->flags =
                static_cast<VkQueueFlagBits>(decoder_family->flags | video_family.flags);
            decoder_family->video_codec_operations |= video_family.video_codec_operations;
        }
    }
    if (plan.queue_families.size() > 64U) return std::nullopt;
    const auto &compute_family = plan.queue_families.front();
    if (compute_family.queue_count > available_compute_queues) return std::nullopt;
    return plan;
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

DeviceCore::DeviceCore(std::shared_ptr<Runtime> runtime, VkDevice device,
                       VolkDeviceTable table) noexcept
    : runtime_(std::move(runtime)), device_(device), table_(table) {
}

DeviceCore::~DeviceCore() {
    // Each consumer establishes quiescence before releasing its children and
    // core reference. A device-wide wait here cannot replace that contract.
    if (device_ != VK_NULL_HANDLE && resource_lifetime_.safe_to_destroy()) {
        table_.vkDestroyDevice(device_, nullptr);
    }
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {
namespace {

[[nodiscard]] std::string device_result_message(const char *operation, VkResult result) {
    std::ostringstream stream;
    stream << operation << " failed with VkResult " << static_cast<int>(result);
    return stream.str();
}

[[nodiscard]] std::optional<std::string> device_canonicalize_uuid(std::string_view text) {
    if (text.empty() || text == "auto") return std::string{};

    std::string hex;
    hex.reserve(32);
    for (const char character : text) {
        if (character == '-') continue;
        const unsigned char byte = static_cast<unsigned char>(character);
        if (std::isxdigit(byte) == 0) return std::nullopt;
        hex.push_back(static_cast<char>(std::tolower(byte)));
    }
    if (hex.size() != 32U) return std::nullopt;

    std::string canonical;
    canonical.reserve(36);
    for (std::size_t index = 0; index < hex.size(); ++index) {
        if (index == 8U || index == 12U || index == 16U || index == 20U) {
            canonical.push_back('-');
        }
        canonical.push_back(hex[index]);
    }
    return canonical;
}

[[nodiscard]] Result<std::vector<VkExtensionProperties>>
device_enumerate_device_extensions(VkPhysicalDevice physical_device) {
    std::uint32_t extension_count = 0U;
    VkResult result =
        vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count, nullptr);
    if (result != VK_SUCCESS) {
        return Result<std::vector<VkExtensionProperties>>::failure(
            {ErrorCode::backend_error,
             device_result_message("vkEnumerateDeviceExtensionProperties", result)});
    }
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (extension_count != 0U) {
        result = vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &extension_count,
                                                      extensions.data());
        if (result != VK_SUCCESS && result != VK_INCOMPLETE) {
            return Result<std::vector<VkExtensionProperties>>::failure(
                {ErrorCode::backend_error,
                 device_result_message("vkEnumerateDeviceExtensionProperties", result)});
        }
        extensions.resize(extension_count);
    }
    return Result<std::vector<VkExtensionProperties>>::success(std::move(extensions));
}

[[nodiscard]] bool device_contains_extension(std::span<const VkExtensionProperties> extensions,
                                             std::string_view extension_name) noexcept {
    return std::any_of(extensions.begin(), extensions.end(),
                       [extension_name](const VkExtensionProperties &extension) {
                           return std::string_view(extension.extensionName) == extension_name;
                       });
}

struct device_VulkanVideoDeviceConfiguration final {
    bool enabled{};
    // Upstream FFmpeg uses video layouts; consumers must publish transitions
    // and timeline signals even if the driver supports unified layouts.
    bool unified_video_image_layouts{};
    VkPhysicalDeviceFeatures base_features{};
    VkPhysicalDeviceVulkan11Features vulkan_11_features{};
    VkPhysicalDeviceVulkan12Features vulkan_12_features{};
    VkPhysicalDeviceVulkan13Features vulkan_13_features{};
    VkPhysicalDeviceHostImageCopyFeaturesEXT host_image_copy_features{};
#ifdef VK_KHR_video_decode_vp9
    VkPhysicalDeviceVideoDecodeVP9FeaturesKHR video_decode_vp9_features{};
#endif
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR unified_image_layouts_features{};
    VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR
        internally_synchronized_queues_features{};
    std::vector<const char *> extension_names;
    std::vector<NativeQueueFamily> video_queue_families;
};

// Appends the optional trailing feature structures behind the Vulkan 1.3
// features so every consumer of the configuration sees one identical chain.
void device_chain_optional_video_features(
    device_VulkanVideoDeviceConfiguration &configuration) noexcept {
    void **tail = &configuration.vulkan_13_features.pNext;
    *tail = nullptr;
    if (configuration.host_image_copy_features.hostImageCopy == VK_TRUE) {
        *tail = &configuration.host_image_copy_features;
        tail = &configuration.host_image_copy_features.pNext;
        *tail = nullptr;
    }
#ifdef VK_KHR_video_decode_vp9
    if (configuration.video_decode_vp9_features.videoDecodeVP9 == VK_TRUE) {
        *tail = &configuration.video_decode_vp9_features;
        tail = &configuration.video_decode_vp9_features.pNext;
        *tail = nullptr;
    }
#endif
    if (configuration.unified_video_image_layouts) {
        *tail = &configuration.unified_image_layouts_features;
        tail = &configuration.unified_image_layouts_features.pNext;
    }
    if (configuration.internally_synchronized_queues_features.internallySynchronizedQueues ==
        VK_TRUE) {
        *tail = &configuration.internally_synchronized_queues_features;
        tail = &configuration.internally_synchronized_queues_features.pNext;
    }
    *tail = nullptr;
}

struct device_VideoCodecExtension final {
    VkVideoCodecOperationFlagBitsKHR operation{};
    const char *extension_name{};
};

[[nodiscard]] device_VulkanVideoDeviceConfiguration
device_query_vulkan_video_configuration(const DeviceCandidate &candidate,
                                        std::span<const VkExtensionProperties> extensions,
                                        VideoImageLayoutPolicy video_image_layout_policy, DeviceRequirements requirements) {
    device_VulkanVideoDeviceConfiguration configuration{};
    if (!requirements.frame_interop && (!requirements.video_decode || !candidate.supports_vulkan_video ||
        !device_contains_extension(extensions, VK_KHR_VIDEO_QUEUE_EXTENSION_NAME) ||
        !device_contains_extension(extensions, VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME))) {
        return configuration;
    }

    if (candidate.info.api_version < VK_API_VERSION_1_3) return configuration;
    std::vector<device_VideoCodecExtension> codec_extensions;
    if (requirements.video_decode && device_contains_extension(extensions, VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME)) {
        codec_extensions.push_back({VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR,
                                    VK_KHR_VIDEO_DECODE_H264_EXTENSION_NAME});
    }
    if (requirements.video_decode && device_contains_extension(extensions, VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME)) {
        codec_extensions.push_back({VK_VIDEO_CODEC_OPERATION_DECODE_H265_BIT_KHR,
                                    VK_KHR_VIDEO_DECODE_H265_EXTENSION_NAME});
    }
    if (requirements.video_decode && device_contains_extension(extensions, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME)) {
        codec_extensions.push_back(
            {VK_VIDEO_CODEC_OPERATION_DECODE_AV1_BIT_KHR, VK_KHR_VIDEO_DECODE_AV1_EXTENSION_NAME});
    }
#ifdef VK_KHR_video_decode_vp9
    if (requirements.video_decode && device_contains_extension(extensions, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME)) {
        codec_extensions.push_back(
            {VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME});
    }
#endif
    if (codec_extensions.empty() && !requirements.frame_interop) return configuration;

    VkPhysicalDeviceFeatures2 supported_features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features supported_vulkan_11_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features supported_vulkan_12_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features supported_vulkan_13_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    supported_features.pNext = &supported_vulkan_11_features;
    supported_vulkan_11_features.pNext = &supported_vulkan_12_features;
    supported_vulkan_12_features.pNext = &supported_vulkan_13_features;
    void **supported_chain_tail = &supported_vulkan_13_features.pNext;
    *supported_chain_tail = nullptr;
    VkPhysicalDeviceHostImageCopyFeaturesEXT supported_host_image_copy_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT};
    const bool host_image_copy_available =
        device_contains_extension(extensions, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
    if (host_image_copy_available) {
        *supported_chain_tail = &supported_host_image_copy_features;
        supported_chain_tail = &supported_host_image_copy_features.pNext;
    }
#ifdef VK_KHR_video_decode_vp9
    VkPhysicalDeviceVideoDecodeVP9FeaturesKHR supported_video_decode_vp9_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR};
    const bool vp9_extension_available =
        device_contains_extension(extensions, VK_KHR_VIDEO_DECODE_VP9_EXTENSION_NAME);
    if (vp9_extension_available) {
        *supported_chain_tail = &supported_video_decode_vp9_features;
        supported_chain_tail = &supported_video_decode_vp9_features.pNext;
        *supported_chain_tail = nullptr;
    }
#endif
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR supported_unified_image_layouts_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR};
    const bool unified_image_layouts_extension_available =
        device_contains_extension(extensions, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME);
    if (unified_image_layouts_extension_available) {
        *supported_chain_tail = &supported_unified_image_layouts_features;
        supported_chain_tail = &supported_unified_image_layouts_features.pNext;
        *supported_chain_tail = nullptr;
    }
    VkPhysicalDeviceInternallySynchronizedQueuesFeaturesKHR
        supported_internally_synchronized_queues_features{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR};
    const bool internal_sync_extension_available =
        device_contains_extension(extensions, VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
    if (internal_sync_extension_available) {
        *supported_chain_tail = &supported_internally_synchronized_queues_features;
    }
    vkGetPhysicalDeviceFeatures2(candidate.physical_device, &supported_features);
    if (supported_vulkan_12_features.timelineSemaphore != VK_TRUE ||
        supported_vulkan_13_features.synchronization2 != VK_TRUE) {
        return configuration;
    }
#ifdef VK_KHR_video_decode_vp9
    if (vp9_extension_available && supported_video_decode_vp9_features.videoDecodeVP9 != VK_TRUE) {
        codec_extensions.erase(
            std::remove_if(codec_extensions.begin(), codec_extensions.end(),
                           [](const device_VideoCodecExtension &codec) {
                               return codec.operation ==
                                      VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
                           }),
            codec_extensions.end());
    }
#endif
    if (codec_extensions.empty() && !requirements.frame_interop) return configuration;

    std::uint32_t queue_family_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties2(candidate.physical_device, &queue_family_count,
                                              nullptr);
    if (queue_family_count == 0U) return configuration;
    std::vector<VkQueueFamilyVideoPropertiesKHR> video_properties(queue_family_count);
    std::vector<VkQueueFamilyProperties2> queue_properties(queue_family_count);
    for (std::uint32_t family_index = 0U; family_index < queue_family_count; ++family_index) {
        video_properties[family_index].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_VIDEO_PROPERTIES_KHR;
        queue_properties[family_index].sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2;
        queue_properties[family_index].pNext = &video_properties[family_index];
    }
    vkGetPhysicalDeviceQueueFamilyProperties2(candidate.physical_device, &queue_family_count,
                                              queue_properties.data());
    queue_properties.resize(queue_family_count);
    video_properties.resize(queue_family_count);

    std::vector<device_VideoCodecExtension> enabled_codecs;
    for (const device_VideoCodecExtension &codec : codec_extensions) {
        auto selected_family = queue_properties.end();
        for (auto family = queue_properties.begin(); family != queue_properties.end(); ++family) {
            const std::size_t family_index =
                static_cast<std::size_t>(family - queue_properties.begin());
            if (family->queueFamilyProperties.queueCount != 0U &&
                (family->queueFamilyProperties.queueFlags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) != 0U &&
                (video_properties[family_index].videoCodecOperations & codec.operation) != 0U) {
                selected_family = family;
                break;
            }
        }
        if (selected_family == queue_properties.end()) continue;

        const std::uint32_t family_index =
            static_cast<std::uint32_t>(selected_family - queue_properties.begin());
        auto existing_family = std::find_if(configuration.video_queue_families.begin(),
                                            configuration.video_queue_families.end(),
                                            [family_index](const NativeQueueFamily &family) {
                                                return family.index == family_index;
                                            });
        if (existing_family == configuration.video_queue_families.end()) {
            const std::uint32_t requested_video_queue_count = std::min(
                selected_family->queueFamilyProperties.queueCount, maximum_lane_queue_count);
            configuration.video_queue_families.push_back(
                {family_index, requested_video_queue_count, VK_QUEUE_VIDEO_DECODE_BIT_KHR,
                 static_cast<VkVideoCodecOperationFlagsKHR>(codec.operation)});
        } else {
            existing_family->video_codec_operations |= codec.operation;
        }
        enabled_codecs.push_back(codec);
    }
    if (enabled_codecs.empty() && !requirements.frame_interop) return configuration;

    configuration.vulkan_11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    configuration.vulkan_12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    configuration.vulkan_13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    configuration.base_features.shaderImageGatherExtended =
        supported_features.features.shaderImageGatherExtended;
    configuration.base_features.shaderStorageImageReadWithoutFormat =
        supported_features.features.shaderStorageImageReadWithoutFormat;
    configuration.base_features.shaderStorageImageWriteWithoutFormat =
        supported_features.features.shaderStorageImageWriteWithoutFormat;
    configuration.base_features.fragmentStoresAndAtomics =
        supported_features.features.fragmentStoresAndAtomics;
    configuration.base_features.vertexPipelineStoresAndAtomics =
        supported_features.features.vertexPipelineStoresAndAtomics;
    configuration.base_features.shaderInt64 = supported_features.features.shaderInt64;
    configuration.base_features.shaderInt16 = supported_features.features.shaderInt16;
    configuration.base_features.shaderFloat64 = supported_features.features.shaderFloat64;

    configuration.vulkan_11_features.samplerYcbcrConversion =
        supported_vulkan_11_features.samplerYcbcrConversion;
    configuration.vulkan_11_features.storagePushConstant16 =
        supported_vulkan_11_features.storagePushConstant16;
    configuration.vulkan_11_features.storageBuffer16BitAccess =
        supported_vulkan_11_features.storageBuffer16BitAccess;
    configuration.vulkan_11_features.uniformAndStorageBuffer16BitAccess =
        supported_vulkan_11_features.uniformAndStorageBuffer16BitAccess;

    configuration.vulkan_12_features.timelineSemaphore =
        supported_vulkan_12_features.timelineSemaphore;
    configuration.vulkan_12_features.scalarBlockLayout =
        supported_vulkan_12_features.scalarBlockLayout;
    configuration.vulkan_12_features.bufferDeviceAddress =
        supported_vulkan_12_features.bufferDeviceAddress;
    configuration.vulkan_12_features.hostQueryReset = supported_vulkan_12_features.hostQueryReset;
    configuration.vulkan_12_features.storagePushConstant8 =
        supported_vulkan_12_features.storagePushConstant8;
    configuration.vulkan_12_features.shaderInt8 = supported_vulkan_12_features.shaderInt8;
    configuration.vulkan_12_features.storageBuffer8BitAccess =
        supported_vulkan_12_features.storageBuffer8BitAccess;
    configuration.vulkan_12_features.uniformAndStorageBuffer8BitAccess =
        supported_vulkan_12_features.uniformAndStorageBuffer8BitAccess;
    configuration.vulkan_12_features.shaderFloat16 = supported_vulkan_12_features.shaderFloat16;
    configuration.vulkan_12_features.shaderBufferInt64Atomics =
        supported_vulkan_12_features.shaderBufferInt64Atomics;
    configuration.vulkan_12_features.shaderSharedInt64Atomics =
        supported_vulkan_12_features.shaderSharedInt64Atomics;
    configuration.vulkan_12_features.vulkanMemoryModel =
        supported_vulkan_12_features.vulkanMemoryModel;
    configuration.vulkan_12_features.vulkanMemoryModelDeviceScope =
        supported_vulkan_12_features.vulkanMemoryModelDeviceScope;
    configuration.vulkan_12_features.vulkanMemoryModelAvailabilityVisibilityChains =
        supported_vulkan_12_features.vulkanMemoryModelAvailabilityVisibilityChains;
    configuration.vulkan_12_features.uniformBufferStandardLayout =
        supported_vulkan_12_features.uniformBufferStandardLayout;
    configuration.vulkan_12_features.runtimeDescriptorArray =
        supported_vulkan_12_features.runtimeDescriptorArray;
    configuration.vulkan_12_features.shaderSubgroupExtendedTypes =
        supported_vulkan_12_features.shaderSubgroupExtendedTypes;
    configuration.vulkan_12_features.shaderUniformBufferArrayNonUniformIndexing =
        supported_vulkan_12_features.shaderUniformBufferArrayNonUniformIndexing;
    configuration.vulkan_12_features.shaderSampledImageArrayNonUniformIndexing =
        supported_vulkan_12_features.shaderSampledImageArrayNonUniformIndexing;
    configuration.vulkan_12_features.shaderStorageBufferArrayNonUniformIndexing =
        supported_vulkan_12_features.shaderStorageBufferArrayNonUniformIndexing;
    configuration.vulkan_12_features.shaderStorageImageArrayNonUniformIndexing =
        supported_vulkan_12_features.shaderStorageImageArrayNonUniformIndexing;

    configuration.vulkan_13_features.dynamicRendering =
        supported_vulkan_13_features.dynamicRendering;
    configuration.vulkan_13_features.maintenance4 = supported_vulkan_13_features.maintenance4;
    configuration.vulkan_13_features.synchronization2 =
        supported_vulkan_13_features.synchronization2;
    configuration.vulkan_13_features.computeFullSubgroups =
        supported_vulkan_13_features.computeFullSubgroups;
    configuration.vulkan_13_features.subgroupSizeControl =
        supported_vulkan_13_features.subgroupSizeControl;
    configuration.vulkan_13_features.shaderZeroInitializeWorkgroupMemory =
        supported_vulkan_13_features.shaderZeroInitializeWorkgroupMemory;
#ifdef VK_KHR_video_decode_vp9
    configuration.video_decode_vp9_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR;
    const bool vp9_codec_enabled = std::any_of(
        enabled_codecs.begin(), enabled_codecs.end(), [](const device_VideoCodecExtension &codec) {
            return codec.operation == VK_VIDEO_CODEC_OPERATION_DECODE_VP9_BIT_KHR;
        });
    configuration.video_decode_vp9_features.videoDecodeVP9 =
        vp9_codec_enabled ? supported_video_decode_vp9_features.videoDecodeVP9 : VK_FALSE;
#endif
    if (!enabled_codecs.empty()) configuration.extension_names.push_back(VK_KHR_VIDEO_QUEUE_EXTENSION_NAME);
    // Match FFmpeg's own device creation: host image copy can restrict frame
    // allocations to the small host-visible VRAM window when ReBAR is disabled.
    // Check our shared device here because importing it bypasses FFmpeg's check.
    if (host_image_copy_available && supported_host_image_copy_features.hostImageCopy == VK_TRUE) {
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(candidate.physical_device, &memory_properties);
        if (has_full_host_visible_vram(memory_properties)) {
            configuration.extension_names.push_back(VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME);
            configuration.host_image_copy_features = supported_host_image_copy_features;
            configuration.host_image_copy_features.pNext = nullptr;
        }
    }
    if (!enabled_codecs.empty()) configuration.extension_names.push_back(VK_KHR_VIDEO_DECODE_QUEUE_EXTENSION_NAME);
    if (requirements.frame_interop) {
        for (const char* name : {
#if defined(_WIN32)
                 "VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32", "VK_KHR_win32_keyed_mutex",
#else
                 "VK_KHR_external_memory_fd", "VK_KHR_external_semaphore_fd", "VK_EXT_external_memory_dma_buf",
                 "VK_EXT_image_drm_format_modifier", "VK_EXT_queue_family_foreign",
#endif
                 "VK_KHR_external_memory", "VK_KHR_external_semaphore"}) {
            if (device_contains_extension(extensions, name)) configuration.extension_names.push_back(name);
        }
    }
    if (!enabled_codecs.empty() && internal_sync_extension_available &&
        supported_internally_synchronized_queues_features.internallySynchronizedQueues == VK_TRUE) {
        configuration.extension_names.push_back(
            VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME);
        configuration.internally_synchronized_queues_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
        configuration.internally_synchronized_queues_features.internallySynchronizedQueues =
            VK_TRUE;
    }
    for (const device_VideoCodecExtension &codec : enabled_codecs) {
        configuration.extension_names.push_back(codec.extension_name);
    }
    configuration.unified_image_layouts_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR;
    // Both policies use the upstream producer contract, not the removed
    // GENERAL-layout patch. Keep the selector for existing internal callers.
    (void)video_image_layout_policy;
    device_chain_optional_video_features(configuration);
    configuration.enabled = true;
    return configuration;
}

class device_PendingDevice final {
  public:
    device_PendingDevice(const VolkDeviceTable &table, VkDevice device) noexcept
        : table_(&table), device_(device) {}

    device_PendingDevice(const device_PendingDevice &) = delete;
    device_PendingDevice &operator=(const device_PendingDevice &) = delete;

    ~device_PendingDevice() {
        if (device_ != VK_NULL_HANDLE) {
            table_->vkDestroyDevice(device_, nullptr);
        }
    }

    void release() noexcept { device_ = VK_NULL_HANDLE; }

  private:
    const VolkDeviceTable *table_{};
    VkDevice device_{VK_NULL_HANDLE};
};

} // namespace

Device::Device(DeviceCandidate candidate, std::shared_ptr<DeviceCore> device_core,
               std::array<VkQueue, maximum_lane_queue_count> lane_queues,
               std::array<std::uint32_t, maximum_lane_queue_count> lane_queue_indices,
               std::uint32_t lane_queue_count, bool shader_int16_enabled,
               bool unified_video_image_layouts) noexcept
    : candidate_(std::move(candidate)), device_core_(std::move(device_core)),
      lane_queues_(lane_queues), lane_queue_indices_(lane_queue_indices),
      lane_queue_count_(lane_queue_count), shader_int16_enabled_(shader_int16_enabled),
      unified_video_image_layouts_(unified_video_image_layouts) {
}

DeviceView Device::device_view() const noexcept {
    const VkPipelineCache pipeline_cache =
        pipeline_cache_ ? pipeline_cache_->handle() : VK_NULL_HANDLE;
    DeviceView view{};
    view.physical_device = candidate_.physical_device;
    view.device = device_core_->device();
    view.queue = lane_queues_[0];
    view.queue_family = candidate_.compute_queue_family;
    view.lane_queues = lane_queues_;
    view.lane_queue_indices = lane_queue_indices_;
    view.lane_queue_count = lane_queue_count_;
    view.table = &device_core_->table();
    view.pipeline_cache = pipeline_cache;
    view.memory_pool = memory_pool_.get();
    view.resource_lifetime = &device_core_->resource_lifetime();
    view.shader_int16_enabled = shader_int16_enabled_;
    view.unified_video_image_layouts = unified_video_image_layouts_;
    view.queue_synchronization = queue_synchronization_;
    return view;
}

Device::~Device() {
    if (!device_core_->resource_lifetime().safe_to_destroy()) {
        // Keep the existing terminal abandonment policy for compute children.
        static_cast<void>(pipeline_cache_.release());
        static_cast<void>(memory_pool_.release());
    }
    pipeline_cache_.reset();
    memory_pool_.reset();
    queue_synchronization_owner_.reset();
    native_capabilities_.reset();
    // Native FFmpeg references can outlive Device and retain the device.
    device_core_.reset();
}

Result<std::unique_ptr<Device>> Device::create(const DeviceSelector &selector,
                                               std::string_view shader_manifest_sha256,
                                               VideoImageLayoutPolicy video_image_layout_policy, DeviceRequirements requirements) {
    const auto canonical_uuid = device_canonicalize_uuid(selector.uuid);
    if (!canonical_uuid) {
        return Result<std::unique_ptr<Device>>::failure(
            {ErrorCode::invalid_argument,
             "device UUID must contain exactly 32 hexadecimal digits"});
    }

    const auto runtime_result = Runtime::acquire();
    if (!runtime_result) {
        return Result<std::unique_ptr<Device>>::failure(runtime_result.error());
    }
    const auto &runtime = runtime_result.value();

    auto candidates_result = runtime->enumerate_candidates();
    if (!candidates_result) {
        return Result<std::unique_ptr<Device>>::failure(candidates_result.error());
    }
    auto candidates = std::move(candidates_result).value();
    if (candidates.empty()) {
        return Result<std::unique_ptr<Device>>::failure(
            {ErrorCode::no_compatible_device,
             "no Vulkan 1.1 discrete or integrated compute GPU is compatible"});
    }

    auto selected = candidates.begin();
    if (!canonical_uuid->empty()) {
        selected = std::find_if(candidates.begin(), candidates.end(),
                                [&](const DeviceCandidate &candidate) {
                                    return candidate.info.uuid == *canonical_uuid;
                                });
        if (selected == candidates.end()) {
            return Result<std::unique_ptr<Device>>::failure(
                {ErrorCode::no_compatible_device,
                 "the requested Vulkan device UUID is not compatible or not present"});
        }
    }

    requirements.video_decode = requirements.video_decode && CROSSVMAF_ENABLE_VULKAN_VIDEO;
    const std::uint32_t compute_only_queue_count =
        std::max(1U, std::min(selected->compute_queue_count, maximum_lane_queue_count));
    auto extensions_result = device_enumerate_device_extensions(selected->physical_device);
    if (!extensions_result) {
        return Result<std::unique_ptr<Device>>::failure(extensions_result.error());
    }
    device_VulkanVideoDeviceConfiguration video_configuration =
        device_query_vulkan_video_configuration(*selected, extensions_result.value(),
                                                video_image_layout_policy, requirements);

    std::optional<VideoQueuePlan> video_queue_plan;
    if (video_configuration.enabled && !video_configuration.video_queue_families.empty()) {
        video_queue_plan =
            plan_video_queues(selected->compute_queue_family, selected->compute_queue_count,
                              video_configuration.video_queue_families,
                              video_configuration.internally_synchronized_queues_features
                                      .internallySynchronizedQueues == VK_TRUE);
        if (!video_queue_plan) video_configuration = {};
    }

    std::vector<std::array<float, maximum_lane_queue_count>> queue_priorities;
    std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
    std::uint32_t runtime_queue_count{};
    VkDeviceQueueGlobalPriorityCreateInfoKHR global_priority_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR,
        nullptr,
        VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR,
    };
    const auto configure_queues = [&](bool enable_global_priority) {
        runtime_queue_count =
            video_queue_plan ? video_queue_plan->runtime_queue_count : compute_only_queue_count;
        const std::size_t family_count =
            video_queue_plan ? video_queue_plan->queue_families.size() : 1U;
        queue_priorities.resize(family_count);
        queue_create_infos.assign(family_count, {});
        for (std::size_t family_index = 0U; family_index < family_count; ++family_index) {
            queue_priorities[family_index].fill(1.0F);
            auto &create_info = queue_create_infos[family_index];
            create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            create_info.flags = video_queue_plan ? video_queue_plan->queue_create_flags : 0U;
            create_info.queueFamilyIndex =
                video_queue_plan ? video_queue_plan->queue_families[family_index].index
                                 : selected->compute_queue_family;
            create_info.queueCount =
                video_queue_plan ? video_queue_plan->queue_families[family_index].queue_count
                                 : runtime_queue_count;
            create_info.pNext = enable_global_priority && create_info.queueFamilyIndex ==
                                                              selected->compute_queue_family
                                    ? &global_priority_info
                                    : nullptr;
            create_info.pQueuePriorities = queue_priorities[family_index].data();
        }
    };
    configure_queues(selected->supports_global_priority);

    VkPhysicalDeviceFeatures supported_base_features{};
    vkGetPhysicalDeviceFeatures(selected->physical_device, &supported_base_features);
    VkPhysicalDeviceFeatures compute_base_features{};
    compute_base_features.shaderInt16 = supported_base_features.shaderInt16;
    VkPhysicalDeviceFeatures enabled_base_features = compute_base_features;
    VkPhysicalDeviceFeatures2 enabled_features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    if (video_configuration.enabled) {
        enabled_base_features = video_configuration.base_features;
        enabled_features2.features = enabled_base_features;
        enabled_features2.pNext = &video_configuration.vulkan_11_features;
        video_configuration.vulkan_11_features.pNext = &video_configuration.vulkan_12_features;
        video_configuration.vulkan_12_features.pNext = &video_configuration.vulkan_13_features;
        device_chain_optional_video_features(video_configuration);
    }

    std::vector<const char *> enabled_extensions;
    if (selected->supports_memory_budget) {
        enabled_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
    if (selected->supports_global_priority) {
        enabled_extensions.push_back(VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME);
    }
    enabled_extensions.insert(enabled_extensions.end(), video_configuration.extension_names.begin(),
                              video_configuration.extension_names.end());
    VkDeviceCreateInfo device_create_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        video_configuration.enabled ? &enabled_features2 : nullptr,
        0,
        static_cast<std::uint32_t>(queue_create_infos.size()),
        queue_create_infos.data(),
        0,
        nullptr,
        static_cast<std::uint32_t>(enabled_extensions.size()),
        enabled_extensions.empty() ? nullptr : enabled_extensions.data(),
        video_configuration.enabled ? nullptr : &enabled_base_features,
    };

    VkDevice device = VK_NULL_HANDLE;
    VkResult create_result =
        vkCreateDevice(selected->physical_device, &device_create_info, nullptr, &device);
    if (create_result != VK_SUCCESS && selected->supports_global_priority) {
        for (VkDeviceQueueCreateInfo &create_info : queue_create_infos) {
            create_info.pNext = nullptr;
        }
        enabled_extensions.erase(std::remove_if(enabled_extensions.begin(),
                                                enabled_extensions.end(),
                                                [](const char *name) {
                                                    return std::string_view(name) ==
                                                           VK_KHR_GLOBAL_PRIORITY_EXTENSION_NAME;
                                                }),
                                 enabled_extensions.end());
        device_create_info.enabledExtensionCount =
            static_cast<std::uint32_t>(enabled_extensions.size());
        device_create_info.ppEnabledExtensionNames =
            enabled_extensions.empty() ? nullptr : enabled_extensions.data();
        device = VK_NULL_HANDLE;
        create_result =
            vkCreateDevice(selected->physical_device, &device_create_info, nullptr, &device);
    }
    if (create_result != VK_SUCCESS && video_configuration.enabled &&
        video_queue_plan && video_queue_plan->queue_create_flags != 0U) {
        // Retry the same queues with FFmpeg's shared host locks. Removing the
        // optional driver feature never requires more queues or fewer lanes.
        video_queue_plan->queue_create_flags = 0U;
        video_configuration.internally_synchronized_queues_features = {};
        device_chain_optional_video_features(video_configuration);
        const auto is_internal_sync_extension = [](const char *name) {
            return std::string_view(name) == VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME;
        };
        std::erase_if(enabled_extensions, is_internal_sync_extension);
        std::erase_if(video_configuration.extension_names, is_internal_sync_extension);
        for (auto &create_info : queue_create_infos) {
            create_info.flags = 0U;
        }
        device_create_info.enabledExtensionCount =
            static_cast<std::uint32_t>(enabled_extensions.size());
        device_create_info.ppEnabledExtensionNames = enabled_extensions.data();
        device = VK_NULL_HANDLE;
        create_result =
            vkCreateDevice(selected->physical_device, &device_create_info, nullptr, &device);
    }
    if (create_result != VK_SUCCESS && video_configuration.enabled) {
        // Drop all optional video features, extensions, and queue flags if
        // their device creation fails. Software decode still uses the original
        // compute-only device, including on drivers with partial support.
        video_configuration = {};
        video_queue_plan.reset();
        configure_queues(false);

        enabled_base_features = compute_base_features;
        enabled_extensions.clear();
        if (selected->supports_memory_budget) {
            enabled_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
        device_create_info.pNext = nullptr;
        device_create_info.queueCreateInfoCount = 1U;
        device_create_info.pQueueCreateInfos = queue_create_infos.data();
        device_create_info.enabledExtensionCount =
            static_cast<std::uint32_t>(enabled_extensions.size());
        device_create_info.ppEnabledExtensionNames =
            enabled_extensions.empty() ? nullptr : enabled_extensions.data();
        device_create_info.pEnabledFeatures = &enabled_base_features;
        device = VK_NULL_HANDLE;
        create_result =
            vkCreateDevice(selected->physical_device, &device_create_info, nullptr, &device);
    }
    if (create_result != VK_SUCCESS) {
        const ErrorCode code = create_result == VK_ERROR_DEVICE_LOST ? ErrorCode::device_lost
                                                                     : ErrorCode::backend_error;
        return Result<std::unique_ptr<Device>>::failure(
            {code, device_result_message("vkCreateDevice", create_result)});
    }

    VolkDeviceTable table{};
    volkLoadDeviceTable(&table, device);
    const PFN_vkGetDeviceQueue2 get_device_queue = table.vkGetDeviceQueue2;
    if (table.vkDestroyDevice == nullptr || get_device_queue == nullptr) {
        vkDestroyDevice(device, nullptr);
        return Result<std::unique_ptr<Device>>::failure(
            {ErrorCode::backend_error, "failed to load required Vulkan device functions"});
    }
    device_PendingDevice pending_device(table, device);
    std::array<VkQueue, maximum_lane_queue_count> lane_queues{};
    std::array<std::uint32_t, maximum_lane_queue_count> lane_queue_indices{};
    // Use the actual creation flags after any compute-only retry, not the
    // originally requested flags (VUID-VkDeviceQueueInfo2-flags-06225).
    const VkDeviceQueueCreateFlags queue_create_flags = queue_create_infos.front().flags;
    for (std::uint32_t lane_index = 0U; lane_index < runtime_queue_count; ++lane_index) {
        const std::uint32_t queue_index = lane_index;
        const VkDeviceQueueInfo2 queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2, nullptr,
                                            queue_create_flags, selected->compute_queue_family,
                                            queue_index};
        get_device_queue(device, &queue_info, &lane_queues[lane_index]);
        if (lane_queues[lane_index] == VK_NULL_HANDLE) {
            return Result<std::unique_ptr<Device>>::failure(
                {ErrorCode::backend_error, "vkGetDeviceQueue2 returned an empty compute queue"});
        }
        lane_queue_indices[lane_index] = queue_index;
    }

    // make_shared allocates before the noexcept constructor adopts the device.
    // Until then device_PendingDevice is the sole native owner.
    auto device_core = std::make_shared<DeviceCore>(runtime, device, table);
    pending_device.release();
    auto context = std::unique_ptr<Device>(
        new Device(std::move(*selected), std::move(device_core), lane_queues, lane_queue_indices,
                   runtime_queue_count, enabled_base_features.shaderInt16 == VK_TRUE,
                   video_configuration.unified_video_image_layouts));
    if (video_configuration.enabled) {
        NativeDeviceCapabilities decode_config{};
        decode_config.device_lifetime = context->device_core_;
        decode_config.get_instance_proc_address = vkGetInstanceProcAddr;
        decode_config.instance = runtime->instance();
        decode_config.physical_device = context->candidate_.physical_device;
        decode_config.device = device;
        decode_config.enabled_base_features = enabled_base_features;
        decode_config.enabled_vulkan_11_features = video_configuration.vulkan_11_features;
        decode_config.enabled_vulkan_12_features = video_configuration.vulkan_12_features;
        decode_config.enabled_vulkan_13_features = video_configuration.vulkan_13_features;
        decode_config.enabled_host_image_copy_features =
            video_configuration.host_image_copy_features;
        decode_config.enabled_host_image_copy_features.pNext = nullptr;
#ifdef VK_KHR_video_decode_vp9
        decode_config.enabled_video_decode_vp9_features =
            video_configuration.video_decode_vp9_features;
#endif
        decode_config.enabled_unified_image_layouts_features =
            video_configuration.unified_image_layouts_features;
        decode_config.enabled_internally_synchronized_queues_features =
            video_configuration.internally_synchronized_queues_features;
        decode_config.queue_create_flags = queue_create_flags;
        decode_config.unified_video_image_layouts = video_configuration.unified_video_image_layouts;
        decode_config.enabled_device_extensions.reserve(enabled_extensions.size());
        for (const char *extension_name : enabled_extensions) {
            decode_config.enabled_device_extensions.emplace_back(extension_name);
        }
        context->video_decode_enabled_ = video_queue_plan.has_value();
        decode_config.queue_families = video_queue_plan ? video_queue_plan->queue_families :
            std::vector<NativeQueueFamily>{{context->candidate_.compute_queue_family, runtime_queue_count,
                                           static_cast<VkQueueFlagBits>(VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT), 0U}};
        decode_config.enabled_vulkan_11_features.pNext = nullptr;
        decode_config.enabled_vulkan_12_features.pNext = nullptr;
        decode_config.enabled_vulkan_13_features.pNext = nullptr;
#ifdef VK_KHR_video_decode_vp9
        decode_config.enabled_video_decode_vp9_features.pNext = nullptr;
#endif
        decode_config.enabled_unified_image_layouts_features.pNext = nullptr;
        decode_config.enabled_internally_synchronized_queues_features.pNext = nullptr;
        context->native_capabilities_ = std::move(decode_config);
    }
    auto memory_pool = DeviceMemoryPool::create(
        context->candidate_.physical_device, device, context->device_core_->table(),
        context->device_core_->resource_lifetime(), context->candidate_.supports_memory_budget);
    if (!memory_pool) {
        return Result<std::unique_ptr<Device>>::failure(memory_pool.error());
    }
    context->memory_pool_ = std::move(memory_pool).value();
    context->pipeline_cache_ =
        PersistentPipelineCache::create(context->device_view(), shader_manifest_sha256);
    const auto self_test = run_device_self_test(context->device_view());
    if (!self_test) {
        const ErrorCode code = self_test.error().code == ErrorCode::device_lost
                                   ? ErrorCode::device_lost
                                   : ErrorCode::device_self_test_failed;
        return Result<std::unique_ptr<Device>>::failure(
            {code, "Vulkan integer self-test failed: " + self_test.error().message});
    }

    return Result<std::unique_ptr<Device>>::success(std::move(context));
}

} // namespace crossvmaf::gpu
