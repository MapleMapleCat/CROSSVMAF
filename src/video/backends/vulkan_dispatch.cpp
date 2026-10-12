#include "video/backends/vulkan_dispatch.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <new>
#include <utility>

namespace crossvmaf::video {
namespace {

// Vulkan resolvers have no opaque argument. Bound entry points avoid tracking
// command-buffer allocation/free or looking up handles on each decoded frame.
constexpr std::size_t dispatch_capacity = 64U;

struct DispatchState final {
    VkInstance instance{};
    VkDevice device{};
    PFN_vkGetInstanceProcAddr instance_proc{};
    PFN_vkGetDeviceProcAddr device_proc{};
    PFN_vkCmdDecodeVideoKHR decode{};
    PFN_vkGetDeviceQueue get_queue{};
    PFN_vkGetDeviceQueue2 get_queue2{};
    std::optional<VulkanDecodeQueueSelection> queue;
    std::shared_ptr<std::mutex> decode_mutex;
    std::size_t slot{dispatch_capacity};
    ~DispatchState();
};

struct DispatchSlot final {
    std::atomic<DispatchState*> active{};
    std::weak_ptr<DispatchState> owner;
};

struct DispatchRegistry final {
    std::mutex mutex;
    std::array<DispatchSlot, dispatch_capacity> slots;
};

DispatchRegistry& registry() {
    // Native FFmpeg references may be released by other static destructors.
    static auto* const value = new DispatchRegistry;
    return *value;
}

DispatchState::~DispatchState() {
    if (slot == dispatch_capacity) return;
    auto& entries = registry();
    std::lock_guard lock(entries.mutex);
    auto& entry = entries.slots[slot];
    assert(entry.active.load(std::memory_order_relaxed) == this);
    entry.active.store(nullptr, std::memory_order_release);
    entry.owner.reset();
}

template<std::size_t Slot>
DispatchState& state() noexcept {
    auto* const active = registry().slots[Slot].active.load(std::memory_order_acquire);
    assert(active != nullptr);
    return *active;
}

template<std::size_t Slot>
VKAPI_ATTR void VKAPI_CALL decode_video(VkCommandBuffer command,
                                        const VkVideoDecodeInfoKHR* information) {
    auto& dispatch = state<Slot>();
    std::lock_guard lock(*dispatch.decode_mutex);
    dispatch.decode(command, information);
}

template<std::size_t Slot>
VKAPI_ATTR void VKAPI_CALL get_queue(VkDevice device, std::uint32_t family,
                                    std::uint32_t index, VkQueue* queue) {
    auto& dispatch = state<Slot>();
    if (device == dispatch.device && family == dispatch.queue->family && index == 0U)
        index = dispatch.queue->index;
    dispatch.get_queue(device, family, index, queue);
}

template<std::size_t Slot>
VKAPI_ATTR void VKAPI_CALL get_queue2(VkDevice device, const VkDeviceQueueInfo2* info,
                                     VkQueue* queue) {
    auto& dispatch = state<Slot>();
    auto mapped = *info;
    if (device == dispatch.device && mapped.queueFamilyIndex == dispatch.queue->family &&
        mapped.queueIndex == 0U)
        mapped.queueIndex = dispatch.queue->index;
    dispatch.get_queue2(device, &mapped, queue);
}

template<std::size_t Slot>
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL device_proc(VkDevice device, const char* name) {
    auto& dispatch = state<Slot>();
    if (device == dispatch.device && std::strcmp(name, "vkCmdDecodeVideoKHR") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(decode_video<Slot>);
    if (device == dispatch.device && dispatch.queue) {
        if (std::strcmp(name, "vkGetDeviceQueue") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(get_queue<Slot>);
        if (std::strcmp(name, "vkGetDeviceQueue2") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(get_queue2<Slot>);
    }
    return dispatch.device_proc(device, name);
}

template<std::size_t Slot>
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL instance_proc(VkInstance instance, const char* name) {
    auto& dispatch = state<Slot>();
    if (instance == dispatch.instance && std::strcmp(name, "vkGetDeviceProcAddr") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(device_proc<Slot>);
    return dispatch.instance_proc(instance, name);
}

template<std::size_t... Slots>
constexpr auto make_entry_points(std::index_sequence<Slots...>) noexcept {
    return std::array<PFN_vkGetInstanceProcAddr, sizeof...(Slots)>{instance_proc<Slots>...};
}

constexpr auto entry_points = make_entry_points(std::make_index_sequence<dispatch_capacity>{});

} // namespace

Result<VulkanDecodeDispatch>
make_vulkan_decode_dispatch(VkInstance instance, VkDevice device,
                           PFN_vkGetInstanceProcAddr get_instance_proc_address,
                           std::optional<VulkanDecodeQueueSelection> queue) {
    if (!instance || !device || !get_instance_proc_address)
        return Result<VulkanDecodeDispatch>::failure(
            {ErrorCode::invalid_argument, "Vulkan decode dispatch requires a device and resolver"});

    const auto get_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        get_instance_proc_address(instance, "vkGetDeviceProcAddr"));
    const auto decode = get_device_proc ? reinterpret_cast<PFN_vkCmdDecodeVideoKHR>(
        get_device_proc(device, "vkCmdDecodeVideoKHR")) : nullptr;
    if (!decode)
        return Result<VulkanDecodeDispatch>::failure(
            {ErrorCode::hardware_decode_unavailable, "Vulkan resolver has no decode command"});

    try {
        auto owner = std::make_shared<DispatchState>();
        owner->instance = instance;
        owner->device = device;
        owner->instance_proc = get_instance_proc_address;
        owner->device_proc = get_device_proc;
        owner->decode = decode;
        owner->queue = queue;
        if (queue) {
            owner->get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(
                get_device_proc(device, "vkGetDeviceQueue"));
            owner->get_queue2 = reinterpret_cast<PFN_vkGetDeviceQueue2>(
                get_device_proc(device, "vkGetDeviceQueue2"));
            if (!owner->get_queue || !owner->get_queue2)
                return Result<VulkanDecodeDispatch>::failure(
                    {ErrorCode::hardware_decode_unavailable, "Vulkan resolver has no queue accessors"});
        }
        auto& entries = registry();
        std::lock_guard lock(entries.mutex);
        std::size_t vacant = dispatch_capacity;
        for (std::size_t slot = 0U; slot < dispatch_capacity; ++slot) {
            auto& entry = entries.slots[slot];
            const auto* const active = entry.active.load(std::memory_order_relaxed);
            if (!active) {
                if (vacant == dispatch_capacity) vacant = slot;
                continue;
            }
            if (active->device != device) continue;
            if (active->instance != instance || active->instance_proc != get_instance_proc_address)
                return Result<VulkanDecodeDispatch>::failure(
                    {ErrorCode::invalid_argument, "Vulkan device has conflicting decode resolvers"});
            owner->decode_mutex = active->decode_mutex;
            if (active->queue != queue) continue;
            if (auto existing = entry.owner.lock()) {
                // Wrappers for the same native device must share the mutex.
                return Result<VulkanDecodeDispatch>::success({std::move(existing), entry_points[slot]});
            }
        }
        if (vacant == dispatch_capacity)
            return Result<VulkanDecodeDispatch>::failure(
                {ErrorCode::hardware_decode_unavailable, "Vulkan decode dispatch device slots exhausted"});
        if (!owner->decode_mutex) owner->decode_mutex = std::make_shared<std::mutex>();
        owner->slot = vacant;
        entries.slots[vacant].owner = owner;
        entries.slots[vacant].active.store(owner.get(), std::memory_order_release);
        return Result<VulkanDecodeDispatch>::success({std::move(owner), entry_points[vacant]});
    } catch (const std::bad_alloc&) {
        return Result<VulkanDecodeDispatch>::failure(
            {ErrorCode::backend_error, "failed to allocate Vulkan decode dispatch"});
    }
}

} // namespace crossvmaf::video
