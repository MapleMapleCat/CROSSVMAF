#include "video/backends/vulkan.hpp"
#include "video/backends/vulkan_dispatch.hpp"
#include "video/ffmpeg_abi.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <memory>
#include <new>
#include <string>
#include <utility>

extern "C" {
#include <libavutil/buffer.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
}

namespace crossvmaf::video {

bool allow_automatic_hardware_decode(const DeviceInfo &device) noexcept {
    // Main10 and AV1 remain unavailable even with serialized decode commands.
    return !requires_serial_decode_commands(device);
}

bool requires_serial_decode_commands(const DeviceInfo &device) noexcept {
#if defined(_WIN32)
    // Concurrent vkCmdDecodeVideoKHR calls corrupt HEVC output on this driver.
    // Queue submission locks do not protect command recording.
    return device.vendor_id == 0x8086U && device.device_id == 0x7d51U &&
           device.driver_version == 1663775U;
#else
    static_cast<void>(device);
    return false;
#endif
}

namespace {

[[nodiscard]] std::string ffmpeg_device_error(int error_code) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> error_buffer{};
    if (av_strerror(error_code, error_buffer.data(), error_buffer.size()) < 0) {
        return "FFmpeg error " + std::to_string(error_code);
    }
    return error_buffer.data();
}

[[nodiscard]] Result<void> validate_config(const VulkanDecodeDeviceConfig &config) {
    if (!config.device_lifetime) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan hardware decoding requires an owning device lifetime"});
    }
    if (config.get_instance_proc_address == nullptr || config.instance == VK_NULL_HANDLE ||
        config.physical_device == VK_NULL_HANDLE || config.device == VK_NULL_HANDLE) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "Vulkan hardware decoding requires complete device handles"});
    }
    if (config.queue_families.empty() || config.queue_families.size() > 64U) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "Vulkan hardware decoding requires 1..64 queue families"});
    }
    for (const VulkanDecodeQueueFamily &queue_family : config.queue_families) {
        if (queue_family.queue_count == 0U || queue_family.flags == 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "Vulkan hardware decoding received an invalid queue family"});
        }
    }
    const bool host_image_copy_extension_enabled =
        std::find(config.enabled_device_extensions.begin(), config.enabled_device_extensions.end(),
                  VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME) != config.enabled_device_extensions.end();
    if (host_image_copy_extension_enabled !=
        (config.enabled_host_image_copy_features.hostImageCopy == VK_TRUE)) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan host image copy requires both the enabled extension and feature"});
    }
    const bool internal_feature_enabled =
        config.enabled_internally_synchronized_queues_features.internallySynchronizedQueues ==
        VK_TRUE;
    const bool internal_extension_enabled =
        std::find(config.enabled_device_extensions.begin(), config.enabled_device_extensions.end(),
                  VK_KHR_INTERNALLY_SYNCHRONIZED_QUEUES_EXTENSION_NAME) !=
        config.enabled_device_extensions.end();
    const bool requests_internal_synchronization =
        internal_feature_enabled || internal_extension_enabled || config.queue_create_flags != 0U;
    if (requests_internal_synchronization &&
        (!internal_feature_enabled || !internal_extension_enabled ||
         config.queue_create_flags != VK_DEVICE_QUEUE_CREATE_INTERNALLY_SYNCHRONIZED_BIT_KHR)) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "shared Vulkan decode queues require the internal synchronization extension, enabled "
             "feature, and matching queue creation flag"});
    }
#if !FF_API_VULKAN_SYNC_QUEUES
    // Never silently lose mutual exclusion when upgrading the pinned FFmpeg.
    if (!requests_internal_synchronization) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "this FFmpeg version does not provide default queue synchronization"});
    }
#endif
    if (config.unified_video_image_layouts &&
        (config.enabled_unified_image_layouts_features.unifiedImageLayouts != VK_TRUE ||
         config.enabled_unified_image_layouts_features.unifiedImageLayoutsVideo != VK_TRUE ||
         std::find(config.enabled_device_extensions.begin(), config.enabled_device_extensions.end(),
                   VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME) ==
             config.enabled_device_extensions.end())) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "unified video image layouts require the enabled extension and both features"});
    }
    return Result<void>::success();
}

} // namespace

class VulkanDecodeDevice::QueueSynchronization final {
  public:
    explicit QueueSynchronization(AVHWDeviceContext *context) noexcept : context_(context) {
#if FF_API_VULKAN_SYNC_QUEUES
        const auto *vulkan_context = static_cast<const AVVulkanDeviceContext *>(context->hwctx);
        // FFmpeg installs its default per-queue mutex callbacks during init.
        // Reading them is the legacy compatibility boundary, not a replacement
        // mutex bank: runtime submissions must take exactly the same locks.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
        lock_ = vulkan_context->lock_queue;
        unlock_ = vulkan_context->unlock_queue;
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif
    }

    [[nodiscard]] bool complete() const noexcept { return lock_ != nullptr && unlock_ != nullptr; }

    [[nodiscard]] VulkanQueueSynchronization callbacks() noexcept {
        return {this,
                [](void *opaque, std::uint32_t family, std::uint32_t index) noexcept {
                    auto &synchronization = *static_cast<QueueSynchronization *>(opaque);
                    synchronization.lock_(synchronization.context_, family, index);
                },
                [](void *opaque, std::uint32_t family, std::uint32_t index) noexcept {
                    auto &synchronization = *static_cast<QueueSynchronization *>(opaque);
                    synchronization.unlock_(synchronization.context_, family, index);
                }};
    }

  private:
    using QueueCallback = void (*)(AVHWDeviceContext *, std::uint32_t, std::uint32_t);
    AVHWDeviceContext *context_{};
    QueueCallback lock_{};
    QueueCallback unlock_{};
};

class VulkanDecodeDevice::NativeState final {
  public:
    explicit NativeState(VulkanDecodeDeviceConfig config) noexcept : config_(std::move(config)) {}
    ~NativeState() { av_buffer_unref(&queue_owner_); }

    static void lock_queue(AVHWDeviceContext* context, std::uint32_t family, std::uint32_t index) {
        auto& state = *static_cast<NativeState*>(context->user_opaque);
        if (state.queue_ && family == state.queue_->family && index == 0U) index = state.queue_->index;
        const auto& callbacks = state.parent_queue_synchronization_;
        callbacks.lock(callbacks.opaque, family, index);
    }

    static void unlock_queue(AVHWDeviceContext* context, std::uint32_t family, std::uint32_t index) {
        auto& state = *static_cast<NativeState*>(context->user_opaque);
        if (state.queue_ && family == state.queue_->family && index == 0U) index = state.queue_->index;
        const auto& callbacks = state.parent_queue_synchronization_;
        callbacks.unlock(callbacks.opaque, family, index);
    }

    // Stable storage for all pointers published into AVVulkanDeviceContext.
    // A bound child retains the parent context for its queue callbacks. The
    // parent never owns its children, so this does not create a cycle.
    VulkanDecodeDeviceConfig config_;
    std::vector<const char *> extension_name_pointers_;
    std::unique_ptr<QueueSynchronization> queue_synchronization_;
    VulkanDecodeDispatch decode_dispatch_;
    std::mutex codec_call_mutex_;
    std::atomic<std::uint32_t> next_decode_queue_{};
    std::optional<VulkanDecodeQueueSelection> queue_;
    AVBufferRef* queue_owner_{};
    VulkanQueueSynchronization parent_queue_synchronization_;
};

VulkanDecodeDevice::VulkanDecodeDevice(void *native_device_context, NativeState *state) noexcept
    : native_device_context_(native_device_context), state_(state) {
}

VulkanDecodeDevice::~VulkanDecodeDevice() {
    auto *native_context = static_cast<AVBufferRef *>(native_device_context_);
    av_buffer_unref(&native_context);
    native_device_context_ = nullptr;
}

Result<std::shared_ptr<VulkanDecodeDevice>>
VulkanDecodeDevice::create(VulkanDecodeDeviceConfig config, bool serialize_decode_commands) {
    return create_context(std::move(config), serialize_decode_commands, nullptr, std::nullopt);
}

Result<std::shared_ptr<VulkanDecodeDevice>> VulkanDecodeDevice::create_local_frame_context() {
    auto config = state_->config_;
    std::erase_if(config.enabled_device_extensions, [](const std::string& name) {
        return name == "VK_KHR_external_memory_win32";
    });
    return create_context(std::move(config), false, this, std::nullopt);
}

Result<std::shared_ptr<VulkanDecodeDevice>>
VulkanDecodeDevice::create_context(VulkanDecodeDeviceConfig config, bool serialize_decode_commands,
                                   VulkanDecodeDevice* queue_owner,
                                   std::optional<VulkanDecodeQueueSelection> queue) {
    const auto compatibility = require_ffmpeg_abi();
    if (!compatibility)
        return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(compatibility.error());
    const auto validation = validate_config(config);
    if (!validation) {
        return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(validation.error());
    }

    try {
        auto state_owner = std::make_unique<NativeState>(std::move(config));
        NativeState *const device = state_owner.get();
        device->queue_ = queue;
        if (queue_owner) {
            device->queue_owner_ = static_cast<AVBufferRef*>(queue_owner->retain_native_device_context());
            if (!device->queue_owner_)
                return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(
                    {ErrorCode::backend_error, "failed to retain Vulkan decode queue owner"});
            device->parent_queue_synchronization_ = queue_owner->queue_synchronization();
        }
        if (serialize_decode_commands) {
            auto dispatch = make_vulkan_decode_dispatch(device->config_.instance, device->config_.device,
                                                        device->config_.get_instance_proc_address, queue);
            if (!dispatch) return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(dispatch.error());
            device->decode_dispatch_ = std::move(dispatch).value();
        }
        device->extension_name_pointers_.reserve(device->config_.enabled_device_extensions.size());
        for (const std::string &extension_name : device->config_.enabled_device_extensions) {
            device->extension_name_pointers_.push_back(extension_name.c_str());
        }

        AVBufferRef *native_context = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_VULKAN);
        if (native_context == nullptr) {
            return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(
                {ErrorCode::backend_error, "failed to allocate FFmpeg Vulkan device context"});
        }
        const auto release_native_context = [](AVBufferRef *reference) noexcept {
            av_buffer_unref(&reference);
        };
        std::unique_ptr<AVBufferRef, decltype(release_native_context)> native_owner(
            native_context, release_native_context);

        AVHWDeviceContext *const device_context =
            reinterpret_cast<AVHWDeviceContext *>(native_context->data);
        device_context->user_opaque = state_owner.release();
        // FFmpeg invokes device_uninit and destroys derived frames contexts
        // before this callback; only then may the supplied VkDevice disappear.
        device_context->free = [](AVHWDeviceContext *context) noexcept {
            delete static_cast<NativeState *>(context->user_opaque);
        };
        auto *const vulkan_context = static_cast<AVVulkanDeviceContext *>(device_context->hwctx);
        vulkan_context->get_proc_addr = device->decode_dispatch_.lifetime
                                           ? device->decode_dispatch_.get_instance_proc_address
                                           : device->config_.get_instance_proc_address;
        vulkan_context->inst = device->config_.instance;
        vulkan_context->phys_dev = device->config_.physical_device;
        vulkan_context->act_dev = device->config_.device;
        vulkan_context->device_features = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        vulkan_context->device_features.features = device->config_.enabled_base_features;

        device->config_.enabled_vulkan_11_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        device->config_.enabled_vulkan_12_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        device->config_.enabled_vulkan_13_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        device->config_.enabled_vulkan_11_features.pNext =
            &device->config_.enabled_vulkan_12_features;
        device->config_.enabled_vulkan_12_features.pNext =
            &device->config_.enabled_vulkan_13_features;
        void **feature_chain_tail = &device->config_.enabled_vulkan_13_features.pNext;
        *feature_chain_tail = nullptr;
        if (device->config_.enabled_host_image_copy_features.hostImageCopy == VK_TRUE) {
            device->config_.enabled_host_image_copy_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
            *feature_chain_tail = &device->config_.enabled_host_image_copy_features;
            feature_chain_tail = &device->config_.enabled_host_image_copy_features.pNext;
            *feature_chain_tail = nullptr;
        }
#ifdef VK_KHR_video_decode_vp9
        device->config_.enabled_video_decode_vp9_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_DECODE_VP9_FEATURES_KHR;
        if (device->config_.enabled_video_decode_vp9_features.videoDecodeVP9 == VK_TRUE) {
            *feature_chain_tail = &device->config_.enabled_video_decode_vp9_features;
            feature_chain_tail = &device->config_.enabled_video_decode_vp9_features.pNext;
            *feature_chain_tail = nullptr;
        }
#endif
        device->config_.enabled_unified_image_layouts_features.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR;
        if (device->config_.unified_video_image_layouts) {
            *feature_chain_tail = &device->config_.enabled_unified_image_layouts_features;
            feature_chain_tail = &device->config_.enabled_unified_image_layouts_features.pNext;
        }
        if (device->config_.enabled_internally_synchronized_queues_features
                .internallySynchronizedQueues == VK_TRUE) {
            device->config_.enabled_internally_synchronized_queues_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_INTERNALLY_SYNCHRONIZED_QUEUES_FEATURES_KHR;
            *feature_chain_tail = &device->config_.enabled_internally_synchronized_queues_features;
            feature_chain_tail =
                &device->config_.enabled_internally_synchronized_queues_features.pNext;
        }
        *feature_chain_tail = nullptr;
        vulkan_context->device_features.pNext = &device->config_.enabled_vulkan_11_features;

        vulkan_context->enabled_dev_extensions = device->extension_name_pointers_.empty()
                                                     ? nullptr
                                                     : device->extension_name_pointers_.data();
        vulkan_context->nb_enabled_dev_extensions =
            static_cast<int>(device->extension_name_pointers_.size());
        vulkan_context->queue_flags = device->config_.queue_create_flags;
#if FF_API_VULKAN_SYNC_QUEUES
        if (queue_owner && device->config_.queue_create_flags == 0U) {
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
            vulkan_context->lock_queue = NativeState::lock_queue;
            vulkan_context->unlock_queue = NativeState::unlock_queue;
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
        }
#endif
        vulkan_context->nb_qf = static_cast<int>(device->config_.queue_families.size());
        for (std::size_t queue_index = 0U; queue_index < device->config_.queue_families.size();
             ++queue_index) {
            const VulkanDecodeQueueFamily &source = device->config_.queue_families[queue_index];
            AVVulkanDeviceQueueFamily &destination = vulkan_context->qf[queue_index];
            destination.idx = static_cast<int>(source.index);
            destination.num = static_cast<int>(source.queue_count);
            destination.flags = source.flags;
            destination.video_caps =
                static_cast<VkVideoCodecOperationFlagBitsKHR>(source.video_codec_operations);
        }

        const int initialization_result = av_hwdevice_ctx_init(native_context);
        if (initialization_result < 0) {
            return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(
                {ErrorCode::backend_error, "failed to initialize FFmpeg Vulkan device context: " +
                                               ffmpeg_device_error(initialization_result)});
        }

        if (device->config_.queue_create_flags == 0U) {
            device->queue_synchronization_ = std::make_unique<QueueSynchronization>(device_context);
            if (!device->queue_synchronization_->complete()) {
                return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(
                    {ErrorCode::backend_error,
                     "FFmpeg did not initialize the shared queue synchronization callbacks"});
            }
        }
        auto wrapper =
            std::unique_ptr<VulkanDecodeDevice>(new VulkanDecodeDevice(native_context, device));
        static_cast<void>(native_owner.release());
        return Result<std::shared_ptr<VulkanDecodeDevice>>::success(
            std::shared_ptr<VulkanDecodeDevice>(std::move(wrapper)));
    } catch (const std::bad_alloc &) {
        return Result<std::shared_ptr<VulkanDecodeDevice>>::failure(
            {ErrorCode::backend_error, "failed to allocate Vulkan hardware decoder state"});
    }
}

void *VulkanDecodeDevice::retain_native_device_context() const noexcept {
    auto *native_context = static_cast<AVBufferRef *>(native_device_context_);
    return native_context == nullptr ? nullptr : av_buffer_ref(native_context);
}

Result<VulkanDecodeDevice::DecoderContext>
VulkanDecodeDevice::retain_decoder_context(bool prefer_dedicated_queue) {
    if (prefer_dedicated_queue && serializes_decode_commands() && !state_->queue_) {
        for (const auto& family : state_->config_.queue_families) {
            if (!(family.flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR) ||
                !(family.video_codec_operations & VK_VIDEO_CODEC_OPERATION_DECODE_H264_BIT_KHR))
                continue;
            // Match FFmpeg's first suitable family. A single queue keeps the
            // existing whole-call H.264 scheduling policy.
            if (family.queue_count <= 1U) break;
            try {
                auto config = state_->config_;
                for (auto& selected : config.queue_families)
                    if (selected.index == family.index) selected.queue_count = 1U;
                const VulkanDecodeQueueSelection queue{
                    family.index,
                    state_->next_decode_queue_.fetch_add(1U, std::memory_order_relaxed) % family.queue_count};
                auto child = create_context(std::move(config), true, this, queue);
                if (!child) return Result<DecoderContext>::failure(child.error());
                auto* const native = child.value()->retain_native_device_context();
                if (!native) return Result<DecoderContext>::failure(
                    {ErrorCode::backend_error, "failed to retain dedicated Vulkan decoder context"});
                return Result<DecoderContext>::success({native, true});
            } catch (const std::bad_alloc&) {
                return Result<DecoderContext>::failure(
                    {ErrorCode::backend_error, "failed to allocate dedicated Vulkan decoder context"});
            }
        }
    }
    auto* const native = retain_native_device_context();
    if (!native) return Result<DecoderContext>::failure(
        {ErrorCode::backend_error, "failed to retain FFmpeg Vulkan device context"});
    return Result<DecoderContext>::success({native, false});
}

VulkanQueueSynchronization VulkanDecodeDevice::queue_synchronization() const noexcept {
    return state_->queue_synchronization_ != nullptr ? state_->queue_synchronization_->callbacks()
                                                     : VulkanQueueSynchronization{};
}

bool VulkanDecodeDevice::unified_video_image_layouts() const noexcept {
    return state_->config_.unified_video_image_layouts;
}

bool VulkanDecodeDevice::serializes_decode_commands() const noexcept {
    return static_cast<bool>(state_->decode_dispatch_.lifetime);
}

std::unique_lock<std::mutex> VulkanDecodeDevice::lock_codec_calls() const {
    return serializes_decode_commands() ? std::unique_lock(state_->codec_call_mutex_)
                                       : std::unique_lock<std::mutex>{};
}

} // namespace crossvmaf::video

namespace crossvmaf::video {

std::shared_ptr<VulkanDecodeDevice> prepare_decode_device(gpu::Device &device) {
    if (device.queue_synchronization_owner()) {
        return std::static_pointer_cast<VulkanDecodeDevice>(device.queue_synchronization_owner());
    }
    if (!device.native_capabilities()) return {};
    auto created = VulkanDecodeDevice::create(*device.native_capabilities(),
                                              device.video_decode_enabled() && requires_serial_decode_commands(device.device_info()));
    if (!created) return {};
    auto decoder_device = std::move(created).value();
    const auto synchronization = decoder_device->queue_synchronization();
    device.bind_queue_synchronization(
        decoder_device, {synchronization.opaque, synchronization.lock, synchronization.unlock});
    return decoder_device;
}

} // namespace crossvmaf::video
