#include "video/transfers/d3d11_vulkan.hpp"
#include "video/backends/ffmpeg_device.hpp"
#include "video/backends/vulkan.hpp"
#include "video/ffmpeg.hpp"
extern "C" {
#include <libavutil/hwcontext_vulkan.h>
}
#if defined(_WIN32)
#include <d3d11.h>
#include <dxgi1_2.h>
#include <vulkan/vulkan_win32.h>
#include <wrl/client.h>
extern "C" {
#include <libavutil/hwcontext_d3d11va.h>
}
#endif
namespace crossvmaf::video {
namespace {
Error unavailable(std::string message) {
    return {ErrorCode::hardware_decode_unavailable, std::move(message)};
}
struct FrameFree {
    void operator()(AVFrame *frame) const { av_frame_free(&frame); }
};
using OwnedFrame = std::unique_ptr<AVFrame, FrameFree>;
} // namespace
class D3d11VulkanTransfer::Impl final {
  public:
#if defined(_WIN32)
    std::shared_ptr<VulkanDecodeDevice> device_owner;
    AVBufferRef *native_device{};
    AVVulkanDeviceContext *vk{};
    VolkDeviceTable table{};
    VkImage image{};
    VkDeviceMemory memory{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkFence fence{};
    VkQueue queue{};
    std::uint32_t family{};
    bool submitted{};
    HANDLE shared_handle{};
    Microsoft::WRL::ComPtr<ID3D11Texture2D> shared_texture;
    Microsoft::WRL::ComPtr<IDXGIKeyedMutex> keyed_mutex;
    OwnedFrame source_owner;
    D3D11_TEXTURE2D_DESC description{};

    ~Impl() {
        if (vk) {
            // Only this transfer uses the scratch image. Compute consumers own
            // independent destination frames and native-device references.
            if (submitted)
                table.vkWaitForFences(vk->act_dev, 1, &fence, VK_TRUE, UINT64_MAX);
            source_owner.reset();
            if (fence)
                table.vkDestroyFence(vk->act_dev, fence, nullptr);
            if (pool)
                table.vkDestroyCommandPool(vk->act_dev, pool, nullptr);
            if (image)
                table.vkDestroyImage(vk->act_dev, image, nullptr);
            if (memory)
                table.vkFreeMemory(vk->act_dev, memory, nullptr);
        }
        // KMT handles are owned by the retained D3D11 resource, not CloseHandle.
        av_buffer_unref(&native_device);
    }
    Result<void> initialize(const AVFrame &child, std::shared_ptr<VulkanDecodeDevice> device) {
        device_owner = std::move(device);
        native_device = static_cast<AVBufferRef *>(device_owner->retain_native_device_context());
        if (!native_device)
            return Result<void>::failure(
                {ErrorCode::backend_error, "retaining D3D11 transfer device"});
        vk = static_cast<AVVulkanDeviceContext *>(
            reinterpret_cast<AVHWDeviceContext *>(native_device->data)->hwctx);
        volkLoadDeviceTable(&table, vk->act_dev);
        bool memory_extension = false, mutex_extension = false;
        for (int i = 0; i < vk->nb_enabled_dev_extensions; ++i) {
            memory_extension |=
                std::string_view(vk->enabled_dev_extensions[i]) == "VK_KHR_external_memory_win32";
            mutex_extension |=
                std::string_view(vk->enabled_dev_extensions[i]) == "VK_KHR_win32_keyed_mutex";
        }
        if (!memory_extension || !mutex_extension)
            return Result<void>::failure(unavailable(
                "D3D11 GPU transfer requires external_memory_win32 and win32_keyed_mutex"));
        auto *texture = reinterpret_cast<ID3D11Texture2D *>(child.data[0]);
        if (!texture || !child.hw_frames_ctx)
            return Result<void>::failure(unavailable("D3D11 returned no D3D11 texture context"));
        texture->GetDesc(&description);
        if (description.Format != DXGI_FORMAT_NV12 && description.Format != DXGI_FORMAT_P010)
            return Result<void>::failure(unavailable("D3D11 shared transfer requires NV12/P010"));
        description.ArraySize = description.MipLevels = 1;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.CPUAccessFlags = 0;
        description.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        description.MiscFlags =
            D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
        auto *frames = reinterpret_cast<AVHWFramesContext *>(child.hw_frames_ctx->data);
        auto *d3d = static_cast<AVD3D11VADeviceContext *>(frames->device_ctx->hwctx);
        HRESULT hr = d3d->device->CreateTexture2D(&description, nullptr, &shared_texture);
        if (FAILED(hr))
            return Result<void>::failure(
                unavailable("creating shared D3D11 texture failed: " + std::to_string(hr)));
        Microsoft::WRL::ComPtr<IDXGIResource> resource;
        if (FAILED(shared_texture.As(&resource)) || FAILED(shared_texture.As(&keyed_mutex)) ||
            FAILED(resource->GetSharedHandle(&shared_handle)))
            return Result<void>::failure(unavailable("exporting shared D3D11 texture failed"));
        const auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties2>(
            vk->get_proc_addr(vk->inst, "vkGetPhysicalDeviceImageFormatProperties2"));
        VkPhysicalDeviceExternalImageFormatInfo external{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO, nullptr,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT};
        VkPhysicalDeviceImageFormatInfo2 format{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2};
        format.pNext = &external;
        format.format = description.Format == DXGI_FORMAT_P010
                            ? VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16
                            : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
        format.type = VK_IMAGE_TYPE_2D;
        format.tiling = VK_IMAGE_TILING_OPTIMAL;
        format.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        format.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
        VkExternalImageFormatProperties external_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
        VkImageFormatProperties2 properties{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                                            &external_properties};
        if (!query || query(vk->phys_dev, &format, &properties) != VK_SUCCESS ||
            !(external_properties.externalMemoryProperties.externalMemoryFeatures &
              VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT))
            return Result<void>::failure(
                unavailable("Vulkan driver cannot import D3D11 shared texture format"));
        VkExternalMemoryImageCreateInfo external_image{
            VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO, nullptr,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT};
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &external_image};
        image_info.flags = format.flags;
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format.format;
        image_info.extent = {description.Width, description.Height, 1};
        image_info.mipLevels = image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = format.usage;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkResult result = table.vkCreateImage(vk->act_dev, &image_info, nullptr, &image);
        if (result != VK_SUCCESS)
            return Result<void>::failure(
                unavailable("creating imported D3D11 image failed: " + std::to_string(result)));
        const auto get_device_proc = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            vk->get_proc_addr(vk->inst, "vkGetDeviceProcAddr"));
        const auto handle_properties = reinterpret_cast<PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            get_device_proc(vk->act_dev, "vkGetMemoryWin32HandlePropertiesKHR"));
        VkMemoryWin32HandlePropertiesKHR handle_info{
            VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
        if (!handle_properties ||
            handle_properties(vk->act_dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT,
                              shared_handle, &handle_info) != VK_SUCCESS)
            return Result<void>::failure(unavailable("querying D3D11 shared texture memory failed"));
        VkMemoryRequirements requirements{};
        table.vkGetImageMemoryRequirements(vk->act_dev, image, &requirements);
        const auto compatible = requirements.memoryTypeBits & handle_info.memoryTypeBits;
        if (!compatible)
            return Result<void>::failure(unavailable("D3D11/Vulkan memory types are incompatible"));
        std::uint32_t type = 0;
        while (!(compatible & (1U << type)))
            ++type;
        VkMemoryDedicatedAllocateInfo dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
                                                nullptr, image, VK_NULL_HANDLE};
        VkImportMemoryWin32HandleInfoKHR import_info{
            VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &dedicated,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_KMT_BIT, shared_handle, nullptr};
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, &import_info,
                                        requirements.size, type};
        result = table.vkAllocateMemory(vk->act_dev, &allocation, nullptr, &memory);
        if (result == VK_SUCCESS)
            result = table.vkBindImageMemory(vk->act_dev, image, memory, 0);
        if (result != VK_SUCCESS)
            return Result<void>::failure(
                unavailable("importing D3D11 shared memory failed: " + std::to_string(result)));
        bool found = false;
        for (int i = 0; i < vk->nb_qf; ++i)
            if (vk->qf[i].flags & VK_QUEUE_COMPUTE_BIT) {
                family = static_cast<std::uint32_t>(vk->qf[i].idx);
                found = true;
                break;
            }
        if (!found)
            return Result<void>::failure(unavailable("D3D11 transfer has no shared compute queue"));
        VkDeviceQueueInfo2 queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2, nullptr,
                                      vk->queue_flags, family, 0};
        table.vkGetDeviceQueue2(vk->act_dev, &queue_info, &queue);
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr,
                                          VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, family};
        result = table.vkCreateCommandPool(vk->act_dev, &pool_info, nullptr, &pool);
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                 nullptr, pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
        if (result == VK_SUCCESS)
            result = table.vkAllocateCommandBuffers(vk->act_dev, &command_info, &command);
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (result == VK_SUCCESS)
            result = table.vkCreateFence(vk->act_dev, &fence_info, nullptr, &fence);
        if (result != VK_SUCCESS)
            return Result<void>::failure(
                {ErrorCode::backend_error, "allocating D3D11 transfer synchronization failed"});
        return Result<void>::success();
    }
    Result<AVFrame *> copy(const AVFrame &child, const AVFrame &source,
                           AVBufferRef *destination_frames) {
        if (submitted) {
            const auto result = table.vkWaitForFences(vk->act_dev, 1, &fence, VK_TRUE, UINT64_MAX);
            if (result != VK_SUCCESS)
                return Result<AVFrame *>::failure(
                    {ErrorCode::device_lost, "waiting for D3D11 transfer failed"});
            submitted = false;
            source_owner.reset();
        }
        OwnedFrame output(av_frame_alloc());
        if (!output)
            return Result<AVFrame *>::failure(
                {ErrorCode::backend_error, "allocating D3D11 transfer frame"});
        int error = av_hwframe_get_buffer(destination_frames, output.get(), 0);
        if (error < 0)
            return Result<AVFrame *>::failure(
                unavailable(ffmpeg_error_message(error, "allocating D3D11 Vulkan destination")));
        error = av_frame_copy_props(output.get(), &source);
        if (error < 0)
            return Result<AVFrame *>::failure({ErrorCode::backend_error, "copying D3D11 metadata"});
        auto *frame = reinterpret_cast<AVVkFrame *>(output->data[0]);
        std::uint32_t image_count = 0;
        while (image_count < AV_NUM_DATA_POINTERS && frame->img[image_count])
            ++image_count;
        if (image_count < 1 || image_count > 2)
            return Result<AVFrame *>::failure(
                unavailable("unsupported D3D11 destination image arrangement"));
        std::array<VkImageMemoryBarrier2, 3> before{};
        before[0] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
        before[0].dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        before[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        before[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        before[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        before[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        before[0].dstQueueFamilyIndex = family;
        before[0].image = image;
        before[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        std::array<std::uint64_t, 2> waits{}, signals{};
        std::array<VkPipelineStageFlags, 2> stages{VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
        for (std::uint32_t i = 0; i < image_count; ++i) {
            if (frame->queue_family[i] != VK_QUEUE_FAMILY_IGNORED &&
                frame->queue_family[i] != family)
                return Result<AVFrame *>::failure(
                    unavailable("D3D11 destination requires cross-family ownership transfer"));
            auto &barrier = before[i + 1];
            barrier = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            barrier.srcAccessMask = frame->access[i];
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.oldLayout = frame->layout[i];
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = frame->img[i];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            waits[i] = frame->sem_value[i];
            if (waits[i] == UINT64_MAX)
                return Result<AVFrame *>::failure(unavailable("D3D11 transfer timeline exhausted"));
            signals[i] = waits[i] + 1;
        }
        source_owner.reset(av_frame_clone(&source));
        if (!source_owner)
            return Result<AVFrame *>::failure({ErrorCode::backend_error, "retaining D3D11 source"});
        if (table.vkResetCommandPool(vk->act_dev, pool, 0) != VK_SUCCESS)
            return Result<AVFrame *>::failure(
                unavailable("resetting D3D11 transfer commands failed"));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, nullptr,
                                       VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VkResult result = table.vkBeginCommandBuffer(command, &begin);
        if (result != VK_SUCCESS)
            return Result<AVFrame *>::failure(
                unavailable("beginning D3D11 transfer commands failed"));
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.imageMemoryBarrierCount = image_count + 1;
        dependency.pImageMemoryBarriers = before.data();
        table.vkCmdPipelineBarrier2(command, &dependency);
        for (std::uint32_t plane = 0; plane < 2; ++plane) {
            VkImageCopy region{};
            region.srcSubresource = {
                static_cast<VkImageAspectFlags>(plane == 0 ? VK_IMAGE_ASPECT_PLANE_0_BIT
                                                           : VK_IMAGE_ASPECT_PLANE_1_BIT),
                0, 0, 1};
            region.dstSubresource = {image_count == 1 ? region.srcSubresource.aspectMask
                                                      : VK_IMAGE_ASPECT_COLOR_BIT,
                                     0, 0, 1};
            region.extent = {static_cast<std::uint32_t>(source.width) >> plane,
                             static_cast<std::uint32_t>(source.height) >> plane, 1};
            table.vkCmdCopyImage(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                 frame->img[image_count == 1 ? 0 : plane],
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
        auto after = before;
        after[0].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        after[0].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        after[0].dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        after[0].dstAccessMask = 0;
        after[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        after[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        after[0].srcQueueFamilyIndex = family;
        after[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_EXTERNAL;
        for (std::uint32_t i = 0; i < image_count; ++i) {
            after[i + 1].srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            after[i + 1].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            after[i + 1].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            after[i + 1].dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
            after[i + 1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            after[i + 1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
        dependency.pImageMemoryBarriers = after.data();
        table.vkCmdPipelineBarrier2(command, &dependency);
        if (table.vkEndCommandBuffer(command) != VK_SUCCESS)
            return Result<AVFrame *>::failure(unavailable("ending D3D11 transfer commands failed"));
        auto *child_frames = reinterpret_cast<AVHWFramesContext *>(child.hw_frames_ctx->data);
        auto *d3d = static_cast<AVD3D11VADeviceContext *>(child_frames->device_ctx->hwctx);
        const auto acquired = keyed_mutex->AcquireSync(0, 10000);
        if (acquired != S_OK)
            return Result<AVFrame *>::failure(unavailable(
                "acquiring D3D11 transfer keyed mutex failed: " + std::to_string(acquired)));
        d3d->lock(d3d->lock_ctx);
        d3d->device_context->CopySubresourceRegion(
            shared_texture.Get(), 0, 0, 0, 0, reinterpret_cast<ID3D11Texture2D *>(child.data[0]),
            static_cast<UINT>(reinterpret_cast<std::uintptr_t>(child.data[1])), nullptr);
        const auto released = keyed_mutex->ReleaseSync(1);
        d3d->device_context->Flush();
        d3d->unlock(d3d->lock_ctx);
        if (FAILED(released))
            return Result<AVFrame *>::failure(
                unavailable("releasing D3D11 transfer keyed mutex failed"));
        const std::uint64_t acquire_key = 1, release_key = 0;
        const std::uint32_t timeout = 10000;
        VkWin32KeyedMutexAcquireReleaseInfoKHR mutex{
            VK_STRUCTURE_TYPE_WIN32_KEYED_MUTEX_ACQUIRE_RELEASE_INFO_KHR,
            nullptr,
            1,
            &memory,
            &acquire_key,
            &timeout,
            1,
            &memory,
            &release_key};
        VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                                               &mutex,
                                               image_count,
                                               waits.data(),
                                               image_count,
                                               signals.data()};
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            &timeline,
                            image_count,
                            frame->sem,
                            stages.data(),
                            1,
                            &command,
                            image_count,
                            frame->sem};
        if (table.vkResetFences(vk->act_dev, 1, &fence) != VK_SUCCESS)
            return Result<AVFrame *>::failure(
                {ErrorCode::device_lost, "resetting D3D11 transfer fence failed"});
        const auto synchronization = device_owner->queue_synchronization();
        if (synchronization.lock)
            synchronization.lock(synchronization.opaque, family, 0);
        result = table.vkQueueSubmit(queue, 1, &submit, fence);
        if (synchronization.unlock)
            synchronization.unlock(synchronization.opaque, family, 0);
        if (result != VK_SUCCESS)
            return Result<AVFrame *>::failure(
                {ErrorCode::device_lost, "submitting D3D11 Vulkan transfer failed"});
        submitted = true;
        for (std::uint32_t i = 0; i < image_count; ++i) {
            frame->sem_value[i] = signals[i];
            frame->layout[i] = VK_IMAGE_LAYOUT_GENERAL;
            frame->access[i] = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        }
        return Result<AVFrame *>::success(output.release());
    }
#endif
};
D3d11VulkanTransfer::D3d11VulkanTransfer() = default;
D3d11VulkanTransfer::~D3d11VulkanTransfer() {
    impl_.reset();
    av_buffer_unref(&child_frames_);
}
Result<AVFrame *> D3d11VulkanTransfer::transfer(const AVFrame &source,
                                                AVBufferRef *destination_frames,
                                                std::shared_ptr<VulkanDecodeDevice> device) {
#if defined(_WIN32)
    OwnedFrame child(av_frame_alloc());
    if (!child)
        return Result<AVFrame *>::failure({ErrorCode::backend_error, "allocating D3D11 child frame"});
    if (!child_frames_) {
        AVBufferRef *child_device = nullptr;
        auto *source_frames = reinterpret_cast<AVHWFramesContext *>(source.hw_frames_ctx->data);
        int error = av_hwdevice_ctx_create_derived(&child_device, AV_HWDEVICE_TYPE_D3D11VA,
                                                   source_frames->device_ref, 0);
        if (error >= 0 && source.format == AV_PIX_FMT_AMF_SURFACE) {
            // AMF's av_hwframe_map maps CPU memory, not a D3D11 surface.
            // Retain the actual child device and borrow the texture directly.
            child_frames_ = av_hwframe_ctx_alloc(child_device);
            if (!child_frames_) {
                error = AVERROR(ENOMEM);
            } else {
                auto *frames = reinterpret_cast<AVHWFramesContext *>(child_frames_->data);
                frames->format = AV_PIX_FMT_D3D11;
                frames->sw_format = source_frames->sw_format;
                frames->width = source_frames->width;
                frames->height = source_frames->height;
                error = av_hwframe_ctx_init(child_frames_);
            }
        } else if (error >= 0)
            error = av_hwframe_ctx_create_derived(&child_frames_, AV_PIX_FMT_D3D11, child_device,
                                                  source.hw_frames_ctx,
                                                  AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
        av_buffer_unref(&child_device);
        if (error < 0) {
            av_buffer_unref(&child_frames_);
            return Result<AVFrame *>::failure(
                {error == AVERROR(ENOMEM) ? ErrorCode::backend_error : ErrorCode::hardware_decode_unavailable,
                 ffmpeg_error_message(error, "deriving decoder D3D11 frames")});
        }
    }
    child->hw_frames_ctx = av_buffer_ref(child_frames_);
    if (!child->hw_frames_ctx)
        return Result<AVFrame *>::failure({ErrorCode::backend_error, "retaining D3D11 frames"});
    child->format = AV_PIX_FMT_D3D11;
    if (source.format == AV_PIX_FMT_AMF_SURFACE) {
        const auto bound = bind_amf_d3d11_surface(source, *child);
        if (!bound) return Result<AVFrame *>::failure(bound.error());
    } else {
        const int error =
            av_hwframe_map(child.get(), &source, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
        if (error < 0)
            return Result<AVFrame *>::failure(
                unavailable(ffmpeg_error_message(error, "mapping QSV to D3D11")));
    }
    if (!impl_) {
        auto pending = std::make_unique<Impl>();
        const auto ready = pending->initialize(*child, std::move(device));
        if (!ready)
            return Result<AVFrame *>::failure(ready.error());
        impl_ = std::move(pending);
    }
    auto result = impl_->copy(*child, source, destination_frames);
    if (result) {
        const std::uint64_t bytes_per_pixel =
            impl_->description.Format == DXGI_FORMAT_P010 ? 2U : 1U;
        last_copy_bytes_ =
            (static_cast<std::uint64_t>(impl_->description.Width) * impl_->description.Height +
             static_cast<std::uint64_t>(source.width) * source.height) *
            3U / 2U * bytes_per_pixel;
    }
    return result;
#else
    static_cast<void>(source);
    static_cast<void>(destination_frames);
    static_cast<void>(device);
    return Result<AVFrame *>::failure(unavailable("D3D11 transfer is Windows-only"));
#endif
}
} // namespace crossvmaf::video
