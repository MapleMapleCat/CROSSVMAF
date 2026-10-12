#include "video/frame_transfer.hpp"
#include "video/ffmpeg.hpp"
#include "video/transfers/d3d11_vulkan.hpp"
extern "C" {
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext_vulkan.h>
}
namespace crossvmaf::video {
namespace {
struct FrameFree {
    void operator()(AVFrame *frame) const { av_frame_free(&frame); }
};
using OwnedFrame = std::unique_ptr<AVFrame, FrameFree>;
void release_source(void *, std::uint8_t *data) {
    auto *frame = reinterpret_cast<AVFrame *>(data);
    av_frame_free(&frame);
}
Result<AVFrame *> failure(int error, std::string_view operation) {
    return Result<AVFrame *>::failure({error == AVERROR(ENOMEM)
                                           ? ErrorCode::backend_error
                                           : ErrorCode::hardware_decode_unavailable,
                                       ffmpeg_error_message(error, operation)});
}
} // namespace
FrameTransfer::~FrameTransfer() {
    d3d11_.reset();
    av_buffer_unref(&qsv_child_frames_);
    av_buffer_unref(&frames_);
}
Result<AVFrame *> FrameTransfer::gpu(const AVFrame &source,
                                     const std::shared_ptr<VulkanDecodeDevice> &destination) {
    if (!destination || !source.hw_frames_ctx)
        return failure(AVERROR(ENOSYS), "GPU transfer has no compatible Vulkan context");
    const auto *source_context =
        reinterpret_cast<const AVHWFramesContext *>(source.hw_frames_ctx->data);
    if (source_context->sw_format != AV_PIX_FMT_NV12 &&
        source_context->sw_format != AV_PIX_FMT_P010LE)
        return failure(AVERROR(ENOSYS), "GPU transfer requires NV12/P010");
    if (source.crop_top || source.crop_bottom || source.crop_left || source.crop_right)
        return failure(AVERROR(ENOSYS), "GPU transfer does not support unapplied crop offsets");
    if (!frames_) {
        auto allocation_device = destination;
#if defined(_WIN32)
        if (source.format == AV_PIX_FMT_QSV) {
            auto local = destination->create_local_frame_context();
            if (!local) return Result<AVFrame*>::failure(local.error());
            allocation_device_ = std::move(local).value();
            allocation_device = allocation_device_;
        }
#endif
        auto *device = static_cast<AVBufferRef *>(allocation_device->retain_native_device_context());
        if (!device)
            return failure(AVERROR(ENOMEM), "retaining transfer device");
        frames_ = av_hwframe_ctx_alloc(device);
        av_buffer_unref(&device);
        if (!frames_)
            return failure(AVERROR(ENOMEM), "allocating Vulkan transfer pool");
        auto *context = reinterpret_cast<AVHWFramesContext *>(frames_->data);
        context->format = AV_PIX_FMT_VULKAN;
        context->sw_format = source_context->sw_format;
        context->width = source.width;
        context->height = source.height;
        bool separate_planes = source.format == AV_PIX_FMT_CUDA;
#if defined(_WIN32)
        separate_planes = separate_planes || source.format == AV_PIX_FMT_QSV;
#endif
        if (separate_planes) {
            auto* native_frames = static_cast<AVVulkanFramesContext*>(context->hwctx);
            native_frames->flags = static_cast<AVVkFrameFlags>(native_frames->flags | AV_VK_FRAME_FLAG_DISABLE_MULTIPLANE);
        }
        const int error = av_hwframe_ctx_init(frames_);
        if (error < 0) {
            av_buffer_unref(&frames_);
            return failure(error, "initializing Vulkan transfer pool");
        }
    }
#if defined(_WIN32)
    if (source.format == AV_PIX_FMT_QSV) {
        if (!d3d11_)
            d3d11_ = std::make_unique<D3d11VulkanTransfer>();
        auto result = d3d11_->transfer(source, frames_, destination);
        if (result) {
            path_ = DecodeTransferPath::gpu_copy;
            last_gpu_copy_bytes_ = d3d11_->last_copy_bytes();
        }
        return result;
    }
#endif
    OwnedFrame output(av_frame_alloc());
    if (!output)
        return failure(AVERROR(ENOMEM), "allocating transferred frame");
    int error = 0;
    if (source.format == AV_PIX_FMT_CUDA) {
        error = av_hwframe_get_buffer(frames_, output.get(), 0);
        if (error >= 0)
            error = av_frame_copy_props(output.get(), &source);
        if (error < 0)
            return failure(error, "copying CUDA frame to Vulkan");
        // transfer_data schedules asynchronous CUDA work. Keep the source
        // surface alive until the resulting Vulkan frame's final consumer.
        auto *retained = av_frame_clone(&source);
        if (!retained)
            return failure(AVERROR(ENOMEM), "retaining GPU transfer source");
        av_buffer_unref(&output->opaque_ref);
        output->opaque_ref = av_buffer_create(reinterpret_cast<std::uint8_t *>(retained),
                                              sizeof(AVFrame), release_source, nullptr, 0);
        if (!output->opaque_ref) {
            av_frame_free(&retained);
            return failure(AVERROR(ENOMEM), "retaining GPU transfer lifetime");
        }
        error = av_hwframe_transfer_data(output.get(), &source, 0);
        if (error < 0)
            return failure(error, "copying CUDA frame to Vulkan");
        last_gpu_copy_bytes_ = static_cast<std::uint64_t>(source.width) * source.height * 3U / 2U *
                               (source_context->sw_format == AV_PIX_FMT_P010LE ? 2U : 1U);
        path_ = DecodeTransferPath::gpu_copy;
    } else {
        OwnedFrame child;
        const AVFrame *mapped_source = &source;
        if (source.format == AV_PIX_FMT_QSV) {
            child.reset(av_frame_alloc());
            if (!child)
                return failure(AVERROR(ENOMEM), "allocating QSV child frame");
#if defined(__linux__)
            if (!qsv_child_frames_) {
                AVBufferRef *child_device = nullptr;
                error = av_hwdevice_ctx_create_derived(&child_device, AV_HWDEVICE_TYPE_VAAPI,
                                                       source_context->device_ref, 0);
                if (error >= 0)
                    error = av_hwframe_ctx_create_derived(
                        &qsv_child_frames_, AV_PIX_FMT_VAAPI, child_device, source.hw_frames_ctx,
                        AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
                av_buffer_unref(&child_device);
                if (error < 0)
                    return failure(error, "deriving QSV VA-API frames");
            }
            child->hw_frames_ctx = av_buffer_ref(qsv_child_frames_);
            if (!child->hw_frames_ctx)
                return failure(AVERROR(ENOMEM), "retaining QSV VA-API frames");
            child->format = AV_PIX_FMT_VAAPI;
#else
            return failure(AVERROR(ENOSYS),
                           "QSV D3D11/Vulkan shared surface transfer is unavailable");
#endif
            error =
                av_hwframe_map(child.get(), &source, AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
            if (error < 0)
                return failure(error, "mapping QSV child surface");
            mapped_source = child.get();
        }
        output->format = AV_PIX_FMT_VULKAN;
        output->hw_frames_ctx = av_buffer_ref(frames_);
        if (!output->hw_frames_ctx)
            return failure(AVERROR(ENOMEM), "retaining Vulkan transfer pool");
        error = av_hwframe_map(output.get(), mapped_source,
                               AV_HWFRAME_MAP_READ | AV_HWFRAME_MAP_DIRECT);
        if (error < 0)
            return failure(error, "mapping hardware surface to Vulkan");
        path_ = DecodeTransferPath::mapped;
    }
    if (source.format != AV_PIX_FMT_CUDA)
        error = av_frame_copy_props(output.get(), &source);
    if (error < 0)
        return failure(error, "copying frame metadata");
    return Result<AVFrame *>::success(output.release());
}
Result<AVFrame *> FrameTransfer::convert(const AVFrame &source, DecodeTransferPolicy policy,
                                         const std::shared_ptr<VulkanDecodeDevice> &destination) {
    const int layout =
        source.hw_frames_ctx
            ? reinterpret_cast<const AVHWFramesContext *>(source.hw_frames_ctx->data)->sw_format
            : source.format;
    if (source_width_ != 0 && (source.width != source_width_ || source.height != source_height_ ||
                               layout != source_layout_))
        return failure(AVERROR(ENOSYS), "decoded layout changed after transfer initialization");
    source_width_ = source.width;
    source_height_ = source.height;
    source_layout_ = layout;
    if (!source.hw_frames_ctx) {
        if (policy == DecodeTransferPolicy::gpu_only)
            return failure(AVERROR(ENOSYS), "GPU-only transfer cannot accept software frames");
        auto *output = av_frame_clone(&source);
        if (!output)
            return failure(AVERROR(ENOMEM), "retaining software frame");
        path_ = DecodeTransferPath::host;
        return Result<AVFrame *>::success(output);
    }
    if (source.format == AV_PIX_FMT_VULKAN && policy != DecodeTransferPolicy::host) {
        auto *output = av_frame_clone(&source);
        if (!output)
            return failure(AVERROR(ENOMEM), "retaining Vulkan frame");
        path_ = DecodeTransferPath::native;
        return Result<AVFrame *>::success(output);
    }
    if (policy != DecodeTransferPolicy::host && (!selected_ || path_ != DecodeTransferPath::host)) {
        auto output = gpu(source, destination);
        if (output) {
            selected_ = true;
            return output;
        }
        if (policy == DecodeTransferPolicy::gpu_only || selected_ ||
            output.error().code != ErrorCode::hardware_decode_unavailable)
            return output;
        // Only the first frame negotiates fallback. A later transfer failure
        // terminates the stream rather than changing the input storage plan.
        fallback_reason_ = output.error().message;
        av_buffer_unref(&frames_);
    }
    if (policy == DecodeTransferPolicy::gpu_only)
        return failure(AVERROR(ENOSYS), "GPU-only transfer unavailable");
    OwnedFrame output(av_frame_alloc());
    if (!output)
        return failure(AVERROR(ENOMEM), "allocating downloaded frame");
    int error = av_hwframe_transfer_data(output.get(), &source, 0);
    if (error < 0)
        return failure(error, "downloading hardware frame");
    error = av_frame_copy_props(output.get(), &source);
    if (error < 0)
        return failure(error, "copying downloaded frame metadata");
    error = av_frame_apply_cropping(output.get(), AV_FRAME_CROP_UNALIGNED);
    if (error < 0)
        return failure(error, "applying decoded frame crop");
    selected_ = true;
    path_ = DecodeTransferPath::host;
    return Result<AVFrame *>::success(output.release());
}
} // namespace crossvmaf::video
