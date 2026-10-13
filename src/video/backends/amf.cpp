#include "video/backends/ffmpeg_device.hpp"
#if defined(_WIN32) && CROSSVMAF_ENABLE_AMF
#include <AMF/core/Surface.h>
#include <d3d11.h>
extern "C" {
#include <libavutil/hwcontext_d3d11va.h>
}
#endif

namespace crossvmaf::video {
const AVCodec *find_amf_codec(int codec_id) {
    const char *name = nullptr;
    switch (codec_id) {
    case AV_CODEC_ID_H264: name = "h264_amf"; break;
    case AV_CODEC_ID_HEVC: name = "hevc_amf"; break;
    case AV_CODEC_ID_AV1: name = "av1_amf"; break;
    case AV_CODEC_ID_VP9: name = "vp9_amf"; break;
    default: return nullptr;
    }
    return avcodec_find_decoder_by_name(name);
}

Result<AVBufferRef *> open_amf_context(const DecodeDeviceInfo &info) {
#if defined(_WIN32) && CROSSVMAF_ENABLE_AMF
    // AMF's direct device creation ignores its selector. Derive from the exact
    // DXGI adapter so "same" and explicit IDs cannot silently choose another GPU.
    auto child = open_native_context(AV_HWDEVICE_TYPE_D3D11VA, info);
    if (!child) return child;
    auto *child_context = child.value();
    AVBufferRef *context = nullptr;
    const int error = av_hwdevice_ctx_create_derived(
        &context, AV_HWDEVICE_TYPE_AMF, child_context, 0);
    av_buffer_unref(&child_context);
    if (error < 0)
        return Result<AVBufferRef *>::failure(
            {error == AVERROR(ENOMEM) ? ErrorCode::backend_error : ErrorCode::hardware_decode_unavailable,
             ffmpeg_error_message(error, "deriving AMF hardware context")});
    return Result<AVBufferRef *>::success(context);
#else
    static_cast<void>(info);
    return Result<AVBufferRef *>::failure(
        {ErrorCode::hardware_decode_unavailable, "AMF requires a Windows build with AMF enabled"});
#endif
}

Result<void> bind_amf_d3d11_surface(const AVFrame &source, AVFrame &child) {
#if defined(_WIN32) && CROSSVMAF_ENABLE_AMF
    auto *surface = reinterpret_cast<amf::AMFSurface *>(source.data[0]);
    if (source.format != AV_PIX_FMT_AMF_SURFACE || !source.hw_frames_ctx ||
        !child.hw_frames_ctx || !surface ||
        surface->GetMemoryType() != amf::AMF_MEMORY_DX11)
        return Result<void>::failure(
            {ErrorCode::hardware_decode_unavailable, "AMF GPU transfer requires a native D3D11 surface"});
    auto *plane = surface->GetPlaneAt(0);
    auto *texture = plane ? static_cast<ID3D11Texture2D *>(plane->GetNative()) : nullptr;
    if (!texture || plane->GetOffsetX() != 0 || plane->GetOffsetY() != 0)
        return Result<void>::failure(
            {ErrorCode::hardware_decode_unavailable, "AMF returned no uncropped D3D11 texture"});
    D3D11_TEXTURE2D_DESC description{};
    texture->GetDesc(&description);
    const auto *frames = reinterpret_cast<const AVHWFramesContext *>(source.hw_frames_ctx->data);
    const auto expected = frames->sw_format == AV_PIX_FMT_NV12 ? DXGI_FORMAT_NV12 :
                          frames->sw_format == AV_PIX_FMT_P010LE ? DXGI_FORMAT_P010 : DXGI_FORMAT_UNKNOWN;
    if (expected == DXGI_FORMAT_UNKNOWN || description.Format != expected ||
        source.width <= 0 || source.height <= 0 ||
        description.Width < static_cast<UINT>(source.width) ||
        description.Height < static_cast<UINT>(source.height))
        return Result<void>::failure(
            {ErrorCode::hardware_decode_unavailable, "AMF texture does not match the decoded frame layout"});
    const auto *child_frames = reinterpret_cast<const AVHWFramesContext *>(child.hw_frames_ctx->data);
    const auto *d3d = static_cast<const AVD3D11VADeviceContext *>(child_frames->device_ctx->hwctx);
    ID3D11Device *texture_device = nullptr;
    texture->GetDevice(&texture_device);
    const bool matches_device = texture_device && texture_device == d3d->device;
    if (texture_device) texture_device->Release();
    if (!matches_device)
        return Result<void>::failure(
            {ErrorCode::hardware_decode_unavailable, "AMF texture is not on the selected D3D11 device"});
    // AMF exposes individual textures. Do not guess a slice if a future
    // runtime returns texture arrays without an explicit subresource index.
    if (description.ArraySize != 1 || description.MipLevels != 1)
        return Result<void>::failure(
            {ErrorCode::hardware_decode_unavailable, "AMF texture arrays are unsupported"});
    child.format = AV_PIX_FMT_D3D11;
    child.width = source.width;
    child.height = source.height;
    child.data[0] = reinterpret_cast<std::uint8_t *>(texture);
    child.data[1] = nullptr;
    return Result<void>::success();
#else
    static_cast<void>(source);
    static_cast<void>(child);
    return Result<void>::failure(
        {ErrorCode::hardware_decode_unavailable, "AMF D3D11 transfer is unavailable"});
#endif
}
} // namespace crossvmaf::video
