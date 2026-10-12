#include "video/backends/ffmpeg_device.hpp"
namespace crossvmaf::video {
const AVCodec *find_qsv_codec(int codec_id) {
    const char *name = nullptr;
    switch (codec_id) {
    case AV_CODEC_ID_H264:
        name = "h264_qsv";
        break;
    case AV_CODEC_ID_HEVC:
        name = "hevc_qsv";
        break;
    case AV_CODEC_ID_AV1:
        name = "av1_qsv";
        break;
    case AV_CODEC_ID_VP9:
        name = "vp9_qsv";
        break;
    default:
        return nullptr;
    }
    return avcodec_find_decoder_by_name(name);
}
Result<AVBufferRef *> open_qsv_context(const DecodeDeviceInfo &info) {
    AVDictionary *options = nullptr;
#if defined(_WIN32)
    constexpr auto type = AV_HWDEVICE_TYPE_D3D11VA;
#else
    constexpr auto type = AV_HWDEVICE_TYPE_VAAPI;
    av_dict_set(&options, "driver", "iHD", 0);
#endif
    auto child = open_native_context(type, info, options);
    av_dict_free(&options);
    if (!child)
        return child;
    AVBufferRef *child_context = child.value();
    AVBufferRef *context = nullptr;
    // Keep the public source-device chain so D3D11/VAAPI frame derivation can
    // retrieve this exact child rather than opening another logical device.
    const int error =
        av_hwdevice_ctx_create_derived(&context, AV_HWDEVICE_TYPE_QSV, child_context, 0);
    av_buffer_unref(&child_context);
    if (error < 0)
        return Result<AVBufferRef *>::failure(
            {error == AVERROR(ENOMEM) ? ErrorCode::backend_error : ErrorCode::hardware_decode_unavailable,
             ffmpeg_error_message(error, "deriving QSV hardware context")});
    return Result<AVBufferRef *>::success(context);
}
} // namespace crossvmaf::video
