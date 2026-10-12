#include "video/frame.hpp"
#include "video/ffmpeg.hpp"
extern "C" {
#include <libavutil/pixdesc.h>
}

namespace crossvmaf::video {
[[nodiscard]] std::string path_as_utf8(const std::filesystem::path &path) {
    const std::u8string utf8_path = path.u8string();
    return std::string(reinterpret_cast<const char *>(utf8_path.data()), utf8_path.size());
}

[[nodiscard]] std::string ffmpeg_error_message(int error_code, std::string_view operation) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> error_buffer{};
    if (av_strerror(error_code, error_buffer.data(), error_buffer.size()) < 0) {
        return std::string(operation) + " failed with FFmpeg error " + std::to_string(error_code);
    }
    return std::string(operation) + " failed: " + error_buffer.data();
}

[[nodiscard]] Result<VideoCodec> video_codec_from_id(AVCodecID codec_id) {
    switch (codec_id) {
    case AV_CODEC_ID_H264:
        return Result<VideoCodec>::success(VideoCodec::h264);
    case AV_CODEC_ID_HEVC:
        return Result<VideoCodec>::success(VideoCodec::hevc);
    case AV_CODEC_ID_AV1:
        return Result<VideoCodec>::success(VideoCodec::av1);
    case AV_CODEC_ID_VP8:
        return Result<VideoCodec>::success(VideoCodec::vp8);
    case AV_CODEC_ID_VP9:
        return Result<VideoCodec>::success(VideoCodec::vp9);
    case AV_CODEC_ID_MPEG1VIDEO:
        return Result<VideoCodec>::success(VideoCodec::mpeg1);
    case AV_CODEC_ID_MPEG2VIDEO:
        return Result<VideoCodec>::success(VideoCodec::mpeg2);
    case AV_CODEC_ID_MPEG4:
        return Result<VideoCodec>::success(VideoCodec::mpeg4);
    case AV_CODEC_ID_VC1:
    case AV_CODEC_ID_WMV3:
        return Result<VideoCodec>::success(VideoCodec::vc1);
    case AV_CODEC_ID_FLV1:
        return Result<VideoCodec>::success(VideoCodec::flv1);
    case AV_CODEC_ID_MJPEG:
        return Result<VideoCodec>::success(VideoCodec::mjpeg);
    default:
        return Result<VideoCodec>::failure(
            {ErrorCode::unsupported_format, "video codec is not supported by CrossVMAF"});
    }
}

[[nodiscard]] Result<FrameDescriptor> descriptor_from_frame(const AVFrame &frame) {
    if (frame.width <= 0 || frame.height <= 0 ||
        static_cast<std::uint64_t>(frame.width) > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::uint64_t>(frame.height) > std::numeric_limits<std::uint32_t>::max()) {
        return Result<FrameDescriptor>::failure(
            {ErrorCode::unsupported_format, "decoded video dimensions exceed the supported range"});
    }

    AVPixelFormat decoded_format = static_cast<AVPixelFormat>(frame.format);
    if (const auto* format = av_pix_fmt_desc_get(decoded_format); format && (format->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
        if (frame.crop_left != 0U || frame.crop_top != 0U || frame.crop_right != 0U ||
            frame.crop_bottom != 0U) {
            return Result<FrameDescriptor>::failure(
                {ErrorCode::unsupported_format,
                 "Vulkan hardware frames with unapplied crop offsets require software fallback"});
        }
        if (frame.hw_frames_ctx == nullptr || frame.hw_frames_ctx->data == nullptr) {
            return Result<FrameDescriptor>::failure(
                {ErrorCode::unsupported_format,
                 "decoded Vulkan frame has no hardware frame context"});
        }
        const auto *hardware_frames =
            reinterpret_cast<const AVHWFramesContext *>(frame.hw_frames_ctx->data);
        decoded_format = hardware_frames->sw_format;
    }

    std::uint8_t bit_depth = 0U;
    PixelFormat pixel_format = PixelFormat::yuv420p;
    switch (decoded_format) {
    case AV_PIX_FMT_NV12:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUVJ420P:
        bit_depth = 8U;
        pixel_format = PixelFormat::yuv420p;
        break;
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUVJ422P:
        bit_depth = 8U;
        pixel_format = PixelFormat::yuv422p;
        break;
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUVJ444P:
        bit_depth = 8U;
        pixel_format = PixelFormat::yuv444p;
        break;
    case AV_PIX_FMT_P010LE:
    case AV_PIX_FMT_YUV420P10LE:
        bit_depth = 10U;
        pixel_format = PixelFormat::yuv420p;
        break;
    case AV_PIX_FMT_YUV422P10LE:
        bit_depth = 10U;
        pixel_format = PixelFormat::yuv422p;
        break;
    case AV_PIX_FMT_YUV444P10LE:
        bit_depth = 10U;
        pixel_format = PixelFormat::yuv444p;
        break;
    case AV_PIX_FMT_YUV420P12LE:
        bit_depth = 12U;
        pixel_format = PixelFormat::yuv420p;
        break;
    case AV_PIX_FMT_YUV422P12LE:
        bit_depth = 12U;
        pixel_format = PixelFormat::yuv422p;
        break;
    case AV_PIX_FMT_YUV444P12LE:
        bit_depth = 12U;
        pixel_format = PixelFormat::yuv444p;
        break;
    case AV_PIX_FMT_YUV420P16LE:
        bit_depth = 16U;
        pixel_format = PixelFormat::yuv420p;
        break;
    case AV_PIX_FMT_YUV422P16LE:
        bit_depth = 16U;
        pixel_format = PixelFormat::yuv422p;
        break;
    case AV_PIX_FMT_YUV444P16LE:
        bit_depth = 16U;
        pixel_format = PixelFormat::yuv444p;
        break;
    default:
        return Result<FrameDescriptor>::failure(
            {ErrorCode::unsupported_format,
             frame.format == AV_PIX_FMT_VULKAN
                 ? "Vulkan hardware decoding must output NV12 or P010"
                 : "decoded video must use supported planar little-endian YUV"});
    }

    FrameDescriptor descriptor{static_cast<std::uint32_t>(frame.width),
                               static_cast<std::uint32_t>(frame.height), bit_depth, pixel_format};
    const auto descriptor_validation = video::validate_frame_layout(descriptor);
    if (!descriptor_validation) {
        return Result<FrameDescriptor>::failure(descriptor_validation.error());
    }
    return Result<FrameDescriptor>::success(descriptor);
}

[[nodiscard]] HardwareSurfaceFormat
hardware_surface_format_from_frame(const AVFrame &frame) noexcept {
    AVPixelFormat layout = static_cast<AVPixelFormat>(frame.format);
    if (frame.hw_frames_ctx && frame.hw_frames_ctx->data) {
        layout = reinterpret_cast<const AVHWFramesContext*>(frame.hw_frames_ctx->data)->sw_format;
    }
    switch (layout) {
    case AV_PIX_FMT_NV12:
        return HardwareSurfaceFormat::nv12;
    case AV_PIX_FMT_P010LE:
        return HardwareSurfaceFormat::p010;
    default:
        return HardwareSurfaceFormat::none;
    }
}
} // namespace crossvmaf::video
