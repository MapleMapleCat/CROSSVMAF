#include "video/ffmpeg.hpp"
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <system_error>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

namespace crossvmaf::video {
namespace {

struct FormatContextCloser final {
    void operator()(AVFormatContext *context) const noexcept {
        if (context != nullptr) avformat_close_input(&context);
    }
};
using FormatContextPtr = std::unique_ptr<AVFormatContext, FormatContextCloser>;

// Average and nominal rates differing by more than this fraction hint at
// variable frame rate content.
constexpr long double variable_frame_rate_tolerance = 1.0e-2L;

[[nodiscard]] std::int64_t modification_time_utc_seconds(const std::filesystem::path &path,
                                                         std::error_code &error) {
    const auto file_time = std::filesystem::last_write_time(path, error);
    if (error) return 0;
    // Portable approximation of the file clock in system time; one-second
    // resolution is enough for change detection.
    const auto system_time = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        file_time - std::filesystem::file_time_type::clock::now() +
        std::chrono::system_clock::now());
    return std::chrono::duration_cast<std::chrono::seconds>(system_time.time_since_epoch()).count();
}

[[nodiscard]] FrameRateMode frame_rate_mode(const Rational &average,
                                            const Rational &nominal) noexcept {
    if (!average.valid()) return FrameRateMode::unknown;
    if (!nominal.valid()) return FrameRateMode::unknown;
    const long double average_value = static_cast<long double>(average.value());
    const long double nominal_value = static_cast<long double>(nominal.value());
    const long double difference = std::fabs(average_value - nominal_value);
    return difference / std::max(average_value, nominal_value) > variable_frame_rate_tolerance
               ? FrameRateMode::variable
               : FrameRateMode::constant;
}

} // namespace

StreamVideoCodec public_stream_codec(VideoCodec codec) noexcept {
    switch (codec) {
    case VideoCodec::h264:
        return StreamVideoCodec::h264;
    case VideoCodec::hevc:
        return StreamVideoCodec::hevc;
    case VideoCodec::av1:
        return StreamVideoCodec::av1;
    case VideoCodec::vp8:
        return StreamVideoCodec::vp8;
    case VideoCodec::vp9:
        return StreamVideoCodec::vp9;
    case VideoCodec::mpeg1:
        return StreamVideoCodec::mpeg1;
    case VideoCodec::mpeg2:
        return StreamVideoCodec::mpeg2;
    case VideoCodec::mpeg4:
        return StreamVideoCodec::mpeg4;
    case VideoCodec::vc1:
        return StreamVideoCodec::vc1;
    case VideoCodec::flv1:
        return StreamVideoCodec::flv1;
    case VideoCodec::mjpeg:
        return StreamVideoCodec::mjpeg;
    }
    return StreamVideoCodec::h264;
}

FileIdentity capture_file_identity(const std::filesystem::path &path) noexcept {
    FileIdentity identity;
    try {
        identity.path = path;
        std::error_code error;
        const std::uintmax_t size = std::filesystem::file_size(path, error);
        if (error) return identity;
        const std::int64_t modification_time = modification_time_utc_seconds(path, error);
        if (error) return identity;
        identity.size_bytes = static_cast<std::uint64_t>(size);
        identity.modification_time_utc_seconds = modification_time;
        identity.captured = true;
    } catch (...) {
        identity.captured = false;
    }
    return identity;
}

VideoMetadata extract_video_metadata(const AVFormatContext *format_context, int video_stream_index,
                                     const std::filesystem::path &path) {
    VideoMetadata metadata;
    metadata.identity = capture_file_identity(path);
    metadata.video_stream_index = video_stream_index;
    if (format_context == nullptr || video_stream_index < 0 ||
        static_cast<unsigned int>(video_stream_index) >= format_context->nb_streams) {
        return metadata;
    }
    const AVStream *const stream = format_context->streams[video_stream_index];
    const AVCodecParameters *const parameters = stream->codecpar;
    if (format_context->iformat != nullptr && format_context->iformat->name != nullptr) {
        metadata.container_format = format_context->iformat->name;
    }
    const auto codec = video_codec_from_id(parameters->codec_id);
    if (codec) metadata.codec = public_stream_codec(codec.value());

    // Reuse the decoded-frame descriptor mapping on a header-only frame so
    // probing and decoding apply identical pixel format rules.
    AVFrame header_frame{};
    header_frame.width = parameters->width;
    header_frame.height = parameters->height;
    header_frame.format = parameters->format;
    const auto descriptor = descriptor_from_frame(header_frame);
    if (descriptor) {
        metadata.descriptor = descriptor.value();
        metadata.descriptor_confirmed = false;
    }

    metadata.time_base = {stream->time_base.num, stream->time_base.den};
    metadata.frame_rate.average = {stream->avg_frame_rate.num, stream->avg_frame_rate.den};
    metadata.frame_rate.nominal = {stream->r_frame_rate.num, stream->r_frame_rate.den};
    metadata.frame_rate.mode =
        frame_rate_mode(metadata.frame_rate.average, metadata.frame_rate.nominal);

    if (stream->duration != AV_NOPTS_VALUE && stream->duration > 0 && metadata.time_base.valid()) {
        metadata.duration_seconds =
            static_cast<double>(stream->duration) * metadata.time_base.value();
    } else if (format_context->duration != AV_NOPTS_VALUE && format_context->duration > 0) {
        metadata.duration_seconds =
            static_cast<double>(format_context->duration) / static_cast<double>(AV_TIME_BASE);
    }

    if (stream->nb_frames > 0) {
        metadata.frame_count = static_cast<std::uint64_t>(stream->nb_frames);
        metadata.frame_count_estimated = false;
    } else if (metadata.duration_seconds.has_value() && *metadata.duration_seconds > 0.0 &&
               metadata.frame_rate.average.valid()) {
        const double estimated_frames =
            *metadata.duration_seconds * metadata.frame_rate.average.value();
        if (estimated_frames >= 0.5) {
            metadata.frame_count = static_cast<std::uint64_t>(std::llround(estimated_frames));
            metadata.frame_count_estimated = true;
        }
    }

    if (parameters->bit_rate > 0) {
        metadata.average_bit_rate = static_cast<std::uint64_t>(parameters->bit_rate);
        metadata.bit_rate_source = BitRateSource::video_stream;
    } else if (format_context->bit_rate > 0) {
        metadata.average_bit_rate = static_cast<std::uint64_t>(format_context->bit_rate);
        metadata.bit_rate_source = BitRateSource::container_overall;
    } else if (metadata.identity.captured && metadata.duration_seconds.has_value() &&
               *metadata.duration_seconds > 0.0) {
        metadata.average_bit_rate = static_cast<std::uint64_t>(
            static_cast<double>(metadata.identity.size_bytes) * 8.0 / *metadata.duration_seconds);
        metadata.bit_rate_source = BitRateSource::file_estimate;
    }
    return metadata;
}

Result<VideoMetadata> probe_video_file(const std::filesystem::path &path,
                                       const ProbeOptions &options) {
    if (path.empty()) {
        return Result<VideoMetadata>::failure(
            {ErrorCode::invalid_argument, "probe_video requires a file path"});
    }
    const auto compatibility = require_ffmpeg_abi();
    if (!compatibility) return Result<VideoMetadata>::failure(compatibility.error());
    FormatContextPtr format_context(avformat_alloc_context());
    if (!format_context) {
        return Result<VideoMetadata>::failure(
            {ErrorCode::backend_error, "failed to allocate FFmpeg format context"});
    }
    if (options.max_analyze_duration_microseconds > 0) {
        format_context->max_analyze_duration = options.max_analyze_duration_microseconds;
    }

    AVDictionary *open_options = nullptr;
    av_dict_set(&open_options, "protocol_whitelist", "file", 0);
    av_dict_set(&open_options, "format_whitelist",
                "h264,hevc,av1,obu,ivf,m4v,mpegvideo,mpeg,mpegts,vc1,"
                "mjpeg,mpjpeg,mov,matroska,webm,avi,asf,flv",
                0);
    const std::string utf8_path = path_as_utf8(path);
    // FFmpeg takes ownership of the supplied context and frees it on failure.
    // Release before opening, then adopt the returned pointer on both paths.
    AVFormatContext *raw_context = format_context.release();
    const int open_result =
        avformat_open_input(&raw_context, utf8_path.c_str(), nullptr, &open_options);
    format_context.reset(raw_context);
    av_dict_free(&open_options);
    if (open_result < 0) {
        return Result<VideoMetadata>::failure(
            {ErrorCode::io_error, ffmpeg_error_message(open_result, "opening video input")});
    }
    const int stream_info_result = avformat_find_stream_info(format_context.get(), nullptr);
    if (stream_info_result < 0) {
        return Result<VideoMetadata>::failure(
            {ErrorCode::unsupported_format,
             ffmpeg_error_message(stream_info_result, "reading video stream metadata")});
    }

    const int video_stream_index =
        av_find_best_stream(format_context.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index < 0) {
        return Result<VideoMetadata>::failure(
            {ErrorCode::unsupported_format, "input does not contain a video stream"});
    }
    const AVCodecParameters *const parameters =
        format_context->streams[video_stream_index]->codecpar;
    const auto codec = video_codec_from_id(parameters->codec_id);
    if (!codec) {
        const char *codec_name = avcodec_get_name(parameters->codec_id);
        return Result<VideoMetadata>::failure(
            {ErrorCode::unsupported_format,
             std::string("video codec is not supported by CrossVMAF: ") +
                 (codec_name != nullptr ? codec_name : "unknown")});
    }
    return Result<VideoMetadata>::success(
        extract_video_metadata(format_context.get(), video_stream_index, path));
}

} // namespace crossvmaf::video

namespace crossvmaf {
Result<VideoMetadata> probe_video(const std::filesystem::path &path, const ProbeOptions &options) try {
    auto result = video::probe_video_file(path, options);
    if (!result) return Result<VideoMetadata>::failure(result.error().with_context(
        "probing " + video::path_as_utf8(path)));
    return result;
} catch (const std::exception &exception) {
    return Result<VideoMetadata>::failure({ErrorCode::backend_error,
        "video probe failed: " + std::string(exception.what())});
} catch (...) {
    return Result<VideoMetadata>::failure({ErrorCode::backend_error, "video probe failed with an unknown exception"});
}
} // namespace crossvmaf
