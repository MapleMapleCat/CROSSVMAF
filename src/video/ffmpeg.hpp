#pragma once
#include "crossvmaf/error.hpp"
#include "video/ffmpeg_abi.hpp"
#include "crossvmaf/frame.hpp"
#include "video/decoder.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include "crossvmaf/video.hpp"

// Shared FFmpeg helpers for the input decoder targets. These functions used
// to live in video_decoder.cpp's anonymous namespace; the lightweight video
// probe reuses them so probing and decoding agree on supported codecs and
// pixel formats.

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
}

namespace crossvmaf::video {

[[nodiscard]] std::string path_as_utf8(const std::filesystem::path &path);

[[nodiscard]] std::string ffmpeg_error_message(int error_code, std::string_view operation);

[[nodiscard]] Result<VideoCodec> video_codec_from_id(AVCodecID codec_id);

[[nodiscard]] Result<FrameDescriptor> descriptor_from_frame(const AVFrame &frame);
[[nodiscard]] Result<void> copy_host_frame(const AVFrame& frame, const FrameDescriptor& descriptor,
                                          const WritableFrameView& destination);

} // namespace crossvmaf::video

namespace crossvmaf::video {
[[nodiscard]] HardwareSurfaceFormat
hardware_surface_format_from_frame(const AVFrame &frame) noexcept;
class Frame::Impl final {
  public:
    Impl(AVFrame *frame, FrameDescriptor descriptor, VideoTimestamp timestamp,
         std::chrono::steady_clock::time_point decode_started_at,
         std::chrono::steady_clock::time_point decode_completed_at,
         VideoDecodeBackend backend = VideoDecodeBackend::software) noexcept
        : frame_(frame), descriptor_(descriptor), timestamp_(timestamp),
          decode_started_at_(decode_started_at), decode_completed_at_(decode_completed_at),
          decode_backend_(backend),
          hardware_surface_format_(frame == nullptr ? HardwareSurfaceFormat::none
                                                    : hardware_surface_format_from_frame(*frame)) {}

    ~Impl() { av_frame_free(&frame_); }

    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    AVFrame *frame_{};
    FrameDescriptor descriptor_{};
    VideoTimestamp timestamp_{};
    std::chrono::steady_clock::time_point decode_started_at_{};
    std::chrono::steady_clock::time_point decode_completed_at_{};
    VideoDecodeBackend decode_backend_{VideoDecodeBackend::software};
    HardwareSurfaceFormat hardware_surface_format_{HardwareSurfaceFormat::none};
};
} // namespace crossvmaf::video

// Lightweight video probing shared by the public probe API and the decoder.
// Decoder snapshots metadata through extract_video_metadata() when it
// opens a file, so runtime-reported metadata and standalone probing always
// agree.

struct AVFormatContext;

namespace crossvmaf::video {

[[nodiscard]] StreamVideoCodec public_stream_codec(VideoCodec codec) noexcept;

// Standalone lightweight probe behind crossvmaf::probe_video(). Reads
// container and stream metadata only; no decoder or evaluation pipeline is
// initialized.
[[nodiscard]] Result<VideoMetadata> probe_video_file(const std::filesystem::path &path,
                                                     const ProbeOptions &options);

// Snapshots metadata from an already-open format context. The returned
// descriptor is header-derived (descriptor_confirmed == false) until the
// caller replaces it with a decoded-frame descriptor.
[[nodiscard]] VideoMetadata extract_video_metadata(const AVFormatContext *format_context,
                                                   int video_stream_index,
                                                   const std::filesystem::path &path);

// Local file identity snapshot: path, size, modification time. Failure to
// read the file leaves captured == false instead of throwing.
[[nodiscard]] FileIdentity capture_file_identity(const std::filesystem::path &path) noexcept;

} // namespace crossvmaf::video
