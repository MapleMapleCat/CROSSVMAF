#pragma once

// Public video container metadata, lightweight probing, and pre-run pair
// validation. These contracts let frontends show real codec/timing
// information after file selection without initializing the full evaluation
// pipeline. All types are free of FFmpeg dependencies.

#include "crossvmaf/error.hpp"
#include "crossvmaf/decode.hpp"
#include "crossvmaf/frame.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace crossvmaf {

enum class StreamVideoCodec : std::uint8_t {
    h264,
    hevc,
    av1,
    vp8,
    vp9,
    mpeg1,
    mpeg2,
    mpeg4,
    vc1,
    flv1,
    mjpeg,
};

enum class StreamHardwarePixelFormat : std::uint8_t {
    none,
    nv12,
    p010,
};

// Local file identity snapshot taken when metadata was captured. It detects
// that the file changed after probing; it is not a content hash.
struct FileIdentity final {
    std::filesystem::path path;
    std::uint64_t size_bytes{};
    // Seconds since the Unix epoch (UTC); 0 when capture failed.
    std::int64_t modification_time_utc_seconds{};
    bool captured{};

    // Same size and modification time means the file very likely did not
    // change since the other snapshot was taken.
    [[nodiscard]] bool matches(const FileIdentity& other) const noexcept {
        return captured && other.captured &&
            size_bytes == other.size_bytes &&
            modification_time_utc_seconds ==
                other.modification_time_utc_seconds;
    }
};

enum class FrameRateMode : std::uint8_t {
    constant,
    variable,
    unknown,
};

struct FrameRateInfo final {
    // Container-reported average frame rate; invalid when unknown.
    Rational average;
    // Container-reported nominal (base) frame rate; invalid when unknown.
    Rational nominal;
    FrameRateMode mode{FrameRateMode::unknown};
};

// Origin of VideoMetadata::average_bit_rate. A container-wide or file-size
// estimate mixes in non-video streams and container overhead; only
// video_stream is the actual coded video rate.
enum class BitRateSource : std::uint8_t {
    video_stream,
    container_overall,
    file_estimate,
};

struct VideoMetadata final {
    FileIdentity identity;

    StreamVideoCodec codec{StreamVideoCodec::h264};
    // Demuxer name as reported by the container (e.g. "mov,mp4,m4a").
    std::string container_format;
    int video_stream_index{-1};

    // Header-derived video spec. Absent when the container does not carry a
    // recognized planar YUV format; descriptor_confirmed is false until a
    // decoded frame confirmed the spec.
    std::optional<FrameDescriptor> descriptor;
    bool descriptor_confirmed{};

    Rational time_base;
    FrameRateInfo frame_rate;

    // Container-reported duration; absent when unknown (never estimated from
    // bit rate, which is too unreliable to present as a duration).
    std::optional<double> duration_seconds;
    // Container-reported frame count, or an estimate from duration and
    // average frame rate (frame_count_estimated is then true).
    std::optional<std::uint64_t> frame_count;
    bool frame_count_estimated{};

    std::optional<std::uint64_t> average_bit_rate;
    BitRateSource bit_rate_source{BitRateSource::video_stream};
};

// Media metadata snapshot of both inputs of an evaluation stream, captured
// when the stream opened the files. Reports keep this snapshot so opening a
// saved report never re-derives old information from the original files.
struct StreamMediaMetadata final {
    VideoMetadata reference;
    VideoMetadata distorted;
};

struct ProbeOptions final {    // Bounds the packet scan avformat_find_stream_info may perform while
    // probing, in microseconds. Zero keeps the FFmpeg default. Some formats
    // can only fill metadata by scanning packets, so probing never promises
    // to read only a fixed-size header.
    std::int64_t max_analyze_duration_microseconds{};
};

// Lightweight single-file probe: container and stream metadata without
// initializing decoders or the evaluation pipeline. I/O failures are
// reported as errors, distinct from a pair mismatch between two successfully
// probed files.
[[nodiscard]] Result<VideoMetadata> probe_video(
    const std::filesystem::path& path,
    const ProbeOptions& options = {});

enum class PairValidationStatus : std::uint8_t {
    // All checked facts match; no runtime surprises expected.
    ready,
    // A hard mismatch was found; starting the evaluation is pointless.
    invalid,
    // Some facts are unknown or can only be verified while decoding; the
    // runtime per-frame and timeline validation remains authoritative.
    needs_runtime_validation,
};

enum class PairIssueSeverity : std::uint8_t {
    error,
    warning,
    info,
};

struct PairIssue final {
    // Stable machine-readable codes, e.g. "dimension_mismatch".
    std::string reason_code;
    // Compared field, e.g. "width", "frame_rate", "duration".
    std::string field;
    PairIssueSeverity severity{PairIssueSeverity::info};
    std::string reference_value;
    std::string distorted_value;
    std::string message;
    // True when the fact cannot be settled by probing and the runtime
    // timeline validation must decide.
    bool requires_runtime_validation{};
};

struct PairValidation final {
    PairValidationStatus status{PairValidationStatus::ready};
    std::vector<PairIssue> issues;

    [[nodiscard]] bool ready() const noexcept {
        return status == PairValidationStatus::ready;
    }
};

// Compares two successfully probed inputs before a run. Hard spec mismatches
// fail early; unknown values stay unknown and defer to runtime validation
// instead of being replaced by unreliable estimates.
[[nodiscard]] Result<PairValidation> validate_video_pair(
    const VideoMetadata& reference,
    const VideoMetadata& distorted);

}  // namespace crossvmaf
