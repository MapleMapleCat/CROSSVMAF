#pragma once
#include "crossvmaf/error.hpp"
#include "video/frame.hpp"
#include "video/decode_device.hpp"
#include "crossvmaf/video.hpp"
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>

namespace crossvmaf::video {

class VulkanDecodeDevice;

enum class VideoCodec : std::uint8_t {
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

// Hardware-capable codecs still fall back to software when the selected
// Vulkan device or stream profile cannot decode them.
[[nodiscard]] bool automatic_decode_should_attempt_hardware(VideoCodec codec) noexcept;

// External software decoders can manage their own worker pools instead of
// using FFmpeg's frame-thread scheduler. Coordinate those pools when multiple
// Decoder instances run concurrently; zero keeps FFmpeg's automatic
// policy when no pair-specific plan applies.
[[nodiscard]] std::size_t
plan_software_decoder_thread_count(VideoCodec codec, std::size_t requested_thread_count,
                                   std::size_t concurrent_decoder_count,
                                   std::size_t logical_processor_count) noexcept;

// Default FFmpeg thread_count for wait-only consumers. This is the decode
// depth on a single decode queue; multiple queues may enlarge the GPU pool,
// and frontends without threading (native AV1) can reduce thread_count to one.
inline constexpr std::size_t default_hardware_decode_depth = 4U;
// Automatic frame-thread cap when consumers extend the decoder timeline and
// decoder run-ahead therefore has to cover reference distances.
inline constexpr std::size_t maximum_automatic_hardware_frame_threads = 8U;

struct HardwareDecoderThreadPlan final {
    // Frame threading trades one retained decode surface and one frame of
    // output delay per thread for decoder run-ahead in front of consumers.
    bool frame_threading{};
    // FFmpeg thread_count; also sizes its in-flight Vulkan decode pool. Zero
    // keeps FFmpeg's automatic count.
    std::size_t thread_count{};
};

// `consumers_extend_decoder_timeline` is false when decoded images can be read
// in VK_IMAGE_LAYOUT_GENERAL without transitioning them, which lets consumers
// wait on the decoder without signalling back into its timeline.
[[nodiscard]] HardwareDecoderThreadPlan
plan_hardware_decoder_threads(std::size_t requested_thread_count,
                              bool consumers_extend_decoder_timeline,
                              std::size_t logical_processor_count,
                              bool dedicated_decode_queue = false) noexcept;

struct SoftwareDecodeWorkload final {
    VideoCodec codec{VideoCodec::h264};
    FrameDescriptor descriptor{};
    // Container metadata is only a startup estimate, not measured CPU cost.
    double encoded_bits_per_frame{};
};

struct DecoderPairThreadPlan final {
    std::size_t reference_thread_count{};
    std::size_t distorted_thread_count{};
};

[[nodiscard]] DecoderPairThreadPlan plan_decoder_pair_threads(
    const SoftwareDecodeWorkload &reference, const SoftwareDecodeWorkload &distorted,
    std::size_t requested_thread_count, std::size_t logical_processor_count) noexcept;

struct VideoDecoderOptions final {
    std::size_t output_queue_capacity{4U};
    // Zero enables automatic selection, including pair-aware H.264 planning.
    // Explicit requests always apply unchanged to each source.
    std::size_t codec_thread_count{};
    std::shared_ptr<DecodeDevice> decode_device;
    std::shared_ptr<VulkanDecodeDevice> transfer_device;
    DecodeTransferPolicy transfer_policy{DecodeTransferPolicy::automatic};
    // A reference/distorted pair sets this to two so external software
    // decoder pools share one host-wide CPU budget.
    std::size_t concurrent_decoder_count{1U};
    // When stop is requested, blocking demux/decode calls abort through the
    // FFmpeg interrupt callback and the decoder reports cancellation.
    std::stop_token stop_token{};
    bool require_hardware_decode{};
    bool respect_automatic_vulkan_policy{};
    std::vector<std::string> open_attempts;
};

enum class VideoFramePopStatus : std::uint8_t {
    frame,
    unavailable,
    end_of_stream,
};

struct VideoFramePopResult final {
    VideoFramePopStatus status{VideoFramePopStatus::unavailable};
    Frame frame;
};

struct VideoDecoderStatistics final {
    VideoCodec codec{VideoCodec::h264};
    VideoDecodeBackend decode_backend{VideoDecodeBackend::software};
    HardwareSurfaceFormat hardware_surface_format{HardwareSurfaceFormat::none};
    std::uint64_t packets_read{};
    std::uint64_t compressed_bytes{};
    std::uint64_t decoded_frames{};
    std::uint64_t decode_active_nanoseconds{};
    std::uint64_t decode_operation_count{};
    std::uint64_t output_queue_blocked_nanoseconds{};
    std::uint64_t output_queue_blocked_wait_count{};
    std::size_t queue_capacity{};
    std::size_t queued_frames{};
    std::size_t codec_thread_count{};
    bool end_of_stream{};
    bool cancellation_requested{};
    DecodeDiagnostics decode;
};

struct VideoDecoderPair;

class Decoder final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Decoder>> open(const std::filesystem::path &path,
                                                               VideoDecoderOptions options = {});

    [[nodiscard]] static Result<VideoDecoderPair>
    open_pair(const std::filesystem::path &reference_path,
              const std::filesystem::path &distorted_path, VideoDecoderOptions options = {});

    ~Decoder();
    Decoder(Decoder &&) noexcept;
    Decoder &operator=(Decoder &&) noexcept;

    Decoder(const Decoder &) = delete;
    Decoder &operator=(const Decoder &) = delete;

    [[nodiscard]] const std::filesystem::path &path() const noexcept;
    [[nodiscard]] const FrameDescriptor &descriptor() const noexcept;
    // Metadata snapshot captured when the decoder opened the file; the
    // descriptor is confirmed once the first frame has been decoded.
    [[nodiscard]] const VideoMetadata &metadata() const noexcept;
    [[nodiscard]] VideoCodec codec() const noexcept;
    [[nodiscard]] VideoDecodeBackend decode_backend() const noexcept;
    [[nodiscard]] HardwareSurfaceFormat hardware_surface_format() const noexcept;
    [[nodiscard]] bool has_vulkan_output() const noexcept;
    [[nodiscard]] std::optional<std::uint64_t> frame_count_hint() const noexcept;
    [[nodiscard]] Result<VideoFramePopResult> try_pop();
    [[nodiscard]] Result<VideoFramePopResult> wait_pop();
    [[nodiscard]] VideoDecoderStatistics statistics() const;

    void request_cancel() noexcept;
    void drain() noexcept;

  private:
    class State;
    explicit Decoder(std::unique_ptr<State> state) noexcept;

    std::unique_ptr<State> state_;
};

struct VideoDecoderPair final {
    std::unique_ptr<Decoder> reference;
    std::unique_ptr<Decoder> distorted;
};

} // namespace crossvmaf::video
