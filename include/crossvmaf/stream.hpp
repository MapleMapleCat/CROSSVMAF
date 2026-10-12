#pragma once

#include "crossvmaf/device.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/result.hpp"
#include "crossvmaf/stream_options.hpp"
#include "crossvmaf/video.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stop_token>
#include <vector>

namespace crossvmaf {

namespace detail {
class EvaluationStreamTelemetryOpener;
}

struct StreamInputStatistics final {
    StreamVideoCodec codec{StreamVideoCodec::h264};
    StreamDecodeBackend decode_backend{StreamDecodeBackend::software};
    StreamHardwarePixelFormat hardware_pixel_format{
        StreamHardwarePixelFormat::none};
    std::uint64_t packets_read{};
    std::uint64_t compressed_bytes{};
    std::uint64_t decoded_frames{};
    std::uint64_t decode_active_nanoseconds{};
    std::uint64_t decode_operation_count{};
    std::uint64_t output_queue_blocked_nanoseconds{};
    std::uint64_t output_queue_blocked_wait_count{};
    std::size_t queue_capacity{};
    std::size_t queued_frames{};
    std::size_t decoder_threads{};
    bool end_of_stream{};
    bool cancellation_requested{};
    DecodeDiagnostics decode;
};

struct EvaluationStreamStatistics final {
    std::optional<std::uint64_t> frame_count_hint;
    std::uint64_t decoded_frame_count{};
    std::uint64_t scheduled_frames{};
    std::uint64_t acquired_frames{};
    std::size_t record_capacity{};
    std::size_t owned_pairs{};
    std::size_t ready_lead_frames{};
    std::uint64_t frame_pair_copy_calls{};
    std::uint64_t frame_pair_copy_bytes{};
    std::uint64_t frame_pair_copy_nanoseconds{};
    std::uint64_t staging_input_wait_nanoseconds{};
    std::uint64_t staging_input_wait_count{};
    std::uint64_t staging_grant_wait_nanoseconds{};
    std::uint64_t staging_grant_wait_count{};
    std::uint64_t coordinator_wait_nanoseconds{};
    std::uint64_t coordinator_wait_count{};
    std::size_t input_lease_capacity{};
    bool scheduling_complete{};
    bool cancellation_requested{};
    StreamInputStatistics reference;
    StreamInputStatistics distorted;
};

class EvaluationStream final {
public:
    [[nodiscard]] static Result<std::unique_ptr<EvaluationStream>> open(
        StreamOptions options);
    // Opens with cooperative cancellation: requesting stop aborts blocking
    // demux/decode work through the FFmpeg interrupt path and fails the
    // open with ErrorCode::cancelled at stage boundaries. Vulkan device
    // initialization itself cannot be interrupted and always completes.
    [[nodiscard]] static Result<std::unique_ptr<EvaluationStream>> open(
        StreamOptions options,
        std::stop_token stop_token);
    [[nodiscard]] static Result<std::vector<DeviceInfo>> enumerate_devices();

    ~EvaluationStream();
    EvaluationStream(EvaluationStream&&) noexcept;
    EvaluationStream& operator=(EvaluationStream&&) noexcept;

    EvaluationStream(const EvaluationStream&) = delete;
    EvaluationStream& operator=(const EvaluationStream&) = delete;

    [[nodiscard]] const DeviceInfo& device_info() const noexcept;
    [[nodiscard]] const FrameDescriptor& descriptor() const noexcept;
    // Actual model, resolved once at stream initialization.
    [[nodiscard]] ModelVariant model() const noexcept;
    // Media metadata snapshot of both inputs, captured when the stream
    // opened the files.
    [[nodiscard]] const StreamMediaMetadata& metadata() const noexcept;
    [[nodiscard]] std::optional<std::uint64_t> frame_count_hint()
        const noexcept;
    [[nodiscard]] std::uint64_t decoded_frame_count() const noexcept;
    [[nodiscard]] EvaluationStreamStatistics statistics() const;

    // Requests cooperative cancellation. Idempotent and safe to call from a
    // thread other than the one running next()/finish(): input waits wake
    // immediately, in-flight GPU work completes, and the worker thread
    // commits the terminal ErrorCode::cancelled state at the next safe
    // point. After that commit, next()/finish() keep reporting the same
    // cancelled terminal instead of a generic backend failure.
    void request_cancel() noexcept;
    [[nodiscard]] bool cancellation_requested() const noexcept;

    // Pumps ordered input and Vulkan work until at least one result is
    // available. An empty successful result means all input was submitted;
    // finish() then drains the remaining GPU and temporal results.
    [[nodiscard]] Result<std::vector<FrameResult>> next();
    [[nodiscard]] Result<std::vector<FrameResult>> finish();

private:
    friend class detail::EvaluationStreamTelemetryOpener;

    class Impl;

    explicit EvaluationStream(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

}  // namespace crossvmaf
