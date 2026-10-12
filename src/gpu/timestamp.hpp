#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include "gpu/device.hpp"
#include "crossvmaf/error.hpp"
#include <volk.h>
#include <memory>

namespace crossvmaf::gpu {

inline constexpr std::uint32_t invalid_telemetry_handle = std::numeric_limits<std::uint32_t>::max();

struct GpuOperationHandle final {
    std::uint32_t value{invalid_telemetry_handle};

    [[nodiscard]] explicit operator bool() const noexcept {
        return value != invalid_telemetry_handle;
    }
};

struct DurationSeriesHandle final {
    std::uint32_t value{invalid_telemetry_handle};

    [[nodiscard]] explicit operator bool() const noexcept {
        return value != invalid_telemetry_handle;
    }
};

struct WaveDurationSeriesHandle final {
    std::uint32_t value{invalid_telemetry_handle};

    [[nodiscard]] explicit operator bool() const noexcept {
        return value != invalid_telemetry_handle;
    }
};

enum class TelemetrySampleOutcome : std::uint8_t {
    success,
    failure,
    skipped,
};

struct HostDurationStatistics final {
    std::uint64_t call_count{};
    std::chrono::nanoseconds total{};
    std::chrono::nanoseconds minimum{};
    std::chrono::nanoseconds maximum{};
};

struct GpuDurationStatistics final {
    // call_count contains only successful duration samples.
    std::uint64_t call_count{};
    std::uint64_t failed_call_count{};
    std::uint64_t skipped_call_count{};
    long double total_nanoseconds{};
    long double minimum_nanoseconds{};
    long double maximum_nanoseconds{};
};

struct GpuWaveDurationStatistics final {
    std::uint64_t wave_count{};
    std::uint64_t failed_wave_count{};
    std::uint64_t task_count{};
    long double total_wave_nanoseconds{};
    long double minimum_wave_nanoseconds{};
    long double maximum_wave_nanoseconds{};
    long double total_amortized_nanoseconds{};
};

struct GpuOperationStatistics final {
    std::uint64_t logical_call_count{};
    std::uint64_t submitted_call_count{};
    std::uint64_t timestamp_unavailable_call_count{};
};

struct GpuTimestampConfiguration final {
    bool device_supported{};
    bool query_pool_available{};
    std::optional<std::uint32_t> queue_family_index{};
    std::optional<std::uint32_t> timestamp_valid_bits{};
    std::optional<long double> timestamp_period_nanoseconds{};
    std::string_view unavailable_reason{};
};

struct GpuTimestampMetadata final {
    bool configured{};
    bool device_supported{};
    bool query_pool_available{};
    std::optional<std::uint32_t> queue_family_index{};
    std::optional<std::uint32_t> timestamp_valid_bits{};
    std::optional<long double> timestamp_period_nanoseconds{};
    std::string unavailable_reason{};
};

struct HostTimelineEvent final {
    std::uint32_t stage_id{};
    std::chrono::steady_clock::time_point started_at{};
    std::chrono::steady_clock::time_point finished_at{};
};

struct PendingSubmissionTimelineEvent final {
    std::chrono::steady_clock::time_point observed_at{};
    std::uint64_t pending_count{};
};

struct GpuTimelineEvent final {
    GpuOperationHandle operation{};
    std::uint32_t kind_id{};
    long double start_nanoseconds{};
    long double end_nanoseconds{};
};

struct InputBufferMemoryConfiguration {
    std::uint32_t memory_type_index{};
    std::uint32_t memory_heap_index{};
    std::uint32_t property_flags{};
    std::uint64_t non_coherent_atom_size{1U};
    bool host_visible{};
    bool host_coherent{};
    bool device_local{};
};

struct InputBufferMemoryPlacement final : InputBufferMemoryConfiguration {
    std::uint64_t buffer_count{};
    bool uniform_across_buffers{true};
};

struct VulkanRuntimeConfiguration final {
    bool configured{};
    std::uint64_t requested_token_depth{};
    std::uint64_t effective_token_depth{};
    std::uint64_t spatial_input_lease_depth{};
    std::uint64_t reference_luma_lease_depth{};
    std::uint64_t maximum_pending_submissions{};
    std::uint64_t depth_scaled_buffer_bytes{};
};

struct InputLeaseRuntimeStatistics final {
    bool configured{};
    std::uint64_t logical_buffer_bytes{};
    std::uint64_t reservation_count{};
    std::uint64_t spatial_release_count{};
    std::uint64_t reference_luma_release_count{};
    std::uint64_t maximum_active_spatial_leases{};
    std::uint64_t maximum_active_reference_luma_leases{};
    std::chrono::nanoseconds spatial_hold_total{};
    std::chrono::nanoseconds reference_luma_hold_total{};
    std::chrono::nanoseconds spatial_hold_maximum{};
    std::chrono::nanoseconds reference_luma_hold_maximum{};
};

struct UploadByteStatistics final {
    std::uint64_t frame_count{};
    std::uint64_t total_bytes{};
    std::uint64_t bytes_per_frame{};
    bool uniform_bytes_per_frame{true};
};

struct SubmissionOverview final {
    std::uint64_t current_pending{};
    std::uint64_t maximum_pending_observed{};
    std::uint64_t pending_count_transition_count{};
    std::chrono::nanoseconds observation_duration{};
    std::chrono::nanoseconds zero_pending_duration{};
    long double average_pending{};
};

struct DurationSeriesSnapshot final {
    std::vector<GpuDurationStatistics> members;
};

struct WaveDurationSeriesSnapshot final {
    std::vector<GpuWaveDurationStatistics> members;
};

struct GpuOperationSnapshot final {
    GpuTimestampMetadata timestamp_metadata;
    GpuDurationStatistics parent_statistics;
    GpuOperationStatistics operation_statistics;
    HostDurationStatistics submission_latency_statistics;
};

struct TelemetrySnapshot final {
    std::vector<HostDurationStatistics> host_stage_statistics;
    std::chrono::steady_clock::time_point timeline_origin{};
    bool timeline_origin_recorded{};
    std::vector<HostTimelineEvent> host_timeline_events;
    std::vector<PendingSubmissionTimelineEvent> pending_submission_timeline_events;
    bool host_timeline_truncated{};
    bool pending_submission_timeline_truncated{};

    std::vector<GpuOperationSnapshot> gpu_operations;
    std::vector<DurationSeriesSnapshot> duration_series;
    std::vector<WaveDurationSeriesSnapshot> wave_duration_series;
    std::vector<GpuTimelineEvent> gpu_timeline_events;
    long double gpu_timeline_period_nanoseconds{};
    std::uint64_t gpu_timeline_rejected_event_count{};
    bool gpu_timeline_origin_recorded{};
    bool gpu_timeline_truncated{};

    InputBufferMemoryPlacement input_buffer_memory_placement;
    VulkanRuntimeConfiguration vulkan_runtime_configuration;
    InputLeaseRuntimeStatistics input_lease_runtime_statistics;
    UploadByteStatistics upload_byte_statistics;
    SubmissionOverview submission_overview;
};

// Generic Vulkan telemetry storage. Callers assign all semantic meaning to
// numeric handles and member ids outside runtime/.
class TelemetryCollector final {
  public:
    explicit TelemetryCollector(bool enabled, std::size_t host_stage_count = 0U);

    TelemetryCollector(const TelemetryCollector &) = delete;
    TelemetryCollector &operator=(const TelemetryCollector &) = delete;

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    [[nodiscard]] GpuOperationHandle register_gpu_operation();
    [[nodiscard]] DurationSeriesHandle register_duration_series(std::size_t member_count);
    [[nodiscard]] WaveDurationSeriesHandle register_wave_duration_series(std::size_t member_count);

    void record_host_duration(std::uint32_t stage_id,
                              std::chrono::steady_clock::time_point started_at,
                              std::chrono::steady_clock::time_point finished_at) noexcept;
    void
    configure_input_buffer_memory(const InputBufferMemoryConfiguration &configuration) noexcept;
    void configure_vulkan_runtime(std::size_t requested_token_depth,
                                  std::size_t effective_token_depth,
                                  std::size_t spatial_input_lease_depth,
                                  std::size_t reference_luma_lease_depth,
                                  std::size_t maximum_pending_submissions,
                                  std::uint64_t depth_scaled_buffer_bytes) noexcept;
    void configure_input_lease_runtime(const InputLeaseRuntimeStatistics &statistics) noexcept;
    void record_upload_bytes(std::uint64_t byte_count) noexcept;
    void record_pending_submissions(std::size_t pending_count) noexcept;

    void configure_gpu_timestamps(GpuOperationHandle operation,
                                  const GpuTimestampConfiguration &configuration) noexcept;
    void record_gpu_submission(GpuOperationHandle operation, bool timestamp_available) noexcept;
    void record_gpu_parent(GpuOperationHandle operation, TelemetrySampleOutcome outcome,
                           long double duration_nanoseconds = 0.0L) noexcept;
    void record_gpu_submission_latency(GpuOperationHandle operation,
                                       std::chrono::nanoseconds duration) noexcept;
    void record_gpu_timeline(GpuOperationHandle operation, std::uint32_t kind_id,
                             std::uint64_t start_timestamp, std::uint64_t end_timestamp,
                             std::uint32_t timestamp_valid_bits,
                             double timestamp_period_nanoseconds) noexcept;

    void record_duration(DurationSeriesHandle series, std::uint32_t member_id,
                         TelemetrySampleOutcome outcome,
                         long double duration_nanoseconds = 0.0L) noexcept;
    void record_wave_duration(WaveDurationSeriesHandle series, std::uint32_t member_id,
                              TelemetrySampleOutcome outcome, long double wave_nanoseconds = 0.0L,
                              std::uint32_t task_count = 0U) noexcept;

    [[nodiscard]] const HostDurationStatistics &
    host_statistics(std::uint32_t stage_id) const noexcept;
    [[nodiscard]] const GpuOperationSnapshot &
    gpu_operation(GpuOperationHandle operation) const noexcept;
    [[nodiscard]] const GpuDurationStatistics &
    duration_statistics(DurationSeriesHandle series, std::uint32_t member_id) const noexcept;
    [[nodiscard]] const GpuWaveDurationStatistics &
    wave_statistics(WaveDurationSeriesHandle series, std::uint32_t member_id) const noexcept;

    [[nodiscard]] TelemetrySnapshot snapshot();

  private:
    void establish_timeline_origin(std::chrono::steady_clock::time_point observation) noexcept;
    void record_pending_submission_timeline(std::chrono::steady_clock::time_point observed_at,
                                            std::uint64_t pending_count) noexcept;
    void accumulate_submission_observation(
        std::chrono::steady_clock::time_point observation_time) noexcept;
    void finalize_submission_observation() noexcept;

    bool enabled_{};
    std::vector<HostDurationStatistics> host_stage_statistics_;
    std::chrono::steady_clock::time_point timeline_origin_{};
    bool timeline_origin_recorded_{};
    std::vector<HostTimelineEvent> host_timeline_events_;
    std::vector<PendingSubmissionTimelineEvent> pending_submission_timeline_events_;
    bool host_timeline_truncated_{};
    bool pending_submission_timeline_truncated_{};

    std::vector<GpuOperationSnapshot> gpu_operations_;
    std::vector<DurationSeriesSnapshot> duration_series_;
    std::vector<WaveDurationSeriesSnapshot> wave_duration_series_;
    std::vector<GpuTimelineEvent> gpu_timeline_events_;
    std::uint64_t gpu_timeline_origin_timestamp_{};
    std::uint64_t gpu_timeline_high_watermark_timestamp_{};
    std::uint64_t gpu_timeline_rejected_event_count_{};
    std::uint32_t gpu_timeline_valid_bits_{};
    double gpu_timeline_period_nanoseconds_{};
    bool gpu_timeline_origin_recorded_{};
    bool gpu_timeline_truncated_{};

    InputBufferMemoryPlacement input_buffer_memory_placement_{};
    VulkanRuntimeConfiguration vulkan_runtime_configuration_{};
    InputLeaseRuntimeStatistics input_lease_runtime_statistics_{};
    UploadByteStatistics upload_byte_statistics_{};
    SubmissionOverview submission_overview_{};
    std::chrono::steady_clock::time_point submission_observation_started_at_{};
    std::chrono::steady_clock::time_point last_pending_transition_at_{};
    long double pending_submission_nanoseconds_{};
    bool submission_observation_active_{};
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

// Execution-shape labels describe generic recorded command intervals.
enum class TimelineSegment : std::uint8_t {
    full,
    prefix,
    suffix,
    wave,
};

[[nodiscard]] constexpr std::uint32_t timeline_segment_id(TimelineSegment segment) noexcept {
    return static_cast<std::uint32_t>(segment);
}

[[nodiscard]] constexpr const char *timeline_segment_name(TimelineSegment segment) noexcept {
    switch (segment) {
    case TimelineSegment::full:
        return "full";
    case TimelineSegment::prefix:
        return "prefix";
    case TimelineSegment::suffix:
        return "suffix";
    case TimelineSegment::wave:
        return "wave";
    }
    return "unknown";
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

struct TimestampQueryCapabilities {
    std::uint32_t timestamp_valid_bits{};
    double timestamp_period_nanoseconds{};

    [[nodiscard]] bool supported() const noexcept { return timestamp_valid_bits != 0U; }
};

struct TimestampQueryResult {
    // Vulkan only defines value when availability is non-zero. Unavailable
    // values returned by TimestampQueryPool are normalized to zero.
    std::uint64_t value{};
    std::uint64_t availability{};

    [[nodiscard]] bool available() const noexcept { return availability != 0U; }
};

struct TimestampDuration {
    std::uint64_t elapsed_ticks{};
    long double nanoseconds{};
    bool wrapped{};
};

// Queries the timestamp properties of DeviceView::queue_family. A valid
// result with supported() == false means the queue cannot write timestamps;
// it does not make the Vulkan device unusable for ordinary computation.
[[nodiscard]] Result<TimestampQueryCapabilities> query_timestamp_capabilities(DeviceView device);

// Converts the valid low timestamp bits using the legacy assumption that the
// interval spans less than one full counter period and can therefore cross at
// most one wrap. Supplying a host measurement-window upper bound makes that
// assumption verifiable; a bound that reaches the full counter period is
// rejected as ambiguous, and a computed duration above the bound is rejected
// as inconsistent with the host submission window. Availability is
// independent of a timestamp value, so zero is a valid start or end tick.
[[nodiscard]] Result<TimestampDuration>
calculate_timestamp_duration(const TimestampQueryResult &start, const TimestampQueryResult &end,
                             std::uint32_t timestamp_valid_bits,
                             long double timestamp_period_nanoseconds,
                             std::optional<long double> maximum_elapsed_nanoseconds = std::nullopt);

[[nodiscard]] Result<TimestampDuration>
calculate_timestamp_duration(const TimestampQueryResult &start, const TimestampQueryResult &end,
                             const TimestampQueryCapabilities &capabilities,
                             std::optional<long double> maximum_elapsed_nanoseconds = std::nullopt);

class TimestampQueryPool final {
  public:
    // Unsupported queue families produce a valid object for which supported()
    // is false and no VkQueryPool is allocated. Vulkan allocation failures are
    // returned to the caller so profiling can independently choose fail-open.
    [[nodiscard]] static Result<std::unique_ptr<TimestampQueryPool>>
    create(DeviceView device, std::uint32_t query_count);

    TimestampQueryPool(const TimestampQueryPool &) = delete;
    TimestampQueryPool &operator=(const TimestampQueryPool &) = delete;
    TimestampQueryPool(TimestampQueryPool &&) = delete;
    TimestampQueryPool &operator=(TimestampQueryPool &&) = delete;
    ~TimestampQueryPool();

    [[nodiscard]] bool supported() const noexcept { return capabilities_.supported(); }
    [[nodiscard]] const TimestampQueryCapabilities &capabilities() const noexcept {
        return capabilities_;
    }

    [[nodiscard]] Result<void> reset_queries(VkCommandBuffer command_buffer,
                                             std::uint32_t first_query,
                                             std::uint32_t query_count) const;

    [[nodiscard]] Result<void> write_timestamp(
        VkCommandBuffer command_buffer, std::uint32_t query_index,
        VkPipelineStageFlagBits pipeline_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) const;

    // The submission containing the timestamp writes must have completed
    // before this call. The implementation reports availability without using
    // VK_QUERY_RESULT_WAIT_BIT. VK_NOT_READY is represented by a successful
    // result containing one or more entries with availability == 0; negative
    // VkResult errors are returned as Result failures.
    [[nodiscard]] Result<std::vector<TimestampQueryResult>>
    read_results(std::uint32_t first_query, std::uint32_t query_count) const;

  private:
    TimestampQueryPool(DeviceView device, std::uint32_t query_count,
                       TimestampQueryCapabilities capabilities) noexcept;

    [[nodiscard]] Result<void> validate_query_range(std::uint32_t first_query,
                                                    std::uint32_t query_count) const;
    [[nodiscard]] Result<void> validate_supported() const;

    DeviceView device_;
    VkQueryPool query_pool_{VK_NULL_HANDLE};
    std::uint32_t query_count_{};
    TimestampQueryCapabilities capabilities_;
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

struct GpuTimestampRecordingState final {
    bool query_pool_available{};
    bool recording_valid{};
};

struct GpuTimestampIntervalRecordingState final {
    GpuTimestampRecordingState parent;
    GpuTimestampRecordingState intervals;
};

// Owns one reusable timestamp query pair for a frame-slot-local executor.
class GpuTimestampPair final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<GpuTimestampPair>>
    create(DeviceView device, TelemetryCollector *telemetry_collector, GpuOperationHandle operation,
           VkPipelineStageFlagBits timestamp_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

    GpuTimestampPair(const GpuTimestampPair &) = delete;
    GpuTimestampPair &operator=(const GpuTimestampPair &) = delete;
    GpuTimestampPair(GpuTimestampPair &&) = delete;
    GpuTimestampPair &operator=(GpuTimestampPair &&) = delete;
    ~GpuTimestampPair() = default;

    [[nodiscard]] GpuTimestampRecordingState
    begin_recording(VkCommandBuffer command_buffer) const noexcept;
    void write_start(VkCommandBuffer command_buffer,
                     GpuTimestampRecordingState &recording_state) const noexcept;
    void write_end(VkCommandBuffer command_buffer,
                   GpuTimestampRecordingState &recording_state) const noexcept;

    [[nodiscard]] Result<void> collect(const GpuTimestampRecordingState &recording_state,
                                       std::chrono::nanoseconds submission_host_elapsed,
                                       std::uint32_t timeline_kind_id = 0U) const;
    void record_skip() const noexcept;

  private:
    GpuTimestampPair(TelemetryCollector *telemetry_collector, GpuOperationHandle operation,
                     VkPipelineStageFlagBits timestamp_stage) noexcept;

    void write_timestamp(VkCommandBuffer command_buffer, std::uint32_t query_index,
                         GpuTimestampRecordingState &recording_state) const noexcept;

    TelemetryCollector *telemetry_collector_{};
    GpuOperationHandle operation_{};
    VkPipelineStageFlagBits timestamp_stage_{VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
    std::unique_ptr<TimestampQueryPool> query_pool_;
};

// Adds adjacent boundary intervals to a parent query pair. Stage count and
// enablement are supplied by the caller, so runtime owns no feature topology.
class GpuTimestampIntervals final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<GpuTimestampIntervals>>
    create(DeviceView device, TelemetryCollector *telemetry_collector, GpuOperationHandle operation,
           DurationSeriesHandle interval_series, std::uint32_t interval_count,
           bool intervals_enabled);

    GpuTimestampIntervals(const GpuTimestampIntervals &) = delete;
    GpuTimestampIntervals &operator=(const GpuTimestampIntervals &) = delete;
    GpuTimestampIntervals(GpuTimestampIntervals &&) = delete;
    GpuTimestampIntervals &operator=(GpuTimestampIntervals &&) = delete;
    ~GpuTimestampIntervals() = default;

    [[nodiscard]] GpuTimestampIntervalRecordingState
    begin_recording(VkCommandBuffer command_buffer) const noexcept;
    void write_boundary(VkCommandBuffer command_buffer, std::uint32_t boundary_index,
                        GpuTimestampIntervalRecordingState &recording_state) const noexcept;

    [[nodiscard]] Result<void> collect(const GpuTimestampIntervalRecordingState &recording_state,
                                       std::chrono::nanoseconds submission_host_elapsed,
                                       std::uint32_t timeline_kind_id = 0U) const;
    void write_parent_start(VkCommandBuffer command_buffer,
                            GpuTimestampIntervalRecordingState &recording_state) const noexcept;
    void write_parent_end(VkCommandBuffer command_buffer,
                          GpuTimestampIntervalRecordingState &recording_state) const noexcept;
    [[nodiscard]] Result<void>
    collect_parent(const GpuTimestampIntervalRecordingState &recording_state,
                   std::chrono::nanoseconds submission_host_elapsed,
                   std::uint32_t timeline_kind_id) const;

  private:
    GpuTimestampIntervals(TelemetryCollector *telemetry_collector,
                          DurationSeriesHandle interval_series,
                          std::uint32_t interval_count) noexcept;

    void record_all_interval_failures() const noexcept;

    TelemetryCollector *telemetry_collector_{};
    DurationSeriesHandle interval_series_{};
    std::uint32_t interval_count_{};
    std::uint32_t query_count_{};
    std::unique_ptr<GpuTimestampPair> parent_timestamp_;
    std::unique_ptr<TimestampQueryPool> query_pool_;
};

} // namespace crossvmaf::gpu
