#pragma once
#include "metrics/cambi/cambi.hpp"
#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include "gpu/device.hpp"
#include "gpu/command.hpp"
#include "video/frame.hpp"
#include "gpu/submission.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/timestamp.hpp"
#include <chrono>
#include <cstddef>
#include <memory>

namespace crossvmaf::metrics::cambi::vulkan {

inline constexpr std::uint32_t scale_count = 5U;
inline constexpr std::uint32_t tile_size = 16U;
inline constexpr std::uint32_t c_value_column_group_size = 32U;
inline constexpr std::uint32_t c_value_tail_scale_segment_rows = 8U;
inline constexpr std::uint32_t c_value_medium_scale_segment_rows = 16U;
inline constexpr std::uint32_t c_value_large_scale_segment_rows = 64U;
inline constexpr std::uint32_t c_value_medium_scale_height_threshold = 256U;
inline constexpr std::uint32_t c_value_large_scale_height_threshold = 512U;
inline constexpr std::uint32_t linear_group_size = 256U;
inline constexpr std::uint32_t radix_values_per_thread = 4U;
// Prefix, rank, K, prefix mask, nonzero count, and exact sparse sum (low/high).
inline constexpr std::uint32_t selection_state_word_count = 7U;
inline constexpr std::uint32_t radix_bits = 8U;
inline constexpr std::uint32_t radix_bin_count = 1U << radix_bits;
inline constexpr std::uint32_t radix_pass_count = 32U / radix_bits;
inline constexpr std::uint16_t base_window_size = 65U;

static_assert((static_cast<std::uint64_t>(base_window_size) * base_window_size << 18U) <
              (std::uint64_t{1} << 31U));

struct CValueSegmentRowsConfiguration {
    std::uint32_t medium_scale_rows{c_value_medium_scale_segment_rows};
    std::uint32_t tail_scale_rows{c_value_tail_scale_segment_rows};

    [[nodiscard]] constexpr bool
    operator==(const CValueSegmentRowsConfiguration &) const noexcept = default;
};

inline constexpr CValueSegmentRowsConfiguration default_c_value_segment_rows_configuration{};

[[nodiscard]] constexpr bool
is_supported_c_value_segment_rows_override(std::uint32_t segment_rows) noexcept {
    return segment_rows == 4U || segment_rows == 8U || segment_rows == 16U || segment_rows == 32U;
}

[[nodiscard]] inline std::optional<CValueSegmentRowsConfiguration>
parse_c_value_segment_rows_configuration(std::string_view text) noexcept {
    const auto trim_ascii_whitespace = [](std::string_view value) noexcept {
        while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
            value.remove_prefix(1U);
        }
        while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
            value.remove_suffix(1U);
        }
        return value;
    };
    const auto parse_segment_rows =
        [trim_ascii_whitespace](std::string_view value) -> std::optional<std::uint32_t> {
        value = trim_ascii_whitespace(value);
        if (value.empty() || value.size() > 2U) return std::nullopt;
        std::uint32_t parsed_value{};
        for (const char character : value) {
            if (character < '0' || character > '9') return std::nullopt;
            parsed_value = parsed_value * 10U + static_cast<std::uint32_t>(character - '0');
        }
        return is_supported_c_value_segment_rows_override(parsed_value)
                   ? std::optional<std::uint32_t>{parsed_value}
                   : std::nullopt;
    };

    text = trim_ascii_whitespace(text);
    if (text == "auto") return default_c_value_segment_rows_configuration;

    const std::size_t separator = text.find(',');
    if (separator == std::string_view::npos) {
        const auto segment_rows = parse_segment_rows(text);
        if (!segment_rows) return std::nullopt;
        return CValueSegmentRowsConfiguration{*segment_rows, *segment_rows};
    }
    if (text.find(',', separator + 1U) != std::string_view::npos) {
        return std::nullopt;
    }
    const auto medium_scale_rows = parse_segment_rows(text.substr(0U, separator));
    const auto tail_scale_rows = parse_segment_rows(text.substr(separator + 1U));
    if (!medium_scale_rows || !tail_scale_rows) return std::nullopt;
    return CValueSegmentRowsConfiguration{*medium_scale_rows, *tail_scale_rows};
}

[[nodiscard]] constexpr std::uint32_t
c_value_segment_rows(std::uint32_t height,
                     CValueSegmentRowsConfiguration configuration =
                         default_c_value_segment_rows_configuration) noexcept {
    const std::uint32_t target_segment_rows =
        height >= c_value_large_scale_height_threshold    ? c_value_large_scale_segment_rows
        : height >= c_value_medium_scale_height_threshold ? configuration.medium_scale_rows
                                                          : configuration.tail_scale_rows;
    return height < target_segment_rows ? height : target_segment_rows;
}

[[nodiscard]] constexpr std::uint32_t
c_value_segment_count(std::uint32_t height,
                      CValueSegmentRowsConfiguration configuration =
                          default_c_value_segment_rows_configuration) noexcept {
    const std::uint32_t segment_rows = c_value_segment_rows(height, configuration);
    return height / segment_rows + (height % segment_rows == 0U ? 0U : 1U);
}

static_assert(c_value_segment_count(68U) == 9U);
static_assert(c_value_segment_count(128U) == 16U);
static_assert(c_value_segment_count(255U) == 32U);
static_assert(c_value_segment_count(256U) == 16U);
static_assert(c_value_segment_count(270U) == 17U);
static_assert(c_value_segment_count(511U) == 32U);
static_assert(c_value_segment_count(512U) == 8U);
static_assert(c_value_segment_count(540U) == 9U);
static_assert(c_value_segment_count(1024U) == 16U);

struct U64Words {
    std::uint32_t low{};
    std::uint32_t high{};
};

struct PoolPartial {
    U64Words sum{};
    std::uint32_t count{};
    std::uint32_t padding{};
};

struct PreprocessParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t stride_bytes{};
    std::uint32_t bit_depth{};
};

struct MaskParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t threshold{};
};

struct DecimateParameters {
    std::uint32_t input_width{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
};

struct ModeParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t produce_next_scale{};
};

struct CValueParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t window_size{};
    std::uint32_t segment_rows{};
    std::uint32_t output_offset{};
    std::uint32_t state_offset{};
};

struct RadixInitParameters {
    std::array<std::uint32_t, scale_count> top_counts{};
};

struct RadixHistogramParameters {
    std::array<std::uint32_t, scale_count> value_offsets{};
    std::array<std::uint32_t, scale_count> value_counts{};
    std::array<std::uint32_t, scale_count> group_offsets{};
    std::array<std::uint32_t, scale_count> partial_offsets{};
    std::uint32_t digit_shift{};
};

struct RadixHistogram2DParameters {
    std::array<std::uint32_t, scale_count> value_offsets{};
    std::array<std::uint32_t, scale_count> value_counts{};
    std::array<std::uint32_t, scale_count> group_offsets{};
    std::array<std::uint32_t, scale_count> partial_offsets{};
    std::uint32_t digit_shift{};
    std::uint32_t group_row_width{};
    std::uint32_t group_count{};
};

struct RadixUpdateParameters {
    std::uint32_t digit_shift{};
    std::array<std::uint32_t, scale_count> partial_offsets{};
};

struct PoolReduceParameters {
    std::uint32_t input_count{};
    std::uint32_t partial_offset{};
};

struct PoolFinalizeParameters {
    std::uint32_t scale{};
    std::uint32_t partial_offset{};
};

struct CombineParameters {
    std::uint32_t window_size{};
    U64Words maximum{};
};

static_assert(sizeof(U64Words) == 8U);
static_assert(sizeof(PoolPartial) == 16U);
static_assert(sizeof(PreprocessParameters) == 16U);
static_assert(sizeof(MaskParameters) == 12U);
static_assert(sizeof(DecimateParameters) == 12U);
static_assert(sizeof(ModeParameters) == 12U);
static_assert(sizeof(CValueParameters) == 24U);
static_assert(sizeof(RadixInitParameters) == 20U);
static_assert(sizeof(RadixHistogramParameters) == 84U);
static_assert(sizeof(RadixHistogram2DParameters) == 92U);
static_assert(sizeof(RadixUpdateParameters) == 24U);
static_assert(sizeof(PoolReduceParameters) == 8U);
static_assert(sizeof(PoolFinalizeParameters) == 8U);
static_assert(sizeof(CombineParameters) == 12U);

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi::vulkan {

enum class GpuStage : std::uint8_t {
    preprocess,
    mask,
    decimate,
    mode,
    c_value,
    radix,
    pool_reduce,
    combine,
    count,
};

inline constexpr std::size_t gpu_stage_count = static_cast<std::size_t>(GpuStage::count);

[[nodiscard]] const char *gpu_stage_name(GpuStage stage) noexcept;

struct CValueScaleConfiguration final {
    bool configured{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint16_t window_size{};
    std::uint32_t segment_rows{};
    std::uint32_t segment_count{};
    std::uint64_t initial_window_source_visits{};
    std::uint32_t dispatch_x{};
    std::uint32_t dispatch_y{};
    std::uint32_t dispatch_z{};
};

struct TelemetrySnapshot final {
    gpu::GpuOperationHandle operation_handle{};
    gpu::GpuOperationSnapshot operation;
    gpu::GpuDurationStatistics latency_span;
    gpu::GpuDurationStatistics prefix_to_suffix_gap;
    std::array<gpu::GpuDurationStatistics, gpu_stage_count> stages{};
    std::array<CValueScaleConfiguration, scale_count> c_value_scale_configurations{};
    std::array<gpu::GpuDurationStatistics, scale_count> c_value_scale_statistics{};
};

class Telemetry final {
  public:
    explicit Telemetry(gpu::TelemetryCollector &collector);

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] gpu::TelemetryCollector &collector() const noexcept;
    [[nodiscard]] gpu::GpuOperationHandle operation_handle() const noexcept;

    void configure_timestamps(const gpu::GpuTimestampConfiguration &configuration) noexcept;
    void record_submission(bool timestamp_available) noexcept;
    void record_parent_success(long double active_duration_nanoseconds,
                               long double latency_span_nanoseconds,
                               long double prefix_to_suffix_gap_nanoseconds) noexcept;
    void record_parent_failure() noexcept;
    void record_parent_skip() noexcept;
    void record_submission_latency(std::chrono::nanoseconds duration) noexcept;
    void record_timeline(std::uint64_t start_timestamp, std::uint64_t end_timestamp,
                         std::uint32_t timestamp_valid_bits, double timestamp_period_nanoseconds,
                         crossvmaf::gpu::TimelineSegment segment) noexcept;

    void record_stage_success(GpuStage stage, long double duration_nanoseconds) noexcept;
    void record_stage_failure(GpuStage stage) noexcept;
    void record_stage_skip(GpuStage stage) noexcept;

    void configure_c_value_scale(std::uint32_t scale_index,
                                 const CValueScaleConfiguration &configuration) noexcept;
    void record_c_value_scale_success(std::uint32_t scale_index,
                                      long double duration_nanoseconds) noexcept;
    void record_c_value_scale_failure(std::uint32_t scale_index) noexcept;
    void record_c_value_scale_skip(std::uint32_t scale_index) noexcept;

    [[nodiscard]] const gpu::GpuOperationSnapshot &operation() const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &latency_span_statistics() const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &
    prefix_to_suffix_gap_statistics() const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &stage_statistics(GpuStage stage) const noexcept;
    [[nodiscard]] const CValueScaleConfiguration &
    c_value_scale_configuration(std::uint32_t scale_index) const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &
    c_value_scale_statistics(std::uint32_t scale_index) const noexcept;
    [[nodiscard]] TelemetrySnapshot snapshot() const;

  private:
    [[nodiscard]] static bool is_valid_stage(GpuStage stage) noexcept;
    [[nodiscard]] static bool is_valid_scale(std::uint32_t scale_index) noexcept;

    gpu::TelemetryCollector *collector_{};
    gpu::GpuOperationHandle operation_{};
    gpu::DurationSeriesHandle parent_detail_series_{};
    gpu::DurationSeriesHandle stage_series_{};
    gpu::DurationSeriesHandle c_value_scale_series_{};
    std::array<CValueScaleConfiguration, scale_count> c_value_scale_configurations_{};
};

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi::vulkan {

class Telemetry;
class Resources;

class Executor final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Executor>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           Telemetry *telemetry = nullptr,
           DetailedDiagnosticsMode diagnostics_mode = DetailedDiagnosticsMode::disabled);

    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;
    ~Executor();

    [[nodiscard]] Result<DetailedResult> compute_detailed(const video::GpuFrameView &distorted);

    [[nodiscard]] Result<VkCommandBuffer> record(const video::GpuFrameView &distorted);
    [[nodiscard]] Result<VkCommandBuffer> record_prefix(const video::GpuFrameView &distorted);
    [[nodiscard]] Result<VkCommandBuffer> record_suffix();
    [[nodiscard]] Result<void> mark_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<void> mark_prefix_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<void> retire_prefix(gpu::SubmissionTicket ticket,
                                             std::chrono::nanoseconds submission_host_elapsed);
    [[nodiscard]] Result<DetailedResult> retire(gpu::SubmissionTicket ticket,
                                                std::chrono::nanoseconds submission_host_elapsed);
    void cancel() noexcept;

  private:
    explicit Executor(std::unique_ptr<Resources> resources) noexcept;

    [[nodiscard]] Result<DetailedResult> readback(std::chrono::nanoseconds submission_host_elapsed);

    std::unique_ptr<Resources> resources_;
    gpu::ConveniencePendingState result_state_;
    gpu::ConveniencePendingState prefix_state_;
    bool pending_timestamp_query_pool_available_{};
    bool pending_prefix_timestamp_recording_valid_{};
    bool pending_timestamp_recording_valid_{};
    std::chrono::nanoseconds pending_prefix_submission_host_elapsed_{};
};

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi {
// Owns this metric's reusable operation slots and sequence-wide state.
// Evaluation borrows operation slots to record/retire commands, but never
// owns their buffers, pipelines, or streaming batch allocations.
class Vulkan final {
  public:
    static Result<std::unique_ptr<Vulkan>> create(gpu::DeviceView device,
                                                  FrameDescriptor descriptor, Config config,
                                                  vulkan::Telemetry *telemetry);
    ~Vulkan();
    Vulkan(const Vulkan &) = delete;
    Vulkan &operator=(const Vulkan &) = delete;
    Result<void> append_slot();
    vulkan::Executor &slot(std::size_t index) noexcept;

  private:
    class Impl;
    explicit Vulkan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace crossvmaf::metrics::cambi
