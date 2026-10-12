#pragma once
#include "metrics/adm3/adm3.hpp"
#include <array>
#include <cstdint>
#include "gpu/device.hpp"
#include "gpu/command.hpp"
#include "video/frame.hpp"
#include "gpu/submission.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/error.hpp"
#include "gpu/timestamp.hpp"
#include <chrono>
#include <cstddef>
#include <string>
#include <memory>
#include "gpu/buffer.hpp"
#include "gpu/pipeline.hpp"
#include <optional>
#include <vector>

namespace crossvmaf::metrics::adm3::vulkan {

enum class ResultMode : std::uint8_t {
    score_only,
    detailed,
};

inline constexpr std::uint32_t scale_count = 4U;
inline constexpr std::uint32_t score_channel_count = 9U;
inline constexpr std::uint32_t detailed_channel_count = 24U;
inline constexpr std::uint32_t channel_count = detailed_channel_count;
inline constexpr std::uint32_t result_slot_count = 109U;
inline constexpr std::uint32_t dwt_tile_size = 16U;
inline constexpr std::uint32_t linear_workgroup_size = 256U;
inline constexpr std::uint32_t metric_segment_width = 32U;
inline constexpr std::uint32_t metric_segments_per_workgroup = 2U;

struct U64Words {
    std::uint32_t low{};
    std::uint32_t high{};
};

struct DwtParameters {
    std::uint32_t input_width{};
    std::uint32_t input_height{};
    std::uint32_t input_stride_bytes{};
    std::uint32_t output_width{};
    std::uint32_t output_height{};
    std::uint32_t bit_depth{};
    std::uint32_t scale{};
    std::uint32_t raw_input{};
};

struct DecoupleParameters {
    std::uint32_t coefficient_count{};
    std::uint32_t scale{};
};

struct MetricSegmentsParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t scale{};
    std::uint32_t horizontal_factor_raw{};
    std::uint32_t diagonal_factor_raw{};
    std::uint32_t row_capacity{};
    std::uint32_t segment_capacity{};
    std::uint32_t segments_per_workgroup{};
};

struct MergeRowsParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t scale{};
    std::uint32_t row_capacity{};
    std::uint32_t segment_count{};
    std::uint32_t segment_capacity{};
};

struct StreamingMergeRowsParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t scale{};
    std::uint32_t row_capacity{};
    std::uint32_t segment_count{};
    std::uint32_t segment_capacity{};
    std::uint32_t packed_row_offset_elements{};
    std::uint32_t packed_row_channel_stride_elements{};
};

struct ReduceParameters {
    std::uint32_t input_count{};
    std::uint32_t channel_capacity{};
};

struct PackScaleParameters {
    std::uint32_t scale{};
    std::uint32_t channel_capacity{};
};

struct StreamingReduceParameters {
    std::uint32_t input_count{};
    std::uint32_t input_task_stride_elements{};
    std::uint32_t input_scale_offset_elements{};
    std::uint32_t input_channel_stride_elements{};
    std::uint32_t output_task_stride_elements{};
    std::uint32_t output_channel_stride_elements{};
};

struct StreamingPackScaleParameters {
    std::uint32_t scale{};
    std::uint32_t input_task_stride_elements{};
    std::uint32_t input_scale_offset_elements{};
    std::uint32_t input_channel_stride_elements{};
    std::uint32_t result_task_stride_elements{};
};

struct FinalizeBatchParameters {
    std::array<std::uint32_t, scale_count> widths{};
    std::array<std::uint32_t, scale_count> heights{};
    std::array<std::uint32_t, scale_count> horizontal_factors_raw{};
    std::array<std::uint32_t, scale_count> diagonal_factors_raw{};
    U64Words noise_weight{};
};

struct CombineParameters {
    U64Words dlm_weight{};
    U64Words minimum{};
};

struct StreamingFinalizeScalesParameters {
    std::array<std::uint32_t, scale_count> widths{};
    std::array<std::uint32_t, scale_count> heights{};
    std::array<std::uint32_t, scale_count> horizontal_factors_raw{};
    std::array<std::uint32_t, scale_count> diagonal_factors_raw{};
    U64Words noise_weight{};
    std::uint32_t result_task_stride_elements{};
};

struct StreamingCombineParameters {
    U64Words dlm_weight{};
    U64Words minimum{};
    std::uint32_t result_task_stride_elements{};
};

static_assert(sizeof(U64Words) == 8U);
static_assert(sizeof(DwtParameters) == 32U);
static_assert(sizeof(DecoupleParameters) == 8U);
static_assert(sizeof(MetricSegmentsParameters) == 32U);
static_assert(sizeof(MergeRowsParameters) == 24U);
static_assert(sizeof(StreamingMergeRowsParameters) == 32U);
static_assert(sizeof(ReduceParameters) == 8U);
static_assert(sizeof(PackScaleParameters) == 8U);
static_assert(sizeof(StreamingReduceParameters) == 24U);
static_assert(sizeof(StreamingPackScaleParameters) == 20U);
static_assert(sizeof(FinalizeBatchParameters) == 72U);
static_assert(sizeof(CombineParameters) == 16U);
static_assert(sizeof(StreamingFinalizeScalesParameters) == 76U);
static_assert(sizeof(StreamingCombineParameters) == 20U);

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {

enum class ScaleOperation : std::uint8_t {
    dwt,
    decouple,
    metric_segments,
    merge_rows,
    reduce,
    pack_scale,
    count,
};

inline constexpr std::uint32_t scale_operation_count =
    static_cast<std::uint32_t>(ScaleOperation::count);
inline constexpr std::uint32_t finalize_stage_id = scale_count * scale_operation_count;
inline constexpr std::size_t gpu_stage_count = static_cast<std::size_t>(finalize_stage_id + 1U);

[[nodiscard]] constexpr std::uint32_t gpu_stage_id(std::uint32_t scale_index,
                                                   ScaleOperation operation) noexcept {
    return scale_index * scale_operation_count + static_cast<std::uint32_t>(operation);
}

[[nodiscard]] std::string gpu_stage_name(std::uint32_t stage_id);
[[nodiscard]] bool stage_timestamps_enabled() noexcept;

struct TelemetrySnapshot final {
    gpu::GpuOperationHandle operation_handle{};
    gpu::GpuOperationSnapshot operation;
    std::array<gpu::GpuDurationStatistics, gpu_stage_count> stages{};
};

class Telemetry final {
  public:
    explicit Telemetry(gpu::TelemetryCollector &collector);

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] gpu::TelemetryCollector &collector() const noexcept;
    [[nodiscard]] gpu::GpuOperationHandle operation_handle() const noexcept;
    [[nodiscard]] gpu::DurationSeriesHandle stage_series_handle() const noexcept;

    void record_submission_latency(std::chrono::nanoseconds duration) noexcept;
    void record_timeline(std::uint64_t start_timestamp, std::uint64_t end_timestamp,
                         std::uint32_t timestamp_valid_bits, double timestamp_period_nanoseconds,
                         crossvmaf::gpu::TimelineSegment segment) noexcept;
    void record_stage_success(std::uint32_t stage_id, long double duration_nanoseconds) noexcept;
    void record_stage_failure(std::uint32_t stage_id) noexcept;

    [[nodiscard]] const gpu::GpuOperationSnapshot &operation() const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &
    stage_statistics(std::uint32_t stage_id) const noexcept;
    [[nodiscard]] TelemetrySnapshot snapshot() const;

  private:
    gpu::TelemetryCollector *collector_{};
    gpu::GpuOperationHandle operation_{};
    gpu::DurationSeriesHandle stage_series_{};
};

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {

class Telemetry;
class Resources;
class StreamingTaskRingResources;
struct StreamingLayout;
struct TaskRingView;
using TaskHandle = gpu::StreamingTaskHandle;

class Executor final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Executor>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           Telemetry *telemetry = nullptr, bool skip_streaming_suffix = false);
    [[nodiscard]] static Result<std::unique_ptr<Executor>>
    create_score_only(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
                      Telemetry *telemetry = nullptr, bool skip_streaming_suffix = false);

    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;
    ~Executor();

    [[nodiscard]] Result<Q32Value> compute(const video::GpuFrameView &reference,
                                           const video::GpuFrameView &distorted);
    [[nodiscard]] Result<DetailedResult> compute_detailed(const video::GpuFrameView &reference,
                                                          const video::GpuFrameView &distorted);

    [[nodiscard]] Result<VkCommandBuffer> record(const video::GpuFrameView &reference,
                                                 const video::GpuFrameView &distorted);
    [[nodiscard]] Result<void> mark_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<DetailedResult> retire(gpu::SubmissionTicket ticket,
                                                std::chrono::nanoseconds submission_host_elapsed);
    [[nodiscard]] Result<VkCommandBuffer> record_stream_prefix(const video::GpuFrameView &reference,
                                                               const video::GpuFrameView &distorted,
                                                               TaskHandle task,
                                                               const TaskRingView &task_ring_view);
    [[nodiscard]] Result<void> mark_stream_prefix_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<void>
    retire_stream_prefix(gpu::SubmissionTicket ticket,
                         std::chrono::nanoseconds submission_host_elapsed);
    [[nodiscard]] bool has_streaming_suffix_work() const noexcept;
    [[nodiscard]] const StreamingLayout &streaming_layout() const noexcept;
    void release_streaming_suffix_resources() noexcept;
    void cancel() noexcept;

  private:
    friend class StreamingTaskRingResources;
    [[nodiscard]] Resources &streaming_resources() noexcept;
    enum class StreamPrefixPhase : std::uint8_t {
        idle,
        prefix_recorded,
        prefix_submitted,
    };

    explicit Executor(std::unique_ptr<Resources> resources) noexcept;

    [[nodiscard]] static Result<std::unique_ptr<Executor>>
    create_with_mode(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
                     ResultMode result_mode, Telemetry *telemetry,
                     bool skip_streaming_suffix = false);

    [[nodiscard]] Result<DetailedResult> readback(std::chrono::nanoseconds submission_host_elapsed);

    std::unique_ptr<Resources> resources_;
    gpu::ConveniencePendingState result_state_;
    StreamPrefixPhase stream_prefix_phase_{StreamPrefixPhase::idle};
};

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {

struct ScaleGeometry {
    std::uint32_t input_width{};
    std::uint32_t input_height{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t pixel_count{};
    std::uint32_t coefficient_count{};
    std::uint32_t region_width{};
    std::uint32_t region_height{};
    std::uint32_t segment_count{};
    gpu::DispatchSize dwt_dispatch{};
    gpu::DispatchSize decouple_dispatch{};
    gpu::DispatchSize metric_dispatch{};
    gpu::DispatchSize merge_dispatch{};
    std::uint32_t packed_row_offset_elements{};
    std::uint32_t packed_row_channel_stride_elements{};
    std::vector<std::uint32_t> reduction_input_counts;
};

struct StreamingScaleLayout final {
    std::uint32_t packed_row_offset_elements{};
    std::uint32_t packed_row_channel_stride_elements{};
};

struct StreamingLayout final {
    ResultMode result_mode{ResultMode::detailed};
    std::uint32_t active_channel_count{detailed_channel_count};
    std::uint32_t packed_row_frame_elements{};
    std::uint32_t reduction_channel_capacity{};
    std::uint32_t reduction_frame_elements{};
    std::uint32_t result_frame_elements{result_slot_count};
    std::array<StreamingScaleLayout, scale_count> scales{};
};

using TaskHandle = gpu::StreamingTaskHandle;

struct TaskRingView final {
    TaskHandle task;
    gpu::Buffer *packed_rows{};
    gpu::Buffer *results{};
    VkDeviceSize packed_rows_offset{};
    VkDeviceSize results_offset{};
    VkDeviceSize packed_rows_range{};
    VkDeviceSize results_range{};
    StreamingLayout layout{};
};

struct StreamingTaskFrameResult final {
    std::uint64_t frame_index{};
    TaskHandle task;
    DetailedResult result;
};

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {
class StreamingTaskRingResources final {
  public:
    static Result<std::unique_ptr<StreamingTaskRingResources>> create(Executor &prototype);
    ~StreamingTaskRingResources();
    StreamingTaskRingResources(const StreamingTaskRingResources &) = delete;
    StreamingTaskRingResources &operator=(const StreamingTaskRingResources &) = delete;
    Result<std::optional<TaskHandle>> reserve(std::uint64_t frame_index);
    Result<TaskRingView> view(TaskHandle task) noexcept;
    Result<void> cancel_prefix(TaskHandle task);
    Result<void> publish_ready(TaskHandle task);
    Result<VkCommandBuffer> record_wave();
    Result<void> mark_wave_submitted(gpu::SubmissionTicket ticket);
    Result<std::vector<StreamingTaskFrameResult>> retire_wave(gpu::SubmissionTicket ticket,
                                                              std::chrono::nanoseconds elapsed);
    std::uint32_t capacity() const noexcept;
    std::uint32_t free_slot_count() const noexcept;
    std::uint32_t prefix_pending_task_count() const noexcept;
    std::uint32_t ready_task_count() const noexcept;
    bool wave_in_flight() const noexcept;
    void cancel() noexcept;
    void cancel_after_device_drain() noexcept;

  private:
    class Impl;
    explicit StreamingTaskRingResources(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3 {
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
    Result<void> initialize_batch();
    vulkan::StreamingTaskRingResources *batch() noexcept;

  private:
    class Impl;
    explicit Vulkan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace crossvmaf::metrics::adm3
