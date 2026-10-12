#pragma once
#include "crossvmaf/error.hpp"
#include "metrics/speed_chroma/speed_chroma.hpp"
#include <cstdint>
#include <vector>
#include "numeric/rounding.hpp"
#include "gpu/device.hpp"
#include "gpu/command.hpp"
#include "video/frame.hpp"
#include "gpu/submission.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/timestamp.hpp"
#include <array>
#include <chrono>
#include <cstddef>
#include <memory>
#include <span>
#include "gpu/buffer.hpp"
#include "gpu/pipeline.hpp"
#include <optional>
#include <bit>

namespace crossvmaf::metrics::speed_chroma::vulkan {

// The shader derives the adjacent bilinear source index from a non-zero
// fraction. A fraction rounded to exactly one is normalized by advancing the
// source index and storing zero.
struct ScaleAxisSample final {
    std::uint32_t source_index{};
    std::uint32_t fraction_q0_32{};
};

struct SpeedChromaScalePlan final {
    std::uint32_t horizontal_sample_count{};
    std::uint32_t vertical_sample_count{};
    std::vector<ScaleAxisSample> axis_samples;
};

[[nodiscard]] Result<SpeedChromaScalePlan>
make_speed_chroma_scale_plan(std::uint32_t source_width, std::uint32_t source_height,
                             std::uint32_t destination_width, std::uint32_t destination_height,
                             PrescaleMethod method);

static_assert(sizeof(ScaleAxisSample) == 2U * sizeof(std::uint32_t));

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {

inline constexpr std::uint32_t tile_size = 16U;
inline constexpr std::uint32_t means_workgroup_size = 64U;
inline constexpr std::uint32_t block_statistics_workgroup_size = 32U;
inline constexpr std::uint32_t block_statistics_scalar_workgroup_size = 64U;
inline constexpr std::uint32_t covariance_sample_lane_count = 64U;
inline constexpr std::uint32_t covariance_entries_per_group = 4U;
inline constexpr std::uint32_t covariance_workgroup_size = covariance_sample_lane_count;
inline constexpr std::uint32_t covariance_displacement_count = 41U;
inline constexpr std::uint32_t covariance_displacement_sample_limit = 1U << 19U;
inline constexpr std::uint32_t covariance_shared_memory_bytes =
    2U * 5U * covariance_sample_lane_count * 4U * sizeof(std::uint32_t);
inline constexpr std::uint32_t compact_tridiagonal_workgroup_size = 64U;
inline constexpr std::uint32_t exact_regularity_workgroup_size = 32U;
inline constexpr std::uint32_t covariance_entry_batch_count =
    (elements_per_block + covariance_entries_per_group - 1U) / covariance_entries_per_group;
inline constexpr std::uint32_t means_shared_row_count = 64U;
inline constexpr std::uint32_t means_shared_memory_bytes =
    means_shared_row_count * 4U * sizeof(std::uint32_t) + sizeof(std::uint32_t);
inline constexpr std::uint32_t parameter_stream_count = 4U;
inline constexpr std::uint32_t channel_count = 2U;

[[nodiscard]] constexpr std::uint32_t
block_statistics_scalar_dispatch_group_count(std::uint32_t block_count) noexcept {
    return (block_count + block_statistics_scalar_workgroup_size - 1U) /
           block_statistics_scalar_workgroup_size;
}

// Frozen intermediate ABI for one score-scratch parameter stream. The
// constants at the top of speed_chroma_score.comp mirror this layout because
// GLSL cannot include this C++ header. Offsets and counts are in uvec2/Q32Words
// elements, not bytes.
inline constexpr std::uint64_t score_covariance_offset = 0U;
inline constexpr std::uint64_t score_covariance_elements = elements_per_block * elements_per_block;
inline constexpr std::uint64_t score_means_offset =
    score_covariance_offset + score_covariance_elements;
inline constexpr std::uint64_t score_mean_elements = elements_per_block;
inline constexpr std::uint64_t score_reflector_tau_offset =
    score_means_offset + score_mean_elements;
inline constexpr std::uint64_t score_reflector_tau_elements = elements_per_block - 2U;
inline constexpr std::uint64_t score_reflector_tails_offset =
    score_reflector_tau_offset + score_reflector_tau_elements;
inline constexpr std::uint64_t score_reflector_tail_elements =
    (elements_per_block - 1U) * (elements_per_block - 2U) / 2U;
inline constexpr std::uint64_t score_factor_diagonal_offset =
    score_reflector_tails_offset + score_reflector_tail_elements;
inline constexpr std::uint64_t score_factor_diagonal_elements = elements_per_block;
inline constexpr std::uint64_t score_factor_subdiagonal_offset =
    score_factor_diagonal_offset + score_factor_diagonal_elements;
inline constexpr std::uint64_t score_factor_subdiagonal_elements = elements_per_block - 1U;
inline constexpr std::uint64_t score_ldl_pivots_offset =
    score_factor_subdiagonal_offset + score_factor_subdiagonal_elements;
inline constexpr std::uint64_t score_ldl_pivot_elements = elements_per_block;
inline constexpr std::uint64_t score_ldl_multipliers_offset =
    score_ldl_pivots_offset + score_ldl_pivot_elements;
inline constexpr std::uint64_t score_ldl_multiplier_elements = elements_per_block - 1U;
inline constexpr std::uint64_t score_tridiagonal_diagonal_offset =
    score_ldl_multipliers_offset + score_ldl_multiplier_elements;
inline constexpr std::uint64_t score_tridiagonal_diagonal_elements = elements_per_block;
inline constexpr std::uint64_t score_tridiagonal_subdiagonal_offset =
    score_tridiagonal_diagonal_offset + score_tridiagonal_diagonal_elements;
inline constexpr std::uint64_t score_tridiagonal_subdiagonal_elements = elements_per_block - 1U;
inline constexpr std::uint64_t score_matrix_exponent_offset =
    score_tridiagonal_subdiagonal_offset + score_tridiagonal_subdiagonal_elements;
inline constexpr std::uint64_t score_matrix_exponent_elements = 1U;
inline constexpr std::uint64_t score_regularity_metadata_offset =
    score_matrix_exponent_offset + score_matrix_exponent_elements;
inline constexpr std::uint64_t score_regularity_metadata_elements = 1U;
inline constexpr std::uint64_t score_factor_liveness_dimensions_offset =
    score_regularity_metadata_offset + score_regularity_metadata_elements;
inline constexpr std::uint64_t score_factor_liveness_dimensions_elements = 1U;
inline constexpr std::uint64_t score_factor_liveness_shape_offset =
    score_factor_liveness_dimensions_offset + score_factor_liveness_dimensions_elements;
inline constexpr std::uint64_t score_factor_liveness_shape_elements = 1U;
inline constexpr std::uint64_t score_scratch_base_elements =
    score_factor_liveness_shape_offset + score_factor_liveness_shape_elements;
inline constexpr std::uint64_t score_block_output_elements = 2U;
inline constexpr std::uint64_t score_block_intermediate_elements = elements_per_block + 1U;
inline constexpr std::uint64_t score_scratch_elements_per_block =
    score_block_output_elements + score_block_intermediate_elements;

// Frozen score-result ABI. Entries 6..9 are per-parameter-stream state words.
inline constexpr std::uint64_t score_packet_result_elements = 6U;
inline constexpr std::uint64_t score_stream_state_offset = score_packet_result_elements;
inline constexpr std::uint64_t score_stream_state_elements = parameter_stream_count;
inline constexpr std::uint64_t score_result_elements =
    score_stream_state_offset + score_stream_state_elements;
inline constexpr std::uint32_t score_stream_singular_flag = 1U;
inline constexpr std::uint32_t score_stream_arithmetic_failure_flag = 4U;

// The compact module uses four tightly packed shared uvec2 arrays, three
// shared uvec2 scalar values, and three uint-sized control values. Vulkan's
// scalar/vector alignment leaves this layout at exactly 5,636 bytes.
inline constexpr std::uint32_t compact_tridiagonal_shared_matrix_elements =
    static_cast<std::uint32_t>(score_covariance_elements);
inline constexpr std::uint32_t compact_tridiagonal_shared_vector_elements = elements_per_block;
inline constexpr std::uint32_t compact_tridiagonal_shared_scalar_elements = 3U;
inline constexpr std::uint32_t compact_tridiagonal_shared_control_words = 3U;
inline constexpr std::uint32_t compact_tridiagonal_shared_memory_bytes =
    (compact_tridiagonal_shared_matrix_elements + 3U * compact_tridiagonal_shared_vector_elements +
     compact_tridiagonal_shared_scalar_elements) *
        2U * sizeof(std::uint32_t) +
    compact_tridiagonal_shared_control_words * sizeof(std::uint32_t);
inline constexpr std::uint32_t compact_tridiagonal_shared_memory_limit_bytes = 12U * 1'024U;

// Pipeline 6 reuses one shared workspace for the 96-element prefix scan,
// lane comparison candidates, and two 84-column product accumulators.
// Every shared value is represented as uint words.
inline constexpr std::uint32_t exact_regularity_limb_count = 80U;
inline constexpr std::uint32_t exact_regularity_wide_slot_count = 5U;
inline constexpr std::uint32_t exact_regularity_prefix_element_count = 96U;
inline constexpr std::uint32_t exact_regularity_prefix_buffer_count = 2U;
inline constexpr std::uint32_t exact_regularity_prefix_pair_words = 2U;
inline constexpr std::uint32_t exact_regularity_product_column_count = 84U;
inline constexpr std::uint32_t exact_regularity_product_count = 2U;
inline constexpr std::uint32_t exact_regularity_product_accumulator_count = 2U;
inline constexpr std::uint32_t exact_regularity_shared_limb_words =
    exact_regularity_limb_count * exact_regularity_wide_slot_count;
inline constexpr std::uint32_t exact_regularity_shared_metadata_words =
    2U * exact_regularity_wide_slot_count;
inline constexpr std::uint32_t exact_regularity_prefix_workspace_words =
    exact_regularity_prefix_element_count * exact_regularity_prefix_buffer_count *
    exact_regularity_prefix_pair_words;
inline constexpr std::uint32_t exact_regularity_product_workspace_words =
    exact_regularity_product_column_count * exact_regularity_product_count *
    exact_regularity_product_accumulator_count;
inline constexpr std::uint32_t exact_regularity_shared_workspace_words =
    exact_regularity_prefix_workspace_words;
static_assert(exact_regularity_shared_workspace_words >= exact_regularity_product_workspace_words);
static_assert(exact_regularity_shared_workspace_words >= exact_regularity_workgroup_size);
inline constexpr std::uint32_t exact_regularity_shared_control_words = 7U;
inline constexpr std::uint32_t exact_regularity_shared_memory_bytes =
    (exact_regularity_shared_limb_words + exact_regularity_shared_metadata_words +
     exact_regularity_shared_workspace_words + exact_regularity_shared_control_words) *
    sizeof(std::uint32_t);
inline constexpr std::uint32_t exact_regularity_shared_memory_limit_bytes = 12U * 1'024U;

// Tiled finalize stages 32-wide per-block contributions for both channels.
// Each contribution is a uvec2 plus a uint validity word; one uvec2 base
// entropy, one uint base-valid flag, and two uvec2 channel score/flags pairs
// complete the bounded 812-byte workspace.
inline constexpr std::uint32_t finalize_workgroup_size = 64U;
inline constexpr std::uint32_t finalize_tile_lane_count = 32U;
inline constexpr std::uint32_t finalize_shared_memory_bytes =
    channel_count * finalize_tile_lane_count * 2U * sizeof(std::uint32_t) +
    channel_count * finalize_tile_lane_count * sizeof(std::uint32_t) +
    5U * 2U * sizeof(std::uint32_t) + sizeof(std::uint32_t);
inline constexpr std::uint32_t finalize_shared_memory_limit_bytes = 12U * 1'024U;

// Pipeline 3 stages one transformed vector and the independent reflector-dot
// products. One shared exponent and one failure flag complete the 416-byte
// cooperative-transform workspace.
inline constexpr std::uint32_t block_statistics_shared_transformed_elements = elements_per_block;
inline constexpr std::uint32_t block_statistics_shared_phase_elements = elements_per_block;
inline constexpr std::uint32_t block_statistics_shared_memory_bytes =
    block_statistics_shared_transformed_elements * 2U * sizeof(std::uint32_t) +
    block_statistics_shared_phase_elements * 2U * sizeof(std::uint32_t) +
    4U * sizeof(std::uint32_t);
inline constexpr std::uint32_t block_statistics_shared_memory_limit_bytes = 12U * 1'024U;

// Packed host diagnostic snapshot layout.
inline constexpr std::uint64_t diagnostic_covariance_elements = score_covariance_elements;
inline constexpr std::uint64_t diagnostic_mean_elements = score_mean_elements;
inline constexpr std::uint64_t diagnostic_tridiagonal_diagonal_elements =
    score_tridiagonal_diagonal_elements;
inline constexpr std::uint64_t diagnostic_tridiagonal_subdiagonal_elements =
    score_tridiagonal_subdiagonal_elements;
inline constexpr std::uint64_t diagnostic_regularity_metadata_elements =
    score_regularity_metadata_elements;
inline constexpr std::uint64_t diagnostic_parameter_elements =
    diagnostic_covariance_elements + diagnostic_mean_elements +
    diagnostic_tridiagonal_diagonal_elements + diagnostic_tridiagonal_subdiagonal_elements +
    diagnostic_regularity_metadata_elements;
inline constexpr std::uint64_t diagnostic_channel_base =
    parameter_stream_count * diagnostic_parameter_elements;

struct ScaleParameters {
    std::uint32_t source_width{};
    std::uint32_t source_height{};
    std::uint32_t source_stride_bytes_0{};
    std::uint32_t source_stride_bytes_1{};
    std::uint32_t source_stride_bytes_2{};
    std::uint32_t source_stride_bytes_3{};
    std::uint32_t destination_width{};
    std::uint32_t destination_height{};
    std::uint32_t bit_depth{};
    std::uint32_t method{};
    std::uint32_t output_offset_elements{};
};

struct VerticalParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t filter_kind{};
};

struct AntialiasParameters {
    std::uint32_t scaled_width{};
    std::uint32_t scaled_height{};
    std::uint32_t operating_width{};
    std::uint32_t operating_height{};
    std::uint32_t source_stride_bytes_0{};
    std::uint32_t source_stride_bytes_1{};
    std::uint32_t source_stride_bytes_2{};
    std::uint32_t source_stride_bytes_3{};
    std::uint32_t bit_depth{};
};

struct LocalParameters {
    std::uint32_t width{};
    std::uint32_t height{};
};

struct ScoreParameters {
    std::uint32_t operating_width{};
    std::uint32_t operating_height{};
    std::uint32_t truncated_width{};
    std::uint32_t truncated_height{};
    std::uint32_t blocks_horizontal{};
    std::uint32_t blocks_vertical{};
    std::uint32_t block_count{};
    std::uint32_t submatrix_width{};
    std::uint32_t submatrix_height{};
    std::uint32_t covariance_sample_count{};
    std::uint32_t reserved{};
    std::uint32_t highpass_frame_stride_elements{};
    std::uint32_t scratch_frame_stride_elements{};
    std::uint32_t result_frame_stride_elements{};
    std::uint32_t status_frame_stride_elements{};
    std::uint32_t sigma_low{};
    std::uint32_t sigma_high{};
    std::uint32_t nn_floor_low{};
    std::uint32_t nn_floor_high{};
    std::uint32_t maximum_low{};
    std::uint32_t maximum_high{};
    numeric::PreparedU32Divider covariance_sample_divider{};
    numeric::PreparedU32Divider block_count_divider{};
};

struct Q32Words {
    std::uint32_t low{};
    std::uint32_t high{};
};

static_assert(sizeof(ScaleParameters) == 44U);
static_assert(sizeof(VerticalParameters) == 12U);
static_assert(sizeof(AntialiasParameters) == 36U);
static_assert(sizeof(LocalParameters) == 8U);
static_assert(sizeof(ScoreParameters) == 116U);
static_assert(sizeof(Q32Words) == 8U);
// The cooperative determinant needs one lane per pivot for the parallel log2.
static_assert(block_statistics_workgroup_size >= elements_per_block);
static_assert(covariance_sample_lane_count == 64U);
static_assert(covariance_entries_per_group == 4U);
static_assert(covariance_workgroup_size == 64U);
static_assert(covariance_entry_batch_count == 7U);
static_assert(means_workgroup_size == means_shared_row_count);
static_assert(parameter_stream_count == 2U * channel_count);
static_assert(compact_tridiagonal_workgroup_size == 64U);
static_assert(exact_regularity_workgroup_size == 32U);
static_assert(compact_tridiagonal_shared_memory_bytes == 5'636U);
static_assert(compact_tridiagonal_shared_memory_bytes <=
              compact_tridiagonal_shared_memory_limit_bytes);
static_assert(exact_regularity_shared_memory_bytes == 3'204U);
static_assert(exact_regularity_shared_memory_bytes <= exact_regularity_shared_memory_limit_bytes);
static_assert(finalize_workgroup_size == 64U);
static_assert(finalize_shared_memory_bytes == 812U);
static_assert(finalize_shared_memory_bytes <= finalize_shared_memory_limit_bytes);
static_assert(block_statistics_scalar_workgroup_size == 64U);
static_assert(block_statistics_shared_memory_bytes == 416U);
static_assert(block_statistics_shared_memory_bytes <= block_statistics_shared_memory_limit_bytes);
static_assert(score_scratch_elements_per_block == 28U);
static_assert(score_covariance_elements == 625U);
static_assert(score_means_offset == 625U);
static_assert(score_reflector_tau_offset == 650U);
static_assert(score_reflector_tau_elements == 23U);
static_assert(score_reflector_tails_offset == 673U);
static_assert(score_reflector_tail_elements == 276U);
static_assert(score_factor_diagonal_offset == 949U);
static_assert(score_factor_subdiagonal_offset == 974U);
static_assert(score_ldl_pivots_offset == 998U);
static_assert(score_ldl_multipliers_offset == 1'023U);
static_assert(score_tridiagonal_diagonal_offset == 1'047U);
static_assert(score_tridiagonal_subdiagonal_offset == 1'072U);
static_assert(score_matrix_exponent_offset == 1'096U);
static_assert(score_regularity_metadata_offset == 1'097U);
static_assert(score_factor_liveness_dimensions_offset == 1'098U);
static_assert(score_factor_liveness_shape_offset == 1'099U);
static_assert(score_scratch_base_elements == 1'100U);
static_assert(score_stream_state_offset == 6U);
static_assert(score_result_elements == 10U);
static_assert(diagnostic_parameter_elements == 700U);

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {

enum class GpuStage : std::uint8_t {
    scale,
    antialias_fused,
    local_vertical,
    local_horizontal_highpass,
    means,
    covariance,
    compact_tridiagonal,
    compact_regularity,
    block_statistics,
    finalize,
    count,
};

inline constexpr std::size_t gpu_stage_count = static_cast<std::size_t>(GpuStage::count);

[[nodiscard]] const char *gpu_stage_name(GpuStage stage) noexcept;

struct DispatchConfiguration final {
    std::uint32_t x{};
    std::uint32_t y{};
    std::uint32_t z{};
};

struct GpuExecutionConfiguration final {
    bool configured{};
    std::int64_t kernel_scale_raw_q32_32{};
    std::int64_t prescale_raw_q32_32{};
    PrescaleMethod prescale_method{PrescaleMethod::nearest};
    std::int64_t sigma_nn_raw_q32_32{};
    std::int64_t nn_floor_raw_q32_32{};
    std::uint32_t weight_variance_mode{};
    std::int64_t maximum_raw_q32_32{};
    std::uint32_t source_width{};
    std::uint32_t source_height{};
    std::uint32_t scaled_width{};
    std::uint32_t scaled_height{};
    std::uint32_t operating_width{};
    std::uint32_t operating_height{};
    std::uint32_t truncated_width{};
    std::uint32_t truncated_height{};
    std::uint32_t block_count{};
    std::uint32_t covariance_sample_count{};
    DispatchConfiguration source_dispatch{};
    DispatchConfiguration scaled_dispatch{};
    DispatchConfiguration operating_dispatch{};
    DispatchConfiguration covariance_dispatch{};
    DispatchConfiguration score_block_dispatch{};
};

struct TelemetrySnapshot final {
    gpu::GpuOperationHandle operation_handle{};
    gpu::GpuOperationSnapshot operation;
    GpuExecutionConfiguration execution_configuration;
    std::array<gpu::GpuDurationStatistics, gpu_stage_count> stages{};
    gpu::GpuDurationStatistics compact_factor;
    std::array<gpu::GpuWaveDurationStatistics, gpu_stage_count> wave_stages{};
};

class Telemetry final {
  public:
    explicit Telemetry(gpu::TelemetryCollector &collector);

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] gpu::TelemetryCollector &collector() const noexcept;
    [[nodiscard]] gpu::GpuOperationHandle operation_handle() const noexcept;

    void configure_timestamps(const gpu::GpuTimestampConfiguration &configuration) noexcept;
    void configure_execution(const GpuExecutionConfiguration &configuration) noexcept;
    void record_submission(bool timestamp_available) noexcept;
    void record_parent_success(long double duration_nanoseconds) noexcept;
    void record_parent_failure() noexcept;
    void record_parent_skip() noexcept;
    void record_submission_latency(std::chrono::nanoseconds duration) noexcept;
    void record_timeline(std::uint64_t start_timestamp, std::uint64_t end_timestamp,
                         std::uint32_t timestamp_valid_bits, double timestamp_period_nanoseconds,
                         crossvmaf::gpu::TimelineSegment segment) noexcept;

    void record_stage_success(GpuStage stage, long double duration_nanoseconds) noexcept;
    void record_stage_failure(GpuStage stage) noexcept;
    void record_stage_skip(GpuStage stage) noexcept;
    void record_wave_stage_success(GpuStage stage, long double wave_nanoseconds,
                                   std::uint32_t task_count) noexcept;
    void record_wave_stage_failure(GpuStage stage) noexcept;
    void record_compact_factor_success(long double duration_nanoseconds) noexcept;
    void record_compact_factor_failure() noexcept;
    void record_compact_factor_skip() noexcept;

    [[nodiscard]] const gpu::GpuOperationSnapshot &operation() const noexcept;
    [[nodiscard]] const GpuExecutionConfiguration &execution_configuration() const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &stage_statistics(GpuStage stage) const noexcept;
    [[nodiscard]] const gpu::GpuWaveDurationStatistics &
    wave_stage_statistics(GpuStage stage) const noexcept;
    [[nodiscard]] const gpu::GpuDurationStatistics &compact_factor_statistics() const noexcept;
    [[nodiscard]] TelemetrySnapshot snapshot() const;

  private:
    [[nodiscard]] static bool is_valid_stage(GpuStage stage) noexcept;

    gpu::TelemetryCollector *collector_{};
    gpu::GpuOperationHandle operation_{};
    gpu::DurationSeriesHandle stage_series_{};
    gpu::DurationSeriesHandle compact_factor_series_{};
    gpu::WaveDurationSeriesHandle wave_stage_series_{};
    GpuExecutionConfiguration execution_configuration_{};
};

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {

class Telemetry;
class Resources;
struct TaskRingView;
using TaskHandle = gpu::StreamingTaskHandle;

struct WaveConformanceResult final {
    std::vector<CompactFactorDiagnostics> stream_diagnostics;
    std::chrono::nanoseconds gpu_elapsed{};
};

class Executor final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Executor>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           Telemetry *telemetry = nullptr, bool skip_streaming_suffix = false);

    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;
    ~Executor();

    [[nodiscard]] Result<Q32Value> compute(const video::GpuFrameView &reference,
                                           const video::GpuFrameView &distorted);
    [[nodiscard]] Result<DetailedResult>
    compute_detailed(const video::GpuFrameView &reference, const video::GpuFrameView &distorted,
                     DetailedDiagnostics *diagnostics = nullptr);

    [[nodiscard]] Result<VkCommandBuffer> record(const video::GpuFrameView &reference,
                                                 const video::GpuFrameView &distorted,
                                                 bool collect_diagnostics = false);
    [[nodiscard]] Result<void> mark_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<DetailedResult> retire(gpu::SubmissionTicket ticket,
                                                std::chrono::nanoseconds submission_host_elapsed,
                                                DetailedDiagnostics *diagnostics = nullptr);
    [[nodiscard]] Result<VkCommandBuffer> record_stream_prefix(const video::GpuFrameView &reference,
                                                               const video::GpuFrameView &distorted,
                                                               TaskHandle task,
                                                               const TaskRingView &task_ring_view);
    [[nodiscard]] Result<void> mark_stream_prefix_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<void>
    retire_stream_prefix(gpu::SubmissionTicket ticket,
                         std::chrono::nanoseconds submission_host_elapsed);
    [[nodiscard]] bool has_streaming_suffix_work() const noexcept;
    [[nodiscard]] bool identity_scale_fused() const noexcept;
    void release_streaming_suffix_resources() noexcept;
    void cancel() noexcept;

    // Internal conformance hook; this header is not part of the public API.
    [[nodiscard]] Result<std::vector<std::int32_t>>
    preprocess_for_conformance(const video::GpuPlaneView &plane);
    [[nodiscard]] Result<CompactFactorDiagnostics>
    factorize_covariance_for_conformance(std::span<const numeric::Q32_32> covariance,
                                         bool begin_with_arithmetic_failure = false);
    [[nodiscard]] Result<WaveConformanceResult>
    factorize_covariance_wave_for_conformance(std::span<const numeric::Q32_32> covariance_matrices,
                                              std::uint32_t frame_count);

  private:
    friend class StreamingTaskRingResources;
    [[nodiscard]] Resources &streaming_resources() noexcept;
    enum class StreamPrefixPhase : std::uint8_t {
        idle,
        prefix_recorded,
        prefix_submitted,
    };

    explicit Executor(std::unique_ptr<Resources> resources) noexcept;

    [[nodiscard]] Result<DetailedResult> readback(std::chrono::nanoseconds submission_host_elapsed,
                                                  DetailedDiagnostics *diagnostics);
    void clear_pending() noexcept;

    std::unique_ptr<Resources> resources_;
    gpu::ConveniencePendingState frame_state_;
    bool pending_guaranteed_singular_{};
    bool pending_collect_diagnostics_{};
    bool pending_timestamp_query_pool_available_{};
    bool pending_timestamp_recording_valid_{};
    StreamPrefixPhase stream_prefix_phase_{StreamPrefixPhase::idle};
};

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {

using TaskHandle = gpu::StreamingTaskHandle;

struct TaskRingView final {
    gpu::Buffer *highpass{};
    gpu::Buffer *score_scratch{};
    gpu::Buffer *score_result{};
    gpu::Buffer *status{};
    VkDeviceSize highpass_offset{};
    VkDeviceSize scratch_offset{};
    VkDeviceSize result_offset{};
    VkDeviceSize status_offset{};
    VkDeviceSize highpass_range{};
    VkDeviceSize scratch_range{};
    VkDeviceSize result_range{};
    VkDeviceSize status_range{};
};

struct StreamingTaskFrameResult final {
    std::uint64_t frame_index{};
    TaskHandle task;
    DetailedResult result;
};

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
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
} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma {
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
} // namespace crossvmaf::metrics::speed_chroma

namespace crossvmaf::metrics::speed_chroma::vulkan {

inline constexpr std::array<std::uint32_t, 5> antialias_unique_coefficients{
    324909248U, 425645824U, 516207552U, 579548672U, 602344640U};
inline constexpr std::size_t antialias_source_value_count = 256U;

[[nodiscard]] constexpr auto make_identity_antialias_products() noexcept {
    std::array<Q32Words, antialias_unique_coefficients.size() * antialias_source_value_count>
        products{};
    for (std::size_t coefficient_index = 0U;
         coefficient_index < antialias_unique_coefficients.size(); ++coefficient_index) {
        for (std::size_t source_value = 0U; source_value < antialias_source_value_count;
             ++source_value) {
            // An 8-bit normalized sample is an exact integer in Q32. Its
            // product needs no Q32 rounding, only the subsequent binary32 RNE.
            const std::int64_t product = (static_cast<std::int64_t>(source_value) - 128) *
                                         antialias_unique_coefficients[coefficient_index];
            std::uint64_t magnitude = static_cast<std::uint64_t>(product < 0 ? -product : product);
            const auto significant_bits = std::bit_width(magnitude);
            if (significant_bits > 24) {
                const auto shift = significant_bits - 24;
                std::uint64_t rounded = magnitude >> shift;
                const std::uint64_t remainder = magnitude & ((std::uint64_t{1} << shift) - 1U);
                const std::uint64_t halfway = std::uint64_t{1} << (shift - 1);
                if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
                    ++rounded;
                }
                magnitude = rounded << shift;
            }
            // The maximum magnitude is below 2^37, so neither signed
            // multiplication nor quantization can overflow or saturate.
            const std::uint64_t bits = product < 0 ? std::uint64_t{0} - magnitude : magnitude;
            products[coefficient_index * antialias_source_value_count + source_value] = {
                static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32U)};
        }
    }
    return products;
}

inline constexpr auto identity_antialias_products = make_identity_antialias_products();
static_assert(sizeof(identity_antialias_products) == 10U * 1024U);

} // namespace crossvmaf::metrics::speed_chroma::vulkan
