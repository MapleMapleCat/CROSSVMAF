#include "video/frame.hpp"
#include "metrics/adm3/vulkan.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "metrics/adm3/adm3.hpp"
#include "gpu/buffer.hpp"
#include "gpu/command.hpp"
#include "gpu/pipeline.hpp"
#include "gpu/device.hpp"
#include "gpu/timestamp.hpp"
#include "gpu/submission.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>
#include "shaders/adm3_combine_spv.hpp"
#include "shaders/adm3_decouple_spv.hpp"
#include "shaders/adm3_dwt_spv.hpp"
#include "shaders/adm3_finalize_batch_spv.hpp"
#include "shaders/adm3_merge_rows_spv.hpp"
#include "shaders/adm3_merge_rows_stream_spv.hpp"
#include "shaders/adm3_metric_segments_detailed_spv.hpp"
#include "shaders/adm3_metric_segments_score_spv.hpp"
#include "shaders/adm3_pack_scale_detailed_spv.hpp"
#include "shaders/adm3_pack_scale_score_spv.hpp"
#include "shaders/adm3_reduce_spv.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <cstddef>
#include "shaders/adm3_combine_wave_spv.hpp"
#include "shaders/adm3_finalize_scales_wave_spv.hpp"
#include "shaders/adm3_pack_scale_detailed_wave_spv.hpp"
#include "shaders/adm3_pack_scale_score_wave_spv.hpp"
#include "shaders/adm3_reduce_wave_spv.hpp"
#include <cstring>
#include <tuple>
#include <cstdlib>
#include <string_view>

namespace crossvmaf::metrics::adm3::vulkan {

class Resources final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Resources>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           ResultMode result_mode, Telemetry *telemetry = nullptr,
           bool skip_streaming_suffix = false);

    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    ResultMode result_mode{ResultMode::detailed};
    std::uint32_t active_channel_count{detailed_channel_count};
    Telemetry *telemetry{};
    std::uint32_t maximum_pixel_count{};
    std::uint32_t row_capacity{};
    std::uint32_t segment_capacity{};
    StreamingLayout streaming_layout{};

    std::array<std::unique_ptr<gpu::Buffer>, 2> reference_low;
    std::array<std::unique_ptr<gpu::Buffer>, 2> distorted_low;
    std::unique_ptr<gpu::Buffer> reference_detail;
    std::unique_ptr<gpu::Buffer> restored_detail;
    std::unique_ptr<gpu::Buffer> additive_detail;
    std::unique_ptr<gpu::Buffer> metric_partials;
    std::unique_ptr<gpu::Buffer> row_sums;
    std::unique_ptr<gpu::Buffer> reduction_a;
    std::unique_ptr<gpu::Buffer> reduction_b;
    std::unique_ptr<gpu::Buffer> results;

    std::unique_ptr<gpu::ComputePipeline> dwt_pipeline;
    std::unique_ptr<gpu::ComputePipeline> decouple_pipeline;
    std::unique_ptr<gpu::ComputePipeline> metric_segments_pipeline;
    std::unique_ptr<gpu::ComputePipeline> merge_rows_pipeline;
    std::unique_ptr<gpu::ComputePipeline> streaming_merge_rows_pipeline;
    std::unique_ptr<gpu::ComputePipeline> reduce_pipeline;
    std::unique_ptr<gpu::ComputePipeline> pack_scale_pipeline;
    std::unique_ptr<gpu::ComputePipeline> finalize_batch_pipeline;
    std::unique_ptr<gpu::ComputePipeline> combine_pipeline;
    std::unique_ptr<gpu::CommandContext> commands;
    std::unique_ptr<gpu::GpuTimestampIntervals> gpu_timestamp;
    gpu::GpuTimestampIntervalRecordingState pending_timestamp_recording;

    std::array<ScaleGeometry, scale_count> geometry;
    std::uint32_t maximum_reduction_passes{};

  private:
    Resources() = default;
};
class StreamingTaskRingResources::Impl final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Impl>> create(Resources &prototype);

    Impl(const Impl &) = delete;
    Impl &operator=(const Impl &) = delete;

    [[nodiscard]] Result<std::optional<TaskHandle>> reserve(std::uint64_t frame_index);
    [[nodiscard]] Result<TaskRingView> view(TaskHandle task) noexcept;
    [[nodiscard]] Result<void> cancel_prefix(TaskHandle task);
    [[nodiscard]] Result<void> publish_ready(TaskHandle task);
    [[nodiscard]] Result<VkCommandBuffer> record_wave();
    [[nodiscard]] Result<void> mark_wave_submitted(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<std::vector<StreamingTaskFrameResult>>
    retire_wave(gpu::SubmissionTicket ticket, std::chrono::nanoseconds submission_host_elapsed);

    [[nodiscard]] std::uint32_t capacity() const noexcept { return core_.capacity(); }
    [[nodiscard]] std::uint32_t free_slot_count() const noexcept;
    [[nodiscard]] std::uint32_t prefix_pending_task_count() const noexcept;
    [[nodiscard]] std::uint32_t ready_task_count() const noexcept;
    [[nodiscard]] bool wave_in_flight() const noexcept { return pending_state_.pending(); }
    [[nodiscard]] const StreamingLayout &layout() const noexcept { return layout_; }
    void cancel() noexcept;
    // SubmissionOwner must be drained before this terminal recovery path is
    // used. It exists for failures after a queue submission was accepted.
    void cancel_after_device_drain() noexcept;

  private:
    enum class SuffixPhaseKind : std::uint8_t {
        reduce,
        pack_scale,
        finalize_scales,
        combine,
    };

    struct SuffixPhase final {
        SuffixPhaseKind kind{SuffixPhaseKind::reduce};
        std::uint32_t scale{};
        std::uint32_t pass{};
        std::uint32_t descriptor_set_index{};
        gpu::DispatchSize dispatch{};
    };

    Impl() = default;

    void fail_retired_wave() noexcept;

    gpu::StreamingTaskRingCore core_;
    gpu::DeviceView device_;
    Config config_;
    ResultMode result_mode_{ResultMode::detailed};
    Telemetry *telemetry_{};
    StreamingLayout layout_{};
    std::array<ScaleGeometry, scale_count> geometry_{};
    VkDeviceSize packed_row_frame_bytes_{};
    VkDeviceSize reduction_frame_bytes_{};
    VkDeviceSize result_frame_bytes_{};
    VkDeviceSize packed_row_frame_stride_{};
    VkDeviceSize reduction_frame_stride_{};
    VkDeviceSize result_frame_stride_{};
    std::vector<SuffixPhase> suffix_phases_;

    std::unique_ptr<gpu::Buffer> wave_slot_buffer_;
    std::unique_ptr<gpu::Buffer> packed_row_buffer_;
    std::unique_ptr<gpu::Buffer> reduction_a_buffer_;
    std::unique_ptr<gpu::Buffer> reduction_b_buffer_;
    std::unique_ptr<gpu::Buffer> result_buffer_;

    std::unique_ptr<gpu::ComputePipeline> reduce_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> pack_scale_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> finalize_scales_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> combine_pipeline_;
    std::unique_ptr<gpu::CommandContext> commands_;
    std::unique_ptr<gpu::TimestampQueryPool> wave_timestamp_query_pool_;
    bool pending_wave_timestamp_recording_valid_{};
    gpu::ConveniencePendingState pending_state_;
};
} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {
namespace {

using gpu::Buffer;
using gpu::BufferBinding;
using gpu::BufferCreateInfo;
using gpu::CommandContext;
using gpu::ComputePipeline;
using gpu::DeviceView;
using gpu::DispatchSize;

[[nodiscard]] constexpr std::uint32_t resources_divide_round_up(std::uint32_t value,
                                                                std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

[[nodiscard]] std::uint32_t resources_border_start(std::uint32_t size) noexcept {
    return size < 5U ? 0U : (size - 5U) / 10U;
}

[[nodiscard]] Result<std::unique_ptr<Buffer>>
resources_create_storage_buffer(DeviceView device, VkDeviceSize size, bool host_visible = false) {
    BufferCreateInfo create_info =
        host_visible ? gpu::host_visible_storage(size) : BufferCreateInfo{};
    if (!host_visible) {
        create_info.size = size;
        create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    }
    return Buffer::create(device, create_info);
}

[[nodiscard]] Result<VkDeviceSize> resources_checked_size(std::uint64_t element_count,
                                                          std::uint64_t element_size,
                                                          VkDeviceSize limit, const char *label) {
    if (element_count == 0U ||
        element_count > std::numeric_limits<std::uint64_t>::max() / element_size) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("ADM3 ") + label + " buffer size overflows"});
    }
    const std::uint64_t bytes = element_count * element_size;
    if (bytes > limit) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("ADM3 ") + label + " buffer exceeds maxStorageBufferRange"});
    }
    return Result<VkDeviceSize>::success(static_cast<VkDeviceSize>(bytes));
}

[[nodiscard]] Result<void> resources_validate_dispatch(const DispatchSize &dispatch,
                                                       const VkPhysicalDeviceLimits &limits,
                                                       const char *label) {
    if (dispatch.x == 0U || dispatch.y == 0U || dispatch.z == 0U ||
        dispatch.x > limits.maxComputeWorkGroupCount[0] ||
        dispatch.y > limits.maxComputeWorkGroupCount[1] ||
        dispatch.z > limits.maxComputeWorkGroupCount[2]) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format,
             std::string("ADM3 ") + label + " dispatch exceeds the selected Vulkan device limits"});
    }
    return Result<void>::success();
}

} // namespace

Result<std::unique_ptr<Resources>> Resources::create(DeviceView device, FrameDescriptor descriptor,
                                                     Config config, ResultMode result_mode,
                                                     Telemetry *telemetry,
                                                     bool skip_streaming_suffix) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    const auto &limits = properties.limits;
    if (limits.maxComputeWorkGroupInvocations < linear_workgroup_size ||
        limits.maxComputeWorkGroupSize[0] < linear_workgroup_size ||
        limits.maxComputeWorkGroupSize[1] < dwt_tile_size ||
        limits.maxPerStageDescriptorStorageBuffers < 4U ||
        limits.maxPushConstantsSize < sizeof(FinalizeBatchParameters)) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet deterministic ADM3 compute limits"});
    }

    auto resources = std::unique_ptr<Resources>(new Resources());
    resources->device = device;
    resources->descriptor = descriptor;
    resources->config = config;
    resources->result_mode = result_mode;
    resources->active_channel_count =
        result_mode == ResultMode::score_only ? score_channel_count : detailed_channel_count;
    resources->telemetry = telemetry;
    resources->streaming_layout.result_mode = result_mode;
    resources->streaming_layout.active_channel_count = resources->active_channel_count;

    std::uint32_t input_width = descriptor.width;
    std::uint32_t input_height = descriptor.height;
    std::uint64_t packed_row_frame_elements = 0U;
    std::uint32_t maximum_first_pass_group_count = 1U;
    for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
        ScaleGeometry &geometry = resources->geometry[scale];
        geometry.input_width = input_width;
        geometry.input_height = input_height;
        geometry.width = (input_width + 1U) / 2U;
        geometry.height = (input_height + 1U) / 2U;
        const std::uint64_t pixel_count =
            static_cast<std::uint64_t>(geometry.width) * geometry.height;
        if (pixel_count > std::numeric_limits<std::uint32_t>::max()) {
            return Result<std::unique_ptr<Resources>>::failure(
                {ErrorCode::unsupported_format,
                 "frame dimensions exceed the ADM3 32-bit address space"});
        }
        geometry.pixel_count = static_cast<std::uint32_t>(pixel_count);
        const std::uint64_t coefficient_count = pixel_count * 3U;
        if (coefficient_count > std::numeric_limits<std::uint32_t>::max()) {
            return Result<std::unique_ptr<Resources>>::failure(
                {ErrorCode::unsupported_format,
                 "frame dimensions exceed the ADM3 coefficient address space"});
        }
        geometry.coefficient_count = static_cast<std::uint32_t>(coefficient_count);
        geometry.region_width = geometry.width - 2U * resources_border_start(geometry.width);
        geometry.region_height = geometry.height - 2U * resources_border_start(geometry.height);
        geometry.segment_count =
            resources_divide_round_up(geometry.region_width, metric_segment_width);
        geometry.packed_row_offset_elements = static_cast<std::uint32_t>(packed_row_frame_elements);
        geometry.packed_row_channel_stride_elements = geometry.region_height;
        const std::uint64_t scale_packed_row_elements =
            static_cast<std::uint64_t>(geometry.region_height) * resources->active_channel_count;
        if (scale_packed_row_elements >
            std::numeric_limits<std::uint32_t>::max() - packed_row_frame_elements) {
            return Result<std::unique_ptr<Resources>>::failure(
                {ErrorCode::unsupported_format,
                 "ADM3 packed row layout exceeds the 32-bit shader address "
                 "space"});
        }
        resources->streaming_layout.scales[scale] = {geometry.packed_row_offset_elements,
                                                     geometry.packed_row_channel_stride_elements};
        packed_row_frame_elements += scale_packed_row_elements;
        maximum_first_pass_group_count =
            std::max(maximum_first_pass_group_count,
                     resources_divide_round_up(geometry.region_height, linear_workgroup_size));
        geometry.dwt_dispatch = {resources_divide_round_up(geometry.width, dwt_tile_size),
                                 resources_divide_round_up(geometry.height, dwt_tile_size), 1U};
        geometry.decouple_dispatch = {
            resources_divide_round_up(geometry.coefficient_count, linear_workgroup_size), 1U, 1U};
        geometry.metric_dispatch = {
            resources_divide_round_up(geometry.segment_count, metric_segments_per_workgroup),
            geometry.region_height, 1U};
        const std::uint32_t merge_lanes_per_row =
            std::min(32U, std::bit_ceil(geometry.segment_count));
        const std::uint32_t merge_rows_per_group = 32U / merge_lanes_per_row;
        geometry.merge_dispatch = {
            resources_divide_round_up(geometry.region_height, merge_rows_per_group),
            resources->active_channel_count, 1U};
        for (const auto [dispatch, label] : std::array<std::pair<DispatchSize, const char *>, 4>{
                 std::pair{geometry.dwt_dispatch, "DWT"},
                 std::pair{geometry.decouple_dispatch, "decouple"},
                 std::pair{geometry.metric_dispatch, "metric segments"},
                 std::pair{geometry.merge_dispatch, "row merge"}}) {
            const auto valid = resources_validate_dispatch(dispatch, limits, label);
            if (!valid) {
                return Result<std::unique_ptr<Resources>>::failure(valid.error());
            }
        }
        std::uint32_t reduction_count = geometry.region_height;
        while (reduction_count > 1U) {
            geometry.reduction_input_counts.push_back(reduction_count);
            const DispatchSize dispatch{
                resources_divide_round_up(reduction_count, linear_workgroup_size),
                resources->active_channel_count, 1U};
            const auto valid = resources_validate_dispatch(dispatch, limits, "reduction");
            if (!valid) {
                return Result<std::unique_ptr<Resources>>::failure(valid.error());
            }
            reduction_count = dispatch.x;
        }
        resources->maximum_reduction_passes =
            std::max(resources->maximum_reduction_passes,
                     static_cast<std::uint32_t>(geometry.reduction_input_counts.size()));
        input_width = geometry.width;
        input_height = geometry.height;
    }
    resources->streaming_layout.packed_row_frame_elements =
        static_cast<std::uint32_t>(packed_row_frame_elements);
    resources->streaming_layout.reduction_channel_capacity = maximum_first_pass_group_count;
    const std::uint64_t reduction_frame_elements =
        static_cast<std::uint64_t>(maximum_first_pass_group_count) *
        resources->active_channel_count;
    if (reduction_frame_elements > std::numeric_limits<std::uint32_t>::max()) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::unsupported_format,
             "ADM3 streaming reduction layout exceeds the 32-bit shader "
             "address space"});
    }
    resources->streaming_layout.reduction_frame_elements =
        static_cast<std::uint32_t>(reduction_frame_elements);
    resources->streaming_layout.result_frame_elements = result_slot_count;
    resources->maximum_pixel_count = resources->geometry[0].pixel_count;
    resources->row_capacity = resources->geometry[0].height;
    resources->segment_capacity = resources->geometry[0].segment_count;

    const auto low_bytes =
        resources_checked_size(resources->maximum_pixel_count, sizeof(std::uint32_t),
                               limits.maxStorageBufferRange, "low-pass");
    const auto detail_bytes =
        resources_checked_size(static_cast<std::uint64_t>(resources->maximum_pixel_count) * 3U,
                               sizeof(std::uint32_t), limits.maxStorageBufferRange, "detail");
    const auto row_bytes = resources_checked_size(
        static_cast<std::uint64_t>(resources->row_capacity) * resources->active_channel_count,
        sizeof(U64Words), limits.maxStorageBufferRange, "row reduction");
    const auto metric_partial_bytes =
        resources_checked_size(static_cast<std::uint64_t>(resources->row_capacity) *
                                   resources->segment_capacity * resources->active_channel_count,
                               sizeof(U64Words), limits.maxStorageBufferRange, "metric partial");
    const auto result_bytes = resources_checked_size(result_slot_count, sizeof(U64Words),
                                                     limits.maxStorageBufferRange, "result");
    if (!low_bytes || !detail_bytes || !row_bytes || !metric_partial_bytes || !result_bytes) {
        const Error error = !low_bytes              ? low_bytes.error()
                            : !detail_bytes         ? detail_bytes.error()
                            : !row_bytes            ? row_bytes.error()
                            : !metric_partial_bytes ? metric_partial_bytes.error()
                                                    : result_bytes.error();
        return Result<std::unique_ptr<Resources>>::failure(error);
    }

    const auto create_into = [&](std::unique_ptr<Buffer> &destination, VkDeviceSize size,
                                 bool host_visible = false) -> Result<void> {
        auto buffer = resources_create_storage_buffer(device, size, host_visible);
        if (!buffer) return Result<void>::failure(buffer.error());
        destination = std::move(buffer).value();
        return Result<void>::success();
    };
    for (auto *buffers : {&resources->reference_low, &resources->distorted_low}) {
        for (auto &buffer : *buffers) {
            const auto created = create_into(buffer, low_bytes.value());
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
        }
    }
    for (auto *buffer :
         {&resources->reference_detail, &resources->restored_detail, &resources->additive_detail}) {
        const auto created = create_into(*buffer, detail_bytes.value());
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
    }
    auto created = create_into(resources->metric_partials, metric_partial_bytes.value());
    if (!created) {
        return Result<std::unique_ptr<Resources>>::failure(created.error());
    }
    if (!skip_streaming_suffix) {
        for (auto *buffer :
             {&resources->row_sums, &resources->reduction_a, &resources->reduction_b}) {
            created = create_into(*buffer, row_bytes.value());
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
        }
        created = create_into(resources->results, result_bytes.value(), true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
    }

    auto dwt = ComputePipeline::create(device, gpu::generated::adm3_dwt_spv, 3,
                                       sizeof(DwtParameters), scale_count * 2U);
    if (!dwt) return Result<std::unique_ptr<Resources>>::failure(dwt.error());
    resources->dwt_pipeline = std::move(dwt).value();
    for (std::uint32_t scale = 1U; scale < scale_count; ++scale) {
        const std::uint32_t input_slot = (scale - 1U) & 1U;
        const std::uint32_t output_slot = scale & 1U;
        const std::array reference_bindings{
            BufferBinding{resources->reference_low[input_slot].get()},
            BufferBinding{resources->reference_low[output_slot].get()},
            BufferBinding{resources->reference_detail.get()}};
        auto updated = resources->dwt_pipeline->update_buffers(scale * 2U, reference_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
        const std::array distorted_bindings{
            BufferBinding{resources->distorted_low[input_slot].get()},
            BufferBinding{resources->distorted_low[output_slot].get()},
            BufferBinding{resources->restored_detail.get()}};
        updated = resources->dwt_pipeline->update_buffers(scale * 2U + 1U, distorted_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
    }

    auto decouple = ComputePipeline::create(device, gpu::generated::adm3_decouple_spv, 3,
                                            sizeof(DecoupleParameters));
    if (!decouple) {
        return Result<std::unique_ptr<Resources>>::failure(decouple.error());
    }
    resources->decouple_pipeline = std::move(decouple).value();
    const std::array decouple_bindings{BufferBinding{resources->reference_detail.get()},
                                       BufferBinding{resources->restored_detail.get()},
                                       BufferBinding{resources->additive_detail.get()}};
    auto updated = resources->decouple_pipeline->update_buffers(0, decouple_bindings);
    if (!updated) {
        return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    const std::span<const std::uint8_t> metric_segments_spirv =
        result_mode == ResultMode::score_only
            ? std::span<const std::uint8_t>(gpu::generated::adm3_metric_segments_score_spv)
            : std::span<const std::uint8_t>(gpu::generated::adm3_metric_segments_detailed_spv);
    auto metric_segments =
        ComputePipeline::create(device, metric_segments_spirv, 4, sizeof(MetricSegmentsParameters));
    if (!metric_segments) {
        return Result<std::unique_ptr<Resources>>::failure(metric_segments.error());
    }
    resources->metric_segments_pipeline = std::move(metric_segments).value();
    const std::array metric_segment_bindings{BufferBinding{resources->reference_detail.get()},
                                             BufferBinding{resources->restored_detail.get()},
                                             BufferBinding{resources->additive_detail.get()},
                                             BufferBinding{resources->metric_partials.get()}};
    updated = resources->metric_segments_pipeline->update_buffers(0, metric_segment_bindings);
    if (!updated) {
        return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    auto streaming_merge_rows =
        ComputePipeline::create(device, gpu::generated::adm3_merge_rows_stream_spv, 2,
                                sizeof(StreamingMergeRowsParameters));
    if (!streaming_merge_rows) {
        return Result<std::unique_ptr<Resources>>::failure(streaming_merge_rows.error());
    }
    resources->streaming_merge_rows_pipeline = std::move(streaming_merge_rows).value();

    if (!skip_streaming_suffix) {
        auto merge_rows = ComputePipeline::create(device, gpu::generated::adm3_merge_rows_spv, 2,
                                                  sizeof(MergeRowsParameters));
        if (!merge_rows) {
            return Result<std::unique_ptr<Resources>>::failure(merge_rows.error());
        }
        resources->merge_rows_pipeline = std::move(merge_rows).value();
        const std::array merge_row_bindings{BufferBinding{resources->metric_partials.get()},
                                            BufferBinding{resources->row_sums.get()}};
        updated = resources->merge_rows_pipeline->update_buffers(0, merge_row_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }

        const std::uint32_t reduction_set_count = std::max(resources->maximum_reduction_passes, 1U);
        auto reduce = ComputePipeline::create(device, gpu::generated::adm3_reduce_spv, 2,
                                              sizeof(ReduceParameters), reduction_set_count);
        if (!reduce) {
            return Result<std::unique_ptr<Resources>>::failure(reduce.error());
        }
        resources->reduce_pipeline = std::move(reduce).value();
        const Buffer *reduction_input = resources->row_sums.get();
        Buffer *reduction_output = resources->reduction_a.get();
        for (std::uint32_t pass = 0; pass < reduction_set_count; ++pass) {
            const std::array bindings{BufferBinding{reduction_input},
                                      BufferBinding{reduction_output}};
            updated = resources->reduce_pipeline->update_buffers(pass, bindings);
            if (!updated) {
                return Result<std::unique_ptr<Resources>>::failure(updated.error());
            }
            reduction_input = reduction_output;
            reduction_output = reduction_output == resources->reduction_a.get()
                                   ? resources->reduction_b.get()
                                   : resources->reduction_a.get();
        }

        const std::span<const std::uint8_t> pack_scale_spirv =
            result_mode == ResultMode::score_only
                ? std::span<const std::uint8_t>(gpu::generated::adm3_pack_scale_score_spv)
                : std::span<const std::uint8_t>(gpu::generated::adm3_pack_scale_detailed_spv);
        auto pack_scale = ComputePipeline::create(device, pack_scale_spirv, 2,
                                                  sizeof(PackScaleParameters), scale_count);
        if (!pack_scale) {
            return Result<std::unique_ptr<Resources>>::failure(pack_scale.error());
        }
        resources->pack_scale_pipeline = std::move(pack_scale).value();
        for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
            const std::size_t pass_count = resources->geometry[scale].reduction_input_counts.size();
            const Buffer *final_buffer =
                pass_count == 0U ? resources->row_sums.get()
                                 : (pass_count % 2U == 1U ? resources->reduction_a.get()
                                                          : resources->reduction_b.get());
            const std::array bindings{BufferBinding{final_buffer},
                                      BufferBinding{resources->results.get()}};
            updated = resources->pack_scale_pipeline->update_buffers(scale, bindings);
            if (!updated) {
                return Result<std::unique_ptr<Resources>>::failure(updated.error());
            }
        }

        const std::array batch_bindings{BufferBinding{resources->results.get()}};
        auto finalize_batch = ComputePipeline::create(
            device, gpu::generated::adm3_finalize_batch_spv, 1, sizeof(FinalizeBatchParameters));
        if (!finalize_batch) {
            return Result<std::unique_ptr<Resources>>::failure(finalize_batch.error());
        }
        resources->finalize_batch_pipeline = std::move(finalize_batch).value();
        updated = resources->finalize_batch_pipeline->update_buffers(0, batch_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }

        auto combine = ComputePipeline::create(device, gpu::generated::adm3_combine_spv, 1,
                                               sizeof(CombineParameters));
        if (!combine) {
            return Result<std::unique_ptr<Resources>>::failure(combine.error());
        }
        resources->combine_pipeline = std::move(combine).value();
        updated = resources->combine_pipeline->update_buffers(0, batch_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
    }

    auto commands = CommandContext::create(device);
    if (!commands) {
        return Result<std::unique_ptr<Resources>>::failure(commands.error());
    }
    resources->commands = std::move(commands).value();
    auto gpu_timestamp = gpu::GpuTimestampIntervals::create(
        device, telemetry != nullptr ? &telemetry->collector() : nullptr,
        telemetry != nullptr ? telemetry->operation_handle() : gpu::GpuOperationHandle{},
        telemetry != nullptr ? telemetry->stage_series_handle() : gpu::DurationSeriesHandle{},
        static_cast<std::uint32_t>(gpu_stage_count),
        telemetry != nullptr && stage_timestamps_enabled());
    if (!gpu_timestamp) {
        return Result<std::unique_ptr<Resources>>::failure(gpu_timestamp.error());
    }
    resources->gpu_timestamp = std::move(gpu_timestamp).value();
    return Result<std::unique_ptr<Resources>>::success(std::move(resources));
}

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {
namespace {

using gpu::BufferBinding;
using gpu::CommandRecordingGuard;
using gpu::DeviceView;
using gpu::DispatchSize;

template <typename T>
[[nodiscard]] std::span<const std::byte> vulkan_bytes_of(const T &value) noexcept {
    return {reinterpret_cast<const std::byte *>(&value), sizeof(T)};
}

[[nodiscard]] constexpr std::uint32_t vulkan_divide_round_up(std::uint32_t value,
                                                             std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

[[nodiscard]] U64Words vulkan_split_i64(std::int64_t value) noexcept {
    const auto bits = static_cast<std::uint64_t>(value);
    return {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32U)};
}

[[nodiscard]] std::int64_t vulkan_join_i64(U64Words value) noexcept {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(value.low) | (static_cast<std::uint64_t>(value.high) << 32U);
    return static_cast<std::int64_t>(bits);
}

[[nodiscard]] Result<void> vulkan_validate_luma(const video::GpuPlaneView &plane,
                                                const FrameDescriptor &descriptor) {
    if (!video::is_compatible_gpu_plane(plane, descriptor.width, descriptor.height,
                                        descriptor.bit_depth)) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "ADM3 received an incompatible GPU luma plane"});
    }
    return Result<void>::success();
}

[[nodiscard]] bool vulkan_streaming_layouts_match(const StreamingLayout &left,
                                                  const StreamingLayout &right) noexcept {
    if (left.result_mode != right.result_mode ||
        left.active_channel_count != right.active_channel_count ||
        left.packed_row_frame_elements != right.packed_row_frame_elements ||
        left.reduction_channel_capacity != right.reduction_channel_capacity ||
        left.reduction_frame_elements != right.reduction_frame_elements ||
        left.result_frame_elements != right.result_frame_elements) {
        return false;
    }
    for (std::size_t scale = 0U; scale < scale_count; ++scale) {
        if (left.scales[scale].packed_row_offset_elements !=
                right.scales[scale].packed_row_offset_elements ||
            left.scales[scale].packed_row_channel_stride_elements !=
                right.scales[scale].packed_row_channel_stride_elements) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] bool vulkan_has_non_streaming_suffix_resources(const Resources &resources) noexcept {
    return resources.row_sums != nullptr && resources.reduction_a != nullptr &&
           resources.reduction_b != nullptr && resources.results != nullptr &&
           resources.reduce_pipeline != nullptr && resources.pack_scale_pipeline != nullptr &&
           resources.finalize_batch_pipeline != nullptr && resources.combine_pipeline != nullptr;
}

} // namespace

Executor::Executor(std::unique_ptr<Resources> resources) noexcept
    : resources_(std::move(resources)) {
}
Executor::~Executor() = default;

Result<std::unique_ptr<Executor>> Executor::create(DeviceView device, FrameDescriptor descriptor,
                                                   Config config, Telemetry *telemetry,
                                                   bool skip_streaming_suffix) {
    return create_with_mode(device, descriptor, config, ResultMode::detailed, telemetry,
                            skip_streaming_suffix);
}

Result<std::unique_ptr<Executor>> Executor::create_score_only(DeviceView device,
                                                              FrameDescriptor descriptor,
                                                              Config config, Telemetry *telemetry,
                                                              bool skip_streaming_suffix) {
    return create_with_mode(device, descriptor, config, ResultMode::score_only, telemetry,
                            skip_streaming_suffix);
}

Result<std::unique_ptr<Executor>> Executor::create_with_mode(DeviceView device,
                                                             FrameDescriptor descriptor,
                                                             Config config, ResultMode result_mode,
                                                             Telemetry *telemetry,
                                                             bool skip_streaming_suffix) {
    if (!device.complete()) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::backend_error, "ADM3 Vulkan executor received an incomplete device"});
    }
    const auto descriptor_validation = video::validate_frame_layout(descriptor);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<Executor>>::failure(descriptor_validation.error());
    }
    if (config.enhancement_gain_limit.raw() != numeric::Q32_32::one_raw || config.csf_mode != 2U) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::unsupported_model,
             "CrossVMAF 1.0.16 ADM3 requires gain limit 1 and legacy blended CSF"});
    }

    auto resources = Resources::create(device, descriptor, config, result_mode, telemetry,
                                       skip_streaming_suffix);
    if (!resources) {
        return Result<std::unique_ptr<Executor>>::failure(resources.error());
    }
    return Result<std::unique_ptr<Executor>>::success(
        std::unique_ptr<Executor>(new Executor(std::move(resources).value())));
}

Result<Q32Value> Executor::compute(const video::GpuFrameView &reference,
                                   const video::GpuFrameView &distorted) {
    auto detailed = compute_detailed(reference, distorted);
    if (!detailed) return Result<Q32Value>::failure(detailed.error());
    return Result<Q32Value>::success(detailed.value().value);
}

Result<DetailedResult> Executor::compute_detailed(const video::GpuFrameView &reference,
                                                  const video::GpuFrameView &distorted) {
    const auto recorded = record(reference, distorted);
    if (!recorded) {
        return Result<DetailedResult>::failure(recorded.error());
    }
    const auto retirement = resources_->commands->submit_finished_and_wait();
    if (!retirement) {
        // Clear ADM3's outer pending state only when CommandContext proved
        // that the queue never observed this command buffer.
        cancel();
        return Result<DetailedResult>::failure(retirement.error());
    }
    result_state_.mark_submission_observed();
    return readback(retirement.value().host_elapsed);
}

Result<VkCommandBuffer> Executor::record(const video::GpuFrameView &reference,
                                         const video::GpuFrameView &distorted) {
    if (result_state_.pending() || stream_prefix_phase_ != StreamPrefixPhase::idle) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "ADM3 already has a recorded result pending retirement"});
    }
    if (!vulkan_has_non_streaming_suffix_resources(*resources_)) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "ADM3 non-streaming suffix resources were released"});
    }
    if (reference.index != distorted.index) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_argument, "ADM3 GPU frame indices must match"});
    }
    const auto reference_validation =
        vulkan_validate_luma(reference.planes[0], resources_->descriptor);
    if (!reference_validation) {
        return Result<VkCommandBuffer>::failure(reference_validation.error());
    }
    const auto distorted_validation =
        vulkan_validate_luma(distorted.planes[0], resources_->descriptor);
    if (!distorted_validation) {
        return Result<VkCommandBuffer>::failure(distorted_validation.error());
    }

    const std::array reference_bindings{BufferBinding{reference.planes[0].buffer,
                                                      reference.planes[0].offset_bytes,
                                                      reference.planes[0].size_bytes},
                                        BufferBinding{resources_->reference_low[0].get()},
                                        BufferBinding{resources_->reference_detail.get()}};
    auto updated = resources_->dwt_pipeline->update_buffers(0, reference_bindings);
    if (!updated) return Result<VkCommandBuffer>::failure(updated.error());
    const std::array distorted_bindings{BufferBinding{distorted.planes[0].buffer,
                                                      distorted.planes[0].offset_bytes,
                                                      distorted.planes[0].size_bytes},
                                        BufferBinding{resources_->distorted_low[0].get()},
                                        BufferBinding{resources_->restored_detail.get()}};
    updated = resources_->dwt_pipeline->update_buffers(1, distorted_bindings);
    if (!updated) return Result<VkCommandBuffer>::failure(updated.error());

    auto begun = resources_->commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    auto timestamp_recording = resources_->gpu_timestamp->begin_recording(command_buffer);
    std::uint32_t timestamp_boundary_index = 0U;
    resources_->commands->memory_barrier(VK_PIPELINE_STAGE_HOST_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    resources_->gpu_timestamp->write_boundary(command_buffer, timestamp_boundary_index,
                                              timestamp_recording);

    for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
        const ScaleGeometry &geometry = resources_->geometry[scale];
        DwtParameters dwt_parameters{};
        dwt_parameters.input_width = geometry.input_width;
        dwt_parameters.input_height = geometry.input_height;
        dwt_parameters.input_stride_bytes = scale == 0U
                                                ? reference.planes[0].stride_bytes
                                                : geometry.input_width * sizeof(std::uint32_t);
        dwt_parameters.output_width = geometry.width;
        dwt_parameters.output_height = geometry.height;
        dwt_parameters.bit_depth = resources_->descriptor.bit_depth;
        dwt_parameters.scale = scale;
        dwt_parameters.raw_input = scale == 0U ? 1U : 0U;
        auto recorded = resources_->dwt_pipeline->record(
            command_buffer, scale * 2U, vulkan_bytes_of(dwt_parameters), geometry.dwt_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        dwt_parameters.input_stride_bytes = scale == 0U
                                                ? distorted.planes[0].stride_bytes
                                                : geometry.input_width * sizeof(std::uint32_t);
        recorded = resources_->dwt_pipeline->record(command_buffer, scale * 2U + 1U,
                                                    vulkan_bytes_of(dwt_parameters),
                                                    geometry.dwt_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const DecoupleParameters decouple_parameters{geometry.coefficient_count, scale};
        recorded = resources_->decouple_pipeline->record(
            command_buffer, 0, vulkan_bytes_of(decouple_parameters), geometry.decouple_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const MetricSegmentsParameters metric_parameters{
            geometry.width,
            geometry.height,
            scale,
            resources_->config.horizontal_csf_raw[scale],
            resources_->config.diagonal_csf_raw[scale],
            resources_->row_capacity,
            resources_->segment_capacity,
            metric_segments_per_workgroup,
        };
        recorded = resources_->metric_segments_pipeline->record(
            command_buffer, 0, vulkan_bytes_of(metric_parameters), geometry.metric_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const MergeRowsParameters merge_parameters{
            geometry.width,           geometry.height,        scale,
            resources_->row_capacity, geometry.segment_count, resources_->segment_capacity,
        };
        recorded = resources_->merge_rows_pipeline->record(
            command_buffer, 0, vulkan_bytes_of(merge_parameters), geometry.merge_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        for (std::uint32_t pass = 0; pass < geometry.reduction_input_counts.size(); ++pass) {
            const ReduceParameters reduce_parameters{geometry.reduction_input_counts[pass],
                                                     resources_->row_capacity};
            const DispatchSize dispatch{
                vulkan_divide_round_up(reduce_parameters.input_count, linear_workgroup_size),
                resources_->active_channel_count, 1U};
            recorded = resources_->reduce_pipeline->record(
                command_buffer, pass, vulkan_bytes_of(reduce_parameters), dispatch);
            if (!recorded) {
                return Result<VkCommandBuffer>::failure(recorded.error());
            }
            resources_->commands->memory_barrier(
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const PackScaleParameters pack_parameters{scale, resources_->row_capacity};
        recorded = resources_->pack_scale_pipeline->record(
            command_buffer, scale, vulkan_bytes_of(pack_parameters), {1U, 1U, 1U});
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);
    }

    std::array<std::uint32_t, scale_count> widths{};
    std::array<std::uint32_t, scale_count> heights{};
    for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
        widths[scale] = resources_->geometry[scale].width;
        heights[scale] = resources_->geometry[scale].height;
    }

    const FinalizeBatchParameters finalize_batch_parameters{
        widths,
        heights,
        resources_->config.horizontal_csf_raw,
        resources_->config.diagonal_csf_raw,
        vulkan_split_i64(resources_->config.noise_weight.raw()),
    };
    auto recorded = resources_->finalize_batch_pipeline->record(
        command_buffer, 0, vulkan_bytes_of(finalize_batch_parameters), {scale_count, 1U, 1U});
    if (!recorded) {
        return Result<VkCommandBuffer>::failure(recorded.error());
    }
    resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    const CombineParameters combine_parameters{
        vulkan_split_i64(resources_->config.dlm_weight.raw()),
        vulkan_split_i64(resources_->config.minimum.raw()),
    };
    recorded = resources_->combine_pipeline->record(
        command_buffer, 0, vulkan_bytes_of(combine_parameters), {1U, 1U, 1U});
    if (!recorded) {
        return Result<VkCommandBuffer>::failure(recorded.error());
    }
    resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                              timestamp_recording);
    resources_->commands->memory_barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT);

    const auto finished = resources_->commands->finish();
    if (!finished) {
        return Result<VkCommandBuffer>::failure(finished.error());
    }
    recording_guard.release();
    result_state_.mark_recorded();
    resources_->pending_timestamp_recording = timestamp_recording;
    return finished;
}

Result<VkCommandBuffer> Executor::record_stream_prefix(const video::GpuFrameView &reference,
                                                       const video::GpuFrameView &distorted,
                                                       TaskHandle task,
                                                       const TaskRingView &task_ring_view) {
    if (result_state_.pending() || stream_prefix_phase_ != StreamPrefixPhase::idle || !task ||
        task_ring_view.task.slot != task.slot ||
        task_ring_view.task.generation != task.generation) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming prefix is already pending or has an invalid "
                                       "task view"});
    }
    if (resources_->streaming_merge_rows_pipeline == nullptr ||
        task_ring_view.packed_rows == nullptr || task_ring_view.results == nullptr ||
        !vulkan_streaming_layouts_match(resources_->streaming_layout, task_ring_view.layout)) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state,
             "ADM3 streaming task view has an incompatible packed layout"});
    }
    const VkDeviceSize expected_packed_row_bytes =
        static_cast<VkDeviceSize>(resources_->streaming_layout.packed_row_frame_elements) *
        sizeof(U64Words);
    const VkDeviceSize expected_result_bytes =
        static_cast<VkDeviceSize>(result_slot_count) * sizeof(U64Words);
    const bool packed_range_valid =
        task_ring_view.packed_rows_range == expected_packed_row_bytes &&
        task_ring_view.packed_rows_offset < task_ring_view.packed_rows->size() &&
        task_ring_view.packed_rows_range <=
            task_ring_view.packed_rows->size() - task_ring_view.packed_rows_offset;
    const bool result_range_valid =
        task_ring_view.results_range == expected_result_bytes &&
        task_ring_view.results_offset < task_ring_view.results->size() &&
        task_ring_view.results_range <=
            task_ring_view.results->size() - task_ring_view.results_offset &&
        (task_ring_view.results->usage() & VK_BUFFER_USAGE_TRANSFER_DST_BIT) != 0U;
    if (!packed_range_valid || !result_range_valid) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state,
             "ADM3 streaming task view contains an invalid buffer slice"});
    }
    if (reference.index != distorted.index) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_argument, "ADM3 GPU frame indices must match"});
    }
    const auto reference_validation =
        vulkan_validate_luma(reference.planes[0], resources_->descriptor);
    if (!reference_validation) {
        return Result<VkCommandBuffer>::failure(reference_validation.error());
    }
    const auto distorted_validation =
        vulkan_validate_luma(distorted.planes[0], resources_->descriptor);
    if (!distorted_validation) {
        return Result<VkCommandBuffer>::failure(distorted_validation.error());
    }

    const std::array reference_bindings{BufferBinding{reference.planes[0].buffer,
                                                      reference.planes[0].offset_bytes,
                                                      reference.planes[0].size_bytes},
                                        BufferBinding{resources_->reference_low[0].get()},
                                        BufferBinding{resources_->reference_detail.get()}};
    auto updated = resources_->dwt_pipeline->update_buffers(0U, reference_bindings);
    if (!updated) return Result<VkCommandBuffer>::failure(updated.error());
    const std::array distorted_bindings{BufferBinding{distorted.planes[0].buffer,
                                                      distorted.planes[0].offset_bytes,
                                                      distorted.planes[0].size_bytes},
                                        BufferBinding{resources_->distorted_low[0].get()},
                                        BufferBinding{resources_->restored_detail.get()}};
    updated = resources_->dwt_pipeline->update_buffers(1U, distorted_bindings);
    if (!updated) return Result<VkCommandBuffer>::failure(updated.error());
    const std::array merge_row_bindings{BufferBinding{resources_->metric_partials.get()},
                                        BufferBinding{task_ring_view.packed_rows,
                                                      task_ring_view.packed_rows_offset,
                                                      task_ring_view.packed_rows_range}};
    updated = resources_->streaming_merge_rows_pipeline->update_buffers(0U, merge_row_bindings);
    if (!updated) return Result<VkCommandBuffer>::failure(updated.error());

    auto begun = resources_->commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    auto timestamp_recording = resources_->gpu_timestamp->begin_recording(command_buffer);
    std::uint32_t timestamp_boundary_index = 0U;
    resources_->gpu_timestamp->write_boundary(command_buffer, timestamp_boundary_index,
                                              timestamp_recording);
    resources_->commands->fill_buffer(*task_ring_view.results, 0U, task_ring_view.results_offset,
                                      task_ring_view.results_range);
    resources_->commands->memory_barrier(VK_PIPELINE_STAGE_HOST_BIT |
                                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        const ScaleGeometry &geometry = resources_->geometry[scale];
        DwtParameters dwt_parameters{};
        dwt_parameters.input_width = geometry.input_width;
        dwt_parameters.input_height = geometry.input_height;
        dwt_parameters.input_stride_bytes = scale == 0U
                                                ? reference.planes[0].stride_bytes
                                                : geometry.input_width * sizeof(std::uint32_t);
        dwt_parameters.output_width = geometry.width;
        dwt_parameters.output_height = geometry.height;
        dwt_parameters.bit_depth = resources_->descriptor.bit_depth;
        dwt_parameters.scale = scale;
        dwt_parameters.raw_input = scale == 0U ? 1U : 0U;
        auto recorded = resources_->dwt_pipeline->record(
            command_buffer, scale * 2U, vulkan_bytes_of(dwt_parameters), geometry.dwt_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        dwt_parameters.input_stride_bytes = scale == 0U
                                                ? distorted.planes[0].stride_bytes
                                                : geometry.input_width * sizeof(std::uint32_t);
        recorded = resources_->dwt_pipeline->record(command_buffer, scale * 2U + 1U,
                                                    vulkan_bytes_of(dwt_parameters),
                                                    geometry.dwt_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const DecoupleParameters decouple_parameters{geometry.coefficient_count, scale};
        recorded = resources_->decouple_pipeline->record(
            command_buffer, 0U, vulkan_bytes_of(decouple_parameters), geometry.decouple_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const MetricSegmentsParameters metric_parameters{
            geometry.width,
            geometry.height,
            scale,
            resources_->config.horizontal_csf_raw[scale],
            resources_->config.diagonal_csf_raw[scale],
            resources_->row_capacity,
            resources_->segment_capacity,
            metric_segments_per_workgroup,
        };
        recorded = resources_->metric_segments_pipeline->record(
            command_buffer, 0U, vulkan_bytes_of(metric_parameters), geometry.metric_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);

        const StreamingMergeRowsParameters merge_parameters{
            geometry.width,
            geometry.height,
            scale,
            resources_->row_capacity,
            geometry.segment_count,
            resources_->segment_capacity,
            geometry.packed_row_offset_elements,
            geometry.packed_row_channel_stride_elements,
        };
        recorded = resources_->streaming_merge_rows_pipeline->record(
            command_buffer, 0U, vulkan_bytes_of(merge_parameters), geometry.merge_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources_->commands->memory_barrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);
        // Streaming reduction and packing run later in a wave. Publish empty
        // intervals so the existing fixed stage schema can still attribute
        // every prefix stage without changing profiler-off command buffers.
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);
        resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                                  timestamp_recording);
    }

    // Finalization and combination also belong to the streaming suffix wave.
    resources_->gpu_timestamp->write_boundary(command_buffer, ++timestamp_boundary_index,
                                              timestamp_recording);
    const auto finished = resources_->commands->finish();
    if (!finished) return Result<VkCommandBuffer>::failure(finished.error());
    recording_guard.release();
    result_state_.mark_recorded();
    resources_->pending_timestamp_recording = timestamp_recording;
    stream_prefix_phase_ = StreamPrefixPhase::prefix_recorded;
    return finished;
}

Result<void> Executor::mark_stream_prefix_submitted(gpu::SubmissionTicket ticket) {
    if (stream_prefix_phase_ != StreamPrefixPhase::prefix_recorded || !result_state_.pending()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming prefix is not recorded"});
    }
    const auto submitted = resources_->commands->mark_submitted(ticket);
    if (submitted) {
        result_state_.mark_submission_observed();
        stream_prefix_phase_ = StreamPrefixPhase::prefix_submitted;
    }
    return submitted;
}

Result<void> Executor::retire_stream_prefix(gpu::SubmissionTicket ticket,
                                            std::chrono::nanoseconds submission_host_elapsed) {
    if (stream_prefix_phase_ != StreamPrefixPhase::prefix_submitted || !result_state_.pending()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming prefix is not submitted"});
    }
    const auto retired = resources_->commands->mark_retired(ticket);
    if (!retired) return retired;
    if (resources_->telemetry != nullptr) {
        resources_->telemetry->record_submission_latency(submission_host_elapsed);
        const auto timestamps_collected = resources_->gpu_timestamp->collect(
            resources_->pending_timestamp_recording, submission_host_elapsed,
            crossvmaf::gpu::timeline_segment_id(gpu::TimelineSegment::prefix));
        if (!timestamps_collected) return timestamps_collected;
    }
    result_state_.clear();
    resources_->pending_timestamp_recording = {};
    stream_prefix_phase_ = StreamPrefixPhase::idle;
    return Result<void>::success();
}

Resources &Executor::streaming_resources() noexcept {
    return *resources_;
}

const StreamingLayout &Executor::streaming_layout() const noexcept {
    return resources_->streaming_layout;
}

void Executor::release_streaming_suffix_resources() noexcept {
    if (result_state_.pending() || stream_prefix_phase_ != StreamPrefixPhase::idle) {
        return;
    }
    resources_->reduce_pipeline.reset();
    resources_->pack_scale_pipeline.reset();
    resources_->finalize_batch_pipeline.reset();
    resources_->combine_pipeline.reset();
    resources_->merge_rows_pipeline.reset();
    resources_->row_sums.reset();
    resources_->reduction_a.reset();
    resources_->reduction_b.reset();
    resources_->results.reset();
}

Result<void> Executor::mark_submitted(gpu::SubmissionTicket ticket) {
    if (!result_state_.pending() || stream_prefix_phase_ != StreamPrefixPhase::idle) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "ADM3 has no recorded result to submit"});
    }
    const auto command_submitted = resources_->commands->mark_submitted(ticket);
    if (command_submitted) result_state_.mark_submission_observed();
    return command_submitted;
}

Result<DetailedResult> Executor::retire(gpu::SubmissionTicket ticket,
                                        std::chrono::nanoseconds submission_host_elapsed) {
    if (stream_prefix_phase_ != StreamPrefixPhase::idle) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming prefix must use prefix retirement"});
    }
    const auto command_retired = resources_->commands->mark_retired(ticket);
    if (!command_retired) {
        return Result<DetailedResult>::failure(command_retired.error());
    }
    return readback(submission_host_elapsed);
}

void Executor::cancel() noexcept {
    resources_->commands->cancel();
    result_state_.rollback_after_cancel(resources_->commands->state());
    if (!result_state_.pending()) {
        resources_->pending_timestamp_recording = {};
        stream_prefix_phase_ = StreamPrefixPhase::idle;
    }
}

Result<DetailedResult> Executor::readback(std::chrono::nanoseconds submission_host_elapsed) {
    if (!result_state_.pending() ||
        resources_->commands->state() != gpu::CommandBufferState::retired) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_state, "ADM3 result is not ready for host readback"});
    }
    const auto timestamp_result = resources_->gpu_timestamp->collect(
        resources_->pending_timestamp_recording, submission_host_elapsed);
    if (!timestamp_result) {
        return Result<DetailedResult>::failure(timestamp_result.error());
    }
    std::array<U64Words, result_slot_count> raw_results{};
    const auto read = resources_->results->read(std::as_writable_bytes(std::span(raw_results)));
    if (!read) return Result<DetailedResult>::failure(read.error());
    DetailedResult output{};
    for (std::size_t scale = 0; scale < scale_count; ++scale) {
        const std::size_t base = scale * 3U;
        output.scales[scale] = {
            numeric::Q32_32::from_raw(vulkan_join_i64(raw_results[base + 0U])),
            numeric::Q32_32::from_raw(vulkan_join_i64(raw_results[base + 1U])),
            numeric::Q32_32::from_raw(vulkan_join_i64(raw_results[base + 2U])),
        };
        if (resources_->result_mode != ResultMode::detailed) continue;
        const std::size_t debug_base = 13U + scale * detailed_channel_count;
        for (std::size_t band = 0; band < 3U; ++band) {
            output.accumulators[scale].denominator[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + band]));
            output.accumulators[scale].numerator[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 3U + band]));
            output.accumulators[scale].aim[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 6U + band]));
            output.accumulators[scale].reference_sum[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 9U + band]));
            output.accumulators[scale].restored_sum[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 12U + band]));
            output.accumulators[scale].additive_sum[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 15U + band]));
            output.accumulators[scale].reference_weighted_sum[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 18U + band]));
            output.accumulators[scale].distorted_weighted_sum[band] =
                static_cast<std::uint64_t>(vulkan_join_i64(raw_results[debug_base + 21U + band]));
        }
    }
    output.value = Q32Value{vulkan_join_i64(raw_results[12U])};
    result_state_.clear();
    resources_->pending_timestamp_recording = {};
    return Result<DetailedResult>::success(output);
}

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {
namespace {

using gpu::Buffer;
using gpu::BufferBinding;
using gpu::BufferCreateInfo;
using gpu::CommandContext;
using gpu::CommandRecordingGuard;
using gpu::ComputePipeline;
using gpu::DispatchSize;

inline constexpr VkDeviceSize streaming_task_ring_task_identity_bytes = 2U * sizeof(std::uint32_t);
inline constexpr std::uint32_t streaming_task_ring_wave_timestamp_query_count = 2U;

static_assert(sizeof(TaskHandle) == streaming_task_ring_task_identity_bytes);

template <typename Value>
[[nodiscard]] std::span<const std::byte> streaming_task_ring_bytes_of(const Value &value) noexcept {
    return {reinterpret_cast<const std::byte *>(&value), sizeof(Value)};
}

[[nodiscard]] constexpr std::uint32_t
streaming_task_ring_divide_round_up(std::uint32_t value, std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

[[nodiscard]] VkDeviceSize streaming_task_ring_align_up(VkDeviceSize value,
                                                        VkDeviceSize alignment) noexcept {
    if (alignment <= 1U) return value;
    const VkDeviceSize remainder = value % alignment;
    return remainder == 0U ? value : value + alignment - remainder;
}

[[nodiscard]] Result<std::unique_ptr<Buffer>>
streaming_task_ring_create_buffer(gpu::DeviceView device, VkDeviceSize size,
                                  VkBufferUsageFlags additional_usage = 0U,
                                  bool host_visible = false) {
    BufferCreateInfo create_info =
        host_visible ? gpu::host_visible_storage(size, additional_usage) : BufferCreateInfo{};
    if (!host_visible) {
        create_info.size = size;
        create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | additional_usage;
    }
    return Buffer::create(device, create_info);
}

[[nodiscard]] U64Words streaming_task_ring_split_i64(std::int64_t value) noexcept {
    const auto bits = static_cast<std::uint64_t>(value);
    return {static_cast<std::uint32_t>(bits), static_cast<std::uint32_t>(bits >> 32U)};
}

[[nodiscard]] std::int64_t streaming_task_ring_join_i64(U64Words value) noexcept {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(value.low) | (static_cast<std::uint64_t>(value.high) << 32U);
    return static_cast<std::int64_t>(bits);
}

[[nodiscard]] DetailedResult
streaming_task_ring_decode_result(std::span<const U64Words, result_slot_count> raw_results,
                                  ResultMode result_mode) noexcept {
    DetailedResult output{};
    for (std::size_t scale = 0U; scale < scale_count; ++scale) {
        const std::size_t scale_base = scale * 3U;
        output.scales[scale] = {
            numeric::Q32_32::from_raw(streaming_task_ring_join_i64(raw_results[scale_base + 0U])),
            numeric::Q32_32::from_raw(streaming_task_ring_join_i64(raw_results[scale_base + 1U])),
            numeric::Q32_32::from_raw(streaming_task_ring_join_i64(raw_results[scale_base + 2U])),
        };
        if (result_mode != ResultMode::detailed) continue;

        const std::size_t accumulator_base = 13U + scale * detailed_channel_count;
        for (std::size_t band = 0U; band < 3U; ++band) {
            output.accumulators[scale].denominator[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + band]));
            output.accumulators[scale].numerator[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 3U + band]));
            output.accumulators[scale].aim[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 6U + band]));
            output.accumulators[scale].reference_sum[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 9U + band]));
            output.accumulators[scale].restored_sum[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 12U + band]));
            output.accumulators[scale].additive_sum[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 15U + band]));
            output.accumulators[scale].reference_weighted_sum[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 18U + band]));
            output.accumulators[scale].distorted_weighted_sum[band] = static_cast<std::uint64_t>(
                streaming_task_ring_join_i64(raw_results[accumulator_base + 21U + band]));
        }
    }
    output.value = Q32Value{streaming_task_ring_join_i64(raw_results[12U])};
    return output;
}

[[nodiscard]] Result<void> streaming_task_ring_validate_frame_stride(VkDeviceSize frame_stride,
                                                                     VkDeviceSize frame_bytes,
                                                                     const char *label) {
    if (frame_bytes == 0U || frame_stride < frame_bytes || frame_stride % sizeof(U64Words) != 0U ||
        frame_stride / sizeof(U64Words) > std::numeric_limits<std::uint32_t>::max()) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format, std::string("ADM3 streaming ") + label +
                                                " stride cannot be addressed by the wave shader"});
    }
    return Result<void>::success();
}

} // namespace

Result<std::unique_ptr<StreamingTaskRingResources::Impl>>
StreamingTaskRingResources::Impl::create(Resources &prototype) {
    const StreamingLayout &prototype_layout = prototype.streaming_layout;
    if (prototype_layout.active_channel_count == 0U ||
        prototype_layout.packed_row_frame_elements == 0U ||
        prototype_layout.reduction_channel_capacity == 0U ||
        prototype_layout.reduction_frame_elements == 0U ||
        prototype_layout.result_frame_elements != result_slot_count) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming requires a complete packed suffix layout"});
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(prototype.device.physical_device, &properties);
    const VkPhysicalDeviceLimits &limits = properties.limits;
    if (limits.maxPerStageDescriptorStorageBuffers < 6U ||
        limits.maxPushConstantsSize < sizeof(StreamingFinalizeScalesParameters) ||
        limits.maxComputeWorkGroupCount[2] == 0U) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet ADM3 streaming wave limits"});
    }

    auto ring =
        std::unique_ptr<StreamingTaskRingResources::Impl>(new StreamingTaskRingResources::Impl());
    ring->device_ = prototype.device;
    ring->config_ = prototype.config;
    ring->result_mode_ = prototype.result_mode;
    ring->telemetry_ = prototype.telemetry;
    ring->layout_ = prototype_layout;
    ring->geometry_ = prototype.geometry;
    ring->packed_row_frame_bytes_ =
        static_cast<VkDeviceSize>(prototype_layout.packed_row_frame_elements) * sizeof(U64Words);
    ring->reduction_frame_bytes_ =
        static_cast<VkDeviceSize>(prototype_layout.reduction_frame_elements) * sizeof(U64Words);
    ring->result_frame_bytes_ =
        static_cast<VkDeviceSize>(prototype_layout.result_frame_elements) * sizeof(U64Words);

    const VkDeviceSize descriptor_alignment =
        std::max<VkDeviceSize>(1U, limits.minStorageBufferOffsetAlignment);
    ring->packed_row_frame_stride_ =
        streaming_task_ring_align_up(ring->packed_row_frame_bytes_, descriptor_alignment);
    ring->reduction_frame_stride_ =
        streaming_task_ring_align_up(ring->reduction_frame_bytes_, descriptor_alignment);
    ring->result_frame_stride_ =
        streaming_task_ring_align_up(ring->result_frame_bytes_, descriptor_alignment);
    for (const auto &[stride, bytes, label] : std::array{
             std::tuple{ring->packed_row_frame_stride_, ring->packed_row_frame_bytes_,
                        "packed row"},
             std::tuple{ring->reduction_frame_stride_, ring->reduction_frame_bytes_, "reduction"},
             std::tuple{ring->result_frame_stride_, ring->result_frame_bytes_, "result"}}) {
        const auto valid = streaming_task_ring_validate_frame_stride(stride, bytes, label);
        if (!valid) {
            return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
                valid.error());
        }
    }

    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        const ScaleGeometry &geometry = ring->geometry_[scale];
        for (std::uint32_t pass = 0U; pass < geometry.reduction_input_counts.size(); ++pass) {
            const std::uint32_t input_count = geometry.reduction_input_counts[pass];
            const DispatchSize dispatch{
                streaming_task_ring_divide_round_up(input_count, linear_workgroup_size),
                prototype_layout.active_channel_count, 1U};
            const std::uint32_t descriptor_set_index =
                pass == 0U ? 0U : ((pass & 1U) == 1U ? 1U : 2U);
            ring->suffix_phases_.push_back(
                {SuffixPhaseKind::reduce, scale, pass, descriptor_set_index, dispatch});
        }

        const std::size_t reduction_pass_count = geometry.reduction_input_counts.size();
        const std::uint32_t pack_descriptor_set_index =
            reduction_pass_count == 0U ? 0U : ((reduction_pass_count & 1U) == 1U ? 1U : 2U);
        ring->suffix_phases_.push_back({SuffixPhaseKind::pack_scale, scale, 0U,
                                        pack_descriptor_set_index, DispatchSize{1U, 1U, 1U}});
    }

    ring->suffix_phases_.push_back(
        {SuffixPhaseKind::finalize_scales, 0U, 0U, 0U, DispatchSize{scale_count, 1U, 1U}});
    ring->suffix_phases_.push_back(
        {SuffixPhaseKind::combine, 0U, 0U, 0U, DispatchSize{1U, 1U, 1U}});

    std::uint64_t capacity = std::min<std::uint64_t>(gpu::maximum_streaming_wave_task_count,
                                                     limits.maxComputeWorkGroupCount[2]);
    for (VkDeviceSize frame_stride : {ring->packed_row_frame_stride_, ring->reduction_frame_stride_,
                                      ring->result_frame_stride_}) {
        capacity = std::min<std::uint64_t>(capacity, limits.maxStorageBufferRange / frame_stride);
    }

    if (capacity == 0U) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            {ErrorCode::invalid_argument, "Vulkan device limits cannot hold one "
                                          "ADM3 streaming task"});
    }
    ring->core_.initialize(static_cast<std::uint32_t>(capacity));

    const auto create_into = [&ring](std::unique_ptr<Buffer> &destination, VkDeviceSize size,
                                     VkBufferUsageFlags additional_usage = 0U,
                                     bool host_visible = false) -> Result<void> {
        auto buffer =
            streaming_task_ring_create_buffer(ring->device_, size, additional_usage, host_visible);
        if (!buffer) return Result<void>::failure(buffer.error());
        destination = std::move(buffer).value();
        return Result<void>::success();
    };

    Result<void> created = create_into(
        ring->wave_slot_buffer_, streaming_task_ring_task_identity_bytes * ring->core_.capacity(),
        VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_into(ring->packed_row_buffer_,
                          ring->packed_row_frame_stride_ * ring->core_.capacity());
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    for (std::unique_ptr<Buffer> *reduction_buffer :
         {&ring->reduction_a_buffer_, &ring->reduction_b_buffer_}) {
        created =
            create_into(*reduction_buffer, ring->reduction_frame_stride_ * ring->core_.capacity());
        if (!created) {
            return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
                created.error());
        }
    }
    created = create_into(ring->result_buffer_, ring->result_frame_stride_ * ring->core_.capacity(),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }

    const auto results_cleared = ring->result_buffer_->fill();
    if (!results_cleared) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            results_cleared.error());
    }

    auto reduce_pipeline =
        ComputePipeline::create(ring->device_, gpu::generated::adm3_reduce_wave_spv, 3U,
                                sizeof(StreamingReduceParameters), 3U);
    if (!reduce_pipeline) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            reduce_pipeline.error());
    }
    ring->reduce_pipeline_ = std::move(reduce_pipeline).value();
    const std::array reduce_first_pass_bindings{
        BufferBinding{ring->packed_row_buffer_.get()},
        BufferBinding{ring->reduction_a_buffer_.get()},
        BufferBinding{ring->wave_slot_buffer_.get()},
    };
    const std::array reduce_a_to_b_bindings{
        BufferBinding{ring->reduction_a_buffer_.get()},
        BufferBinding{ring->reduction_b_buffer_.get()},
        BufferBinding{ring->wave_slot_buffer_.get()},
    };
    const std::array reduce_b_to_a_bindings{
        BufferBinding{ring->reduction_b_buffer_.get()},
        BufferBinding{ring->reduction_a_buffer_.get()},
        BufferBinding{ring->wave_slot_buffer_.get()},
    };
    for (const auto [descriptor_set_index, bindings] :
         std::array{std::pair{0U, std::span<const BufferBinding>(reduce_first_pass_bindings)},
                    std::pair{1U, std::span<const BufferBinding>(reduce_a_to_b_bindings)},
                    std::pair{2U, std::span<const BufferBinding>(reduce_b_to_a_bindings)}}) {
        created = ring->reduce_pipeline_->update_buffers(descriptor_set_index, bindings);
        if (!created) {
            return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
                created.error());
        }
    }

    const std::span<const std::uint8_t> pack_scale_spirv =
        ring->result_mode_ == ResultMode::score_only
            ? std::span<const std::uint8_t>(gpu::generated::adm3_pack_scale_score_wave_spv)
            : std::span<const std::uint8_t>(gpu::generated::adm3_pack_scale_detailed_wave_spv);
    auto pack_scale_pipeline = ComputePipeline::create(ring->device_, pack_scale_spirv, 3U,
                                                       sizeof(StreamingPackScaleParameters), 3U);
    if (!pack_scale_pipeline) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            pack_scale_pipeline.error());
    }
    ring->pack_scale_pipeline_ = std::move(pack_scale_pipeline).value();
    const std::array pack_sources{ring->packed_row_buffer_.get(), ring->reduction_a_buffer_.get(),
                                  ring->reduction_b_buffer_.get()};
    for (std::uint32_t descriptor_set_index = 0U; descriptor_set_index < pack_sources.size();
         ++descriptor_set_index) {
        const std::array pack_bindings{
            BufferBinding{pack_sources[descriptor_set_index]},
            BufferBinding{ring->result_buffer_.get()},
            BufferBinding{ring->wave_slot_buffer_.get()},
        };
        created = ring->pack_scale_pipeline_->update_buffers(descriptor_set_index, pack_bindings);
        if (!created) {
            return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
                created.error());
        }
    }

    const std::array result_wave_bindings{
        BufferBinding{ring->result_buffer_.get()},
        BufferBinding{ring->wave_slot_buffer_.get()},
    };
    auto finalize_scales_pipeline =
        ComputePipeline::create(ring->device_, gpu::generated::adm3_finalize_scales_wave_spv, 2U,
                                sizeof(StreamingFinalizeScalesParameters));
    if (!finalize_scales_pipeline) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            finalize_scales_pipeline.error());
    }
    ring->finalize_scales_pipeline_ = std::move(finalize_scales_pipeline).value();
    created = ring->finalize_scales_pipeline_->update_buffers(0U, result_wave_bindings);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }

    auto combine_pipeline =
        ComputePipeline::create(ring->device_, gpu::generated::adm3_combine_wave_spv, 2U,
                                sizeof(StreamingCombineParameters));
    if (!combine_pipeline) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            combine_pipeline.error());
    }
    ring->combine_pipeline_ = std::move(combine_pipeline).value();
    created = ring->combine_pipeline_->update_buffers(0U, result_wave_bindings);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }

    auto commands = CommandContext::create(ring->device_);
    if (!commands) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(commands.error());
    }
    ring->commands_ = std::move(commands).value();
    if (ring->telemetry_ != nullptr && ring->telemetry_->enabled()) {
        auto timestamp_pool = gpu::TimestampQueryPool::create(
            ring->device_, streaming_task_ring_wave_timestamp_query_count);
        if (!timestamp_pool) {
            return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
                timestamp_pool.error());
        }
        ring->wave_timestamp_query_pool_ = std::move(timestamp_pool).value();
    }
    return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::success(std::move(ring));
}

std::uint32_t StreamingTaskRingResources::Impl::free_slot_count() const noexcept {
    return core_.free_slot_count();
}

std::uint32_t StreamingTaskRingResources::Impl::prefix_pending_task_count() const noexcept {
    return core_.prefix_pending_task_count();
}

std::uint32_t StreamingTaskRingResources::Impl::ready_task_count() const noexcept {
    return core_.ready_task_count();
}

Result<std::optional<TaskHandle>>
StreamingTaskRingResources::Impl::reserve(std::uint64_t frame_index) {
    return core_.reserve(frame_index, "ADM3");
}

Result<TaskRingView> StreamingTaskRingResources::Impl::view(TaskHandle task) noexcept {
    using TaskState = gpu::StreamingTaskRingCore::TaskState;
    if (!task || task.slot >= core_.tasks().size() ||
        core_.task(task.slot).generation != task.generation ||
        core_.task(task.slot).state != TaskState::prefix_pending) {
        return Result<TaskRingView>::failure(
            {ErrorCode::invalid_state, "ADM3 task ring view references a stale prefix task"});
    }
    return Result<TaskRingView>::success({
        task,
        packed_row_buffer_.get(),
        result_buffer_.get(),
        static_cast<VkDeviceSize>(task.slot) * packed_row_frame_stride_,
        static_cast<VkDeviceSize>(task.slot) * result_frame_stride_,
        packed_row_frame_bytes_,
        result_frame_bytes_,
        layout_,
    });
}

Result<void> StreamingTaskRingResources::Impl::cancel_prefix(TaskHandle task) {
    return core_.cancel_prefix(task, "ADM3");
}

Result<void> StreamingTaskRingResources::Impl::publish_ready(TaskHandle task) {
    return core_.publish_ready(task, "ADM3");
}

Result<VkCommandBuffer> StreamingTaskRingResources::Impl::record_wave() {
    if (pending_state_.pending()) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming suffix wave is not ready to record"});
    }
    const auto snapshotted = core_.snapshot_ready_wave("ADM3");
    if (!snapshotted) {
        return Result<VkCommandBuffer>::failure(snapshotted.error());
    }
    const std::uint32_t snapshot_task_count =
        static_cast<std::uint32_t>(core_.pending_wave_tasks().size());

    auto begun = commands_->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*commands_);
    pending_wave_timestamp_recording_valid_ =
        wave_timestamp_query_pool_ != nullptr && wave_timestamp_query_pool_->supported();
    if (pending_wave_timestamp_recording_valid_) {
        const auto reset = wave_timestamp_query_pool_->reset_queries(
            command_buffer, 0U, streaming_task_ring_wave_timestamp_query_count);
        pending_wave_timestamp_recording_valid_ = static_cast<bool>(reset);
        if (pending_wave_timestamp_recording_valid_) {
            const auto written = wave_timestamp_query_pool_->write_timestamp(command_buffer, 0U);
            if (!written) pending_wave_timestamp_recording_valid_ = false;
        }
    }
    commands_->update_buffer(*wave_slot_buffer_,
                             std::as_bytes(std::span(core_.pending_wave_tasks())));
    commands_->memory_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    auto recorded = Result<void>::success();
    std::array<std::uint32_t, scale_count> widths{};
    std::array<std::uint32_t, scale_count> heights{};
    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        widths[scale] = geometry_[scale].width;
        heights[scale] = geometry_[scale].height;
    }
    const std::uint32_t packed_row_task_stride_elements =
        static_cast<std::uint32_t>(packed_row_frame_stride_ / sizeof(U64Words));
    const std::uint32_t reduction_task_stride_elements =
        static_cast<std::uint32_t>(reduction_frame_stride_ / sizeof(U64Words));
    const std::uint32_t result_task_stride_elements =
        static_cast<std::uint32_t>(result_frame_stride_ / sizeof(U64Words));

    for (const SuffixPhase &phase : suffix_phases_) {
        const DispatchSize wave_dispatch{phase.dispatch.x, phase.dispatch.y, snapshot_task_count};
        switch (phase.kind) {
        case SuffixPhaseKind::reduce: {
            const bool first_pass = phase.pass == 0U;
            const StreamingReduceParameters parameters{
                geometry_[phase.scale].reduction_input_counts[phase.pass],
                first_pass ? packed_row_task_stride_elements : reduction_task_stride_elements,
                first_pass ? geometry_[phase.scale].packed_row_offset_elements : 0U,
                first_pass ? geometry_[phase.scale].packed_row_channel_stride_elements
                           : layout_.reduction_channel_capacity,
                reduction_task_stride_elements,
                layout_.reduction_channel_capacity,
            };
            recorded =
                reduce_pipeline_->record(command_buffer, phase.descriptor_set_index,
                                         streaming_task_ring_bytes_of(parameters), wave_dispatch);
            break;
        }
        case SuffixPhaseKind::pack_scale: {
            const std::size_t reduction_pass_count =
                geometry_[phase.scale].reduction_input_counts.size();
            const bool reads_packed_rows = reduction_pass_count == 0U;
            const StreamingPackScaleParameters parameters{
                phase.scale,
                reads_packed_rows ? packed_row_task_stride_elements
                                  : reduction_task_stride_elements,
                reads_packed_rows ? geometry_[phase.scale].packed_row_offset_elements : 0U,
                reads_packed_rows ? geometry_[phase.scale].packed_row_channel_stride_elements
                                  : layout_.reduction_channel_capacity,
                result_task_stride_elements,
            };
            recorded = pack_scale_pipeline_->record(command_buffer, phase.descriptor_set_index,
                                                    streaming_task_ring_bytes_of(parameters),
                                                    wave_dispatch);
            break;
        }
        case SuffixPhaseKind::finalize_scales: {
            const StreamingFinalizeScalesParameters parameters{
                widths,
                heights,
                config_.horizontal_csf_raw,
                config_.diagonal_csf_raw,
                streaming_task_ring_split_i64(config_.noise_weight.raw()),
                result_task_stride_elements,
            };
            recorded = finalize_scales_pipeline_->record(
                command_buffer, 0U, streaming_task_ring_bytes_of(parameters), wave_dispatch);
            break;
        }
        case SuffixPhaseKind::combine: {
            const StreamingCombineParameters parameters{
                streaming_task_ring_split_i64(config_.dlm_weight.raw()),
                streaming_task_ring_split_i64(config_.minimum.raw()),
                result_task_stride_elements,
            };
            recorded = combine_pipeline_->record(
                command_buffer, 0U, streaming_task_ring_bytes_of(parameters), wave_dispatch);
            break;
        }
        }
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        commands_->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    commands_->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);

    if (pending_wave_timestamp_recording_valid_) {
        const auto written = wave_timestamp_query_pool_->write_timestamp(command_buffer, 1U);
        if (!written) pending_wave_timestamp_recording_valid_ = false;
    }

    const auto finished = commands_->finish();
    if (!finished) return Result<VkCommandBuffer>::failure(finished.error());
    recording_guard.release();
    core_.mark_wave_recorded();
    pending_state_.mark_recorded();
    return finished;
}

Result<void> StreamingTaskRingResources::Impl::mark_wave_submitted(gpu::SubmissionTicket ticket) {
    if (core_.failed() || !pending_state_.pending()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "ADM3 streaming suffix wave has not been recorded"});
    }
    const auto submitted = commands_->mark_submitted(ticket);
    if (submitted) pending_state_.mark_submission_observed();
    return submitted;
}

Result<std::vector<StreamingTaskFrameResult>>
StreamingTaskRingResources::Impl::retire_wave(gpu::SubmissionTicket ticket,
                                              std::chrono::nanoseconds submission_host_elapsed) {
    const auto retired = commands_->mark_retired(ticket);
    if (!retired) {
        return Result<std::vector<StreamingTaskFrameResult>>::failure(retired.error());
    }

    const auto fail_wave = [this](Error error) {
        fail_retired_wave();
        return Result<std::vector<StreamingTaskFrameResult>>::failure(std::move(error));
    };

    try {
        using TaskState = gpu::StreamingTaskRingCore::TaskState;
        const auto mapped_results = result_buffer_->readable_span(result_buffer_->size());
        if (!mapped_results) return fail_wave(mapped_results.error());
        const std::span<const std::byte> result_bytes = mapped_results.value();

        std::vector<StreamingTaskFrameResult> frame_results;
        frame_results.reserve(core_.pending_wave_tasks().size());
        for (const TaskHandle expected_task : core_.pending_wave_tasks()) {
            if (expected_task.slot >= core_.tasks().size()) {
                return fail_wave({ErrorCode::invalid_state,
                                  "ADM3 submitted wave contains an invalid task slot"});
            }
            const auto &task = core_.task(expected_task.slot);
            if (task.state != TaskState::wave || task.generation != expected_task.generation) {
                return fail_wave({ErrorCode::invalid_state,
                                  "ADM3 task generation changed during suffix execution"});
            }

            std::array<U64Words, result_slot_count> raw_results{};
            const VkDeviceSize result_offset =
                static_cast<VkDeviceSize>(expected_task.slot) * result_frame_stride_;
            if (result_offset + sizeof(raw_results) > result_bytes.size()) {
                return fail_wave(
                    {ErrorCode::invalid_state, "ADM3 result slot is outside the mapped range"});
            }
            std::memcpy(raw_results.data(),
                        result_bytes.data() + static_cast<std::size_t>(result_offset),
                        sizeof(raw_results));
            frame_results.push_back({
                task.frame_index,
                expected_task,
                streaming_task_ring_decode_result(
                    std::span<const U64Words, result_slot_count>(raw_results), result_mode_),
            });
        }

        core_.complete_retired_wave();

        pending_state_.clear();
        if (telemetry_ != nullptr) {
            telemetry_->record_submission_latency(submission_host_elapsed);
            if (telemetry_->enabled() && pending_wave_timestamp_recording_valid_ &&
                wave_timestamp_query_pool_ != nullptr) {
                const auto timestamp_results = wave_timestamp_query_pool_->read_results(
                    0U, streaming_task_ring_wave_timestamp_query_count);
                if (timestamp_results && timestamp_results.value().size() ==
                                             streaming_task_ring_wave_timestamp_query_count) {
                    telemetry_->record_timeline(
                        timestamp_results.value().front().value,
                        timestamp_results.value().back().value,
                        wave_timestamp_query_pool_->capabilities().timestamp_valid_bits,
                        wave_timestamp_query_pool_->capabilities().timestamp_period_nanoseconds,
                        crossvmaf::gpu::TimelineSegment::wave);
                }
            }
        }
        pending_wave_timestamp_recording_valid_ = false;
        return Result<std::vector<StreamingTaskFrameResult>>::success(std::move(frame_results));
    } catch (...) {
        fail_retired_wave();
        throw;
    }
}

void StreamingTaskRingResources::Impl::fail_retired_wave() noexcept {
    core_.fail_retired_wave();
    pending_wave_timestamp_recording_valid_ = false;
    pending_state_.clear();
}

void StreamingTaskRingResources::Impl::cancel() noexcept {
    if (commands_ != nullptr) commands_->cancel();
    pending_state_.rollback_after_cancel(commands_ != nullptr ? commands_->state()
                                                              : gpu::CommandBufferState::failed);
    if (pending_state_.pending()) return;
    core_.restore_unsubmitted_wave();
}

void StreamingTaskRingResources::Impl::cancel_after_device_drain() noexcept {
    if (commands_ != nullptr) commands_->cancel();
    core_.cancel_after_device_drain();
    pending_wave_timestamp_recording_valid_ = false;
    pending_state_.clear();
}

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {
namespace {

constexpr std::string_view telemetry_stage_timestamp_environment_variable =
    "CROSSVMAF_VULKAN_ADM3_STAGE_TIMESTAMPS";

constexpr std::array<std::string_view, scale_operation_count> telemetry_operation_names{
    "dwt", "decouple", "metric_segments", "merge_rows", "reduce", "pack_scale",
};

} // namespace

std::string gpu_stage_name(std::uint32_t stage_id) {
    if (stage_id == finalize_stage_id) {
        return "finalize_batch_combine";
    }
    if (stage_id >= gpu_stage_count) return "unknown";

    const std::uint32_t scale_index = stage_id / scale_operation_count;
    const std::uint32_t operation_index = stage_id % scale_operation_count;
    return std::string(telemetry_operation_names[operation_index]) + ".s" +
           std::to_string(scale_index);
}

bool stage_timestamps_enabled() noexcept {
#if defined(_WIN32)
    char *value = nullptr;
    std::size_t value_length = 0U;
    const int status =
        _dupenv_s(&value, &value_length, telemetry_stage_timestamp_environment_variable.data());
    if (status != 0 || value == nullptr) {
        std::free(value);
        return false;
    }
    const bool enabled = value[0] != '\0' && std::string_view(value) != "0";
    std::free(value);
    return enabled;
#else
    const char *const value = std::getenv(telemetry_stage_timestamp_environment_variable.data());
    return value != nullptr && value[0] != '\0' && std::string_view(value) != "0";
#endif
}

Telemetry::Telemetry(gpu::TelemetryCollector &collector)
    : collector_(&collector), operation_(collector.register_gpu_operation()),
      stage_series_(collector.register_duration_series(gpu_stage_count)) {
}

bool Telemetry::enabled() const noexcept {
    return collector_ != nullptr && collector_->enabled();
}

gpu::TelemetryCollector &Telemetry::collector() const noexcept {
    return *collector_;
}

gpu::GpuOperationHandle Telemetry::operation_handle() const noexcept {
    return operation_;
}

gpu::DurationSeriesHandle Telemetry::stage_series_handle() const noexcept {
    return stage_series_;
}

void Telemetry::record_submission_latency(std::chrono::nanoseconds duration) noexcept {
    collector_->record_gpu_submission_latency(operation_, duration);
}

void Telemetry::record_timeline(std::uint64_t start_timestamp, std::uint64_t end_timestamp,
                                std::uint32_t timestamp_valid_bits,
                                double timestamp_period_nanoseconds,
                                crossvmaf::gpu::TimelineSegment segment) noexcept {
    collector_->record_gpu_timeline(operation_, crossvmaf::gpu::timeline_segment_id(segment),
                                    start_timestamp, end_timestamp, timestamp_valid_bits,
                                    timestamp_period_nanoseconds);
}

void Telemetry::record_stage_success(std::uint32_t stage_id,
                                     long double duration_nanoseconds) noexcept {
    collector_->record_duration(stage_series_, stage_id, gpu::TelemetrySampleOutcome::success,
                                duration_nanoseconds);
}

void Telemetry::record_stage_failure(std::uint32_t stage_id) noexcept {
    collector_->record_duration(stage_series_, stage_id, gpu::TelemetrySampleOutcome::failure);
}

const gpu::GpuOperationSnapshot &Telemetry::operation() const noexcept {
    return collector_->gpu_operation(operation_);
}

const gpu::GpuDurationStatistics &
Telemetry::stage_statistics(std::uint32_t stage_id) const noexcept {
    return collector_->duration_statistics(stage_series_, stage_id);
}

TelemetrySnapshot Telemetry::snapshot() const {
    TelemetrySnapshot result{};
    result.operation_handle = operation_;
    result.operation = operation();
    for (std::uint32_t stage_id = 0U; stage_id < gpu_stage_count; ++stage_id) {
        result.stages[stage_id] = stage_statistics(stage_id);
    }
    return result;
}

} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3::vulkan {
bool Executor::has_streaming_suffix_work() const noexcept {
    return resources_->streaming_layout.active_channel_count != 0U &&
           resources_->streaming_layout.packed_row_frame_elements != 0U &&
           resources_->streaming_layout.reduction_frame_elements != 0U;
}
StreamingTaskRingResources::StreamingTaskRingResources(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {
}
StreamingTaskRingResources::~StreamingTaskRingResources() = default;
Result<std::unique_ptr<StreamingTaskRingResources>>
StreamingTaskRingResources::create(Executor &prototype) {
    auto impl = Impl::create(prototype.streaming_resources());
    if (!impl) return Result<std::unique_ptr<StreamingTaskRingResources>>::failure(impl.error());
    return Result<std::unique_ptr<StreamingTaskRingResources>>::success(
        std::unique_ptr<StreamingTaskRingResources>(
            new StreamingTaskRingResources(std::move(impl).value())));
}
Result<std::optional<TaskHandle>> StreamingTaskRingResources::reserve(std::uint64_t frame_index) {
    return impl_->reserve(frame_index);
}
Result<TaskRingView> StreamingTaskRingResources::view(TaskHandle task) noexcept {
    return impl_->view(task);
}
Result<void> StreamingTaskRingResources::cancel_prefix(TaskHandle task) {
    return impl_->cancel_prefix(task);
}
Result<void> StreamingTaskRingResources::publish_ready(TaskHandle task) {
    return impl_->publish_ready(task);
}
Result<VkCommandBuffer> StreamingTaskRingResources::record_wave() {
    return impl_->record_wave();
}
Result<void> StreamingTaskRingResources::mark_wave_submitted(gpu::SubmissionTicket ticket) {
    return impl_->mark_wave_submitted(ticket);
}
Result<std::vector<StreamingTaskFrameResult>>
StreamingTaskRingResources::retire_wave(gpu::SubmissionTicket ticket,
                                        std::chrono::nanoseconds elapsed) {
    return impl_->retire_wave(ticket, elapsed);
}
std::uint32_t StreamingTaskRingResources::capacity() const noexcept {
    return impl_->capacity();
}
std::uint32_t StreamingTaskRingResources::free_slot_count() const noexcept {
    return impl_->free_slot_count();
}
std::uint32_t StreamingTaskRingResources::prefix_pending_task_count() const noexcept {
    return impl_->prefix_pending_task_count();
}
std::uint32_t StreamingTaskRingResources::ready_task_count() const noexcept {
    return impl_->ready_task_count();
}
bool StreamingTaskRingResources::wave_in_flight() const noexcept {
    return impl_->wave_in_flight();
}
void StreamingTaskRingResources::cancel() noexcept {
    return impl_->cancel();
}
void StreamingTaskRingResources::cancel_after_device_drain() noexcept {
    return impl_->cancel_after_device_drain();
}
} // namespace crossvmaf::metrics::adm3::vulkan

namespace crossvmaf::metrics::adm3 {
class Vulkan::Impl final {
  public:
    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    vulkan::Telemetry *telemetry{};
    std::vector<std::unique_ptr<vulkan::Executor>> slots;
    std::unique_ptr<vulkan::StreamingTaskRingResources> batch;
};
Vulkan::Vulkan(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
}
Vulkan::~Vulkan() = default;
Result<std::unique_ptr<Vulkan>> Vulkan::create(gpu::DeviceView device, FrameDescriptor descriptor,
                                               Config config, vulkan::Telemetry *telemetry) {
    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->descriptor = descriptor;
    impl->config = config;
    impl->telemetry = telemetry;
    return Result<std::unique_ptr<Vulkan>>::success(
        std::unique_ptr<Vulkan>(new Vulkan(std::move(impl))));
}
vulkan::Executor &Vulkan::slot(std::size_t index) noexcept {
    return *impl_->slots[index];
}
Result<void> Vulkan::append_slot() {
    auto created = vulkan::Executor::create_score_only(impl_->device, impl_->descriptor,
                                                       impl_->config, impl_->telemetry, true);
    if (!created) return Result<void>::failure(created.error());
    impl_->slots.push_back(std::move(created).value());
    return Result<void>::success();
}
Result<void> Vulkan::initialize_batch() {
    if (impl_->slots.empty() || impl_->batch)
        return Result<void>::failure(
            {ErrorCode::invalid_state, "metric batch requires initialized operation slots"});
    if (!impl_->slots.front()->has_streaming_suffix_work()) return Result<void>::success();
    for (auto &slot : impl_->slots)
        slot->release_streaming_suffix_resources();
    auto batch = vulkan::StreamingTaskRingResources::create(*impl_->slots.front());
    if (!batch) return Result<void>::failure(batch.error());
    impl_->batch = std::move(batch).value();
    return Result<void>::success();
}
vulkan::StreamingTaskRingResources *Vulkan::batch() noexcept {
    return impl_->batch.get();
}
} // namespace crossvmaf::metrics::adm3
