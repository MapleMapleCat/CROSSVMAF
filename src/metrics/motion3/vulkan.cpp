#include "video/frame.hpp"
#include "metrics/motion3/vulkan.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/buffer.hpp"
#include "gpu/command.hpp"
#include "gpu/pipeline.hpp"
#include "gpu/device.hpp"
#include "gpu/timestamp.hpp"
#include <cstdint>
#include <memory>
#include <vector>
#include "shaders/filter_sad_spv.hpp"
#include "shaders/finalize_spv.hpp"
#include "shaders/reduce_spv.hpp"
#include "shaders/temporal_spv.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <utility>
#include <span>

namespace crossvmaf::metrics::motion3::vulkan {

struct ReductionPass {
    std::uint32_t input_count{};
    gpu::DispatchSize dispatch{};
};

// Sequence-wide temporal buffers are shared by all operation slots. Queue
// submission order serializes their mutations while each slot keeps private
// command, descriptor, scratch, result, and reduction resources.
class SharedState final {
  public:
    gpu::DeviceView device;
    std::unique_ptr<gpu::Buffer> score_ring;
    std::unique_ptr<gpu::Buffer> temporal_state;
    std::uint64_t next_index{};
    bool flushed{};
};

class Resources final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Resources>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           std::shared_ptr<SharedState> shared_state, Telemetry *telemetry = nullptr);

    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    std::shared_ptr<SharedState> shared_state;
    Telemetry *telemetry{};
    std::uint32_t partial_count{};
    gpu::DispatchSize filter_dispatch{};

    std::unique_ptr<gpu::Buffer> partial_a;
    std::unique_ptr<gpu::Buffer> partial_b;
    std::unique_ptr<gpu::Buffer> results;

    std::unique_ptr<gpu::ComputePipeline> filter_pipeline;
    std::unique_ptr<gpu::ComputePipeline> reduce_pipeline;
    std::unique_ptr<gpu::ComputePipeline> finalize_pipeline;
    std::unique_ptr<gpu::ComputePipeline> temporal_pipeline;
    std::unique_ptr<gpu::CommandContext> commands;
    std::unique_ptr<gpu::GpuTimestampPair> gpu_timestamp;
    gpu::GpuTimestampRecordingState pending_timestamp_recording;
    std::vector<ReductionPass> reduction_passes;

  private:
    Resources() = default;
};

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3::vulkan {
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

[[nodiscard]] Result<std::unique_ptr<Buffer>> resources_create_storage_buffer(DeviceView device,
                                                                              VkDeviceSize size) {
    BufferCreateInfo create_info{};
    create_info.size = size;
    create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    return Buffer::create(device, create_info);
}

} // namespace

Result<std::unique_ptr<Resources>> Resources::create(DeviceView device, FrameDescriptor descriptor,
                                                     Config config,
                                                     std::shared_ptr<SharedState> shared_state,
                                                     Telemetry *telemetry) {
    if (shared_state == nullptr || shared_state->score_ring == nullptr ||
        shared_state->temporal_state == nullptr) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::invalid_argument,
             "Motion3 operation slot requires complete shared temporal state"});
    }
    const std::uint64_t pixel_count_u64 =
        static_cast<std::uint64_t>(descriptor.width) * descriptor.height;
    if (pixel_count_u64 > std::numeric_limits<std::uint32_t>::max()) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::unsupported_format,
             "frame dimensions exceed the deterministic Motion3 address space"});
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    if (properties.limits.maxComputeWorkGroupInvocations < workgroup_size ||
        properties.limits.maxComputeWorkGroupSize[0] < workgroup_size ||
        properties.limits.maxComputeWorkGroupSize[1] < 16U) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not support Motion3 filter/reduction workgroups"});
    }

    auto resources = std::unique_ptr<Resources>(new Resources());
    resources->device = device;
    resources->descriptor = descriptor;
    resources->config = config;
    resources->shared_state = std::move(shared_state);
    resources->telemetry = telemetry;
    const auto make_dispatch = [&](std::uint32_t group_count) -> Result<DispatchSize> {
        const std::uint32_t maximum_x = properties.limits.maxComputeWorkGroupCount[0];
        const std::uint32_t maximum_y = properties.limits.maxComputeWorkGroupCount[1];
        if (maximum_x == 0U || maximum_y == 0U) {
            return Result<DispatchSize>::failure(
                {ErrorCode::no_compatible_device,
                 "Vulkan device exposes an empty compute dispatch grid"});
        }
        const std::uint32_t x = std::min(group_count, maximum_x);
        const std::uint32_t y = resources_divide_round_up(group_count, x);
        if (y > maximum_y) {
            return Result<DispatchSize>::failure(
                {ErrorCode::unsupported_format,
                 "frame requires more compute workgroups than the device supports"});
        }
        return Result<DispatchSize>::success({x, y, 1U});
    };

    const std::uint32_t filter_groups_x = resources_divide_round_up(descriptor.width, 16U);
    const std::uint32_t filter_groups_y = resources_divide_round_up(descriptor.height, 16U);
    if (filter_groups_x > properties.limits.maxComputeWorkGroupCount[0] ||
        filter_groups_y > properties.limits.maxComputeWorkGroupCount[1]) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::unsupported_format,
             "frame requires a Motion3 filter dispatch larger than the device supports"});
    }
    resources->filter_dispatch = {filter_groups_x, filter_groups_y, 1U};
    resources->partial_count = filter_groups_x * filter_groups_y;

    std::uint32_t reduction_input_count = resources->partial_count;
    while (reduction_input_count > 1U) {
        const std::uint32_t output_count =
            resources_divide_round_up(reduction_input_count, workgroup_size);
        auto dispatch = make_dispatch(output_count);
        if (!dispatch) {
            return Result<std::unique_ptr<Resources>>::failure(dispatch.error());
        }
        resources->reduction_passes.push_back({reduction_input_count, dispatch.value()});
        reduction_input_count = output_count;
    }

    const VkDeviceSize partial_bytes =
        static_cast<VkDeviceSize>(resources->partial_count) * sizeof(U64Words);
    auto partial_a = resources_create_storage_buffer(device, partial_bytes);
    if (!partial_a) {
        return Result<std::unique_ptr<Resources>>::failure(partial_a.error());
    }
    resources->partial_a = std::move(partial_a).value();
    auto partial_b = resources_create_storage_buffer(device, partial_bytes);
    if (!partial_b) {
        return Result<std::unique_ptr<Resources>>::failure(partial_b.error());
    }
    resources->partial_b = std::move(partial_b).value();
    auto results = gpu::Buffer::create(
        device,
        gpu::host_visible_storage(static_cast<VkDeviceSize>(result_capacity) * sizeof(U64Words)));
    if (!results) {
        return Result<std::unique_ptr<Resources>>::failure(results.error());
    }
    resources->results = std::move(results).value();

    const auto results_cleared = resources->results->fill();
    if (!results_cleared) {
        return Result<std::unique_ptr<Resources>>::failure(results_cleared.error());
    }

    auto filter_pipeline = ComputePipeline::create(device, gpu::generated::motion3_filter_sad_spv,
                                                   3, sizeof(FilterParameters));
    if (!filter_pipeline) {
        return Result<std::unique_ptr<Resources>>::failure(filter_pipeline.error());
    }
    resources->filter_pipeline = std::move(filter_pipeline).value();

    const std::uint32_t reduction_set_count =
        static_cast<std::uint32_t>(std::max<std::size_t>(resources->reduction_passes.size(), 1U));
    auto reduce_pipeline = ComputePipeline::create(device, gpu::generated::motion3_reduce_spv, 2,
                                                   sizeof(ReduceParameters), reduction_set_count);
    if (!reduce_pipeline) {
        return Result<std::unique_ptr<Resources>>::failure(reduce_pipeline.error());
    }
    resources->reduce_pipeline = std::move(reduce_pipeline).value();

    Buffer *reduction_input = resources->partial_a.get();
    Buffer *reduction_output = resources->partial_b.get();
    for (std::size_t index = 0; index < resources->reduction_passes.size(); ++index) {
        const std::array bindings{BufferBinding{reduction_input}, BufferBinding{reduction_output}};
        const auto updated =
            resources->reduce_pipeline->update_buffers(static_cast<std::uint32_t>(index), bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
        std::swap(reduction_input, reduction_output);
    }
    Buffer *const total_sad = reduction_input;

    auto finalize_pipeline = ComputePipeline::create(device, gpu::generated::motion3_finalize_spv,
                                                     2, sizeof(FinalizeParameters));
    if (!finalize_pipeline) {
        return Result<std::unique_ptr<Resources>>::failure(finalize_pipeline.error());
    }
    resources->finalize_pipeline = std::move(finalize_pipeline).value();
    const std::array finalize_bindings{BufferBinding{total_sad, 0, sizeof(U64Words)},
                                       BufferBinding{resources->shared_state->score_ring.get()}};
    auto updated = resources->finalize_pipeline->update_buffers(0, finalize_bindings);
    if (!updated) {
        return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    auto temporal_pipeline = ComputePipeline::create(device, gpu::generated::motion3_temporal_spv,
                                                     3, sizeof(TemporalParameters));
    if (!temporal_pipeline) {
        return Result<std::unique_ptr<Resources>>::failure(temporal_pipeline.error());
    }
    resources->temporal_pipeline = std::move(temporal_pipeline).value();
    const std::array temporal_bindings{BufferBinding{resources->shared_state->score_ring.get()},
                                       BufferBinding{resources->shared_state->temporal_state.get()},
                                       BufferBinding{resources->results.get()}};
    updated = resources->temporal_pipeline->update_buffers(0, temporal_bindings);
    if (!updated) {
        return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    auto commands = CommandContext::create(device);
    if (!commands) {
        return Result<std::unique_ptr<Resources>>::failure(commands.error());
    }
    resources->commands = std::move(commands).value();
    auto gpu_timestamp = gpu::GpuTimestampPair::create(
        device, telemetry != nullptr ? &telemetry->collector() : nullptr,
        telemetry != nullptr ? telemetry->operation_handle() : gpu::GpuOperationHandle{});
    if (!gpu_timestamp) {
        return Result<std::unique_ptr<Resources>>::failure(gpu_timestamp.error());
    }
    resources->gpu_timestamp = std::move(gpu_timestamp).value();
    return Result<std::unique_ptr<Resources>>::success(std::move(resources));
}

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3::vulkan {
namespace {

using gpu::BufferBinding;
using gpu::CommandRecordingGuard;
using gpu::DeviceView;
using gpu::SubmissionOwner;

template <typename T>
[[nodiscard]] std::span<const std::byte> vulkan_bytes_of(const T &value) noexcept {
    return {reinterpret_cast<const std::byte *>(&value), sizeof(T)};
}

[[nodiscard]] U64Words vulkan_split_u64(std::uint64_t value) noexcept {
    return {static_cast<std::uint32_t>(value), static_cast<std::uint32_t>(value >> 32U)};
}

[[nodiscard]] std::int64_t vulkan_join_i64(U64Words value) noexcept {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(value.low) | (static_cast<std::uint64_t>(value.high) << 32U);
    return static_cast<std::int64_t>(bits);
}

struct vulkan_TemporalSchedule {
    bool active{};
    TemporalParameters parameters{};
    std::array<std::uint64_t, result_capacity> output_indices{};
};

} // namespace

Executor::Executor(std::unique_ptr<Resources> resources) noexcept
    : resources_(std::move(resources)) {
}
Executor::~Executor() = default;

Result<std::unique_ptr<Executor>> Executor::create(DeviceView device, FrameDescriptor descriptor,
                                                   Config config, Telemetry *telemetry) {
    auto shared_state = create_shared_state(device);
    if (!shared_state) {
        return Result<std::unique_ptr<Executor>>::failure(shared_state.error());
    }
    return create_with_shared_state(device, descriptor, config, std::move(shared_state).value(),
                                    telemetry);
}

Result<std::shared_ptr<SharedState>> Executor::create_shared_state(DeviceView device) {
    if (!device.complete()) {
        return Result<std::shared_ptr<SharedState>>::failure(
            {ErrorCode::backend_error, "Motion3 shared state received an incomplete device"});
    }
    auto shared_state = std::make_shared<SharedState>();
    shared_state->device = device;
    gpu::BufferCreateInfo buffer_create_info{};
    buffer_create_info.usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_create_info.size = static_cast<VkDeviceSize>(score_ring_size) * sizeof(U64Words);
    auto score_ring = gpu::Buffer::create(device, buffer_create_info);
    if (!score_ring) {
        return Result<std::shared_ptr<SharedState>>::failure(score_ring.error());
    }
    shared_state->score_ring = std::move(score_ring).value();
    buffer_create_info.size = sizeof(U64Words);
    auto temporal_state = gpu::Buffer::create(device, buffer_create_info);
    if (!temporal_state) {
        return Result<std::shared_ptr<SharedState>>::failure(temporal_state.error());
    }
    shared_state->temporal_state = std::move(temporal_state).value();
    auto commands = gpu::CommandContext::create(device);
    if (!commands) {
        return Result<std::shared_ptr<SharedState>>::failure(commands.error());
    }
    auto begun = commands.value()->begin();
    if (!begun) {
        return Result<std::shared_ptr<SharedState>>::failure(begun.error());
    }
    commands.value()->fill_buffer(*shared_state->score_ring);
    commands.value()->fill_buffer(*shared_state->temporal_state);
    auto finished = commands.value()->finish();
    if (!finished) {
        return Result<std::shared_ptr<SharedState>>::failure(finished.error());
    }
    auto waited = commands.value()->submit_and_wait();
    if (!waited) {
        return Result<std::shared_ptr<SharedState>>::failure(waited.error());
    }
    return Result<std::shared_ptr<SharedState>>::success(std::move(shared_state));
}

Result<std::unique_ptr<Executor>>
Executor::create_with_shared_state(DeviceView device, FrameDescriptor descriptor, Config config,
                                   std::shared_ptr<SharedState> shared_state,
                                   Telemetry *telemetry) {
    if (!device.complete() || shared_state == nullptr) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::backend_error, "Motion3 Vulkan executor received incomplete shared state"});
    }
    const auto descriptor_validation = video::validate_frame_layout(descriptor);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<Executor>>::failure(descriptor_validation.error());
    }
    if (config.maximum < numeric::Q32_32{}) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::model_error, "Motion3 maximum must be non-negative"});
    }

    auto resources =
        Resources::create(device, descriptor, config, std::move(shared_state), telemetry);
    if (!resources) {
        return Result<std::unique_ptr<Executor>>::failure(resources.error());
    }
    return Result<std::unique_ptr<Executor>>::success(
        std::unique_ptr<Executor>(new Executor(std::move(resources).value())));
}

namespace {

vulkan_TemporalSchedule vulkan_schedule_push(const Resources &resources,
                                             std::uint64_t index) noexcept {
    vulkan_TemporalSchedule schedule{};
    if (!resources.config.five_frame_window) {
        if (index == 0U) return schedule;
        schedule.active = true;
        schedule.parameters.output_count = 1;
        if (index == 1U) {
            schedule.parameters.mode = 0;
            schedule.parameters.first_slot = 1U;
            schedule.output_indices[0] = 0;
        } else {
            schedule.parameters.mode = resources.config.moving_average ? 2U : 1U;
            schedule.parameters.first_slot =
                static_cast<std::uint32_t>((index - 1U) % score_ring_size);
            schedule.parameters.second_slot = static_cast<std::uint32_t>(index % score_ring_size);
            schedule.output_indices[0] = index - 1U;
        }
        return schedule;
    }

    if (index < 2U) return schedule;
    schedule.active = true;
    if (index == 2U) {
        schedule.parameters.mode = 0;
        schedule.parameters.first_slot = 2U;
        schedule.parameters.output_count = 2;
        schedule.output_indices[0] = 0;
        schedule.output_indices[1] = 1;
    } else if (index == 3U) {
        schedule.parameters.mode = resources.config.moving_average ? 2U : 1U;
        schedule.parameters.first_slot = 3U;
        schedule.parameters.output_count = 1;
        schedule.output_indices[0] = 2;
    } else {
        schedule.parameters.mode = resources.config.moving_average ? 2U : 1U;
        schedule.parameters.first_slot = static_cast<std::uint32_t>((index - 2U) % score_ring_size);
        schedule.parameters.second_slot = static_cast<std::uint32_t>(index % score_ring_size);
        schedule.parameters.output_count = 1;
        schedule.output_indices[0] = index - 1U;
    }
    return schedule;
}

vulkan_TemporalSchedule vulkan_schedule_flush(const Resources &resources) noexcept {
    vulkan_TemporalSchedule schedule{};
    const std::uint64_t next_index = resources.shared_state->next_index;
    if (next_index == 0U) return schedule;
    schedule.active = true;
    if (!resources.config.five_frame_window) {
        schedule.parameters.mode = resources.config.moving_average ? 2U : 1U;
        schedule.parameters.first_slot =
            static_cast<std::uint32_t>((next_index - 1U) % score_ring_size);
        schedule.parameters.output_count = 1;
        schedule.output_indices[0] = next_index - 1U;
    } else if (next_index < 3U) {
        schedule.parameters.mode = 3;
        schedule.parameters.first_slot = 0;
        schedule.parameters.output_count = static_cast<std::uint32_t>(next_index);
        for (std::uint32_t index = 0; index < schedule.parameters.output_count; ++index) {
            schedule.output_indices[index] = index;
        }
    } else {
        schedule.parameters.mode = resources.config.moving_average ? 2U : 1U;
        schedule.parameters.first_slot =
            static_cast<std::uint32_t>((next_index - 1U) % score_ring_size);
        schedule.parameters.output_count = 1;
        schedule.output_indices[0] = next_index - 1U;
    }
    return schedule;
}

Result<VkCommandBuffer> vulkan_record_execute(Resources &resources, bool calculate_score,
                                              const video::GpuPlaneView *previous_luma,
                                              const video::GpuPlaneView *current_luma,
                                              const vulkan_TemporalSchedule &schedule) {
    if (!schedule.active || schedule.parameters.output_count == 0U ||
        schedule.parameters.output_count > result_capacity) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "invalid Motion3 temporal dispatch schedule"});
    }

    if (calculate_score) {
        if (previous_luma == nullptr || current_luma == nullptr ||
            previous_luma->buffer == nullptr || current_luma->buffer == nullptr) {
            return Result<VkCommandBuffer>::failure(
                {ErrorCode::invalid_argument,
                 "Motion3 score calculation requires two GPU luma planes"});
        }
        const std::array filter_bindings{
            BufferBinding{previous_luma->buffer, previous_luma->offset_bytes,
                          previous_luma->size_bytes},
            BufferBinding{current_luma->buffer, current_luma->offset_bytes,
                          current_luma->size_bytes},
            BufferBinding{resources.partial_a.get()}};
        const auto updated = resources.filter_pipeline->update_buffers(0, filter_bindings);
        if (!updated) {
            return Result<VkCommandBuffer>::failure(updated.error());
        }
    }

    auto begun = resources.commands->begin();
    if (!begun) {
        return Result<VkCommandBuffer>::failure(begun.error());
    }
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources.commands);
    auto timestamp_recording = resources.gpu_timestamp->begin_recording(command_buffer);
    resources.commands->memory_barrier(
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT);
    resources.gpu_timestamp->write_start(command_buffer, timestamp_recording);

    if (calculate_score) {
        FilterParameters filter_parameters{};
        filter_parameters.width = resources.descriptor.width;
        filter_parameters.height = resources.descriptor.height;
        filter_parameters.previous_stride_bytes = previous_luma->stride_bytes;
        filter_parameters.current_stride_bytes = current_luma->stride_bytes;
        filter_parameters.bit_depth = resources.descriptor.bit_depth;
        filter_parameters.dispatch_groups_x = resources.filter_dispatch.x;
        auto recorded = resources.filter_pipeline->record(
            command_buffer, 0, vulkan_bytes_of(filter_parameters), resources.filter_dispatch);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        for (std::size_t index = 0; index < resources.reduction_passes.size(); ++index) {
            ReduceParameters reduce_parameters{};
            reduce_parameters.input_count = resources.reduction_passes[index].input_count;
            reduce_parameters.dispatch_groups_x = resources.reduction_passes[index].dispatch.x;
            recorded = resources.reduce_pipeline->record(
                command_buffer, static_cast<std::uint32_t>(index),
                vulkan_bytes_of(reduce_parameters), resources.reduction_passes[index].dispatch);
            if (!recorded) {
                return Result<VkCommandBuffer>::failure(recorded.error());
            }
            resources.commands->memory_barrier(
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }

        const std::uint64_t denominator =
            std::uint64_t{256} * resources.descriptor.width * resources.descriptor.height;
        FinalizeParameters finalize_parameters{};
        finalize_parameters.denominator = vulkan_split_u64(denominator);
        finalize_parameters.maximum =
            vulkan_split_u64(static_cast<std::uint64_t>(resources.config.maximum.raw()));
        finalize_parameters.output_slot =
            static_cast<std::uint32_t>(resources.shared_state->next_index % score_ring_size);
        recorded = resources.finalize_pipeline->record(
            command_buffer, 0, vulkan_bytes_of(finalize_parameters), {1, 1, 1});
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    const auto temporal_recorded = resources.temporal_pipeline->record(
        command_buffer, 0, vulkan_bytes_of(schedule.parameters), {1, 1, 1});
    if (!temporal_recorded) {
        return Result<VkCommandBuffer>::failure(temporal_recorded.error());
    }
    resources.gpu_timestamp->write_end(command_buffer, timestamp_recording);
    resources.commands->memory_barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT);

    const auto finished = resources.commands->finish();
    if (!finished) {
        return Result<VkCommandBuffer>::failure(finished.error());
    }
    recording_guard.release();
    resources.pending_timestamp_recording = timestamp_recording;
    return finished;
}

} // namespace

Result<std::vector<MotionScore>> Executor::push(const video::GpuFrameView &reference,
                                                const video::GpuPlaneView *previous_luma) {
    const auto recorded = record_push(reference, previous_luma);
    if (!recorded) {
        return Result<std::vector<MotionScore>>::failure(recorded.error());
    }
    return submit_pending_for_convenience();
}

Result<std::vector<MotionScore>> Executor::flush() {
    const auto recorded = record_flush();
    if (!recorded) {
        return Result<std::vector<MotionScore>>::failure(recorded.error());
    }
    return submit_pending_for_convenience();
}

Result<std::optional<VkCommandBuffer>>
Executor::record_push(const video::GpuFrameView &reference,
                      const video::GpuPlaneView *previous_luma) {
    if (pending_operation_ != PendingOperation::none) {
        return Result<std::optional<VkCommandBuffer>>::failure(
            {ErrorCode::invalid_state, "Motion3 already has recorded work pending retirement"});
    }
    if (resources_->shared_state->flushed) {
        return Result<std::optional<VkCommandBuffer>>::failure(
            {ErrorCode::invalid_state, "Motion3 Vulkan executor is already flushed"});
    }
    if (reference.index != resources_->shared_state->next_index) {
        return Result<std::optional<VkCommandBuffer>>::failure(
            {ErrorCode::invalid_argument,
             "Motion3 Vulkan frame indices must be contiguous and start at zero"});
    }
    const auto &current_luma = reference.planes[0];
    if (!video::is_compatible_gpu_plane(current_luma, resources_->descriptor.width,
                                        resources_->descriptor.height,
                                        resources_->descriptor.bit_depth)) {
        return Result<std::optional<VkCommandBuffer>>::failure(
            {ErrorCode::invalid_argument, "Motion3 received an incompatible GPU luma plane"});
    }
    const vulkan_TemporalSchedule schedule =
        vulkan_schedule_push(*resources_, resources_->shared_state->next_index);
    if (!schedule.active) {
        pending_operation_ = PendingOperation::push;
        return Result<std::optional<VkCommandBuffer>>::success(std::nullopt);
    }

    const auto recorded =
        vulkan_record_execute(*resources_, true, previous_luma, &current_luma, schedule);
    if (!recorded) {
        return Result<std::optional<VkCommandBuffer>>::failure(recorded.error());
    }
    pending_operation_ = PendingOperation::push;
    pending_command_buffer_ = recorded.value();
    pending_output_indices_ = schedule.output_indices;
    pending_output_count_ = schedule.parameters.output_count;
    pending_has_command_ = true;
    return Result<std::optional<VkCommandBuffer>>::success(recorded.value());
}

Result<std::optional<VkCommandBuffer>> Executor::record_flush() {
    if (pending_operation_ != PendingOperation::none) {
        return Result<std::optional<VkCommandBuffer>>::failure(
            {ErrorCode::invalid_state, "Motion3 already has recorded work pending retirement"});
    }
    if (resources_->shared_state->flushed) {
        return Result<std::optional<VkCommandBuffer>>::failure(
            {ErrorCode::invalid_state, "Motion3 Vulkan flush may only be called once"});
    }
    const vulkan_TemporalSchedule schedule = vulkan_schedule_flush(*resources_);
    if (!schedule.active) {
        pending_operation_ = PendingOperation::flush;
        return Result<std::optional<VkCommandBuffer>>::success(std::nullopt);
    }

    const auto recorded = vulkan_record_execute(*resources_, false, nullptr, nullptr, schedule);
    if (!recorded) {
        return Result<std::optional<VkCommandBuffer>>::failure(recorded.error());
    }
    pending_operation_ = PendingOperation::flush;
    pending_command_buffer_ = recorded.value();
    pending_output_indices_ = schedule.output_indices;
    pending_output_count_ = schedule.parameters.output_count;
    pending_has_command_ = true;
    return Result<std::optional<VkCommandBuffer>>::success(recorded.value());
}

Result<void> Executor::commit_submission(gpu::SubmissionTicket ticket) {
    if (pending_operation_ == PendingOperation::none || pending_committed_ || !ticket) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Motion3 has no uncommitted recorded operation"});
    }
    if (pending_has_command_) {
        const auto command_submitted = resources_->commands->mark_submitted(ticket);
        if (!command_submitted) return command_submitted;
    }

    if (pending_operation_ == PendingOperation::push) {
        ++resources_->shared_state->next_index;
    } else {
        resources_->shared_state->flushed = true;
    }
    pending_ticket_ = ticket;
    pending_committed_ = true;
    return Result<void>::success();
}

Result<std::vector<MotionScore>>
Executor::retire(gpu::SubmissionTicket ticket, std::chrono::nanoseconds submission_host_elapsed) {
    if (!pending_committed_ || !ticket || ticket.serial != pending_ticket_.serial) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state,
             "Motion3 retirement serial does not match its committed operation"});
    }
    if (!pending_has_command_) {
        resources_->gpu_timestamp->record_skip();
        clear_pending();
        return Result<std::vector<MotionScore>>::success({});
    }

    const auto command_retired = resources_->commands->mark_retired(ticket);
    if (!command_retired) {
        return Result<std::vector<MotionScore>>::failure(command_retired.error());
    }
    if (pending_output_count_ == 0U) {
        clear_pending();
        return Result<std::vector<MotionScore>>::success({});
    }
    return readback(submission_host_elapsed);
}

Result<std::vector<MotionScore>> Executor::complete_without_submission() {
    if (pending_operation_ == PendingOperation::none || pending_has_command_ ||
        pending_committed_) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state, "Motion3 pending work requires a queue submission"});
    }

    if (pending_operation_ == PendingOperation::push) {
        ++resources_->shared_state->next_index;
    } else {
        resources_->shared_state->flushed = true;
    }
    resources_->gpu_timestamp->record_skip();
    clear_pending();
    return Result<std::vector<MotionScore>>::success({});
}

Result<std::vector<MotionScore>> Executor::submit_pending_for_convenience() {
    if (pending_operation_ == PendingOperation::none) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state, "Motion3 has no recorded work to submit"});
    }
    if (!pending_has_command_) return complete_without_submission();

    auto submission_owner = SubmissionOwner::create(resources_->device, 1U, 1U, 0U, true);
    if (!submission_owner) {
        cancel();
        return Result<std::vector<MotionScore>>::failure(submission_owner.error());
    }
    const std::array command_buffers{pending_command_buffer_};
    const auto submitted = submission_owner.value()->submit(command_buffers);
    if (!submitted) {
        if (!submission_owner.value()->last_submit_reached_queue()) {
            cancel();
        }
        return Result<std::vector<MotionScore>>::failure(submitted.error());
    }
    const auto committed = commit_submission(submitted.value());
    if (!committed) {
        static_cast<void>(submission_owner.value()->drain());
        return Result<std::vector<MotionScore>>::failure(committed.error());
    }
    const auto submission_retired = submission_owner.value()->retire(submitted.value());
    if (!submission_retired) {
        static_cast<void>(submission_owner.value()->drain());
        return Result<std::vector<MotionScore>>::failure(submission_retired.error());
    }
    return retire(submitted.value(), submission_retired.value().host_elapsed);
}

Result<std::vector<MotionScore>>
Executor::readback(std::chrono::nanoseconds submission_host_elapsed) {
    if (!pending_has_command_ || !pending_committed_ || pending_output_count_ == 0U ||
        resources_->commands->state() != gpu::CommandBufferState::retired) {
        return Result<std::vector<MotionScore>>::failure(
            {ErrorCode::invalid_state, "Motion3 result is not ready for host readback"});
    }

    const auto timestamp_result = resources_->gpu_timestamp->collect(
        resources_->pending_timestamp_recording, submission_host_elapsed);
    if (!timestamp_result) {
        return Result<std::vector<MotionScore>>::failure(timestamp_result.error());
    }

    std::array<U64Words, result_capacity> raw_results{};
    std::span<std::byte> output_bytes = std::as_writable_bytes(std::span(raw_results));
    output_bytes =
        output_bytes.first(static_cast<std::size_t>(pending_output_count_) * sizeof(U64Words));
    const auto read = resources_->results->read(output_bytes);
    if (!read) {
        return Result<std::vector<MotionScore>>::failure(read.error());
    }

    std::vector<MotionScore> output;
    output.reserve(pending_output_count_);
    for (std::uint32_t output_index = 0U; output_index < pending_output_count_; ++output_index) {
        output.push_back({pending_output_indices_[output_index],
                          Q32Value{vulkan_join_i64(raw_results[output_index])}});
    }
    clear_pending();
    return Result<std::vector<MotionScore>>::success(std::move(output));
}

void Executor::cancel() noexcept {
    if (pending_operation_ == PendingOperation::none) return;
    if (pending_has_command_) resources_->commands->cancel();
    if (gpu::ConveniencePendingState::can_rollback_uncommitted_operation(
            pending_committed_, pending_has_command_, resources_->commands->state())) {
        clear_pending();
    }
}

void Executor::clear_pending() noexcept {
    pending_operation_ = PendingOperation::none;
    pending_command_buffer_ = VK_NULL_HANDLE;
    pending_output_indices_ = {};
    pending_output_count_ = 0U;
    pending_has_command_ = false;
    pending_committed_ = false;
    pending_ticket_ = {};
    resources_->pending_timestamp_recording = {};
}

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3::vulkan {

Telemetry::Telemetry(gpu::TelemetryCollector &collector)
    : collector_(&collector), operation_(collector.register_gpu_operation()) {
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

void Telemetry::record_submission_latency(std::chrono::nanoseconds duration) noexcept {
    collector_->record_gpu_submission_latency(operation_, duration);
}

TelemetrySnapshot Telemetry::snapshot() const {
    return {
        operation_,
        collector_->gpu_operation(operation_),
    };
}

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3 {
class Vulkan::Impl final {
  public:
    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    vulkan::Telemetry *telemetry{};
    std::shared_ptr<vulkan::SharedState> temporal_state;
    std::vector<std::unique_ptr<vulkan::Executor>> slots;
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
    auto temporal_state = vulkan::Executor::create_shared_state(device);
    if (!temporal_state) return Result<std::unique_ptr<Vulkan>>::failure(temporal_state.error());
    impl->temporal_state = std::move(temporal_state).value();
    return Result<std::unique_ptr<Vulkan>>::success(
        std::unique_ptr<Vulkan>(new Vulkan(std::move(impl))));
}
vulkan::Executor &Vulkan::slot(std::size_t index) noexcept {
    return *impl_->slots[index];
}
Result<void> Vulkan::append_slot() {
    auto created = vulkan::Executor::create_with_shared_state(
        impl_->device, impl_->descriptor, impl_->config, impl_->temporal_state, impl_->telemetry);
    if (!created) return Result<void>::failure(created.error());
    impl_->slots.push_back(std::move(created).value());
    return Result<void>::success();
}
} // namespace crossvmaf::metrics::motion3
