#include "metrics/speed_chroma/vulkan.hpp"
#include "video/frame.hpp"
#include "numeric/q32.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "metrics/speed_chroma/speed_chroma.hpp"
#include "gpu/buffer.hpp"
#include "gpu/command.hpp"
#include "gpu/pipeline.hpp"
#include "gpu/device.hpp"
#include "gpu/submission.hpp"
#include "gpu/timestamp.hpp"
#include <chrono>
#include <memory>
#include <optional>
#include <vector>
#include "shaders/speed_chroma_antialias_fused_spv.hpp"
#include "shaders/speed_chroma_antialias_identity_spv.hpp"
#include "shaders/speed_chroma_antialias_identity_products_spv.hpp"
#include "shaders/speed_chroma_block_statistics_scalar_spv.hpp"
#include "shaders/speed_chroma_compact_blocks_spv.hpp"
#include "shaders/speed_chroma_compact_tridiagonal_spv.hpp"
#include "shaders/speed_chroma_covariance_spv.hpp"
#include "shaders/speed_chroma_exact_regularity_spv.hpp"
#include "shaders/speed_chroma_filter_vertical_spv.hpp"
#include "shaders/speed_chroma_finalize_spv.hpp"
#include "shaders/speed_chroma_local_horizontal_spv.hpp"
#include "shaders/speed_chroma_means_spv.hpp"
#include "shaders/speed_chroma_scale_spv.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <string>
#include "shaders/speed_chroma_compact_tridiagonal_wave_spv.hpp"
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include "shaders/speed_chroma_block_statistics_scalar_wave_spv.hpp"
#include "shaders/speed_chroma_compact_blocks_wave_spv.hpp"
#include "shaders/speed_chroma_exact_regularity_wave_spv.hpp"
#include "shaders/speed_chroma_finalize_wave_spv.hpp"
#include <cstring>

namespace crossvmaf::metrics::speed_chroma::vulkan {
namespace {

[[nodiscard]] Result<void>
scale_plan_make_nearest_axis_samples(std::uint32_t source_size, std::uint32_t destination_size,
                                     std::span<ScaleAxisSample> output) noexcept {
    for (std::uint32_t destination_index = 0U; destination_index < destination_size;
         ++destination_index) {
        const std::uint64_t product = static_cast<std::uint64_t>(destination_index) * source_size;
        const std::uint64_t source_index = product / destination_size;
        if (source_index >= source_size) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format, "Speed Chroma nearest coordinate is out of range"});
        }
        output[destination_index] = {static_cast<std::uint32_t>(source_index), 0U};
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void>
scale_plan_make_bilinear_axis_samples(std::uint32_t source_size, std::uint32_t destination_size,
                                      std::span<ScaleAxisSample> output) noexcept {
    if (source_size < destination_size) {
        return Result<void>::failure({ErrorCode::unsupported_format,
                                      "Speed Chroma bilinear scaling does not support upscaling"});
    }

    const std::uint64_t denominator = static_cast<std::uint64_t>(destination_size) * 2U;
    for (std::uint32_t destination_index = 0U; destination_index < destination_size;
         ++destination_index) {
        const std::uint64_t coordinate_factor =
            static_cast<std::uint64_t>(destination_index) * 2U + 1U;
        if (coordinate_factor > std::numeric_limits<std::uint64_t>::max() / source_size) {
            return Result<void>::failure({ErrorCode::unsupported_format,
                                          "Speed Chroma bilinear coordinate overflows uint64"});
        }
        const std::uint64_t scaled_coordinate = coordinate_factor * source_size;
        if (scaled_coordinate < destination_size) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format, "Speed Chroma bilinear coordinate is negative"});
        }

        const std::uint64_t numerator = scaled_coordinate - destination_size;
        std::uint64_t source_index = numerator / denominator;
        const std::uint64_t remainder = numerator - source_index * denominator;
        if (source_index >= source_size) {
            return Result<void>::failure({ErrorCode::unsupported_format,
                                          "Speed Chroma bilinear coordinate is out of range"});
        }

        const auto fraction_result = numeric::Q32_32::from_ratio(
            static_cast<std::int64_t>(remainder), static_cast<std::int64_t>(denominator));
        if (!fraction_result) {
            return Result<void>::failure(fraction_result.error());
        }
        const std::int64_t fraction_raw = fraction_result.value().raw();
        if (fraction_raw < 0 || fraction_raw > numeric::Q32_32::one_raw) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "Speed Chroma bilinear fraction is outside [0, 1]"});
        }

        std::uint32_t fraction_q0_32{};
        if (fraction_raw == numeric::Q32_32::one_raw) {
            ++source_index;
            if (source_index >= source_size) {
                return Result<void>::failure(
                    {ErrorCode::unsupported_format,
                     "Speed Chroma rounded bilinear coordinate is out of range"});
            }
        } else {
            fraction_q0_32 = static_cast<std::uint32_t>(fraction_raw);
            const std::uint64_t adjacent_source_index =
                source_index + (fraction_q0_32 == 0U ? 0U : 1U);
            if (adjacent_source_index >= source_size) {
                return Result<void>::failure({ErrorCode::unsupported_format,
                                              "Speed Chroma bilinear neighbor is out of range"});
            }
        }
        output[destination_index] = {static_cast<std::uint32_t>(source_index), fraction_q0_32};
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void>
scale_plan_make_axis_samples(std::uint32_t source_size, std::uint32_t destination_size,
                             PrescaleMethod method, std::span<ScaleAxisSample> output) noexcept {
    if (source_size == 0U || destination_size == 0U || output.size() != destination_size) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma scale axis geometry is invalid"});
    }
    if (method == PrescaleMethod::nearest) {
        return scale_plan_make_nearest_axis_samples(source_size, destination_size, output);
    }
    if (method == PrescaleMethod::bilinear) {
        return scale_plan_make_bilinear_axis_samples(source_size, destination_size, output);
    }
    return Result<void>::failure(
        {ErrorCode::unsupported_model, "Speed Chroma scale method is unsupported"});
}

} // namespace

Result<SpeedChromaScalePlan> make_speed_chroma_scale_plan(std::uint32_t source_width,
                                                          std::uint32_t source_height,
                                                          std::uint32_t destination_width,
                                                          std::uint32_t destination_height,
                                                          PrescaleMethod method) {
    const std::uint64_t axis_sample_count =
        static_cast<std::uint64_t>(destination_width) + destination_height;
    if (axis_sample_count == 0U ||
        axis_sample_count > std::numeric_limits<std::size_t>::max() / sizeof(ScaleAxisSample)) {
        return Result<SpeedChromaScalePlan>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma scale plan exceeds the host address space"});
    }

    SpeedChromaScalePlan plan{};
    plan.horizontal_sample_count = destination_width;
    plan.vertical_sample_count = destination_height;
    plan.axis_samples.resize(static_cast<std::size_t>(axis_sample_count));

    const auto horizontal_result =
        scale_plan_make_axis_samples(source_width, destination_width, method,
                                     std::span(plan.axis_samples).first(destination_width));
    if (!horizontal_result) {
        return Result<SpeedChromaScalePlan>::failure(horizontal_result.error());
    }
    const auto vertical_result =
        scale_plan_make_axis_samples(source_height, destination_height, method,
                                     std::span(plan.axis_samples).subspan(destination_width));
    if (!vertical_result) {
        return Result<SpeedChromaScalePlan>::failure(vertical_result.error());
    }
    return Result<SpeedChromaScalePlan>::success(std::move(plan));
}

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
enum class GpuTimestampQuerySlot : std::uint32_t {
    scale_start = 0U,
    scale_end = 1U,
    antialias_fused_start = 2U,
    antialias_fused_end = 3U,
    local_vertical_start = 4U,
    local_vertical_end = 5U,
    local_horizontal_highpass_start = 6U,
    local_horizontal_highpass_end = 7U,
    means_start = 8U,
    means_end = 9U,
    covariance_start = 10U,
    covariance_end = 11U,
    compact_tridiagonal_start = 12U,
    compact_tridiagonal_end = 13U,
    compact_regularity_start = 14U,
    compact_regularity_end = 15U,
    block_statistics_start = 16U,
    block_statistics_end = 17U,
    finalize_start = 18U,
    finalize_end = 19U,
    count = 20U,
};

inline constexpr std::uint32_t gpu_timestamp_query_count =
    static_cast<std::uint32_t>(GpuTimestampQuerySlot::count);
class Resources final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Resources>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           Telemetry *telemetry = nullptr, bool skip_streaming_suffix = false);

    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    Dimensions dimensions;
    bool identity_scale_fused{};
    Telemetry *telemetry{};
    gpu::DispatchSize scaled_dispatch;
    gpu::DispatchSize source_dispatch;
    gpu::DispatchSize antialias_dispatch;
    gpu::DispatchSize operating_dispatch;
    gpu::DispatchSize covariance_dispatch;
    gpu::DispatchSize score_block_dispatch;
    gpu::DispatchSize score_scalar_dispatch;
    VkDeviceSize scaled_sample_offset_bytes{};
    VkDeviceSize scaled_sample_bytes{};
    std::uint32_t scaled_sample_offset_elements{};
    VkDeviceSize highpass_frame_bytes{};
    VkDeviceSize scratch_frame_bytes{};
    VkDeviceSize result_frame_bytes{};
    VkDeviceSize status_frame_bytes{};

    std::unique_ptr<gpu::Buffer> scaled;
    std::unique_ptr<gpu::Buffer> antialias_products;
    std::unique_ptr<gpu::Buffer> decimated;
    std::unique_ptr<gpu::Buffer> local_vertical;
    std::unique_ptr<gpu::Buffer> highpass;
    std::unique_ptr<gpu::Buffer> status;
    std::unique_ptr<gpu::Buffer> score_scratch;
    std::unique_ptr<gpu::Buffer> score_result;
    std::unique_ptr<gpu::Buffer> diagnostics_snapshot;

    std::unique_ptr<gpu::ComputePipeline> scale_pipeline;
    std::unique_ptr<gpu::ComputePipeline> vertical_pipeline;
    std::unique_ptr<gpu::ComputePipeline> antialias_fused_pipeline;
    std::unique_ptr<gpu::ComputePipeline> local_pipeline;
    std::unique_ptr<gpu::ComputePipeline> means_pipeline;
    std::unique_ptr<gpu::ComputePipeline> covariance_pipeline;
    std::unique_ptr<gpu::ComputePipeline> compact_tridiagonal_pipeline;
    std::unique_ptr<gpu::ComputePipeline> exact_regularity_pipeline;
    std::unique_ptr<gpu::ComputePipeline> compact_blocks_pipeline;
    std::unique_ptr<gpu::ComputePipeline> block_statistics_scalar_pipeline;
    std::unique_ptr<gpu::ComputePipeline> finalize_pipeline;
    std::unique_ptr<gpu::CommandContext> commands;
    std::unique_ptr<gpu::TimestampQueryPool> gpu_timestamp_query_pool;

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
    void cancel() noexcept;
    // SubmissionOwner must be drained before this terminal recovery path is
    // used. It exists for failures after a queue submission was accepted.
    void cancel_after_device_drain() noexcept;

  private:
    Impl() = default;

    void fail_retired_wave() noexcept;

    gpu::StreamingTaskRingCore core_;
    gpu::DeviceView device_;
    Dimensions dimensions_;
    Config config_;
    Telemetry *telemetry_{};
    VkDeviceSize highpass_frame_bytes_{};
    VkDeviceSize scratch_frame_bytes_{};
    VkDeviceSize result_frame_bytes_{};
    VkDeviceSize status_frame_bytes_{};
    VkDeviceSize highpass_frame_stride_{};
    VkDeviceSize scratch_frame_stride_{};
    VkDeviceSize result_frame_stride_{};
    VkDeviceSize status_frame_stride_{};

    std::unique_ptr<gpu::Buffer> wave_slot_buffer_;
    std::unique_ptr<gpu::Buffer> highpass_buffer_;
    std::unique_ptr<gpu::Buffer> scratch_buffer_;
    std::unique_ptr<gpu::Buffer> result_buffer_;
    std::unique_ptr<gpu::Buffer> status_buffer_;

    std::unique_ptr<gpu::ComputePipeline> compact_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> exact_regularity_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> block_statistics_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> block_statistics_scalar_pipeline_;
    std::unique_ptr<gpu::ComputePipeline> finalize_pipeline_;
    std::unique_ptr<gpu::CommandContext> commands_;
    std::unique_ptr<gpu::TimestampQueryPool> wave_timestamp_query_pool_;
    bool pending_wave_timestamp_recording_valid_{};
    gpu::ConveniencePendingState pending_state_;
};
} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
namespace {

using gpu::Buffer;
using gpu::BufferBinding;
using gpu::BufferCreateInfo;
using gpu::CommandContext;
using gpu::CommandRecordingGuard;
using gpu::ComputePipeline;
using gpu::DeviceView;
using gpu::GpuTimestampConfiguration;
using gpu::query_timestamp_capabilities;
using gpu::TimestampQueryPool;

[[nodiscard]] Result<Dimensions> resources_make_dimensions(std::uint32_t width,
                                                           std::uint32_t height,
                                                           numeric::Q32_32 prescale) noexcept {
    if (width == 0U || height == 0U || prescale.raw() <= 0 ||
        prescale > numeric::Q32_32::from_integer(1)) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_model,
             "CrossVMAF 1.0.16 Speed Chroma requires 0 < prescale <= 1"});
    }
    const auto scaled_dimension = [prescale](std::uint32_t value) -> Result<std::uint32_t> {
        const std::uint64_t scale = static_cast<std::uint64_t>(prescale.raw());
        constexpr std::uint64_t rounding = std::uint64_t{1} << 31U;
        if (value > (std::numeric_limits<std::uint64_t>::max() - rounding) / scale) {
            return Result<std::uint32_t>::failure(
                {ErrorCode::unsupported_format, "Speed Chroma scaled dimension overflows"});
        }
        const std::uint64_t product = static_cast<std::uint64_t>(value) * scale + rounding;
        const std::uint64_t scaled = product >> 32U;
        if (scaled == 0U || scaled > std::numeric_limits<std::uint32_t>::max()) {
            return Result<std::uint32_t>::failure(
                {ErrorCode::unsupported_format, "Speed Chroma scaled dimension is out of range"});
        }
        return Result<std::uint32_t>::success(static_cast<std::uint32_t>(scaled));
    };

    const auto scaled_width = scaled_dimension(width);
    const auto scaled_height = scaled_dimension(height);
    if (!scaled_width || !scaled_height) {
        return Result<Dimensions>::failure(!scaled_width ? scaled_width.error()
                                                         : scaled_height.error());
    }
    Dimensions dimensions{};
    dimensions.original_width = width;
    dimensions.original_height = height;
    dimensions.scaled_width = scaled_width.value();
    dimensions.scaled_height = scaled_height.value();
    dimensions.operating_width = dimensions.scaled_width >> 4U;
    dimensions.operating_height = dimensions.scaled_height >> 4U;
    dimensions.truncated_width = (dimensions.operating_width / block_size) * block_size;
    dimensions.truncated_height = (dimensions.operating_height / block_size) * block_size;
    if (dimensions.truncated_width == 0U || dimensions.truncated_height == 0U) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma image is too small after prescale and 16x decimation"});
    }
    dimensions.blocks_horizontal = dimensions.truncated_width / block_size;
    dimensions.blocks_vertical = dimensions.truncated_height / block_size;
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(dimensions.blocks_horizontal) * dimensions.blocks_vertical;
    if (block_count > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma block count exceeds deterministic Q32 limits"});
    }
    dimensions.block_count = static_cast<std::uint32_t>(block_count);
    dimensions.submatrix_width = dimensions.truncated_width - block_size + 1U;
    dimensions.submatrix_height = dimensions.truncated_height - block_size + 1U;
    const std::uint64_t covariance_sample_count =
        static_cast<std::uint64_t>(dimensions.submatrix_width) * dimensions.submatrix_height;
    if (covariance_sample_count == 0U ||
        covariance_sample_count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        return Result<Dimensions>::failure(
            {ErrorCode::unsupported_format,
             "Speed Chroma covariance sample count exceeds Q32 limits"});
    }
    dimensions.covariance_sample_count = static_cast<std::uint32_t>(covariance_sample_count);
    return Result<Dimensions>::success(dimensions);
}

[[nodiscard]] constexpr std::uint32_t resources_divide_round_up(std::uint32_t value,
                                                                std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

[[nodiscard]] Result<std::unique_ptr<Buffer>>
resources_create_storage_buffer(DeviceView device, VkDeviceSize size,
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

[[nodiscard]] Result<VkDeviceSize> resources_checked_size(std::uint64_t element_count,
                                                          std::uint64_t element_size,
                                                          VkDeviceSize limit, const char *label) {
    if (element_count == 0U ||
        element_count > std::numeric_limits<std::uint64_t>::max() / element_size) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("Speed Chroma ") + label + " buffer size overflows"});
    }
    const std::uint64_t size = element_count * element_size;
    if (size > limit) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("Speed Chroma ") + label + " buffer exceeds maxStorageBufferRange"});
    }
    return Result<VkDeviceSize>::success(static_cast<VkDeviceSize>(size));
}

[[nodiscard]] Result<VkDeviceSize>
resources_checked_align_up(VkDeviceSize value, VkDeviceSize alignment, const char *label) {
    if (alignment <= 1U) return Result<VkDeviceSize>::success(value);
    const VkDeviceSize remainder = value % alignment;
    if (remainder == 0U) return Result<VkDeviceSize>::success(value);
    const VkDeviceSize padding = alignment - remainder;
    if (value > std::numeric_limits<VkDeviceSize>::max() - padding) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("Speed Chroma ") + label + " alignment overflows"});
    }
    return Result<VkDeviceSize>::success(value + padding);
}

[[nodiscard]] Result<void> resources_upload_scale_plan(Resources &resources,
                                                       const SpeedChromaScalePlan &scale_plan) {
    auto begun = resources.commands->begin();
    if (!begun) return Result<void>::failure(begun.error());
    CommandRecordingGuard recording_guard(*resources.commands);
    resources.commands->update_buffer(*resources.scaled,
                                      std::as_bytes(std::span(scale_plan.axis_samples)));
    resources.commands->memory_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    const auto submitted = resources.commands->submit_and_wait();
    recording_guard.release();
    return submitted;
}

[[nodiscard]] Result<void> resources_initialize_gpu_timestamp_queries(Resources &resources) {
    Telemetry *const telemetry = resources.telemetry;
    if (telemetry == nullptr || !telemetry->enabled()) {
        return Result<void>::success();
    }

    GpuTimestampConfiguration configuration{};
    configuration.queue_family_index = resources.device.queue_family;

    const auto capabilities_result = query_timestamp_capabilities(resources.device);
    if (!capabilities_result) {
        const Error &capability_error = capabilities_result.error();
        configuration.unavailable_reason = capability_error.message;
        telemetry->configure_timestamps(configuration);
        if (capability_error.code == ErrorCode::device_lost) {
            return Result<void>::failure(capability_error);
        }
        return Result<void>::success();
    }

    const auto capabilities = capabilities_result.value();
    configuration.timestamp_valid_bits = capabilities.timestamp_valid_bits;
    configuration.timestamp_period_nanoseconds =
        static_cast<long double>(capabilities.timestamp_period_nanoseconds);
    configuration.device_supported = capabilities.supported();
    if (!capabilities.supported()) {
        configuration.unavailable_reason =
            "selected Vulkan queue family reports timestampValidBits == 0";
        telemetry->configure_timestamps(configuration);
        return Result<void>::success();
    }

    auto query_pool_result =
        TimestampQueryPool::create(resources.device, gpu_timestamp_query_count);
    if (!query_pool_result) {
        const Error &query_pool_error = query_pool_result.error();
        configuration.unavailable_reason = query_pool_error.message;
        telemetry->configure_timestamps(configuration);
        if (query_pool_error.code == ErrorCode::device_lost) {
            return Result<void>::failure(query_pool_error);
        }
        return Result<void>::success();
    }

    std::unique_ptr<TimestampQueryPool> query_pool = std::move(query_pool_result).value();
    const auto &query_pool_capabilities = query_pool->capabilities();
    configuration.timestamp_valid_bits = query_pool_capabilities.timestamp_valid_bits;
    configuration.timestamp_period_nanoseconds =
        static_cast<long double>(query_pool_capabilities.timestamp_period_nanoseconds);
    configuration.device_supported = query_pool->supported();
    configuration.query_pool_available = query_pool->supported();
    if (!query_pool->supported()) {
        configuration.unavailable_reason =
            "selected Vulkan queue family reports timestampValidBits == 0";
    } else {
        resources.gpu_timestamp_query_pool = std::move(query_pool);
    }
    telemetry->configure_timestamps(configuration);
    return Result<void>::success();
}

void resources_configure_gpu_execution_report(Resources &resources) noexcept {
    Telemetry *const telemetry = resources.telemetry;
    if (telemetry == nullptr || !telemetry->enabled()) return;

    const auto make_dispatch_configuration = [](gpu::DispatchSize dispatch) noexcept {
        return DispatchConfiguration{dispatch.x, dispatch.y, dispatch.z};
    };
    telemetry->configure_execution(GpuExecutionConfiguration{
        true,
        resources.config.kernel_scale.raw(),
        resources.config.prescale.raw(),
        resources.config.prescale_method,
        resources.config.sigma_nn.raw(),
        resources.config.nn_floor.raw(),
        resources.config.weight_variance_mode,
        resources.config.maximum.raw(),
        resources.dimensions.original_width,
        resources.dimensions.original_height,
        resources.dimensions.scaled_width,
        resources.dimensions.scaled_height,
        resources.dimensions.operating_width,
        resources.dimensions.operating_height,
        resources.dimensions.truncated_width,
        resources.dimensions.truncated_height,
        resources.dimensions.block_count,
        resources.dimensions.covariance_sample_count,
        make_dispatch_configuration(resources.source_dispatch),
        make_dispatch_configuration(resources.scaled_dispatch),
        make_dispatch_configuration(resources.operating_dispatch),
        make_dispatch_configuration(resources.covariance_dispatch),
        make_dispatch_configuration(resources.score_block_dispatch),
    });
}

} // namespace

Result<std::unique_ptr<Resources>> Resources::create(DeviceView device, FrameDescriptor descriptor,
                                                     Config config, Telemetry *telemetry,
                                                     bool skip_streaming_suffix) {
    const auto dimensions = resources_make_dimensions(
        plane_width(descriptor, 1U), plane_height(descriptor, 1U), config.prescale);
    if (!dimensions) {
        return Result<std::unique_ptr<Resources>>::failure(dimensions.error());
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    const auto &limits = properties.limits;
    constexpr std::uint32_t maximum_linear_score_workgroup_size =
        std::max({means_workgroup_size, block_statistics_workgroup_size,
                  covariance_workgroup_size});
    if (limits.maxComputeWorkGroupInvocations < tile_size * tile_size ||
        limits.maxComputeWorkGroupSize[0] < tile_size ||
        limits.maxComputeWorkGroupSize[1] < tile_size ||
        limits.maxComputeWorkGroupCount[2] < parameter_stream_count ||
        limits.maxPerStageDescriptorStorageBuffers < 6U ||
        limits.maxPushConstantsSize < sizeof(ScoreParameters)) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet deterministic Speed Chroma limits"});
    }
    if (limits.maxComputeWorkGroupInvocations < maximum_linear_score_workgroup_size ||
        limits.maxComputeWorkGroupSize[0] < maximum_linear_score_workgroup_size) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet selected Speed Chroma linear "
             "workgroup limits"});
    }
    if (limits.maxComputeSharedMemorySize <
        std::max(means_shared_memory_bytes, covariance_shared_memory_bytes)) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet Speed Chroma statistics shared-memory "
             "limits"});
    }
    if (limits.maxComputeWorkGroupInvocations < compact_tridiagonal_workgroup_size ||
        limits.maxComputeWorkGroupSize[0] < compact_tridiagonal_workgroup_size ||
        limits.maxComputeSharedMemorySize < compact_tridiagonal_shared_memory_bytes) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device, "Vulkan device does not meet Speed Chroma compact "
                                              "tridiagonal workgroup or shared-memory limits"});
    }
    if (limits.maxComputeWorkGroupInvocations < exact_regularity_workgroup_size ||
        limits.maxComputeWorkGroupSize[0] < exact_regularity_workgroup_size ||
        limits.maxComputeSharedMemorySize < exact_regularity_shared_memory_bytes) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet Speed Chroma exact regularity "
             "workgroup or shared-memory limits"});
    }
    if (limits.maxComputeWorkGroupInvocations < finalize_workgroup_size ||
        limits.maxComputeWorkGroupSize[0] < finalize_workgroup_size ||
        limits.maxComputeSharedMemorySize < finalize_shared_memory_bytes) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet Speed Chroma finalize workgroup "
             "or shared-memory limits"});
    }
    if (limits.maxComputeWorkGroupInvocations < block_statistics_workgroup_size ||
        limits.maxComputeWorkGroupSize[0] < block_statistics_workgroup_size ||
        limits.maxComputeSharedMemorySize < block_statistics_shared_memory_bytes) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet Speed Chroma block statistics "
             "workgroup or shared-memory limits"});
    }

    auto resources = std::unique_ptr<Resources>(new Resources());
    resources->telemetry = telemetry;
    resources->device = device;
    resources->descriptor = descriptor;
    resources->config = config;
    resources->dimensions = dimensions.value();
    SpeedChromaScalePlan scale_plan{};
    const bool requires_scale_axis_map =
        resources->dimensions.original_width != resources->dimensions.scaled_width ||
        resources->dimensions.original_height != resources->dimensions.scaled_height;
    resources->identity_scale_fused = !requires_scale_axis_map;
    if (requires_scale_axis_map) {
        auto scale_plan_result = make_speed_chroma_scale_plan(
            resources->dimensions.original_width, resources->dimensions.original_height,
            resources->dimensions.scaled_width, resources->dimensions.scaled_height,
            resources->config.prescale_method);
        if (!scale_plan_result) {
            return Result<std::unique_ptr<Resources>>::failure(scale_plan_result.error());
        }
        scale_plan = std::move(scale_plan_result).value();
    }
    resources->scaled_dispatch = {
        resources_divide_round_up(resources->dimensions.scaled_width, tile_size),
        resources_divide_round_up(resources->dimensions.scaled_height, tile_size),
        parameter_stream_count};
    resources->source_dispatch = {
        resources_divide_round_up(resources->dimensions.original_width, tile_size),
        resources_divide_round_up(resources->dimensions.original_height, tile_size),
        parameter_stream_count};
    resources->antialias_dispatch = {
        resources_divide_round_up(resources->dimensions.operating_width, tile_size),
        resources->dimensions.operating_height, parameter_stream_count};
    resources->operating_dispatch = {
        resources_divide_round_up(resources->dimensions.operating_width, tile_size),
        resources_divide_round_up(resources->dimensions.operating_height, tile_size),
        parameter_stream_count};
    resources->covariance_dispatch = {resources->dimensions.covariance_sample_count <=
                                              covariance_displacement_sample_limit
                                          ? covariance_displacement_count
                                          : elements_per_block * covariance_entry_batch_count,
                                      1U, parameter_stream_count};
    if (resources->scaled_dispatch.x > limits.maxComputeWorkGroupCount[0] ||
        resources->scaled_dispatch.y > limits.maxComputeWorkGroupCount[1] ||
        resources->source_dispatch.x > limits.maxComputeWorkGroupCount[0] ||
        resources->source_dispatch.y > limits.maxComputeWorkGroupCount[1] ||
        resources->antialias_dispatch.x > limits.maxComputeWorkGroupCount[0] ||
        resources->antialias_dispatch.y > limits.maxComputeWorkGroupCount[1] ||
        resources->operating_dispatch.x > limits.maxComputeWorkGroupCount[0] ||
        resources->operating_dispatch.y > limits.maxComputeWorkGroupCount[1] ||
        resources->dimensions.block_count > limits.maxComputeWorkGroupCount[0] ||
        block_statistics_scalar_dispatch_group_count(resources->dimensions.block_count) >
            limits.maxComputeWorkGroupCount[0]) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::unsupported_format, "Speed Chroma dispatch exceeds Vulkan device limits"});
    }
    resources->score_block_dispatch = {resources->dimensions.block_count, 1U,
                                       parameter_stream_count};
    resources->score_scalar_dispatch = {
        block_statistics_scalar_dispatch_group_count(resources->dimensions.block_count), 1U,
        parameter_stream_count};

    const std::uint64_t scaled_count =
        static_cast<std::uint64_t>(resources->dimensions.scaled_width) *
        resources->dimensions.scaled_height;
    const std::uint64_t operating_count =
        static_cast<std::uint64_t>(resources->dimensions.operating_width) *
        resources->dimensions.operating_height;
    const auto scaled_q32_bytes =
        resources->identity_scale_fused
            ? Result<VkDeviceSize>::success(0U)
            : resources_checked_size(scaled_count, parameter_stream_count * sizeof(std::uint64_t),
                                     limits.maxStorageBufferRange, "scaled samples");
    const auto scale_axis_map_bytes =
        scale_plan.axis_samples.empty()
            ? Result<VkDeviceSize>::success(0U)
            : resources_checked_size(scale_plan.axis_samples.size(), sizeof(ScaleAxisSample),
                                     limits.maxStorageBufferRange, "scale axis map");
    const auto operating_q32_bytes =
        resources_checked_size(operating_count, parameter_stream_count * sizeof(std::uint64_t),
                               limits.maxStorageBufferRange, "operating samples");
    const auto local_q32_bytes =
        resources_checked_size(operating_count, parameter_stream_count * sizeof(std::uint64_t),
                               limits.maxStorageBufferRange, "local filter");
    const auto highpass_q32_bytes =
        resources_checked_size(operating_count, parameter_stream_count * sizeof(std::uint64_t),
                               limits.maxStorageBufferRange, "highpass samples");
    const std::uint64_t score_scratch_elements =
        score_scratch_base_elements +
        score_scratch_elements_per_block * resources->dimensions.block_count;
    const auto score_scratch_bytes = resources_checked_size(
        score_scratch_elements, parameter_stream_count * sizeof(std::uint64_t),
        limits.maxStorageBufferRange, "score scratch");
    const auto score_result_bytes = resources_checked_size(
        score_result_elements, sizeof(std::uint64_t), limits.maxStorageBufferRange, "score result");
    const auto diagnostics_snapshot_bytes = resources_checked_size(
        diagnostic_channel_base + 8U * resources->dimensions.block_count +
            score_packet_result_elements,
        sizeof(std::uint64_t), limits.maxStorageBufferRange, "diagnostics snapshot");
    if (!scaled_q32_bytes || !scale_axis_map_bytes || !operating_q32_bytes || !local_q32_bytes ||
        !highpass_q32_bytes || !score_scratch_bytes || !score_result_bytes ||
        !diagnostics_snapshot_bytes) {
        const Error error = !scaled_q32_bytes       ? scaled_q32_bytes.error()
                            : !scale_axis_map_bytes ? scale_axis_map_bytes.error()
                            : !operating_q32_bytes  ? operating_q32_bytes.error()
                            : !local_q32_bytes      ? local_q32_bytes.error()
                            : !highpass_q32_bytes   ? highpass_q32_bytes.error()
                            : !score_scratch_bytes  ? score_scratch_bytes.error()
                            : !score_result_bytes   ? score_result_bytes.error()
                                                    : diagnostics_snapshot_bytes.error();
        return Result<std::unique_ptr<Resources>>::failure(error);
    }
    const VkDeviceSize scaled_workspace_alignment =
        std::max<VkDeviceSize>(sizeof(Q32Words), limits.minStorageBufferOffsetAlignment);
    const auto scaled_sample_offset = resources_checked_align_up(
        scale_axis_map_bytes.value(), scaled_workspace_alignment, "scaled sample offset");
    if (!scaled_sample_offset ||
        scaled_sample_offset.value() > limits.maxStorageBufferRange - scaled_q32_bytes.value() ||
        scaled_sample_offset.value() / sizeof(Q32Words) >
            std::numeric_limits<std::uint32_t>::max()) {
        return Result<std::unique_ptr<Resources>>::failure(
            !scaled_sample_offset ? scaled_sample_offset.error()
                                  : Error{ErrorCode::unsupported_format,
                                          "Speed Chroma scale workspace exceeds shader or "
                                          "maxStorageBufferRange limits"});
    }
    const VkDeviceSize scaled_workspace_bytes =
        scaled_sample_offset.value() + scaled_q32_bytes.value();
    resources->scaled_sample_offset_bytes = scaled_sample_offset.value();
    resources->scaled_sample_bytes = scaled_q32_bytes.value();
    resources->scaled_sample_offset_elements =
        static_cast<std::uint32_t>(scaled_sample_offset.value() / sizeof(Q32Words));
    resources->highpass_frame_bytes = highpass_q32_bytes.value();
    resources->scratch_frame_bytes = score_scratch_bytes.value();
    resources->result_frame_bytes = score_result_bytes.value();
    resources->status_frame_bytes = sizeof(std::uint32_t);

    const bool requires_score_resources = !guarantees_singular_covariance(resources->dimensions);
    const bool skip_suffix_buffers = skip_streaming_suffix && requires_score_resources;

    const auto create_into = [&](std::unique_ptr<Buffer> &destination, VkDeviceSize size,
                                 VkBufferUsageFlags additional_usage = 0U,
                                 bool host_visible = false) -> Result<void> {
        auto buffer = resources_create_storage_buffer(device, size, additional_usage, host_visible);
        if (!buffer) return Result<void>::failure(buffer.error());
        destination = std::move(buffer).value();
        return Result<void>::success();
    };
    if (!resources->identity_scale_fused) {
        auto scaled_buffer = resources_create_storage_buffer(device, scaled_workspace_bytes,
                                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        if (!scaled_buffer) {
            return Result<std::unique_ptr<Resources>>::failure(scaled_buffer.error());
        }
        resources->scaled = std::move(scaled_buffer).value();
    }
    const bool use_identity_products = resources->identity_scale_fused &&
                                       descriptor.bit_depth == 8U &&
                                       limits.maxPerStageDescriptorStorageBuffers >= 7U &&
                                       limits.maxDescriptorSetStorageBuffers >= 7U;
    if (use_identity_products) {
        const auto created = create_into(resources->antialias_products,
                                         sizeof(identity_antialias_products), 0U, true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
        const auto written = resources->antialias_products->write(
            std::as_bytes(std::span(identity_antialias_products)));
        if (!written) {
            return Result<std::unique_ptr<Resources>>::failure(written.error());
        }
    }
    for (const auto [destination, size] :
         std::array<std::pair<std::unique_ptr<Buffer> *, VkDeviceSize>, 2>{
             std::pair{&resources->decimated, operating_q32_bytes.value()},
             std::pair{&resources->local_vertical, local_q32_bytes.value()}}) {
        const auto created = create_into(*destination, size);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
    }
    if (!skip_suffix_buffers) {
        auto created = create_into(resources->highpass, highpass_q32_bytes.value(),
                                   VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
        created =
            create_into(resources->status, static_cast<VkDeviceSize>(sizeof(std::uint32_t)),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
    }

    if (requires_score_resources && !skip_suffix_buffers) {
        auto created =
            create_into(resources->score_scratch, score_scratch_bytes.value(),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
        created = create_into(resources->diagnostics_snapshot, diagnostics_snapshot_bytes.value(),
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
        created =
            create_into(resources->score_result, score_result_bytes.value(),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, true);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
    }

    if (!resources->identity_scale_fused) {
        auto scale = ComputePipeline::create(device, gpu::generated::speed_chroma_scale_spv, 6U,
                                             sizeof(ScaleParameters));
        if (!scale) {
            return Result<std::unique_ptr<Resources>>::failure(scale.error());
        }
        resources->scale_pipeline = std::move(scale).value();
    }

    auto vertical =
        ComputePipeline::create(device, gpu::generated::speed_chroma_filter_vertical_spv, 2U,
                                sizeof(VerticalParameters), 2U);
    if (!vertical) {
        return Result<std::unique_ptr<Resources>>::failure(vertical.error());
    }
    resources->vertical_pipeline = std::move(vertical).value();
    auto updated = resources->vertical_pipeline->update_buffers(
        1U, std::array{BufferBinding{resources->decimated.get()},
                       BufferBinding{resources->local_vertical.get()}});
    if (!updated) {
        return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    std::span<const std::uint8_t> antialias_spirv =
        resources->identity_scale_fused
            ? std::span<const std::uint8_t>(gpu::generated::speed_chroma_antialias_identity_spv)
            : std::span<const std::uint8_t>(gpu::generated::speed_chroma_antialias_fused_spv);
    std::uint32_t antialias_binding_count = resources->identity_scale_fused ? 6U : 2U;
    if (use_identity_products) {
        antialias_spirv = gpu::generated::speed_chroma_antialias_identity_products_spv;
        antialias_binding_count = 7U;
    }
    auto antialias_fused = ComputePipeline::create(device, antialias_spirv, antialias_binding_count,
                                                   sizeof(AntialiasParameters));
    if (!antialias_fused) {
        return Result<std::unique_ptr<Resources>>::failure(antialias_fused.error());
    }
    resources->antialias_fused_pipeline = std::move(antialias_fused).value();
    if (!resources->identity_scale_fused) {
        updated = resources->antialias_fused_pipeline->update_buffers(
            0U,
            std::array{BufferBinding{resources->scaled.get(), resources->scaled_sample_offset_bytes,
                                     resources->scaled_sample_bytes},
                       BufferBinding{resources->decimated.get()}});
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
    }

    auto local = ComputePipeline::create(device, gpu::generated::speed_chroma_local_horizontal_spv,
                                         3U, sizeof(LocalParameters));
    if (!local) return Result<std::unique_ptr<Resources>>::failure(local.error());
    resources->local_pipeline = std::move(local).value();
    if (resources->highpass != nullptr) {
        updated = resources->local_pipeline->update_buffers(
            0U, std::array{BufferBinding{resources->decimated.get()},
                           BufferBinding{resources->local_vertical.get()},
                           BufferBinding{resources->highpass.get()}});
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
    }

    if (requires_score_resources) {
        const auto create_unbound_score_pipeline =
            [&]<typename Spirv>(std::unique_ptr<ComputePipeline> &destination,
                                const Spirv &spirv) -> Result<void> {
            auto pipeline = ComputePipeline::create(device, spirv, 4U, sizeof(ScoreParameters));
            if (!pipeline) return Result<void>::failure(pipeline.error());
            destination = std::move(pipeline).value();
            return Result<void>::success();
        };

        Result<void> created =
            create_unbound_score_pipeline(resources->means_pipeline,
                                          gpu::generated::speed_chroma_means_spv);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
        created = create_unbound_score_pipeline(resources->covariance_pipeline,
                                                gpu::generated::speed_chroma_covariance_spv);
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }

        if (!skip_suffix_buffers) {
            const std::array score_bindings{BufferBinding{resources->highpass.get()},
                                            BufferBinding{resources->score_scratch.get()},
                                            BufferBinding{resources->score_result.get()},
                                            BufferBinding{resources->status.get()}};
            const auto bind_score_pipeline = [&](ComputePipeline &pipeline) -> Result<void> {
                return pipeline.update_buffers(0U, score_bindings);
            };

            created = bind_score_pipeline(*resources->means_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = bind_score_pipeline(*resources->covariance_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created =
                create_unbound_score_pipeline(resources->compact_tridiagonal_pipeline,
                                              gpu::generated::speed_chroma_compact_tridiagonal_spv);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = bind_score_pipeline(*resources->compact_tridiagonal_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created =
                create_unbound_score_pipeline(resources->exact_regularity_pipeline,
                                              gpu::generated::speed_chroma_exact_regularity_spv);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = bind_score_pipeline(*resources->exact_regularity_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created =
                create_unbound_score_pipeline(resources->compact_blocks_pipeline,
                                              gpu::generated::speed_chroma_compact_blocks_spv);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = bind_score_pipeline(*resources->compact_blocks_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = create_unbound_score_pipeline(
                resources->block_statistics_scalar_pipeline,
                gpu::generated::speed_chroma_block_statistics_scalar_spv);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = bind_score_pipeline(*resources->block_statistics_scalar_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = create_unbound_score_pipeline(resources->finalize_pipeline,
                                                    gpu::generated::speed_chroma_finalize_spv);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
            created = bind_score_pipeline(*resources->finalize_pipeline);
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
        }
    }

    auto commands = CommandContext::create(device);
    if (!commands) {
        return Result<std::unique_ptr<Resources>>::failure(commands.error());
    }
    resources->commands = std::move(commands).value();

    if (!scale_plan.axis_samples.empty()) {
        const auto scale_plan_uploaded = resources_upload_scale_plan(*resources, scale_plan);
        if (!scale_plan_uploaded) {
            return Result<std::unique_ptr<Resources>>::failure(scale_plan_uploaded.error());
        }
    }

    resources_configure_gpu_execution_report(*resources);
    const auto timestamps_initialized = resources_initialize_gpu_timestamp_queries(*resources);
    if (!timestamps_initialized) {
        return Result<std::unique_ptr<Resources>>::failure(timestamps_initialized.error());
    }
    return Result<std::unique_ptr<Resources>>::success(std::move(resources));
}

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
namespace {

using gpu::Buffer;
using gpu::BufferBinding;
using gpu::BufferCreateInfo;
using gpu::BufferMemoryBarrier;
using gpu::calculate_timestamp_duration;
using gpu::CommandContext;
using gpu::CommandRecordingGuard;
using gpu::ComputePipeline;
using gpu::DeviceView;
using gpu::DispatchSize;
using gpu::TimestampQueryPool;

using SpeedChromaGpuPhase = GpuStage;

[[nodiscard]] bool vulkan_trace_timings() noexcept {
    static const bool enabled = [] {
#if defined(_WIN32)
        char *value = nullptr;
        std::size_t length = 0U;
        const int status = _dupenv_s(&value, &length, "CROSSVMAF_VULKAN_TRACE");
        const bool result = status == 0 && value != nullptr && std::string_view(value) != "0";
        std::free(value);
        return result;
#else
        const char *const value = std::getenv("CROSSVMAF_VULKAN_TRACE");
        return value != nullptr && std::string_view(value) != "0";
#endif
    }();
    return enabled;
}

void vulkan_trace_elapsed(const char *label, std::uint32_t phase,
                          std::chrono::steady_clock::time_point start) noexcept {
    if (!vulkan_trace_timings()) return;
    const double milliseconds =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::fprintf(stderr, "[crossvmaf-vulkan] speed_chroma.%s phase=%u %.3f ms\n", label, phase,
                 milliseconds);
}

template <typename T>
[[nodiscard]] std::span<const std::byte> vulkan_bytes_of(const T &value) noexcept {
    return {reinterpret_cast<const std::byte *>(&value), sizeof(T)};
}

[[nodiscard]] Result<void> vulkan_validate_chroma_plane(const video::GpuPlaneView &plane,
                                                        const FrameDescriptor &descriptor) {
    const std::uint32_t width = plane_width(descriptor, 1U);
    const std::uint32_t height = plane_height(descriptor, 1U);
    if (!video::is_compatible_gpu_plane(plane, width, height, descriptor.bit_depth)) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "Speed Chroma received an incompatible GPU chroma plane"});
    }
    return Result<void>::success();
}

[[nodiscard]] numeric::Q32_32 vulkan_decode_q32(Q32Words words) noexcept {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(words.low) | (static_cast<std::uint64_t>(words.high) << 32U);
    return numeric::Q32_32::from_raw(std::bit_cast<std::int64_t>(bits));
}

[[nodiscard]] Q32Words vulkan_encode_q32(numeric::Q32_32 value) noexcept {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value.raw());
    return {
        static_cast<std::uint32_t>(bits),
        static_cast<std::uint32_t>(bits >> 32U),
    };
}

[[nodiscard]] std::int64_t vulkan_decode_i64(Q32Words words) noexcept {
    return vulkan_decode_q32(words).raw();
}

struct vulkan_GpuTimestampPhaseQueryPair final {
    SpeedChromaGpuPhase phase;
    GpuTimestampQuerySlot start;
    GpuTimestampQuerySlot end;
};

inline constexpr std::array vulkan_gpu_timestamp_phase_query_pairs{
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::scale,
                                      GpuTimestampQuerySlot::scale_start,
                                      GpuTimestampQuerySlot::scale_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::antialias_fused,
                                      GpuTimestampQuerySlot::antialias_fused_start,
                                      GpuTimestampQuerySlot::antialias_fused_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::local_vertical,
                                      GpuTimestampQuerySlot::local_vertical_start,
                                      GpuTimestampQuerySlot::local_vertical_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::local_horizontal_highpass,
                                      GpuTimestampQuerySlot::local_horizontal_highpass_start,
                                      GpuTimestampQuerySlot::local_horizontal_highpass_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::means,
                                      GpuTimestampQuerySlot::means_start,
                                      GpuTimestampQuerySlot::means_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::covariance,
                                      GpuTimestampQuerySlot::covariance_start,
                                      GpuTimestampQuerySlot::covariance_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::compact_tridiagonal,
                                      GpuTimestampQuerySlot::compact_tridiagonal_start,
                                      GpuTimestampQuerySlot::compact_tridiagonal_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::compact_regularity,
                                      GpuTimestampQuerySlot::compact_regularity_start,
                                      GpuTimestampQuerySlot::compact_regularity_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::block_statistics,
                                      GpuTimestampQuerySlot::block_statistics_start,
                                      GpuTimestampQuerySlot::block_statistics_end},
    vulkan_GpuTimestampPhaseQueryPair{SpeedChromaGpuPhase::finalize,
                                      GpuTimestampQuerySlot::finalize_start,
                                      GpuTimestampQuerySlot::finalize_end},
};

static_assert(gpu_timestamp_query_count == vulkan_gpu_timestamp_phase_query_pairs.size() * 2U);
static_assert(vulkan_gpu_timestamp_phase_query_pairs.size() ==
              static_cast<std::size_t>(SpeedChromaGpuPhase::count));

[[nodiscard]] constexpr std::uint32_t
vulkan_timestamp_query_index(GpuTimestampQuerySlot slot) noexcept {
    return static_cast<std::uint32_t>(slot);
}

[[nodiscard]] constexpr const vulkan_GpuTimestampPhaseQueryPair &
vulkan_timestamp_queries_for_phase(SpeedChromaGpuPhase phase) noexcept {
    return vulkan_gpu_timestamp_phase_query_pairs[static_cast<std::size_t>(phase)];
}

[[nodiscard]] constexpr bool vulkan_is_preprocess_gpu_phase(SpeedChromaGpuPhase phase) noexcept {
    return phase <= SpeedChromaGpuPhase::local_horizontal_highpass;
}

class vulkan_SpeedChromaGpuTimestampRecording final {
  public:
    vulkan_SpeedChromaGpuTimestampRecording(TimestampQueryPool *query_pool,
                                            VkCommandBuffer command_buffer,
                                            bool recording_valid = false) noexcept
        : query_pool_(query_pool), command_buffer_(command_buffer),
          recording_valid_(recording_valid) {}

    void reset() {
        if (query_pool_ == nullptr) return;
        const auto reset_result =
            query_pool_->reset_queries(command_buffer_, 0U, gpu_timestamp_query_count);
        recording_valid_ = static_cast<bool>(reset_result);
    }

    void write_phase_start(SpeedChromaGpuPhase phase) {
        write_timestamp(vulkan_timestamp_queries_for_phase(phase).start);
    }

    void write_phase_end(SpeedChromaGpuPhase phase) {
        write_timestamp(vulkan_timestamp_queries_for_phase(phase).end);
    }

    [[nodiscard]] bool query_pool_available() const noexcept { return query_pool_ != nullptr; }

    [[nodiscard]] bool recording_valid() const noexcept { return recording_valid_; }

  private:
    void write_timestamp(GpuTimestampQuerySlot slot) {
        if (!recording_valid_) return;
        const auto write_result =
            query_pool_->write_timestamp(command_buffer_, vulkan_timestamp_query_index(slot),
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        // Profiling failure invalidates this frame without interrupting the
        // Speed Chroma command stream.
        if (!write_result) recording_valid_ = false;
    }

    TimestampQueryPool *query_pool_{};
    VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
    bool recording_valid_{};
};

void vulkan_record_guaranteed_singular_gpu_phase_skips(Telemetry &telemetry) noexcept {
    for (const vulkan_GpuTimestampPhaseQueryPair &phase_queries :
         vulkan_gpu_timestamp_phase_query_pairs) {
        if (vulkan_is_preprocess_gpu_phase(phase_queries.phase)) continue;
        telemetry.record_stage_skip(phase_queries.phase);
    }
    telemetry.record_compact_factor_skip();
}

void vulkan_record_gpu_timestamp_frame_failure(Resources &resources,
                                               bool guaranteed_singular) noexcept {
    Telemetry &telemetry = *resources.telemetry;
    telemetry.record_parent_failure();
    for (const vulkan_GpuTimestampPhaseQueryPair &phase_queries :
         vulkan_gpu_timestamp_phase_query_pairs) {
        const bool phase_executed =
            (!guaranteed_singular || vulkan_is_preprocess_gpu_phase(phase_queries.phase)) &&
            !(resources.identity_scale_fused && phase_queries.phase == SpeedChromaGpuPhase::scale);
        if (phase_executed) {
            telemetry.record_stage_failure(phase_queries.phase);
        }
    }
    if (!guaranteed_singular) {
        telemetry.record_compact_factor_failure();
    }
}

[[nodiscard]] Result<void>
vulkan_collect_gpu_timestamp_results(Resources &resources, bool query_pool_available,
                                     bool recording_valid, bool guaranteed_singular,
                                     long double maximum_elapsed_nanoseconds) {
    Telemetry *const telemetry = resources.telemetry;
    if (telemetry == nullptr || !telemetry->enabled()) {
        return Result<void>::success();
    }

    telemetry->record_submission(query_pool_available);

    if (resources.identity_scale_fused) {
        telemetry->record_stage_skip(SpeedChromaGpuPhase::scale);
    }
    if (guaranteed_singular) {
        vulkan_record_guaranteed_singular_gpu_phase_skips(*telemetry);
    }
    if (!query_pool_available) {
        return Result<void>::success();
    }
    if (!recording_valid) {
        vulkan_record_gpu_timestamp_frame_failure(resources, guaranteed_singular);
        return Result<void>::success();
    }

    if (resources.gpu_timestamp_query_pool == nullptr) {
        vulkan_record_gpu_timestamp_frame_failure(resources, guaranteed_singular);
        return Result<void>::success();
    }
    TimestampQueryPool &query_pool = *resources.gpu_timestamp_query_pool;
    const vulkan_GpuTimestampPhaseQueryPair &parent_start_phase =
        vulkan_gpu_timestamp_phase_query_pairs.front();
    const vulkan_GpuTimestampPhaseQueryPair &parent_end_phase =
        guaranteed_singular
            ? vulkan_timestamp_queries_for_phase(SpeedChromaGpuPhase::local_horizontal_highpass)
            : vulkan_gpu_timestamp_phase_query_pairs.back();
    const std::uint32_t query_result_count =
        vulkan_timestamp_query_index(parent_end_phase.end) + 1U;
    const auto query_results_result = query_pool.read_results(0U, query_result_count);
    if (!query_results_result) {
        const Error &read_error = query_results_result.error();
        vulkan_record_gpu_timestamp_frame_failure(resources, guaranteed_singular);
        if (read_error.code == ErrorCode::device_lost) {
            return Result<void>::failure(read_error);
        }
        return Result<void>::success();
    }

    const auto &query_results = query_results_result.value();
    if (query_results.size() != static_cast<std::size_t>(query_result_count)) {
        vulkan_record_gpu_timestamp_frame_failure(resources, guaranteed_singular);
        return Result<void>::success();
    }

    const std::optional<long double> duration_upper_bound{maximum_elapsed_nanoseconds};
    const auto parent_duration = calculate_timestamp_duration(
        query_results[vulkan_timestamp_query_index(parent_start_phase.start)],
        query_results[vulkan_timestamp_query_index(parent_end_phase.end)],
        query_pool.capabilities(), duration_upper_bound);
    if (parent_duration) {
        telemetry->record_parent_success(parent_duration.value().nanoseconds);
    } else {
        telemetry->record_parent_failure();
    }
    if (parent_duration) {
        telemetry->record_timeline(
            query_results[vulkan_timestamp_query_index(parent_start_phase.start)].value,
            query_results[vulkan_timestamp_query_index(parent_end_phase.end)].value,
            query_pool.capabilities().timestamp_valid_bits,
            query_pool.capabilities().timestamp_period_nanoseconds,
            crossvmaf::gpu::TimelineSegment::full);
    }

    std::optional<long double> compact_tridiagonal_nanoseconds;
    std::optional<long double> compact_regularity_nanoseconds;
    for (const vulkan_GpuTimestampPhaseQueryPair &phase_queries :
         vulkan_gpu_timestamp_phase_query_pairs) {
        if (resources.identity_scale_fused && phase_queries.phase == SpeedChromaGpuPhase::scale) {
            continue;
        }
        if (guaranteed_singular && !vulkan_is_preprocess_gpu_phase(phase_queries.phase)) {
            continue;
        }
        const auto phase_duration = calculate_timestamp_duration(
            query_results[vulkan_timestamp_query_index(phase_queries.start)],
            query_results[vulkan_timestamp_query_index(phase_queries.end)],
            query_pool.capabilities(), duration_upper_bound);
        if (phase_duration) {
            const long double phase_nanoseconds = phase_duration.value().nanoseconds;
            telemetry->record_stage_success(phase_queries.phase, phase_nanoseconds);
            if (phase_queries.phase == SpeedChromaGpuPhase::compact_tridiagonal) {
                compact_tridiagonal_nanoseconds = phase_nanoseconds;
            } else if (phase_queries.phase == SpeedChromaGpuPhase::compact_regularity) {
                compact_regularity_nanoseconds = phase_nanoseconds;
            }
        } else {
            telemetry->record_stage_failure(phase_queries.phase);
        }
    }
    if (!guaranteed_singular) {
        if (compact_tridiagonal_nanoseconds.has_value() &&
            compact_regularity_nanoseconds.has_value()) {
            telemetry->record_compact_factor_success(*compact_tridiagonal_nanoseconds +
                                                     *compact_regularity_nanoseconds);
        } else {
            telemetry->record_compact_factor_failure();
        }
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void> vulkan_collect_stream_prefix_gpu_timestamp_results(
    Resources &resources, bool query_pool_available, bool recording_valid,
    std::chrono::nanoseconds submission_host_elapsed) {
    Telemetry *const telemetry = resources.telemetry;
    if (telemetry == nullptr || !telemetry->enabled()) {
        return Result<void>::success();
    }
    telemetry->record_submission(query_pool_available);
    if (resources.identity_scale_fused) {
        telemetry->record_stage_skip(SpeedChromaGpuPhase::scale);
    }
    if (!query_pool_available) return Result<void>::success();

    constexpr std::size_t prefix_phase_count = 6U;
    const std::size_t first_executed_phase = resources.identity_scale_fused ? 1U : 0U;
    if (!recording_valid || resources.gpu_timestamp_query_pool == nullptr) {
        for (std::size_t phase_index = first_executed_phase; phase_index < prefix_phase_count;
             ++phase_index) {
            telemetry->record_stage_failure(
                vulkan_gpu_timestamp_phase_query_pairs[phase_index].phase);
        }
        return Result<void>::success();
    }

    const std::uint32_t query_result_count =
        vulkan_timestamp_query_index(
            vulkan_gpu_timestamp_phase_query_pairs[prefix_phase_count - 1U].end) +
        1U;
    const auto query_results =
        resources.gpu_timestamp_query_pool->read_results(0U, query_result_count);
    if (!query_results || query_results.value().size() != query_result_count) {
        for (std::size_t phase_index = first_executed_phase; phase_index < prefix_phase_count;
             ++phase_index) {
            telemetry->record_stage_failure(
                vulkan_gpu_timestamp_phase_query_pairs[phase_index].phase);
        }
        return Result<void>::success();
    }

    const long double duration_upper_bound =
        std::chrono::duration<long double, std::nano>(submission_host_elapsed).count();
    for (std::size_t phase_index = first_executed_phase; phase_index < prefix_phase_count;
         ++phase_index) {
        const vulkan_GpuTimestampPhaseQueryPair &phase_queries =
            vulkan_gpu_timestamp_phase_query_pairs[phase_index];
        const auto duration = calculate_timestamp_duration(
            query_results.value()[vulkan_timestamp_query_index(phase_queries.start)],
            query_results.value()[vulkan_timestamp_query_index(phase_queries.end)],
            resources.gpu_timestamp_query_pool->capabilities(), duration_upper_bound);
        if (duration) {
            telemetry->record_stage_success(phase_queries.phase, duration.value().nanoseconds);
        } else {
            telemetry->record_stage_failure(phase_queries.phase);
        }
    }
    const auto parent_duration = calculate_timestamp_duration(
        query_results.value()[vulkan_timestamp_query_index(
            vulkan_gpu_timestamp_phase_query_pairs.front().start)],
        query_results.value()[vulkan_timestamp_query_index(
            vulkan_gpu_timestamp_phase_query_pairs[prefix_phase_count - 1U].end)],
        resources.gpu_timestamp_query_pool->capabilities(), duration_upper_bound);
    if (parent_duration) {
        telemetry->record_timeline(
            query_results
                .value()[vulkan_timestamp_query_index(
                    vulkan_gpu_timestamp_phase_query_pairs.front().start)]
                .value,
            query_results
                .value()[vulkan_timestamp_query_index(
                    vulkan_gpu_timestamp_phase_query_pairs[prefix_phase_count - 1U].end)]
                .value,
            resources.gpu_timestamp_query_pool->capabilities().timestamp_valid_bits,
            resources.gpu_timestamp_query_pool->capabilities().timestamp_period_nanoseconds,
            crossvmaf::gpu::TimelineSegment::prefix);
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void>
vulkan_configure_preprocess_bindings(Resources &resources, std::array<BufferBinding, 6> bindings) {
    if (resources.identity_scale_fused) {
        bindings[4] = BufferBinding{resources.decimated.get()};
        if (resources.antialias_products != nullptr) {
            const std::array product_bindings{bindings[0],
                                              bindings[1],
                                              bindings[2],
                                              bindings[3],
                                              bindings[4],
                                              bindings[5],
                                              BufferBinding{resources.antialias_products.get()}};
            return resources.antialias_fused_pipeline->update_buffers(0U, product_bindings);
        }
        return resources.antialias_fused_pipeline->update_buffers(0U, bindings);
    }
    return resources.scale_pipeline->update_buffers(0U, bindings);
}

Result<void> vulkan_preprocess(Resources &resources, const video::GpuPlaneView &plane,
                               Buffer &destination) {
    const auto trace_start = std::chrono::steady_clock::now();
    const std::array scale_bindings{
        BufferBinding{plane.buffer, plane.offset_bytes, plane.size_bytes},
        BufferBinding{plane.buffer, plane.offset_bytes, plane.size_bytes},
        BufferBinding{plane.buffer, plane.offset_bytes, plane.size_bytes},
        BufferBinding{plane.buffer, plane.offset_bytes, plane.size_bytes},
        BufferBinding{resources.scaled.get()},
        BufferBinding{resources.status.get()}};
    auto updated = vulkan_configure_preprocess_bindings(resources, scale_bindings);
    if (!updated) {
        return Result<void>::failure(updated.error());
    }
    updated = resources.local_pipeline->update_buffers(
        0U, std::array{BufferBinding{resources.decimated.get()},
                       BufferBinding{resources.local_vertical.get()}, BufferBinding{&destination}});
    if (!updated) {
        return Result<void>::failure(updated.error());
    }
    const auto cleared = resources.status->fill(std::byte{0});
    if (!cleared) {
        return Result<void>::failure(cleared.error());
    }
    auto begun = resources.commands->begin();
    if (!begun) {
        return Result<void>::failure(begun.error());
    }
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources.commands);
    resources.commands->memory_barrier(
        VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    const ScaleParameters scale_parameters{
        resources.dimensions.original_width,
        resources.dimensions.original_height,
        plane.stride_bytes,
        plane.stride_bytes,
        plane.stride_bytes,
        plane.stride_bytes,
        resources.dimensions.scaled_width,
        resources.dimensions.scaled_height,
        resources.descriptor.bit_depth,
        resources.config.prescale_method == PrescaleMethod::bilinear ? 1U : 0U,
        resources.scaled_sample_offset_elements,
    };
    if (!resources.identity_scale_fused) {
        const auto recorded = resources.scale_pipeline->record(
            command_buffer, 0U, vulkan_bytes_of(scale_parameters), resources.source_dispatch);
        if (!recorded) return Result<void>::failure(recorded.error());
        resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                           VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    const AntialiasParameters antialias_parameters{
        resources.dimensions.scaled_width,
        resources.dimensions.scaled_height,
        resources.dimensions.operating_width,
        resources.dimensions.operating_height,
        plane.stride_bytes,
        plane.stride_bytes,
        plane.stride_bytes,
        plane.stride_bytes,
        resources.descriptor.bit_depth,
    };
    auto recorded = resources.antialias_fused_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(antialias_parameters), resources.antialias_dispatch);
    if (!recorded) {
        return Result<void>::failure(recorded.error());
    }
    resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    const VerticalParameters local_vertical_parameters{resources.dimensions.operating_width,
                                                       resources.dimensions.operating_height, 1U};
    recorded = resources.vertical_pipeline->record(command_buffer, 1U,
                                                   vulkan_bytes_of(local_vertical_parameters),
                                                   resources.operating_dispatch);
    if (!recorded) {
        return Result<void>::failure(recorded.error());
    }
    resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    const LocalParameters local_parameters{resources.dimensions.operating_width,
                                           resources.dimensions.operating_height};
    recorded = resources.local_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(local_parameters), resources.operating_dispatch);
    if (!recorded) {
        return Result<void>::failure(recorded.error());
    }
    resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_ACCESS_HOST_READ_BIT);
    const auto submitted = resources.commands->submit_and_wait();
    recording_guard.release();
    if (!submitted) {
        return Result<void>::failure(submitted.error());
    }
    vulkan_trace_elapsed("vulkan_preprocess", 0U, trace_start);

    std::uint32_t flags{};
    auto read = resources.status->read(std::as_writable_bytes(std::span(&flags, 1U)));
    if (!read) return Result<void>::failure(read.error());
    if ((flags & 1U) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma sample exceeds the declared bit depth"});
    }
    return Result<void>::success();
}

void vulkan_record_compute_barrier(CommandContext &commands) noexcept {
    commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

void vulkan_record_compute_barrier(Resources &resources) noexcept {
    vulkan_record_compute_barrier(*resources.commands);
}

void vulkan_record_score_buffer_compute_barriers(Resources &resources,
                                                 CommandContext &commands) noexcept {
    const std::array score_buffer_barriers{
        BufferMemoryBarrier{resources.score_scratch.get(), VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT},
        BufferMemoryBarrier{resources.score_result.get(), VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT},
    };
    commands.buffer_memory_barriers(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, score_buffer_barriers);
}

inline constexpr std::int64_t vulkan_canonical_sigma_nn_raw = 816'043'776LL;
inline constexpr std::int64_t vulkan_canonical_nn_floor_raw = 429'496'736LL;
inline constexpr std::int64_t vulkan_log2_two_pi_e_raw = 17'584'417'180LL;

[[nodiscard]] constexpr std::uint64_t vulkan_q32_magnitude(std::int64_t value) noexcept {
    const auto bits = static_cast<std::uint64_t>(value);
    return value < 0 ? (~bits + 1U) : bits;
}

[[nodiscard]] numeric::Q32_32 vulkan_quantize_binary32_compatible(numeric::Q32_32 value) noexcept {
    if (value.raw() == 0) return value;
    const bool negative = value.raw() < 0;
    const std::uint64_t absolute = vulkan_q32_magnitude(value.raw());
    const unsigned highest_bit = 63U - std::countl_zero(absolute);
    if (highest_bit <= 23U) return value;
    const unsigned shift = highest_bit - 23U;
    std::uint64_t rounded = absolute >> shift;
    const std::uint64_t mask = (std::uint64_t{1} << shift) - 1U;
    const std::uint64_t remainder = absolute & mask;
    const std::uint64_t halfway = std::uint64_t{1} << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U)) {
        ++rounded;
    }
    if (rounded > (static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) >> shift)) {
        return numeric::Q32_32::from_raw(negative ? std::numeric_limits<std::int64_t>::min()
                                                  : std::numeric_limits<std::int64_t>::max());
    }
    const std::uint64_t raw = rounded << shift;
    if (!negative) {
        return numeric::Q32_32::from_raw(static_cast<std::int64_t>(raw));
    }
    const std::uint64_t negative_limit =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1U;
    return numeric::Q32_32::from_raw(raw >= negative_limit
                                         ? std::numeric_limits<std::int64_t>::min()
                                         : -static_cast<std::int64_t>(raw));
}

[[nodiscard]] numeric::Q32_32 vulkan_binary32_compatible_add(numeric::Q32_32 lhs,
                                                             numeric::Q32_32 rhs) noexcept {
    return vulkan_quantize_binary32_compatible(numeric::Q32_32::add_saturating(lhs, rhs));
}

[[nodiscard]] Result<numeric::Q32_32> vulkan_make_guaranteed_singular_entropy() noexcept {
    const numeric::Q32_32 rounded_sigma_nn = vulkan_binary32_compatible_add(
        numeric::Q32_32{}, numeric::Q32_32::from_raw(vulkan_canonical_sigma_nn_raw));
    const auto logarithm = numeric::Q32_32::log2(rounded_sigma_nn);
    if (!logarithm) {
        return Result<numeric::Q32_32>::failure(logarithm.error());
    }
    const numeric::Q32_32 entropy_term = numeric::Q32_32::add_saturating(
        logarithm.value(), numeric::Q32_32::from_raw(vulkan_log2_two_pi_e_raw));
    numeric::Q32_32 entropy{};
    for (std::uint32_t element_index = 0U; element_index < elements_per_block; ++element_index) {
        entropy = vulkan_binary32_compatible_add(entropy, entropy_term);
    }
    return Result<numeric::Q32_32>::success(entropy);
}

[[nodiscard]] std::uint64_t
vulkan_score_scratch_stream_elements(const Resources &resources) noexcept {
    return score_scratch_base_elements +
           score_scratch_elements_per_block * resources.dimensions.block_count;
}

void vulkan_record_diagnostic_copy(Resources &resources, std::uint32_t parameter_stream_index,
                                   std::uint64_t source_element_offset,
                                   std::uint64_t destination_element_offset,
                                   std::uint64_t element_count) noexcept {
    resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_ACCESS_TRANSFER_READ_BIT);
    resources.commands->copy_buffer(*resources.score_scratch, *resources.diagnostics_snapshot,
                                    (static_cast<std::uint64_t>(parameter_stream_index) *
                                         vulkan_score_scratch_stream_elements(resources) +
                                     source_element_offset) *
                                        sizeof(Q32Words),
                                    destination_element_offset * sizeof(Q32Words),
                                    element_count * sizeof(Q32Words));
    resources.commands->memory_barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

[[nodiscard]] Result<void>
vulkan_configure_frame_descriptors(Resources &resources, const video::GpuFrameView &reference,
                                   const video::GpuFrameView &distorted) {
    return vulkan_configure_preprocess_bindings(
        resources,
        std::array{BufferBinding{reference.planes[1].buffer, reference.planes[1].offset_bytes,
                                 reference.planes[1].size_bytes},
                   BufferBinding{distorted.planes[1].buffer, distorted.planes[1].offset_bytes,
                                 distorted.planes[1].size_bytes},
                   BufferBinding{reference.planes[2].buffer, reference.planes[2].offset_bytes,
                                 reference.planes[2].size_bytes},
                   BufferBinding{distorted.planes[2].buffer, distorted.planes[2].offset_bytes,
                                 distorted.planes[2].size_bytes},
                   BufferBinding{resources.scaled.get()}, BufferBinding{resources.status.get()}});
}

[[nodiscard]] Result<void> vulkan_configure_stream_prefix_descriptors(
    Resources &resources, const video::GpuFrameView &reference,
    const video::GpuFrameView &distorted, const TaskRingView &task_ring_view) {
    const auto scale_updated = vulkan_configure_preprocess_bindings(
        resources,
        std::array{BufferBinding{reference.planes[1].buffer, reference.planes[1].offset_bytes,
                                 reference.planes[1].size_bytes},
                   BufferBinding{distorted.planes[1].buffer, distorted.planes[1].offset_bytes,
                                 distorted.planes[1].size_bytes},
                   BufferBinding{reference.planes[2].buffer, reference.planes[2].offset_bytes,
                                 reference.planes[2].size_bytes},
                   BufferBinding{distorted.planes[2].buffer, distorted.planes[2].offset_bytes,
                                 distorted.planes[2].size_bytes},
                   BufferBinding{resources.scaled.get()},
                   BufferBinding{task_ring_view.status, task_ring_view.status_offset,
                                 task_ring_view.status_range}});
    if (!scale_updated) return scale_updated;

    const auto local_updated = resources.local_pipeline->update_buffers(
        0U, std::array{BufferBinding{resources.decimated.get()},
                       BufferBinding{resources.local_vertical.get()},
                       BufferBinding{task_ring_view.highpass, task_ring_view.highpass_offset,
                                     task_ring_view.highpass_range}});
    if (!local_updated) return local_updated;

    const std::array score_bindings{
        BufferBinding{task_ring_view.highpass, task_ring_view.highpass_offset,
                      task_ring_view.highpass_range},
        BufferBinding{task_ring_view.score_scratch, task_ring_view.scratch_offset,
                      task_ring_view.scratch_range},
        BufferBinding{task_ring_view.score_result, task_ring_view.result_offset,
                      task_ring_view.result_range},
        BufferBinding{task_ring_view.status, task_ring_view.status_offset,
                      task_ring_view.status_range}};
    auto score_updated = resources.means_pipeline->update_buffers(0U, score_bindings);
    if (!score_updated) return score_updated;
    return resources.covariance_pipeline->update_buffers(0U, score_bindings);
}

[[nodiscard]] Result<void> vulkan_record_preprocess(
    Resources &resources, VkCommandBuffer command_buffer,
    const std::array<std::uint32_t, parameter_stream_count> &source_stride_bytes,
    vulkan_SpeedChromaGpuTimestampRecording &timestamp_recording) {
    const ScaleParameters scale_parameters{
        resources.dimensions.original_width,
        resources.dimensions.original_height,
        source_stride_bytes[0],
        source_stride_bytes[1],
        source_stride_bytes[2],
        source_stride_bytes[3],
        resources.dimensions.scaled_width,
        resources.dimensions.scaled_height,
        resources.descriptor.bit_depth,
        resources.config.prescale_method == PrescaleMethod::bilinear ? 1U : 0U,
        resources.scaled_sample_offset_elements,
    };
    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::scale);
    if (!resources.identity_scale_fused) {
        const auto recorded = resources.scale_pipeline->record(
            command_buffer, 0U, vulkan_bytes_of(scale_parameters), resources.source_dispatch);
        if (!recorded) return Result<void>::failure(recorded.error());
    }
    // Keep the query layout contiguous, but report the absent scale as skipped.
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::scale);
    if (!resources.identity_scale_fused) vulkan_record_compute_barrier(resources);

    const AntialiasParameters antialias_parameters{
        resources.dimensions.scaled_width,
        resources.dimensions.scaled_height,
        resources.dimensions.operating_width,
        resources.dimensions.operating_height,
        source_stride_bytes[0],
        source_stride_bytes[1],
        source_stride_bytes[2],
        source_stride_bytes[3],
        resources.descriptor.bit_depth,
    };
    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::antialias_fused);
    auto recorded = resources.antialias_fused_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(antialias_parameters), resources.antialias_dispatch);
    if (!recorded) return Result<void>::failure(recorded.error());
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::antialias_fused);
    vulkan_record_compute_barrier(resources);

    const VerticalParameters local_vertical_parameters{resources.dimensions.operating_width,
                                                       resources.dimensions.operating_height, 1U};
    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::local_vertical);
    recorded = resources.vertical_pipeline->record(command_buffer, 1U,
                                                   vulkan_bytes_of(local_vertical_parameters),
                                                   resources.operating_dispatch);
    if (!recorded) return Result<void>::failure(recorded.error());
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::local_vertical);
    vulkan_record_compute_barrier(resources);

    const LocalParameters local_parameters{resources.dimensions.operating_width,
                                           resources.dimensions.operating_height};
    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::local_horizontal_highpass);
    recorded = resources.local_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(local_parameters), resources.operating_dispatch);
    if (!recorded) return Result<void>::failure(recorded.error());
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::local_horizontal_highpass);
    vulkan_record_compute_barrier(resources);
    return Result<void>::success();
}

[[nodiscard]] ScoreParameters vulkan_make_score_parameters(const Resources &resources) noexcept {
    const std::uint64_t sigma_bits = std::bit_cast<std::uint64_t>(vulkan_canonical_sigma_nn_raw);
    const std::uint64_t nn_floor_bits = std::bit_cast<std::uint64_t>(vulkan_canonical_nn_floor_raw);
    const std::uint64_t maximum_bits = std::bit_cast<std::uint64_t>(resources.config.maximum.raw());
    return {
        resources.dimensions.operating_width,
        resources.dimensions.operating_height,
        resources.dimensions.truncated_width,
        resources.dimensions.truncated_height,
        resources.dimensions.blocks_horizontal,
        resources.dimensions.blocks_vertical,
        resources.dimensions.block_count,
        resources.dimensions.submatrix_width,
        resources.dimensions.submatrix_height,
        resources.dimensions.covariance_sample_count,
        0U,
        0U,
        0U,
        0U,
        0U,
        static_cast<std::uint32_t>(sigma_bits),
        static_cast<std::uint32_t>(sigma_bits >> 32U),
        static_cast<std::uint32_t>(nn_floor_bits),
        static_cast<std::uint32_t>(nn_floor_bits >> 32U),
        static_cast<std::uint32_t>(maximum_bits),
        static_cast<std::uint32_t>(maximum_bits >> 32U),
        numeric::prepare_u32_divider(resources.dimensions.covariance_sample_count),
        numeric::prepare_u32_divider(resources.dimensions.block_count),
    };
}

enum class vulkan_ScorePipelineBarrier : std::uint8_t {
    // A full compute-to-compute barrier for the next pipeline in the score
    // stream (means, covariance, and block statistics).
    generic_compute = 0U,
    // Two buffer-scoped barriers covering only the score scratch and result
    // buffers (compact tridiagonal and exact regularity).
    score_buffers = 1U,
    // No compute consumer follows; the only remaining dependencies are the
    // final host-visibility and optional diagnostic transfer barriers.
    none = 2U,
};

[[nodiscard]] Result<void> vulkan_record_score_pipeline(
    Resources &resources, VkCommandBuffer command_buffer, ComputePipeline &pipeline,
    DispatchSize dispatch,
    vulkan_ScorePipelineBarrier barrier = vulkan_ScorePipelineBarrier::generic_compute) {
    CommandContext &commands = *resources.commands;
    ScoreParameters parameters = vulkan_make_score_parameters(resources);
    const auto recorded =
        pipeline.record(command_buffer, 0U, vulkan_bytes_of(parameters), dispatch);
    if (!recorded) return Result<void>::failure(recorded.error());
    switch (barrier) {
    case vulkan_ScorePipelineBarrier::score_buffers:
        vulkan_record_score_buffer_compute_barriers(resources, commands);
        break;
    case vulkan_ScorePipelineBarrier::generic_compute:
        vulkan_record_compute_barrier(commands);
        break;
    case vulkan_ScorePipelineBarrier::none:
        break;
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void>
vulkan_record_means_and_covariance(Resources &resources, VkCommandBuffer command_buffer,
                                   bool capture_diagnostics,
                                   vulkan_SpeedChromaGpuTimestampRecording &timestamp_recording) {
    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::means);
    auto recorded = vulkan_record_score_pipeline(
        resources, command_buffer, *resources.means_pipeline,
        DispatchSize{elements_per_block, 1U, parameter_stream_count});
    if (!recorded) return recorded;
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::means);

    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::covariance);
    recorded = vulkan_record_score_pipeline(
        resources, command_buffer, *resources.covariance_pipeline, resources.covariance_dispatch);
    if (!recorded) return recorded;
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::covariance);
    if (capture_diagnostics) {
        for (std::uint32_t parameter_stream_index = 0U;
             parameter_stream_index < parameter_stream_count; ++parameter_stream_index) {
            const std::uint64_t diagnostic_parameter_base =
                static_cast<std::uint64_t>(parameter_stream_index) * diagnostic_parameter_elements;
            vulkan_record_diagnostic_copy(resources, parameter_stream_index,
                                          score_covariance_offset, diagnostic_parameter_base,
                                          diagnostic_covariance_elements);
            vulkan_record_diagnostic_copy(resources, parameter_stream_index, score_means_offset,
                                          diagnostic_parameter_base +
                                              diagnostic_covariance_elements,
                                          diagnostic_mean_elements);
        }
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void>
vulkan_record_parameter_estimation(Resources &resources, VkCommandBuffer command_buffer,
                                   bool capture_diagnostics,
                                   vulkan_SpeedChromaGpuTimestampRecording &timestamp_recording) {
    auto recorded = vulkan_record_means_and_covariance(resources, command_buffer,
                                                       capture_diagnostics, timestamp_recording);
    if (!recorded) return recorded;

    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::compact_tridiagonal);
    recorded = vulkan_record_score_pipeline(
        resources, command_buffer, *resources.compact_tridiagonal_pipeline,
        DispatchSize{1U, 1U, parameter_stream_count}, vulkan_ScorePipelineBarrier::score_buffers);
    if (!recorded) return recorded;
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::compact_tridiagonal);

    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::compact_regularity);
    recorded = vulkan_record_score_pipeline(
        resources, command_buffer, *resources.exact_regularity_pipeline,
        DispatchSize{1U, 1U, parameter_stream_count}, vulkan_ScorePipelineBarrier::score_buffers);
    if (!recorded) return recorded;
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::compact_regularity);
    if (capture_diagnostics) {
        for (std::uint32_t parameter_stream_index = 0U;
             parameter_stream_index < parameter_stream_count; ++parameter_stream_index) {
            const std::uint64_t diagnostic_parameter_base =
                static_cast<std::uint64_t>(parameter_stream_index) * diagnostic_parameter_elements;
            vulkan_record_diagnostic_copy(
                resources, parameter_stream_index, score_tridiagonal_diagonal_offset,
                diagnostic_parameter_base + diagnostic_covariance_elements +
                    diagnostic_mean_elements,
                diagnostic_tridiagonal_diagonal_elements);
            vulkan_record_diagnostic_copy(
                resources, parameter_stream_index, score_tridiagonal_subdiagonal_offset,
                diagnostic_parameter_base + diagnostic_covariance_elements +
                    diagnostic_mean_elements + diagnostic_tridiagonal_diagonal_elements,
                diagnostic_tridiagonal_subdiagonal_elements);
            vulkan_record_diagnostic_copy(
                resources, parameter_stream_index, score_regularity_metadata_offset,
                diagnostic_parameter_base + diagnostic_covariance_elements +
                    diagnostic_mean_elements + diagnostic_tridiagonal_diagonal_elements +
                    diagnostic_tridiagonal_subdiagonal_elements,
                diagnostic_regularity_metadata_elements);
        }
    }

    timestamp_recording.write_phase_start(SpeedChromaGpuPhase::block_statistics);
    recorded =
        vulkan_record_score_pipeline(resources, command_buffer, *resources.compact_blocks_pipeline,
                                     resources.score_block_dispatch);
    if (!recorded) return recorded;
    recorded = vulkan_record_score_pipeline(resources, command_buffer,
                                            *resources.block_statistics_scalar_pipeline,
                                            resources.score_scalar_dispatch);
    if (!recorded) return recorded;
    timestamp_recording.write_phase_end(SpeedChromaGpuPhase::block_statistics);
    return Result<void>::success();
}

void vulkan_record_channel_diagnostics(Resources &resources,
                                       std::uint32_t output_channel) noexcept {
    const std::uint64_t block_count = resources.dimensions.block_count;
    const std::uint64_t reference_entropy_offset = score_scratch_base_elements;
    const std::uint64_t destination_offset =
        diagnostic_channel_base + static_cast<std::uint64_t>(output_channel) * 4U * block_count;
    const std::uint32_t reference_stream = output_channel * 2U;
    const std::uint32_t distorted_stream = reference_stream + 1U;
    vulkan_record_diagnostic_copy(resources, reference_stream, reference_entropy_offset,
                                  destination_offset, 2U * block_count);
    vulkan_record_diagnostic_copy(resources, distorted_stream, reference_entropy_offset,
                                  destination_offset + 2U * block_count, 2U * block_count);
}

void vulkan_record_packet_diagnostics(Resources &resources) noexcept {
    const std::uint64_t packet_destination_offset =
        diagnostic_channel_base + 8U * resources.dimensions.block_count;
    resources.commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                       VK_ACCESS_TRANSFER_READ_BIT);
    resources.commands->copy_buffer(*resources.score_result, *resources.diagnostics_snapshot, 0U,
                                    packet_destination_offset * sizeof(Q32Words),
                                    score_packet_result_elements * sizeof(Q32Words));
}

} // namespace

Executor::Executor(std::unique_ptr<Resources> resources) noexcept
    : resources_(std::move(resources)) {
}
Executor::~Executor() = default;

Result<std::unique_ptr<Executor>> Executor::create(DeviceView device, FrameDescriptor descriptor,
                                                   Config config, Telemetry *telemetry,
                                                   bool skip_streaming_suffix) {
    if (!device.complete()) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::backend_error,
             "Speed Chroma Vulkan executor received an incomplete device"});
    }
    const auto descriptor_validation = video::validate_frame_layout(descriptor);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<Executor>>::failure(descriptor_validation.error());
    }
    if (config.kernel_scale != numeric::Q32_32::from_integer(1) || config.sigma_nn.raw() <= 0 ||
        config.nn_floor.raw() < 0 || config.weight_variance_mode != 5U ||
        config.maximum.raw() < 0) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::unsupported_model,
             "unsupported Speed Chroma parameters for pinned VMAF 1.0.16 models"});
    }
    auto resources =
        Resources::create(device, descriptor, config, telemetry, skip_streaming_suffix);
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
    return Result<Q32Value>::success(detailed.value().uv);
}

Result<std::vector<std::int32_t>>
Executor::preprocess_for_conformance(const video::GpuPlaneView &plane) {
    const auto processed = vulkan_preprocess(*resources_, plane, *resources_->highpass);
    if (!processed) {
        return Result<std::vector<std::int32_t>>::failure(processed.error());
    }
    const std::size_t count = static_cast<std::size_t>(resources_->dimensions.operating_width) *
                              resources_->dimensions.operating_height;
    std::vector<std::int32_t> samples(count);
    const auto read = resources_->highpass->read(std::as_writable_bytes(std::span(samples)));
    if (!read) {
        return Result<std::vector<std::int32_t>>::failure(read.error());
    }
    return Result<std::vector<std::int32_t>>::success(std::move(samples));
}

Result<CompactFactorDiagnostics>
Executor::factorize_covariance_for_conformance(std::span<const numeric::Q32_32> covariance,
                                               bool begin_with_arithmetic_failure) {
    if (covariance.size() != elements_per_block * elements_per_block ||
        resources_->score_scratch == nullptr || resources_->score_result == nullptr) {
        return Result<CompactFactorDiagnostics>::failure(
            {ErrorCode::invalid_argument,
             "Speed Chroma compact conformance covariance has invalid geometry"});
    }

    std::vector<Q32Words> encoded_covariance;
    encoded_covariance.reserve(covariance.size());
    for (const numeric::Q32_32 value : covariance) {
        encoded_covariance.push_back(vulkan_encode_q32(value));
    }
    auto prepared = resources_->score_scratch->fill();
    if (!prepared) {
        return Result<CompactFactorDiagnostics>::failure(prepared.error());
    }
    prepared = resources_->score_result->fill();
    if (!prepared) {
        return Result<CompactFactorDiagnostics>::failure(prepared.error());
    }
    if (begin_with_arithmetic_failure) {
        const Q32Words initial_stream_state{score_stream_arithmetic_failure_flag, 0U};
        prepared = resources_->score_result->write(vulkan_bytes_of(initial_stream_state),
                                                   score_stream_state_offset * sizeof(Q32Words));
        if (!prepared) {
            return Result<CompactFactorDiagnostics>::failure(prepared.error());
        }
    }
    prepared = resources_->score_scratch->write(std::as_bytes(std::span(encoded_covariance)));
    if (!prepared) {
        return Result<CompactFactorDiagnostics>::failure(prepared.error());
    }
    prepared = resources_->compact_tridiagonal_pipeline->update_buffers(
        0U, std::array{BufferBinding{resources_->highpass.get()},
                       BufferBinding{resources_->score_scratch.get()},
                       BufferBinding{resources_->score_result.get()},
                       BufferBinding{resources_->status.get()}});
    if (!prepared) {
        return Result<CompactFactorDiagnostics>::failure(prepared.error());
    }
    prepared = resources_->exact_regularity_pipeline->update_buffers(
        0U, std::array{BufferBinding{resources_->highpass.get()},
                       BufferBinding{resources_->score_scratch.get()},
                       BufferBinding{resources_->score_result.get()},
                       BufferBinding{resources_->status.get()}});
    if (!prepared) {
        return Result<CompactFactorDiagnostics>::failure(prepared.error());
    }

    auto begun = resources_->commands->begin();
    if (!begun) {
        return Result<CompactFactorDiagnostics>::failure(begun.error());
    }
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    resources_->commands->memory_barrier(
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    ScoreParameters score_parameters = vulkan_make_score_parameters(*resources_);
    auto recorded = resources_->compact_tridiagonal_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(score_parameters), DispatchSize{1U, 1U, 1U});
    if (!recorded) {
        return Result<CompactFactorDiagnostics>::failure(recorded.error());
    }
    vulkan_record_compute_barrier(*resources_);
    score_parameters = vulkan_make_score_parameters(*resources_);
    recorded = resources_->exact_regularity_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(score_parameters), DispatchSize{1U, 1U, 1U});
    if (!recorded) {
        return Result<CompactFactorDiagnostics>::failure(recorded.error());
    }
    resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_HOST_READ_BIT);
    const auto submitted = resources_->commands->submit_and_wait();
    recording_guard.release();
    if (!submitted) {
        return Result<CompactFactorDiagnostics>::failure(submitted.error());
    }

    std::array<Q32Words, score_result_elements> result_words{};
    auto read = resources_->score_result->read(std::as_writable_bytes(std::span(result_words)));
    if (!read) {
        return Result<CompactFactorDiagnostics>::failure(read.error());
    }
    const Q32Words stream_state = result_words[score_stream_state_offset];
    CompactFactorDiagnostics diagnostics{};
    diagnostics.stream_status_flags = stream_state.low;
    if ((stream_state.low & score_stream_arithmetic_failure_flag) != 0U) {
        diagnostics.status = CompactFactorStatus::arithmetic_failure;
        return Result<CompactFactorDiagnostics>::success(diagnostics);
    }

    std::vector<Q32Words> scratch_words(score_scratch_base_elements);
    read = resources_->score_scratch->read(std::as_writable_bytes(std::span(scratch_words)));
    if (!read) {
        return Result<CompactFactorDiagnostics>::failure(read.error());
    }

    diagnostics.status = (stream_state.low & score_stream_singular_flag) != 0U
                             ? CompactFactorStatus::singular
                             : CompactFactorStatus::computed;
    diagnostics.factorization_payload_available = true;
    diagnostics.regularity_payload_available = true;
    diagnostics.matrix_exponent =
        static_cast<std::int32_t>(scratch_words[score_matrix_exponent_offset].low);
    diagnostics.singular = (stream_state.low & score_stream_singular_flag) != 0U;
    diagnostics.sign_variation_count = scratch_words[score_regularity_metadata_offset].low;
    diagnostics.exact_zero_count = scratch_words[score_regularity_metadata_offset].high;
    for (std::size_t index = 0U; index < score_reflector_tau_elements; ++index) {
        diagnostics.reflector_tau_raw[index] =
            vulkan_decode_i64(scratch_words[score_reflector_tau_offset + index]);
    }
    for (std::size_t index = 0U; index < score_reflector_tail_elements; ++index) {
        diagnostics.reflector_tails_raw[index] =
            vulkan_decode_i64(scratch_words[score_reflector_tails_offset + index]);
    }
    for (std::size_t index = 0U; index < elements_per_block; ++index) {
        diagnostics.diagonal_raw[index] =
            vulkan_decode_i64(scratch_words[score_factor_diagonal_offset + index]);
        diagnostics.diagnostic_diagonal_raw[index] =
            vulkan_decode_i64(scratch_words[score_tridiagonal_diagonal_offset + index]);
    }
    for (std::size_t index = 0U; index + 1U < elements_per_block; ++index) {
        diagnostics.subdiagonal_raw[index] =
            vulkan_decode_i64(scratch_words[score_factor_subdiagonal_offset + index]);
        diagnostics.diagnostic_subdiagonal_raw[index] =
            vulkan_decode_i64(scratch_words[score_tridiagonal_subdiagonal_offset + index]);
    }
    if (!diagnostics.singular) {
        diagnostics.ldl_payload_available = true;
        for (std::size_t index = 0U; index < elements_per_block; ++index) {
            diagnostics.ldl_pivots_raw[index] =
                vulkan_decode_i64(scratch_words[score_ldl_pivots_offset + index]);
        }
        for (std::size_t index = 0U; index + 1U < elements_per_block; ++index) {
            diagnostics.ldl_multipliers_raw[index] =
                vulkan_decode_i64(scratch_words[score_ldl_multipliers_offset + index]);
        }
    }
    return Result<CompactFactorDiagnostics>::success(diagnostics);
}

Result<WaveConformanceResult> Executor::factorize_covariance_wave_for_conformance(
    std::span<const numeric::Q32_32> covariance_matrices, std::uint32_t frame_count) {
    constexpr std::uint32_t maximum_conformance_wave_frame_count = 32U;
    const std::uint64_t matrix_element_count = static_cast<std::uint64_t>(frame_count) *
                                               parameter_stream_count * score_covariance_elements;
    if (frame_state_.pending() || frame_count == 0U ||
        frame_count > maximum_conformance_wave_frame_count ||
        covariance_matrices.size() != matrix_element_count ||
        resources_->score_scratch == nullptr || resources_->score_result == nullptr) {
        return Result<WaveConformanceResult>::failure(
            {ErrorCode::invalid_argument,
             "Speed Chroma streaming wave has invalid dimensions or state"});
    }

    const std::uint64_t scratch_stream_elements = vulkan_score_scratch_stream_elements(*resources_);
    const std::uint64_t scratch_frame_elements = parameter_stream_count * scratch_stream_elements;
    const std::uint64_t batch_scratch_element_count =
        static_cast<std::uint64_t>(frame_count) * scratch_frame_elements;
    const std::uint64_t batch_result_element_count =
        static_cast<std::uint64_t>(frame_count) * score_result_elements;
    const auto make_batch_buffer =
        [&](std::uint64_t element_count) -> Result<std::unique_ptr<Buffer>> {
        if (element_count > std::numeric_limits<VkDeviceSize>::max() / sizeof(Q32Words)) {
            return Result<std::unique_ptr<Buffer>>::failure(
                {ErrorCode::unsupported_format,
                 "Speed Chroma streaming wave buffer size overflows"});
        }
        BufferCreateInfo create_info =
            gpu::host_visible_storage(static_cast<VkDeviceSize>(element_count * sizeof(Q32Words)));
        return Buffer::create(resources_->device, create_info);
    };

    auto batch_scratch_result = make_batch_buffer(batch_scratch_element_count);
    if (!batch_scratch_result) {
        return Result<WaveConformanceResult>::failure(batch_scratch_result.error());
    }
    auto batch_result_result = make_batch_buffer(batch_result_element_count);
    if (!batch_result_result) {
        return Result<WaveConformanceResult>::failure(batch_result_result.error());
    }
    std::unique_ptr<Buffer> batch_scratch = std::move(batch_scratch_result).value();
    std::unique_ptr<Buffer> batch_result = std::move(batch_result_result).value();
    auto batch_pipeline_result = ComputePipeline::create(
        resources_->device, gpu::generated::speed_chroma_compact_tridiagonal_wave_spv, 5U,
        sizeof(ScoreParameters));
    if (!batch_pipeline_result) {
        return Result<WaveConformanceResult>::failure(batch_pipeline_result.error());
    }
    std::unique_ptr<ComputePipeline> batch_pipeline = std::move(batch_pipeline_result).value();
    BufferCreateInfo wave_slot_create_info =
        gpu::host_visible_storage(static_cast<VkDeviceSize>(frame_count) * sizeof(TaskHandle));
    auto wave_slot_result = Buffer::create(resources_->device, wave_slot_create_info);
    if (!wave_slot_result) {
        return Result<WaveConformanceResult>::failure(wave_slot_result.error());
    }
    std::unique_ptr<Buffer> wave_slots = std::move(wave_slot_result).value();
    std::vector<TaskHandle> wave_task_handles(frame_count);
    for (std::uint32_t frame_index = 0U; frame_index < frame_count; ++frame_index) {
        wave_task_handles[frame_index] = {frame_index, 1U};
    }
    auto prepared = wave_slots->write(std::as_bytes(std::span(wave_task_handles)));
    if (!prepared) {
        return Result<WaveConformanceResult>::failure(prepared.error());
    }

    std::vector<Q32Words> scratch_words(static_cast<std::size_t>(batch_scratch_element_count));
    for (std::uint64_t matrix_index = 0U;
         matrix_index < static_cast<std::uint64_t>(frame_count) * parameter_stream_count;
         ++matrix_index) {
        const std::uint64_t source_base = matrix_index * score_covariance_elements;
        const std::uint64_t destination_base =
            matrix_index * scratch_stream_elements + score_covariance_offset;
        for (std::uint64_t covariance_index = 0U; covariance_index < score_covariance_elements;
             ++covariance_index) {
            scratch_words[static_cast<std::size_t>(destination_base + covariance_index)] =
                vulkan_encode_q32(
                    covariance_matrices[static_cast<std::size_t>(source_base + covariance_index)]);
        }
    }
    prepared = batch_scratch->write(std::as_bytes(std::span(scratch_words)));
    if (!prepared) {
        return Result<WaveConformanceResult>::failure(prepared.error());
    }
    prepared = batch_result->fill();
    if (!prepared) {
        return Result<WaveConformanceResult>::failure(prepared.error());
    }
    prepared = batch_pipeline->update_buffers(
        0U, std::array{BufferBinding{resources_->highpass.get()},
                       BufferBinding{batch_scratch.get()}, BufferBinding{batch_result.get()},
                       BufferBinding{resources_->status.get()}, BufferBinding{wave_slots.get()}});
    if (!prepared) {
        return Result<WaveConformanceResult>::failure(prepared.error());
    }

    auto timestamp_pool_result = TimestampQueryPool::create(resources_->device, 2U);
    if (!timestamp_pool_result) {
        return Result<WaveConformanceResult>::failure(timestamp_pool_result.error());
    }
    std::unique_ptr<TimestampQueryPool> timestamp_pool = std::move(timestamp_pool_result).value();

    auto begun = resources_->commands->begin();
    if (!begun) {
        return Result<WaveConformanceResult>::failure(begun.error());
    }
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    resources_->commands->memory_barrier(
        VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_HOST_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    if (timestamp_pool->supported()) {
        prepared = timestamp_pool->reset_queries(command_buffer, 0U, 2U);
        if (!prepared) {
            return Result<WaveConformanceResult>::failure(prepared.error());
        }
        prepared = timestamp_pool->write_timestamp(command_buffer, 0U);
        if (!prepared) {
            return Result<WaveConformanceResult>::failure(prepared.error());
        }
    }
    ScoreParameters score_parameters = vulkan_make_score_parameters(*resources_);
    score_parameters.highpass_frame_stride_elements =
        static_cast<std::uint32_t>(resources_->highpass->size() / sizeof(Q32Words));
    score_parameters.scratch_frame_stride_elements =
        static_cast<std::uint32_t>(scratch_frame_elements);
    score_parameters.result_frame_stride_elements =
        static_cast<std::uint32_t>(score_result_elements);
    score_parameters.status_frame_stride_elements = 1U;
    auto recorded = batch_pipeline->record(command_buffer, 0U, vulkan_bytes_of(score_parameters),
                                           DispatchSize{1U, frame_count, parameter_stream_count});
    if (!recorded) {
        return Result<WaveConformanceResult>::failure(recorded.error());
    }
    if (timestamp_pool->supported()) {
        prepared = timestamp_pool->write_timestamp(command_buffer, 1U);
        if (!prepared) {
            return Result<WaveConformanceResult>::failure(prepared.error());
        }
    }
    resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_HOST_READ_BIT);
    const auto submitted = resources_->commands->submit_and_wait();
    recording_guard.release();
    if (!submitted) {
        return Result<WaveConformanceResult>::failure(submitted.error());
    }

    WaveConformanceResult output{};
    if (timestamp_pool->supported()) {
        const auto timestamp_results = timestamp_pool->read_results(0U, 2U);
        if (!timestamp_results) {
            return Result<WaveConformanceResult>::failure(timestamp_results.error());
        }
        if (timestamp_results.value().size() != 2U || !timestamp_results.value()[0].available() ||
            !timestamp_results.value()[1].available()) {
            return Result<WaveConformanceResult>::failure(
                {ErrorCode::backend_error,
                 "Speed Chroma streaming wave timestamps are unavailable"});
        }
        const auto duration =
            calculate_timestamp_duration(timestamp_results.value()[0], timestamp_results.value()[1],
                                         timestamp_pool->capabilities());
        if (!duration) {
            return Result<WaveConformanceResult>::failure(duration.error());
        }
        output.gpu_elapsed = std::chrono::nanoseconds{
            static_cast<std::chrono::nanoseconds::rep>(duration.value().nanoseconds)};
    }

    std::vector<Q32Words> result_words(static_cast<std::size_t>(batch_result_element_count));
    auto read = batch_result->read(std::as_writable_bytes(std::span(result_words)));
    if (!read) {
        return Result<WaveConformanceResult>::failure(read.error());
    }
    read = batch_scratch->read(std::as_writable_bytes(std::span(scratch_words)));
    if (!read) {
        return Result<WaveConformanceResult>::failure(read.error());
    }

    const std::size_t output_stream_count =
        static_cast<std::size_t>(frame_count) * parameter_stream_count;
    output.stream_diagnostics.resize(output_stream_count);
    for (std::size_t output_stream_index = 0U; output_stream_index < output_stream_count;
         ++output_stream_index) {
        const std::size_t frame_index = output_stream_index / parameter_stream_count;
        const std::size_t parameter_stream = output_stream_index % parameter_stream_count;
        const Q32Words stream_state = result_words[frame_index * score_result_elements +
                                                   score_stream_state_offset + parameter_stream];
        CompactFactorDiagnostics &diagnostics = output.stream_diagnostics[output_stream_index];
        diagnostics.stream_status_flags = stream_state.low;
        if ((stream_state.low & score_stream_arithmetic_failure_flag) != 0U) {
            diagnostics.status = CompactFactorStatus::arithmetic_failure;
            continue;
        }

        diagnostics.status = CompactFactorStatus::computed;
        diagnostics.factorization_payload_available = true;
        const std::size_t scratch_base =
            static_cast<std::size_t>(output_stream_index * scratch_stream_elements);
        diagnostics.matrix_exponent = static_cast<std::int32_t>(
            scratch_words[scratch_base + score_matrix_exponent_offset].low);
        for (std::size_t index = 0U; index < score_reflector_tau_elements; ++index) {
            diagnostics.reflector_tau_raw[index] =
                vulkan_decode_i64(scratch_words[scratch_base + score_reflector_tau_offset + index]);
        }
        for (std::size_t index = 0U; index < score_reflector_tail_elements; ++index) {
            diagnostics.reflector_tails_raw[index] = vulkan_decode_i64(
                scratch_words[scratch_base + score_reflector_tails_offset + index]);
        }
        for (std::size_t index = 0U; index < elements_per_block; ++index) {
            diagnostics.diagonal_raw[index] = vulkan_decode_i64(
                scratch_words[scratch_base + score_factor_diagonal_offset + index]);
            diagnostics.diagnostic_diagonal_raw[index] = vulkan_decode_i64(
                scratch_words[scratch_base + score_tridiagonal_diagonal_offset + index]);
        }
        for (std::size_t index = 0U; index + 1U < elements_per_block; ++index) {
            diagnostics.subdiagonal_raw[index] = vulkan_decode_i64(
                scratch_words[scratch_base + score_factor_subdiagonal_offset + index]);
            diagnostics.diagnostic_subdiagonal_raw[index] = vulkan_decode_i64(
                scratch_words[scratch_base + score_tridiagonal_subdiagonal_offset + index]);
        }
    }
    return Result<WaveConformanceResult>::success(std::move(output));
}

Result<DetailedResult> Executor::compute_detailed(const video::GpuFrameView &reference,
                                                  const video::GpuFrameView &distorted,
                                                  DetailedDiagnostics *diagnostics) {
    const auto recorded = record(reference, distorted, diagnostics != nullptr);
    if (!recorded) {
        return Result<DetailedResult>::failure(recorded.error());
    }
    const auto retirement = resources_->commands->submit_finished_and_wait();
    if (!retirement) {
        // Keep outer frame state aligned with a retry-safe CommandContext
        // rollback. A command that reached the queue remains failed/pending.
        cancel();
        return Result<DetailedResult>::failure(retirement.error());
    }
    frame_state_.mark_submission_observed();
    return readback(retirement.value().host_elapsed, diagnostics);
}

Result<VkCommandBuffer> Executor::record(const video::GpuFrameView &reference,
                                         const video::GpuFrameView &distorted,
                                         bool collect_diagnostics) {
    if (frame_state_.pending()) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state,
             "Speed Chroma already has a recorded frame pending retirement"});
    }
    if (reference.index != distorted.index) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma GPU frame indices must match"});
    }
    for (std::size_t plane = 1U; plane < 3U; ++plane) {
        const auto reference_validation =
            vulkan_validate_chroma_plane(reference.planes[plane], resources_->descriptor);
        const auto distorted_validation =
            vulkan_validate_chroma_plane(distorted.planes[plane], resources_->descriptor);
        if (!reference_validation || !distorted_validation) {
            return Result<VkCommandBuffer>::failure(!reference_validation
                                                        ? reference_validation.error()
                                                        : distorted_validation.error());
        }
    }
    const auto trace_start = std::chrono::steady_clock::now();
    const auto descriptors_configured =
        vulkan_configure_frame_descriptors(*resources_, reference, distorted);
    if (!descriptors_configured) {
        return Result<VkCommandBuffer>::failure(descriptors_configured.error());
    }

    auto begun = resources_->commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    vulkan_SpeedChromaGpuTimestampRecording timestamp_recording(
        resources_->gpu_timestamp_query_pool.get(), command_buffer);
    timestamp_recording.reset();

    resources_->commands->fill_buffer(*resources_->status);
    if (resources_->score_result != nullptr) {
        resources_->commands->fill_buffer(*resources_->score_result);
    }
    resources_->commands->memory_barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    const std::array<std::uint32_t, parameter_stream_count> source_strides{
        reference.planes[1].stride_bytes, distorted.planes[1].stride_bytes,
        reference.planes[2].stride_bytes, distorted.planes[2].stride_bytes};
    auto recorded =
        vulkan_record_preprocess(*resources_, command_buffer, source_strides, timestamp_recording);
    if (!recorded) {
        return Result<VkCommandBuffer>::failure(recorded.error());
    }
    const bool guaranteed_singular = guarantees_singular_covariance(resources_->dimensions);
    if (!guaranteed_singular) {
        recorded = vulkan_record_parameter_estimation(*resources_, command_buffer,
                                                      collect_diagnostics, timestamp_recording);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        if (collect_diagnostics) {
            vulkan_record_channel_diagnostics(*resources_, 0U);
            vulkan_record_channel_diagnostics(*resources_, 1U);
        }
        timestamp_recording.write_phase_start(SpeedChromaGpuPhase::finalize);
        recorded = vulkan_record_score_pipeline(
            *resources_, command_buffer, *resources_->finalize_pipeline, DispatchSize{1U, 1U, 1U},
            vulkan_ScorePipelineBarrier::none);
        if (!recorded) {
            return Result<VkCommandBuffer>::failure(recorded.error());
        }
        timestamp_recording.write_phase_end(SpeedChromaGpuPhase::finalize);
        if (collect_diagnostics) {
            vulkan_record_packet_diagnostics(*resources_);
        }
    }

    resources_->commands->memory_barrier(
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_HOST_READ_BIT);

    const auto finished = resources_->commands->finish();
    if (!finished) {
        return Result<VkCommandBuffer>::failure(finished.error());
    }
    recording_guard.release();
    frame_state_.mark_recorded();
    pending_guaranteed_singular_ = guaranteed_singular;
    pending_collect_diagnostics_ = collect_diagnostics;
    pending_timestamp_query_pool_available_ = timestamp_recording.query_pool_available();
    pending_timestamp_recording_valid_ = timestamp_recording.recording_valid();
    stream_prefix_phase_ = StreamPrefixPhase::idle;
    vulkan_trace_elapsed("record", 0U, trace_start);
    return finished;
}

Result<VkCommandBuffer> Executor::record_stream_prefix(const video::GpuFrameView &reference,
                                                       const video::GpuFrameView &distorted,
                                                       TaskHandle task,
                                                       const TaskRingView &task_ring_view) {
    if (frame_state_.pending() || stream_prefix_phase_ != StreamPrefixPhase::idle || !task) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "Speed Chroma streaming prefix is already pending or has an "
                                       "invalid task handle"});
    }
    if (reference.index != distorted.index) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma GPU frame indices must match"});
    }
    if (guarantees_singular_covariance(resources_->dimensions) ||
        task_ring_view.highpass == nullptr || task_ring_view.score_scratch == nullptr ||
        task_ring_view.score_result == nullptr || task_ring_view.status == nullptr ||
        task_ring_view.highpass_range != resources_->highpass_frame_bytes ||
        task_ring_view.scratch_range != resources_->scratch_frame_bytes ||
        task_ring_view.result_range != resources_->result_frame_bytes ||
        task_ring_view.status_range != resources_->status_frame_bytes) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "Speed Chroma streaming task has incompatible ring slices"});
    }
    for (std::size_t plane = 1U; plane < 3U; ++plane) {
        const auto reference_validation =
            vulkan_validate_chroma_plane(reference.planes[plane], resources_->descriptor);
        const auto distorted_validation =
            vulkan_validate_chroma_plane(distorted.planes[plane], resources_->descriptor);
        if (!reference_validation || !distorted_validation) {
            return Result<VkCommandBuffer>::failure(!reference_validation
                                                        ? reference_validation.error()
                                                        : distorted_validation.error());
        }
    }

    const auto descriptors_configured = vulkan_configure_stream_prefix_descriptors(
        *resources_, reference, distorted, task_ring_view);
    if (!descriptors_configured) {
        return Result<VkCommandBuffer>::failure(descriptors_configured.error());
    }

    auto begun = resources_->commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    vulkan_SpeedChromaGpuTimestampRecording timestamp_recording(
        resources_->gpu_timestamp_query_pool.get(), command_buffer);
    timestamp_recording.reset();

    resources_->commands->fill_buffer(*task_ring_view.status, 0U, task_ring_view.status_offset,
                                      task_ring_view.status_range);
    resources_->commands->fill_buffer(*task_ring_view.score_result, 0U,
                                      task_ring_view.result_offset, task_ring_view.result_range);
    resources_->commands->memory_barrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    const std::array<std::uint32_t, parameter_stream_count> source_strides{
        reference.planes[1].stride_bytes, distorted.planes[1].stride_bytes,
        reference.planes[2].stride_bytes, distorted.planes[2].stride_bytes};
    auto recorded =
        vulkan_record_preprocess(*resources_, command_buffer, source_strides, timestamp_recording);
    if (!recorded) {
        return Result<VkCommandBuffer>::failure(recorded.error());
    }
    recorded =
        vulkan_record_means_and_covariance(*resources_, command_buffer, false, timestamp_recording);
    if (!recorded) {
        return Result<VkCommandBuffer>::failure(recorded.error());
    }

    const auto finished = resources_->commands->finish();
    if (!finished) return Result<VkCommandBuffer>::failure(finished.error());
    recording_guard.release();
    frame_state_.mark_recorded();
    pending_guaranteed_singular_ = false;
    pending_collect_diagnostics_ = false;
    pending_timestamp_query_pool_available_ = timestamp_recording.query_pool_available();
    pending_timestamp_recording_valid_ = timestamp_recording.recording_valid();
    stream_prefix_phase_ = StreamPrefixPhase::prefix_recorded;
    return finished;
}

Result<void> Executor::mark_stream_prefix_submitted(gpu::SubmissionTicket ticket) {
    if (stream_prefix_phase_ != StreamPrefixPhase::prefix_recorded) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Speed Chroma streaming prefix is not recorded"});
    }
    const auto submitted = resources_->commands->mark_submitted(ticket);
    if (submitted) {
        frame_state_.mark_submission_observed();
        stream_prefix_phase_ = StreamPrefixPhase::prefix_submitted;
    }
    return submitted;
}

Result<void> Executor::retire_stream_prefix(gpu::SubmissionTicket ticket,
                                            std::chrono::nanoseconds submission_host_elapsed) {
    if (stream_prefix_phase_ != StreamPrefixPhase::prefix_submitted) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Speed Chroma streaming prefix is not submitted"});
    }
    const auto retired = resources_->commands->mark_retired(ticket);
    if (!retired) return retired;
    const auto timestamps_collected = vulkan_collect_stream_prefix_gpu_timestamp_results(
        *resources_, pending_timestamp_query_pool_available_, pending_timestamp_recording_valid_,
        submission_host_elapsed);
    if (!timestamps_collected) return timestamps_collected;
    frame_state_.clear();
    clear_pending();
    return Result<void>::success();
}

Resources &Executor::streaming_resources() noexcept {
    return *resources_;
}

void Executor::release_streaming_suffix_resources() noexcept {
    if (frame_state_.pending()) return;
    resources_->compact_tridiagonal_pipeline.reset();
    resources_->exact_regularity_pipeline.reset();
    resources_->compact_blocks_pipeline.reset();
    resources_->finalize_pipeline.reset();
    resources_->diagnostics_snapshot.reset();
    resources_->highpass.reset();
    resources_->score_scratch.reset();
    resources_->score_result.reset();
    resources_->status.reset();
}

Result<void> Executor::mark_submitted(gpu::SubmissionTicket ticket) {
    if (!frame_state_.pending()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Speed Chroma has no recorded frame to submit"});
    }
    const auto command_submitted = resources_->commands->mark_submitted(ticket);
    if (command_submitted) {
        frame_state_.mark_submission_observed();
    }
    return command_submitted;
}

Result<DetailedResult> Executor::retire(gpu::SubmissionTicket ticket,
                                        std::chrono::nanoseconds submission_host_elapsed,
                                        DetailedDiagnostics *diagnostics) {
    CommandContext *const submitted_commands = resources_->commands.get();
    if (submitted_commands == nullptr) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_state, "Speed Chroma result has no submitted command context"});
    }
    const auto command_retired = submitted_commands->mark_retired(ticket);
    if (!command_retired) {
        return Result<DetailedResult>::failure(command_retired.error());
    }
    return readback(submission_host_elapsed, diagnostics);
}

void Executor::cancel() noexcept {
    resources_->commands->cancel();
    frame_state_.rollback_after_cancel(resources_->commands->state());
    if (!frame_state_.pending()) {
        clear_pending();
    }
}

Result<DetailedResult> Executor::readback(std::chrono::nanoseconds submission_host_elapsed,
                                          DetailedDiagnostics *diagnostics) {
    const bool diagnostics_match_recording =
        pending_collect_diagnostics_ == (diagnostics != nullptr);
    const CommandContext *const result_commands = resources_->commands.get();
    if (!frame_state_.pending() || !diagnostics_match_recording || result_commands == nullptr ||
        result_commands->state() != gpu::CommandBufferState::retired) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_state,
             "Speed Chroma result is not ready for matching host readback"});
    }

    const auto readback_started_at = std::chrono::steady_clock::now();
    const bool guaranteed_singular = pending_guaranteed_singular_;
    const long double maximum_elapsed_nanoseconds =
        std::chrono::duration<long double, std::nano>(submission_host_elapsed).count();

    const auto timestamps_collected = vulkan_collect_gpu_timestamp_results(
        *resources_, pending_timestamp_query_pool_available_, pending_timestamp_recording_valid_,
        guaranteed_singular, maximum_elapsed_nanoseconds);
    if (!timestamps_collected) {
        return Result<DetailedResult>::failure(timestamps_collected.error());
    }

    if (guaranteed_singular) {
        std::uint32_t preprocessing_flags{};
        const auto read =
            resources_->status->read(std::as_writable_bytes(std::span(&preprocessing_flags, 1U)));
        if (!read) return Result<DetailedResult>::failure(read.error());
        if ((preprocessing_flags & 1U) != 0U) {
            return Result<DetailedResult>::failure(
                {ErrorCode::invalid_argument,
                 "Speed Chroma sample exceeds the declared bit depth"});
        }
        if (preprocessing_flags != 0U) {
            return Result<DetailedResult>::failure(
                {ErrorCode::backend_error,
                 "Speed Chroma GPU returned an invalid vulkan_preprocess status"});
        }
        if (diagnostics != nullptr) {
            const auto entropy = vulkan_make_guaranteed_singular_entropy();
            if (!entropy) {
                return Result<DetailedResult>::failure(entropy.error());
            }
            *diagnostics = {};
            for (ChannelDiagnostics *channel_diagnostics :
                 std::array{&diagnostics->u, &diagnostics->v}) {
                channel_diagnostics->reference_parameters.status =
                    EstimationStatus::guaranteed_singular;
                channel_diagnostics->distorted_parameters.status =
                    EstimationStatus::guaranteed_singular;
                channel_diagnostics->reference_parameters.covariance_sample_count =
                    resources_->dimensions.covariance_sample_count;
                channel_diagnostics->distorted_parameters.covariance_sample_count =
                    resources_->dimensions.covariance_sample_count;
                channel_diagnostics->reference_entropies.assign(resources_->dimensions.block_count,
                                                                entropy.value());
                channel_diagnostics->reference_variances.assign(resources_->dimensions.block_count,
                                                                numeric::Q32_32{});
                channel_diagnostics->distorted_entropies.assign(resources_->dimensions.block_count,
                                                                entropy.value());
                channel_diagnostics->distorted_variances.assign(resources_->dimensions.block_count,
                                                                numeric::Q32_32{});
            }
        }
        DetailedResult output{Q32Value{}, Q32Value{}, Q32Value{},
                              true,       true,       resources_->dimensions};
        clear_pending();
        vulkan_trace_elapsed("readback", 0U, readback_started_at);
        return Result<DetailedResult>::success(std::move(output));
    }

    const std::uint64_t diagnostic_data_element_count =
        diagnostic_channel_base + 8U * resources_->dimensions.block_count;
    std::vector<Q32Words> snapshot;
    std::array<Q32Words, score_packet_result_elements> packet{};
    if (diagnostics != nullptr) {
        snapshot.resize(diagnostic_data_element_count + packet.size());
        const auto read =
            resources_->diagnostics_snapshot->read(std::as_writable_bytes(std::span(snapshot)));
        if (!read) return Result<DetailedResult>::failure(read.error());
        for (std::size_t packet_index = 0U; packet_index < packet.size(); ++packet_index) {
            packet[packet_index] = snapshot[diagnostic_data_element_count + packet_index];
        }
    } else {
        const auto read = resources_->score_result->read(std::as_writable_bytes(std::span(packet)));
        if (!read) return Result<DetailedResult>::failure(read.error());
    }
    if ((packet[5].low & 1U) != 0U) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_argument, "Speed Chroma sample exceeds the declared bit depth"});
    }
    if (packet[5].low != 0U) {
        return Result<DetailedResult>::failure(
            {ErrorCode::backend_error,
             "Speed Chroma GPU returned an invalid vulkan_preprocess status"});
    }
    if (packet[5].high != 0U) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_state, "Speed Chroma GPU final packet arithmetic failed"});
    }

    const auto validate_channel_state = [](std::uint32_t state, const char *label) -> Result<void> {
        if ((state & 4U) != 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, std::string("Speed Chroma GPU ") + label +
                                               " compact tridiagonal arithmetic failed"});
        }
        if ((state & 2U) != 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 std::string("Speed Chroma GPU ") + label + " fixed-point scoring failed"});
        }
        if ((state & ~7U) != 0U) {
            return Result<void>::failure(
                {ErrorCode::backend_error,
                 std::string("Speed Chroma GPU ") + label + " returned an invalid score status"});
        }
        return Result<void>::success();
    };
    const auto u_state_validation = validate_channel_state(packet[1].low, "U");
    const auto v_state_validation = validate_channel_state(packet[3].low, "V");
    if (!u_state_validation || !v_state_validation) {
        return Result<DetailedResult>::failure(!u_state_validation ? u_state_validation.error()
                                                                   : v_state_validation.error());
    }

    if (diagnostics != nullptr) {
        *diagnostics = {};
        const auto decode_snapshot_range = [&snapshot](std::uint64_t offset, std::uint64_t count) {
            std::vector<numeric::Q32_32> values;
            values.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t index = 0U; index < count; ++index) {
                values.push_back(vulkan_decode_q32(snapshot[offset + index]));
            }
            return values;
        };
        const std::array<ParameterDiagnostics *, 4U> parameters{
            &diagnostics->u.reference_parameters, &diagnostics->u.distorted_parameters,
            &diagnostics->v.reference_parameters, &diagnostics->v.distorted_parameters};
        for (std::uint64_t parameter_index = 0U; parameter_index < parameters.size();
             ++parameter_index) {
            ParameterDiagnostics &parameter = *parameters[parameter_index];
            const std::uint64_t parameter_base = parameter_index * diagnostic_parameter_elements;
            parameter.covariance_sample_count = resources_->dimensions.covariance_sample_count;
            parameter.covariance =
                decode_snapshot_range(parameter_base, diagnostic_covariance_elements);
            parameter.means = decode_snapshot_range(parameter_base + diagnostic_covariance_elements,
                                                    diagnostic_mean_elements);
            parameter.tridiagonal_diagonal = decode_snapshot_range(
                parameter_base + diagnostic_covariance_elements + diagnostic_mean_elements,
                diagnostic_tridiagonal_diagonal_elements);
            parameter.tridiagonal_subdiagonal = decode_snapshot_range(
                parameter_base + diagnostic_covariance_elements + diagnostic_mean_elements +
                    diagnostic_tridiagonal_diagonal_elements,
                diagnostic_tridiagonal_subdiagonal_elements);
            const Q32Words regularity_metadata =
                snapshot[parameter_base + diagnostic_covariance_elements +
                         diagnostic_mean_elements + diagnostic_tridiagonal_diagonal_elements +
                         diagnostic_tridiagonal_subdiagonal_elements];
            parameter.negative_pivot_count = regularity_metadata.low;
            parameter.stabilized_pivot_count = regularity_metadata.high;
        }

        const std::uint64_t block_count = resources_->dimensions.block_count;
        const auto populate_channel_arrays = [&](ChannelDiagnostics &channel_diagnostics,
                                                 std::uint32_t output_channel) {
            const std::uint64_t channel_base =
                diagnostic_channel_base +
                static_cast<std::uint64_t>(output_channel) * 4U * block_count;
            channel_diagnostics.reference_entropies =
                decode_snapshot_range(channel_base, block_count);
            channel_diagnostics.reference_variances =
                decode_snapshot_range(channel_base + block_count, block_count);
            channel_diagnostics.distorted_entropies =
                decode_snapshot_range(channel_base + 2U * block_count, block_count);
            channel_diagnostics.distorted_variances =
                decode_snapshot_range(channel_base + 3U * block_count, block_count);
        };
        populate_channel_arrays(diagnostics->u, 0U);
        populate_channel_arrays(diagnostics->v, 1U);
    }

    DetailedResult output{
        Q32Value{vulkan_decode_q32(packet[0]).raw()},
        Q32Value{vulkan_decode_q32(packet[2]).raw()},
        Q32Value{vulkan_decode_q32(packet[4]).raw()},
        (packet[1].low & 1U) != 0U,
        (packet[3].low & 1U) != 0U,
        resources_->dimensions,
    };
    clear_pending();
    vulkan_trace_elapsed("readback", 0U, readback_started_at);
    return Result<DetailedResult>::success(std::move(output));
}

void Executor::clear_pending() noexcept {
    frame_state_.clear();
    pending_guaranteed_singular_ = false;
    pending_collect_diagnostics_ = false;
    pending_timestamp_query_pool_available_ = false;
    pending_timestamp_recording_valid_ = false;
    stream_prefix_phase_ = StreamPrefixPhase::idle;
}

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
namespace {

using gpu::Buffer;
using gpu::BufferBinding;
using gpu::BufferCreateInfo;
using gpu::CommandContext;
using gpu::CommandRecordingGuard;
using gpu::ComputePipeline;
using gpu::DispatchSize;

inline constexpr VkDeviceSize streaming_task_ring_task_identity_bytes = 2U * sizeof(std::uint32_t);
inline constexpr std::uint32_t streaming_task_ring_suffix_phase_count = 4U;
inline constexpr std::uint32_t streaming_task_ring_wave_timestamp_query_count =
    streaming_task_ring_suffix_phase_count * 2U;
inline constexpr std::array streaming_task_ring_wave_gpu_phases{
    GpuStage::compact_tridiagonal,
    GpuStage::compact_regularity,
    GpuStage::block_statistics,
    GpuStage::finalize,
};

template <typename Value>
[[nodiscard]] std::span<const std::byte> streaming_task_ring_bytes_of(const Value &value) noexcept {
    return {reinterpret_cast<const std::byte *>(&value), sizeof(Value)};
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

[[nodiscard]] numeric::Q32_32 streaming_task_ring_decode_packet_score(Q32Words words) noexcept {
    const std::uint64_t bits =
        static_cast<std::uint64_t>(words.low) | (static_cast<std::uint64_t>(words.high) << 32U);
    return numeric::Q32_32::from_raw(std::bit_cast<std::int64_t>(bits));
}

[[nodiscard]] ScoreParameters streaming_task_ring_make_wave_score_parameters(
    const Dimensions &dimensions, const Config &config, VkDeviceSize highpass_frame_stride,
    VkDeviceSize scratch_frame_stride, VkDeviceSize result_frame_stride,
    VkDeviceSize status_frame_stride) noexcept {
    constexpr std::int64_t canonical_sigma_nn_raw = 816'043'776LL;
    constexpr std::int64_t canonical_nn_floor_raw = 429'496'736LL;
    const std::uint64_t sigma_bits = std::bit_cast<std::uint64_t>(canonical_sigma_nn_raw);
    const std::uint64_t nn_floor_bits = std::bit_cast<std::uint64_t>(canonical_nn_floor_raw);
    const std::uint64_t maximum_bits = std::bit_cast<std::uint64_t>(config.maximum.raw());
    return {
        dimensions.operating_width,
        dimensions.operating_height,
        dimensions.truncated_width,
        dimensions.truncated_height,
        dimensions.blocks_horizontal,
        dimensions.blocks_vertical,
        dimensions.block_count,
        dimensions.submatrix_width,
        dimensions.submatrix_height,
        dimensions.covariance_sample_count,
        0U,
        static_cast<std::uint32_t>(highpass_frame_stride / sizeof(Q32Words)),
        static_cast<std::uint32_t>(scratch_frame_stride / sizeof(Q32Words)),
        static_cast<std::uint32_t>(result_frame_stride / sizeof(Q32Words)),
        static_cast<std::uint32_t>(status_frame_stride / sizeof(std::uint32_t)),
        static_cast<std::uint32_t>(sigma_bits),
        static_cast<std::uint32_t>(sigma_bits >> 32U),
        static_cast<std::uint32_t>(nn_floor_bits),
        static_cast<std::uint32_t>(nn_floor_bits >> 32U),
        static_cast<std::uint32_t>(maximum_bits),
        static_cast<std::uint32_t>(maximum_bits >> 32U),
        numeric::prepare_u32_divider(dimensions.covariance_sample_count),
        numeric::prepare_u32_divider(dimensions.block_count),
    };
}

[[nodiscard]] Result<void> streaming_task_ring_validate_channel_state(std::uint32_t state,
                                                                      const char *label) {
    if ((state & score_stream_arithmetic_failure_flag) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             std::string("Speed Chroma GPU ") + label + " compact tridiagonal arithmetic failed"});
    }
    if ((state & 2U) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             std::string("Speed Chroma GPU ") + label + " fixed-point scoring failed"});
    }
    if ((state & ~7U) != 0U) {
        return Result<void>::failure(
            {ErrorCode::backend_error,
             std::string("Speed Chroma GPU ") + label + " returned an invalid score status"});
    }
    return Result<void>::success();
}

} // namespace

Result<std::unique_ptr<StreamingTaskRingResources::Impl>>
StreamingTaskRingResources::Impl::create(Resources &prototype) {
    if (prototype.highpass_frame_bytes == 0U || prototype.scratch_frame_bytes == 0U ||
        prototype.result_frame_bytes == 0U || prototype.status_frame_bytes == 0U) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            {ErrorCode::invalid_state, "Speed Chroma streaming requires score resources"});
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(prototype.device.physical_device, &properties);
    const VkDeviceSize descriptor_alignment =
        std::max<VkDeviceSize>(1U, properties.limits.minStorageBufferOffsetAlignment);

    auto ring =
        std::unique_ptr<StreamingTaskRingResources::Impl>(new StreamingTaskRingResources::Impl());
    ring->device_ = prototype.device;
    ring->dimensions_ = prototype.dimensions;
    ring->config_ = prototype.config;
    ring->telemetry_ = prototype.telemetry;
    ring->highpass_frame_bytes_ = prototype.highpass_frame_bytes;
    ring->scratch_frame_bytes_ = prototype.scratch_frame_bytes;
    ring->result_frame_bytes_ = prototype.result_frame_bytes;
    ring->status_frame_bytes_ = prototype.status_frame_bytes;
    ring->highpass_frame_stride_ =
        streaming_task_ring_align_up(ring->highpass_frame_bytes_, descriptor_alignment);
    ring->scratch_frame_stride_ =
        streaming_task_ring_align_up(ring->scratch_frame_bytes_, descriptor_alignment);
    ring->result_frame_stride_ =
        streaming_task_ring_align_up(ring->result_frame_bytes_, descriptor_alignment);
    ring->status_frame_stride_ =
        streaming_task_ring_align_up(ring->status_frame_bytes_, descriptor_alignment);

    const std::array frame_strides{ring->highpass_frame_stride_, ring->scratch_frame_stride_,
                                   ring->result_frame_stride_, ring->status_frame_stride_};
    std::uint64_t capacity = gpu::maximum_streaming_wave_task_count;
    for (VkDeviceSize frame_stride : frame_strides) {
        if (frame_stride == 0U) {
            return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
                {ErrorCode::invalid_state, "Speed Chroma streaming frame stride is zero"});
        }
        capacity = std::min<std::uint64_t>(capacity,
                                           properties.limits.maxStorageBufferRange / frame_stride);
    }
    if (capacity == 0U) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(
            {ErrorCode::invalid_argument, "Vulkan maxStorageBufferRange cannot hold one "
                                          "Speed Chroma streaming task"});
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
    created =
        create_into(ring->highpass_buffer_, ring->highpass_frame_stride_ * ring->core_.capacity(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created =
        create_into(ring->scratch_buffer_, ring->scratch_frame_stride_ * ring->core_.capacity(),
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_into(ring->result_buffer_, ring->result_frame_stride_ * ring->core_.capacity(),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_into(ring->status_buffer_, ring->status_frame_stride_ * ring->core_.capacity(),
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, true);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    const std::array wave_bindings{
        BufferBinding{ring->highpass_buffer_.get()},  BufferBinding{ring->scratch_buffer_.get()},
        BufferBinding{ring->result_buffer_.get()},    BufferBinding{ring->status_buffer_.get()},
        BufferBinding{ring->wave_slot_buffer_.get()},
    };
    const auto create_wave_pipeline =
        [&ring, &wave_bindings]<typename Spirv>(std::unique_ptr<ComputePipeline> &destination,
                                                const Spirv &spirv) -> Result<void> {
        auto pipeline = ComputePipeline::create(ring->device_, spirv, 5U, sizeof(ScoreParameters));
        if (!pipeline) return Result<void>::failure(pipeline.error());
        destination = std::move(pipeline).value();
        return destination->update_buffers(0U, wave_bindings);
    };
    created = create_wave_pipeline(ring->compact_pipeline_,
                                   gpu::generated::speed_chroma_compact_tridiagonal_wave_spv);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_wave_pipeline(ring->exact_regularity_pipeline_,
                                   gpu::generated::speed_chroma_exact_regularity_wave_spv);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_wave_pipeline(ring->block_statistics_pipeline_,
                                   gpu::generated::speed_chroma_compact_blocks_wave_spv);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_wave_pipeline(ring->block_statistics_scalar_pipeline_,
                                   gpu::generated::speed_chroma_block_statistics_scalar_wave_spv);
    if (!created) {
        return Result<std::unique_ptr<StreamingTaskRingResources::Impl>>::failure(created.error());
    }
    created = create_wave_pipeline(ring->finalize_pipeline_,
                                   gpu::generated::speed_chroma_finalize_wave_spv);
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
    return core_.reserve(frame_index, "Speed Chroma");
}

Result<TaskRingView> StreamingTaskRingResources::Impl::view(TaskHandle task) noexcept {
    using TaskState = gpu::StreamingTaskRingCore::TaskState;
    if (task.slot >= core_.tasks().size() || !task ||
        core_.task(task.slot).generation != task.generation ||
        core_.task(task.slot).state != TaskState::prefix_pending) {
        return Result<TaskRingView>::failure(
            {ErrorCode::invalid_state, "Speed Chroma task ring view references a stale task"});
    }
    return Result<TaskRingView>::success({
        highpass_buffer_.get(),
        scratch_buffer_.get(),
        result_buffer_.get(),
        status_buffer_.get(),
        static_cast<VkDeviceSize>(task.slot) * highpass_frame_stride_,
        static_cast<VkDeviceSize>(task.slot) * scratch_frame_stride_,
        static_cast<VkDeviceSize>(task.slot) * result_frame_stride_,
        static_cast<VkDeviceSize>(task.slot) * status_frame_stride_,
        highpass_frame_bytes_,
        scratch_frame_bytes_,
        result_frame_bytes_,
        status_frame_bytes_,
    });
}

Result<void> StreamingTaskRingResources::Impl::cancel_prefix(TaskHandle task) {
    return core_.cancel_prefix(task, "Speed Chroma");
}

Result<void> StreamingTaskRingResources::Impl::publish_ready(TaskHandle task) {
    return core_.publish_ready(task, "Speed Chroma");
}

Result<VkCommandBuffer> StreamingTaskRingResources::Impl::record_wave() {
    if (pending_state_.pending()) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "Speed Chroma streaming wave is not ready to record"});
    }
    const auto snapshotted = core_.snapshot_ready_wave("Speed Chroma");
    if (!snapshotted) {
        return Result<VkCommandBuffer>::failure(snapshotted.error());
    }

    const std::uint32_t wave_task_count =
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
    }
    const auto write_wave_timestamp = [this, command_buffer](std::uint32_t query_index) {
        if (!pending_wave_timestamp_recording_valid_) return;
        const auto written =
            wave_timestamp_query_pool_->write_timestamp(command_buffer, query_index);
        if (!written) pending_wave_timestamp_recording_valid_ = false;
    };
    commands_->update_buffer(*wave_slot_buffer_,
                             std::as_bytes(std::span(core_.pending_wave_tasks())));
    commands_->memory_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    auto recorded = Result<void>::success();
    const auto record_phase = [this, command_buffer, &write_wave_timestamp](
                                  ComputePipeline &pipeline, std::uint32_t phase_index,
                                  DispatchSize dispatch) -> Result<void> {
        const ScoreParameters parameters = streaming_task_ring_make_wave_score_parameters(
            dimensions_, config_, highpass_frame_stride_, scratch_frame_stride_,
            result_frame_stride_, status_frame_stride_);
        write_wave_timestamp(phase_index * 2U);
        const auto dispatched =
            pipeline.record(command_buffer, 0U, streaming_task_ring_bytes_of(parameters), dispatch);
        if (!dispatched) return dispatched;
        write_wave_timestamp(phase_index * 2U + 1U);
        commands_->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        return Result<void>::success();
    };

    recorded = record_phase(*compact_pipeline_, 0U, {1U, wave_task_count, parameter_stream_count});
    if (!recorded) return Result<VkCommandBuffer>::failure(recorded.error());
    recorded = record_phase(*exact_regularity_pipeline_, 1U,
                            {1U, wave_task_count, parameter_stream_count});
    if (!recorded) return Result<VkCommandBuffer>::failure(recorded.error());
    const ScoreParameters block_statistics_parameters =
        streaming_task_ring_make_wave_score_parameters(dimensions_, config_, highpass_frame_stride_,
                                                       scratch_frame_stride_, result_frame_stride_,
                                                       status_frame_stride_);
    write_wave_timestamp(4U);
    recorded = block_statistics_pipeline_->record(
        command_buffer, 0U, streaming_task_ring_bytes_of(block_statistics_parameters),
        {dimensions_.block_count, wave_task_count, parameter_stream_count});
    if (!recorded) return Result<VkCommandBuffer>::failure(recorded.error());
    commands_->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    recorded = block_statistics_scalar_pipeline_->record(
        command_buffer, 0U, streaming_task_ring_bytes_of(block_statistics_parameters),
        {block_statistics_scalar_dispatch_group_count(dimensions_.block_count), wave_task_count,
         parameter_stream_count});
    if (!recorded) return Result<VkCommandBuffer>::failure(recorded.error());
    write_wave_timestamp(5U);
    commands_->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    recorded = record_phase(*finalize_pipeline_, 3U, {1U, wave_task_count, 1U});
    if (!recorded) return Result<VkCommandBuffer>::failure(recorded.error());

    commands_->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);

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
            {ErrorCode::invalid_state, "Speed Chroma streaming wave has not been recorded"});
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
        if (telemetry_ != nullptr && telemetry_->enabled()) {
            if (!pending_wave_timestamp_recording_valid_ || wave_timestamp_query_pool_ == nullptr) {
                for (const auto phase : streaming_task_ring_wave_gpu_phases) {
                    telemetry_->record_wave_stage_failure(phase);
                }
            } else {
                const auto timestamp_results = wave_timestamp_query_pool_->read_results(
                    0U, streaming_task_ring_wave_timestamp_query_count);
                if (!timestamp_results || timestamp_results.value().size() !=
                                              streaming_task_ring_wave_timestamp_query_count) {
                    for (const auto phase : streaming_task_ring_wave_gpu_phases) {
                        telemetry_->record_wave_stage_failure(phase);
                    }
                } else {
                    const std::optional<long double> duration_upper_bound{
                        std::chrono::duration<long double, std::nano>(submission_host_elapsed)
                            .count()};
                    for (std::uint32_t phase_index = 0U;
                         phase_index < streaming_task_ring_suffix_phase_count; ++phase_index) {
                        const auto duration = gpu::calculate_timestamp_duration(
                            timestamp_results.value()[phase_index * 2U],
                            timestamp_results.value()[phase_index * 2U + 1U],
                            wave_timestamp_query_pool_->capabilities(), duration_upper_bound);
                        if (duration) {
                            telemetry_->record_wave_stage_success(
                                streaming_task_ring_wave_gpu_phases[phase_index],
                                duration.value().nanoseconds,
                                static_cast<std::uint32_t>(core_.pending_wave_tasks().size()));
                        } else {
                            telemetry_->record_wave_stage_failure(
                                streaming_task_ring_wave_gpu_phases[phase_index]);
                        }
                    }
                    telemetry_->record_timeline(
                        timestamp_results.value().front().value,
                        timestamp_results.value().back().value,
                        wave_timestamp_query_pool_->capabilities().timestamp_valid_bits,
                        wave_timestamp_query_pool_->capabilities().timestamp_period_nanoseconds,
                        crossvmaf::gpu::TimelineSegment::wave);
                }
            }
        }

        using TaskState = gpu::StreamingTaskRingCore::TaskState;
        const auto mapped_results = result_buffer_->readable_span(result_buffer_->size());
        if (!mapped_results) return fail_wave(mapped_results.error());
        const std::span<const std::byte> result_bytes = mapped_results.value();

        std::vector<StreamingTaskFrameResult> frame_results;
        frame_results.reserve(core_.pending_wave_tasks().size());
        for (const TaskHandle completed_task : core_.pending_wave_tasks()) {
            if (completed_task.slot >= core_.tasks().size()) {
                return fail_wave({ErrorCode::invalid_state,
                                  "Speed Chroma submitted wave contains an invalid task"});
            }
            const auto &task = core_.task(completed_task.slot);
            if (task.state != TaskState::wave || task.generation != completed_task.generation) {
                return fail_wave({ErrorCode::invalid_state,
                                  "Speed Chroma task generation changed during a wave"});
            }

            std::array<Q32Words, score_packet_result_elements> packet{};
            const VkDeviceSize packet_offset =
                static_cast<VkDeviceSize>(completed_task.slot) * result_frame_stride_;
            if (packet_offset + sizeof(packet) > result_bytes.size()) {
                return fail_wave({ErrorCode::invalid_state,
                                  "Speed Chroma result slot is outside the mapped range"});
            }
            std::memcpy(packet.data(),
                        result_bytes.data() + static_cast<std::size_t>(packet_offset),
                        sizeof(packet));
            if ((packet[5].low & 1U) != 0U) {
                return fail_wave({ErrorCode::invalid_argument,
                                  "Speed Chroma sample exceeds the declared bit depth"});
            }
            if (packet[5].low != 0U || packet[5].high != 0U) {
                return fail_wave(
                    {packet[5].low != 0U ? ErrorCode::backend_error : ErrorCode::invalid_state,
                     "Speed Chroma streaming wave returned an invalid final "
                     "packet"});
            }
            const auto u_state = streaming_task_ring_validate_channel_state(packet[1].low, "U");
            const auto v_state = streaming_task_ring_validate_channel_state(packet[3].low, "V");
            if (!u_state || !v_state) {
                return fail_wave(!u_state ? u_state.error() : v_state.error());
            }

            frame_results.push_back({
                task.frame_index,
                completed_task,
                DetailedResult{
                    Q32Value{streaming_task_ring_decode_packet_score(packet[0]).raw()},
                    Q32Value{streaming_task_ring_decode_packet_score(packet[2]).raw()},
                    Q32Value{streaming_task_ring_decode_packet_score(packet[4]).raw()},
                    (packet[1].low & score_stream_singular_flag) != 0U,
                    (packet[3].low & score_stream_singular_flag) != 0U,
                    dimensions_,
                },
            });
        }

        core_.complete_retired_wave();
        pending_wave_timestamp_recording_valid_ = false;
        pending_state_.clear();
        if (telemetry_ != nullptr) {
            telemetry_->record_submission_latency(submission_host_elapsed);
        }
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
    if (!pending_state_.pending()) {
        core_.restore_unsubmitted_wave();
    }
}

void StreamingTaskRingResources::Impl::cancel_after_device_drain() noexcept {
    if (commands_ != nullptr) commands_->cancel();
    core_.cancel_after_device_drain();
    pending_wave_timestamp_recording_valid_ = false;
    pending_state_.clear();
}

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
namespace {

[[nodiscard]] constexpr std::uint32_t telemetry_stage_id(GpuStage stage) noexcept {
    return static_cast<std::uint32_t>(stage);
}

} // namespace

const char *gpu_stage_name(GpuStage stage) noexcept {
    switch (stage) {
    case GpuStage::scale:
        return "scale";
    case GpuStage::antialias_fused:
        return "antialias_fused_9tap_decimate";
    case GpuStage::local_vertical:
        return "local_vertical_3tap";
    case GpuStage::local_horizontal_highpass:
        return "local_horizontal_3tap_highpass";
    case GpuStage::means:
        return "means";
    case GpuStage::covariance:
        return "covariance";
    case GpuStage::compact_tridiagonal:
        return "compact_tridiagonal";
    case GpuStage::compact_regularity:
        return "compact_regularity";
    case GpuStage::block_statistics:
        return "block_statistics";
    case GpuStage::finalize:
        return "finalize";
    case GpuStage::count:
        break;
    }
    return "unknown";
}

Telemetry::Telemetry(gpu::TelemetryCollector &collector)
    : collector_(&collector), operation_(collector.register_gpu_operation()),
      stage_series_(collector.register_duration_series(gpu_stage_count)),
      compact_factor_series_(collector.register_duration_series(1U)),
      wave_stage_series_(collector.register_wave_duration_series(gpu_stage_count)) {
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

void Telemetry::configure_timestamps(const gpu::GpuTimestampConfiguration &configuration) noexcept {
    collector_->configure_gpu_timestamps(operation_, configuration);
}

void Telemetry::configure_execution(const GpuExecutionConfiguration &configuration) noexcept {
    if (!enabled()) return;
    execution_configuration_ = configuration;
    execution_configuration_.configured = true;
}

void Telemetry::record_submission(bool timestamp_available) noexcept {
    collector_->record_gpu_submission(operation_, timestamp_available);
}

void Telemetry::record_parent_success(long double duration_nanoseconds) noexcept {
    collector_->record_gpu_parent(operation_, gpu::TelemetrySampleOutcome::success,
                                  duration_nanoseconds);
}

void Telemetry::record_parent_failure() noexcept {
    collector_->record_gpu_parent(operation_, gpu::TelemetrySampleOutcome::failure);
}

void Telemetry::record_parent_skip() noexcept {
    collector_->record_gpu_parent(operation_, gpu::TelemetrySampleOutcome::skipped);
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

void Telemetry::record_stage_success(GpuStage stage, long double duration_nanoseconds) noexcept {
    if (!is_valid_stage(stage)) return;
    collector_->record_duration(stage_series_, telemetry_stage_id(stage),
                                gpu::TelemetrySampleOutcome::success, duration_nanoseconds);
}

void Telemetry::record_stage_failure(GpuStage stage) noexcept {
    if (!is_valid_stage(stage)) return;
    collector_->record_duration(stage_series_, telemetry_stage_id(stage),
                                gpu::TelemetrySampleOutcome::failure);
}

void Telemetry::record_stage_skip(GpuStage stage) noexcept {
    if (!is_valid_stage(stage)) return;
    collector_->record_duration(stage_series_, telemetry_stage_id(stage),
                                gpu::TelemetrySampleOutcome::skipped);
}

void Telemetry::record_wave_stage_success(GpuStage stage, long double wave_nanoseconds,
                                          std::uint32_t task_count) noexcept {
    if (!is_valid_stage(stage)) return;
    collector_->record_wave_duration(wave_stage_series_, telemetry_stage_id(stage),
                                     gpu::TelemetrySampleOutcome::success, wave_nanoseconds,
                                     task_count);
}

void Telemetry::record_wave_stage_failure(GpuStage stage) noexcept {
    if (!is_valid_stage(stage)) return;
    collector_->record_wave_duration(wave_stage_series_, telemetry_stage_id(stage),
                                     gpu::TelemetrySampleOutcome::failure);
}

void Telemetry::record_compact_factor_success(long double duration_nanoseconds) noexcept {
    collector_->record_duration(compact_factor_series_, 0U, gpu::TelemetrySampleOutcome::success,
                                duration_nanoseconds);
}

void Telemetry::record_compact_factor_failure() noexcept {
    collector_->record_duration(compact_factor_series_, 0U, gpu::TelemetrySampleOutcome::failure);
}

void Telemetry::record_compact_factor_skip() noexcept {
    collector_->record_duration(compact_factor_series_, 0U, gpu::TelemetrySampleOutcome::skipped);
}

const gpu::GpuOperationSnapshot &Telemetry::operation() const noexcept {
    return collector_->gpu_operation(operation_);
}

const GpuExecutionConfiguration &Telemetry::execution_configuration() const noexcept {
    return execution_configuration_;
}

const gpu::GpuDurationStatistics &Telemetry::stage_statistics(GpuStage stage) const noexcept {
    if (!is_valid_stage(stage)) {
        static constexpr gpu::GpuDurationStatistics empty_statistics{};
        return empty_statistics;
    }
    return collector_->duration_statistics(stage_series_, telemetry_stage_id(stage));
}

const gpu::GpuWaveDurationStatistics &
Telemetry::wave_stage_statistics(GpuStage stage) const noexcept {
    if (!is_valid_stage(stage)) {
        static constexpr gpu::GpuWaveDurationStatistics empty_statistics{};
        return empty_statistics;
    }
    return collector_->wave_statistics(wave_stage_series_, telemetry_stage_id(stage));
}

const gpu::GpuDurationStatistics &Telemetry::compact_factor_statistics() const noexcept {
    return collector_->duration_statistics(compact_factor_series_, 0U);
}

TelemetrySnapshot Telemetry::snapshot() const {
    TelemetrySnapshot result{};
    result.operation_handle = operation_;
    result.operation = operation();
    result.execution_configuration = execution_configuration_;
    result.compact_factor = compact_factor_statistics();
    for (std::size_t stage_index = 0U; stage_index < gpu_stage_count; ++stage_index) {
        const GpuStage stage = static_cast<GpuStage>(stage_index);
        result.stages[stage_index] = stage_statistics(stage);
        result.wave_stages[stage_index] = wave_stage_statistics(stage);
    }
    return result;
}

bool Telemetry::is_valid_stage(GpuStage stage) noexcept {
    return telemetry_stage_id(stage) < gpu_stage_count;
}

} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
bool Executor::has_streaming_suffix_work() const noexcept {
    return !guarantees_singular_covariance(resources_->dimensions) &&
           resources_->highpass_frame_bytes != 0U && resources_->scratch_frame_bytes != 0U &&
           resources_->result_frame_bytes != 0U && resources_->status_frame_bytes != 0U;
}
bool Executor::identity_scale_fused() const noexcept {
    return resources_->identity_scale_fused && !resources_->scaled && !resources_->scale_pipeline;
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
} // namespace crossvmaf::metrics::speed_chroma::vulkan

namespace crossvmaf::metrics::speed_chroma {
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
    auto created = vulkan::Executor::create(impl_->device, impl_->descriptor, impl_->config,
                                            impl_->telemetry, true);
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
} // namespace crossvmaf::metrics::speed_chroma
