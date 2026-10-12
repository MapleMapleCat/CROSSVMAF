#include "video/frame.hpp"
#include "metrics/cambi/vulkan.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "metrics/cambi/cambi.hpp"
#include "gpu/buffer.hpp"
#include "gpu/command.hpp"
#include "gpu/pipeline.hpp"
#include "gpu/device.hpp"
#include "gpu/timestamp.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
#include "shaders/cambi_c_values_rows_spv.hpp"
#include "shaders/cambi_c_values_rows_u16_spv.hpp"
#include "shaders/cambi_combine_spv.hpp"
#include "shaders/cambi_decimate_spv.hpp"
#include "shaders/cambi_mask_spv.hpp"
#include "shaders/cambi_mode_spv.hpp"
#include "shaders/cambi_pool_finalize_spv.hpp"
#include "shaders/cambi_pool_reduce_spv.hpp"
#include "shaders/cambi_preprocess_spv.hpp"
#include "shaders/cambi_radix_histogram_spv.hpp"
#include "shaders/cambi_radix_histogram_2d_spv.hpp"
#include "shaders/cambi_radix_init_spv.hpp"
#include "shaders/cambi_radix_update_spv.hpp"
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cmath>

namespace crossvmaf::metrics::cambi::vulkan {

struct ScaleGeometry {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t value_count{};
    std::uint32_t c_value_offset{};
    std::uint32_t top_count{};
    bool decimate{};
    std::uint32_t decimate_set{};
    std::uint32_t mode_set{};
    std::uint32_t c_value_set{};
    std::uint32_t image_input_index{};
    std::uint32_t mask_index{};
    std::uint32_t c_value_segment_rows{};
    std::uint32_t c_value_segment_count{};
    std::uint64_t c_value_initial_window_source_visits{};
    gpu::DispatchSize two_dimensional_dispatch{};
    gpu::DispatchSize c_value_dispatch{};
    std::uint32_t radix_group_count{};
    std::vector<std::uint32_t> reduction_input_counts;
    std::uint32_t final_partial_index{};
    // Per-scale partition base inside the pool_partials ping-pong buffers.
    // Disjoint partitions allow all scales' reduce/finalize dispatches to be
    // issued in barrier-free lockstep rounds.
    std::uint32_t pool_partial_offset{};
};

class Resources final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Resources>>
    create(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
           Telemetry *telemetry = nullptr,
           DetailedDiagnosticsMode diagnostics_mode = DetailedDiagnosticsMode::disabled);

    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    Telemetry *telemetry{};
    DetailedDiagnosticsMode diagnostics_mode{};
    bool high_res_speedup{};
    std::uint16_t window_size{};
    std::array<ScaleGeometry, scale_count> geometry;
    gpu::DispatchSize radix_histogram_dispatch{};
    bool radix_histogram_uses_2d_fallback{};

    std::array<std::unique_ptr<gpu::Buffer>, 2> image;
    // One mode-filter output per scale. The two-entry image ping-pong cannot
    // hold these: scale s+1 overwrites the slot that scale s wrote, which
    // forces every c-value dispatch to run before the next scale's mode pass.
    // Dedicated buffers keep each scale's source live so all five c-value
    // dispatches can be issued as one concurrent group.
    std::array<std::unique_ptr<gpu::Buffer>, scale_count> mode_output;
    std::array<std::unique_ptr<gpu::Buffer>, 2> mask;
    std::unique_ptr<gpu::Buffer> c_values;
    std::array<std::unique_ptr<gpu::Buffer>, scale_count> c_value_diagnostics_readback;
    std::unique_ptr<gpu::Buffer> radix_counts;
    std::unique_ptr<gpu::Buffer> selection_state;
    std::array<std::unique_ptr<gpu::Buffer>, 2> pool_partials;
    std::unique_ptr<gpu::Buffer> results;

    std::unique_ptr<gpu::ComputePipeline> preprocess_pipeline;
    std::unique_ptr<gpu::ComputePipeline> mask_pipeline;
    std::unique_ptr<gpu::ComputePipeline> decimate_pipeline;
    std::unique_ptr<gpu::ComputePipeline> mode_pipeline;
    std::unique_ptr<gpu::ComputePipeline> c_value_pipeline;
    std::unique_ptr<gpu::ComputePipeline> radix_init_pipeline;
    std::unique_ptr<gpu::ComputePipeline> radix_histogram_pipeline;
    std::unique_ptr<gpu::ComputePipeline> radix_update_pipeline;
    std::unique_ptr<gpu::ComputePipeline> pool_reduce_pipeline;
    std::unique_ptr<gpu::ComputePipeline> pool_finalize_pipeline;
    std::unique_ptr<gpu::ComputePipeline> combine_pipeline;
    std::unique_ptr<gpu::CommandContext> commands;
    std::unique_ptr<gpu::CommandContext> prefix_commands;
    std::vector<GpuStage> gpu_timestamp_interval_stages;
    std::unique_ptr<gpu::TimestampQueryPool> gpu_timestamp_query_pool;

  private:
    Resources() = default;
};

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi::vulkan {
namespace {

using gpu::Buffer;
using gpu::BufferBinding;
using gpu::BufferCreateInfo;
using gpu::CommandContext;
using gpu::ComputePipeline;
using gpu::DeviceView;
using gpu::DispatchSize;
using gpu::GpuTimestampConfiguration;
using gpu::query_timestamp_capabilities;
using gpu::TimestampQueryPool;

using CambiGpuCValueScaleConfiguration = CValueScaleConfiguration;
using CambiGpuStage = GpuStage;

constexpr std::string_view resources_c_value_segment_rows_environment_variable =
    "CROSSVMAF_CAMBI_CVALUE_SEGMENT_ROWS";
constexpr std::string_view resources_c_value_histogram_environment_variable =
    "CROSSVMAF_CAMBI_CVALUE_HISTOGRAM";

[[nodiscard]] Result<std::optional<std::string>>
resources_read_environment_value(std::string_view variable_name) {
#if defined(_WIN32)
    char *raw_environment_value = nullptr;
    std::size_t environment_value_length = 0U;
    const int environment_status =
        _dupenv_s(&raw_environment_value, &environment_value_length, variable_name.data());
    if (environment_status != 0) {
        std::free(raw_environment_value);
        return Result<std::optional<std::string>>::failure(
            {ErrorCode::backend_error, "failed to read a CAMBI C-value diagnostic override"});
    }
    if (raw_environment_value == nullptr) {
        return Result<std::optional<std::string>>::success(std::nullopt);
    }
    std::string environment_value(raw_environment_value);
    std::free(raw_environment_value);
    return Result<std::optional<std::string>>::success(std::move(environment_value));
#else
    const char *const raw_environment_value = std::getenv(variable_name.data());
    if (raw_environment_value == nullptr) {
        return Result<std::optional<std::string>>::success(std::nullopt);
    }
    return Result<std::optional<std::string>>::success(std::string(raw_environment_value));
#endif
}

[[nodiscard]] Result<CValueSegmentRowsConfiguration>
resources_read_c_value_segment_rows_configuration() {
    const auto environment_value_result =
        resources_read_environment_value(resources_c_value_segment_rows_environment_variable);
    if (!environment_value_result) {
        return Result<CValueSegmentRowsConfiguration>::failure(environment_value_result.error());
    }
    if (!environment_value_result.value()) {
        return Result<CValueSegmentRowsConfiguration>::success(
            default_c_value_segment_rows_configuration);
    }

    const auto configuration =
        parse_c_value_segment_rows_configuration(*environment_value_result.value());
    if (!configuration) {
        return Result<CValueSegmentRowsConfiguration>::failure(
            {ErrorCode::invalid_argument,
             "CROSSVMAF_CAMBI_CVALUE_SEGMENT_ROWS must be auto, one of "
             "4/8/16/32, or a medium,tail pair using those values"});
    }
    return Result<CValueSegmentRowsConfiguration>::success(*configuration);
}

[[nodiscard]] Result<bool> resources_read_c_value_uses_uint16_histogram(bool shader_int16_enabled) {
    const auto environment_value_result =
        resources_read_environment_value(resources_c_value_histogram_environment_variable);
    if (!environment_value_result) {
        return Result<bool>::failure(environment_value_result.error());
    }
    const std::string_view environment_value =
        environment_value_result.value() ? std::string_view(*environment_value_result.value())
                                         : std::string_view("auto");
    if (environment_value == "auto") {
        return Result<bool>::success(shader_int16_enabled);
    }
    if (environment_value == "u32") {
        return Result<bool>::success(false);
    }
    if (environment_value != "u16") {
        return Result<bool>::failure(
            {ErrorCode::invalid_argument,
             "CROSSVMAF_CAMBI_CVALUE_HISTOGRAM must be auto, u32, or u16"});
    }
    if (!shader_int16_enabled) {
        return Result<bool>::failure({ErrorCode::no_compatible_device,
                                      "CAMBI uint16 histogram requires shaderInt16 support"});
    }
    return Result<bool>::success(true);
}

[[nodiscard]] std::uint16_t resources_adjusted_window_size(std::uint16_t base, std::uint32_t width,
                                                           std::uint32_t height,
                                                           bool high_res_speedup) noexcept {
    std::uint64_t adjusted = static_cast<std::uint64_t>(base) * (width + height);
    adjusted = (adjusted / 375U) >> 4U;
    if (high_res_speedup) adjusted = (adjusted + 1U) >> 1U;
    adjusted |= 1U;
    return static_cast<std::uint16_t>(
        std::min<std::uint64_t>(adjusted, std::numeric_limits<std::uint16_t>::max()));
}

[[nodiscard]] constexpr std::uint32_t resources_divide_round_up(std::uint32_t value,
                                                                std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

[[nodiscard]] constexpr DispatchSize
resources_calculate_batched_radix_dispatch(std::uint32_t group_count,
                                           std::uint32_t maximum_group_count_x) noexcept {
    const std::uint32_t row_count = resources_divide_round_up(group_count, maximum_group_count_x);
    const std::uint32_t row_width = resources_divide_round_up(group_count, row_count);
    return {
        row_width,
        row_count,
        1U,
    };
}

static_assert(resources_calculate_batched_radix_dispatch(76'725U, 65'535U).x == 38'363U);
static_assert(resources_calculate_batched_radix_dispatch(76'725U, 65'535U).y == 2U);
static_assert(resources_calculate_batched_radix_dispatch(4'096U, 65'535U).x == 4'096U);
static_assert(resources_calculate_batched_radix_dispatch(4'096U, 65'535U).y == 1U);

[[nodiscard]] std::uint64_t
resources_initial_window_source_visits(std::uint32_t width, std::uint32_t height,
                                       std::uint32_t window_size, std::uint32_t segment_rows,
                                       std::uint32_t segment_count) noexcept {
    const std::int64_t radius = window_size >> 1U;
    std::uint64_t horizontal_visits{};
    for (std::uint32_t column = 0U; column < width; ++column) {
        const std::int64_t left =
            std::max<std::int64_t>(static_cast<std::int64_t>(column) - radius, 0);
        const std::int64_t right = std::min<std::int64_t>(
            static_cast<std::int64_t>(column) + radius + 1, static_cast<std::int64_t>(width));
        horizontal_visits += static_cast<std::uint64_t>(right - left);
    }

    std::uint64_t vertical_visits{};
    for (std::uint32_t segment_index = 0U; segment_index < segment_count; ++segment_index) {
        const std::int64_t segment_begin = static_cast<std::int64_t>(segment_index) * segment_rows;
        const std::int64_t top = std::max(segment_begin - radius, 0LL);
        const std::int64_t bottom =
            std::min(segment_begin + radius + 1, static_cast<std::int64_t>(height));
        vertical_visits += static_cast<std::uint64_t>(bottom - top);
    }
    return horizontal_visits * vertical_visits;
}

[[nodiscard]] Result<std::unique_ptr<Buffer>>
resources_create_storage_buffer(DeviceView device, VkDeviceSize size,
                                VkBufferUsageFlags additional_usage = 0U) {
    BufferCreateInfo create_info{};
    create_info.size = size;
    create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | additional_usage;
    return Buffer::create(device, create_info);
}

[[nodiscard]] Result<std::unique_ptr<Buffer>>
resources_create_diagnostics_buffer(DeviceView device, VkDeviceSize size) {
    return Buffer::create(device,
                          gpu::host_visible_storage(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT));
}

[[nodiscard]] Result<VkDeviceSize> resources_checked_size(std::uint64_t element_count,
                                                          std::uint64_t element_size,
                                                          VkDeviceSize limit, const char *label) {
    if (element_count == 0U ||
        element_count > std::numeric_limits<std::uint64_t>::max() / element_size) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("CAMBI ") + label + " buffer size overflows"});
    }
    const std::uint64_t bytes = element_count * element_size;
    if (bytes > limit) {
        return Result<VkDeviceSize>::failure(
            {ErrorCode::unsupported_format,
             std::string("CAMBI ") + label + " buffer exceeds maxStorageBufferRange"});
    }
    return Result<VkDeviceSize>::success(static_cast<VkDeviceSize>(bytes));
}

[[nodiscard]] Result<void> resources_validate_dispatch(DispatchSize dispatch,
                                                       const VkPhysicalDeviceLimits &limits,
                                                       const char *label) {
    if (dispatch.x == 0U || dispatch.y == 0U || dispatch.z == 0U ||
        dispatch.x > limits.maxComputeWorkGroupCount[0] ||
        dispatch.y > limits.maxComputeWorkGroupCount[1] ||
        dispatch.z > limits.maxComputeWorkGroupCount[2]) {
        return Result<void>::failure({ErrorCode::unsupported_format,
                                      std::string("CAMBI ") + label +
                                          " dispatch exceeds the selected Vulkan device limits"});
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void> resources_initialize_gpu_timestamp_queries(Resources &resources) {
    Telemetry *const telemetry = resources.telemetry;
    if (telemetry == nullptr || !telemetry->enabled()) {
        return Result<void>::success();
    }

    for (std::uint32_t scale_index = 0U; scale_index < scale_count; ++scale_index) {
        const ScaleGeometry &geometry = resources.geometry[scale_index];
        telemetry->configure_c_value_scale(scale_index,
                                           CambiGpuCValueScaleConfiguration{
                                               true,
                                               geometry.width,
                                               geometry.height,
                                               resources.window_size,
                                               geometry.c_value_segment_rows,
                                               geometry.c_value_segment_count,
                                               geometry.c_value_initial_window_source_visits,
                                               geometry.c_value_dispatch.x,
                                               geometry.c_value_dispatch.y,
                                               geometry.c_value_dispatch.z,
                                           });
    }

    resources.gpu_timestamp_interval_stages.reserve(3U + scale_count * 5U);
    resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::preprocess);
    resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::mask);
    // Outside c-value diagnostics the five c-value dispatches are issued as one
    // concurrent group, so they share a single interval and cannot be attributed
    // per scale. Diagnostics mode keeps them serialized behind per-scale
    // readback copies, where per-scale intervals remain meaningful.
    const bool per_scale_c_value_intervals =
        resources.diagnostics_mode == DetailedDiagnosticsMode::c_value_maps;
    for (const ScaleGeometry &geometry : resources.geometry) {
        if (geometry.decimate) {
            resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::decimate);
        }
        resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::mode);
        if (per_scale_c_value_intervals) {
            resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::c_value);
        }
    }
    if (!per_scale_c_value_intervals) {
        resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::c_value);
    }
    resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::radix);
    // The pool section records as one barrier-free lockstep group in both
    // modes, so a single interval covers it (same treatment as the grouped
    // c-value dispatches).
    resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::pool_reduce);
    resources.gpu_timestamp_interval_stages.push_back(CambiGpuStage::combine);

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

    const std::uint32_t query_count =
        static_cast<std::uint32_t>(resources.gpu_timestamp_interval_stages.size() * 2U);
    auto query_pool_result = TimestampQueryPool::create(resources.device, query_count);
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

} // namespace

Result<std::unique_ptr<Resources>> Resources::create(DeviceView device, FrameDescriptor descriptor,
                                                     Config config, Telemetry *telemetry,
                                                     DetailedDiagnosticsMode diagnostics_mode) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    const auto &limits = properties.limits;
    if (limits.maxComputeWorkGroupInvocations < linear_group_size ||
        limits.maxComputeWorkGroupSize[0] < linear_group_size ||
        limits.maxComputeWorkGroupSize[1] < tile_size ||
        limits.maxPerStageDescriptorStorageBuffers < 5U ||
        limits.maxPushConstantsSize < sizeof(RadixHistogram2DParameters)) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::no_compatible_device,
             "Vulkan device does not meet deterministic CAMBI compute limits"});
    }

    auto resources = std::unique_ptr<Resources>(new Resources());
    resources->device = device;
    resources->descriptor = descriptor;
    resources->config = config;
    resources->telemetry = telemetry;
    resources->diagnostics_mode = diagnostics_mode;
    const std::uint64_t original_count =
        static_cast<std::uint64_t>(descriptor.width) * descriptor.height;
    resources->high_res_speedup =
        config.high_res_speedup == 1080U && original_count >= std::uint64_t{1920} * 1080U;
    resources->window_size = resources_adjusted_window_size(
        base_window_size, descriptor.width, descriptor.height, resources->high_res_speedup);
    if (static_cast<std::uint32_t>(resources->window_size) * resources->window_size >= 4226U) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::unsupported_format,
             "CAMBI adjusted window exceeds the deterministic histogram limit"});
    }

    const auto segment_rows_configuration_result =
        resources_read_c_value_segment_rows_configuration();
    if (!segment_rows_configuration_result) {
        return Result<std::unique_ptr<Resources>>::failure(
            segment_rows_configuration_result.error());
    }
    const CValueSegmentRowsConfiguration segment_rows_configuration =
        segment_rows_configuration_result.value();
    const auto use_uint16_histogram_result =
        resources_read_c_value_uses_uint16_histogram(device.shader_int16_enabled);
    if (!use_uint16_histogram_result) {
        return Result<std::unique_ptr<Resources>>::failure(use_uint16_histogram_result.error());
    }
    const bool use_uint16_histogram = use_uint16_histogram_result.value();

    std::uint32_t width = descriptor.width;
    std::uint32_t height = descriptor.height;
    std::uint32_t image_index = 0U;
    std::uint32_t mask_index = 0U;
    std::uint32_t decimate_set_index = 0U;
    std::uint64_t packed_c_value_count = 0U;
    std::uint32_t pool_partial_count = 0U;
    for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
        auto &geometry = resources->geometry[scale];
        geometry.decimate = scale == 0U && resources->high_res_speedup;
        if (geometry.decimate) geometry.decimate_set = decimate_set_index++;
        if (scale != 0U || resources->high_res_speedup) {
            image_index = 1U - image_index;
            mask_index = 1U - mask_index;
            width = (width + 1U) >> 1U;
            height = (height + 1U) >> 1U;
        }
        geometry.width = width;
        geometry.height = height;
        const std::uint64_t value_count = static_cast<std::uint64_t>(width) * height;
        if (value_count > std::numeric_limits<std::uint32_t>::max()) {
            return Result<std::unique_ptr<Resources>>::failure(
                {ErrorCode::unsupported_format,
                 "CAMBI frame dimensions exceed the 32-bit address space"});
        }
        geometry.value_count = static_cast<std::uint32_t>(value_count);
        if (packed_c_value_count > std::numeric_limits<std::uint32_t>::max() - value_count) {
            return Result<std::unique_ptr<Resources>>::failure(
                {ErrorCode::unsupported_format,
                 "CAMBI packed C-value offsets exceed 32-bit indexing"});
        }
        geometry.c_value_offset = static_cast<std::uint32_t>(packed_c_value_count);
        packed_c_value_count += value_count;
        geometry.top_count =
            std::clamp<std::uint32_t>((geometry.value_count * 3U) / 5U, 1U, geometry.value_count);
        geometry.image_input_index = image_index;
        geometry.mask_index = mask_index;
        geometry.mode_set = scale;
        geometry.c_value_set = scale;
        geometry.c_value_segment_rows = c_value_segment_rows(height, segment_rows_configuration);
        geometry.c_value_segment_count = c_value_segment_count(height, segment_rows_configuration);
        geometry.two_dimensional_dispatch = {resources_divide_round_up(width, tile_size),
                                             resources_divide_round_up(height, tile_size), 1U};
        geometry.c_value_dispatch = {resources_divide_round_up(width, c_value_column_group_size),
                                     geometry.c_value_segment_count, 1U};
        geometry.radix_group_count = resources_divide_round_up(
            geometry.value_count, linear_group_size * radix_values_per_thread);
        for (const auto [dispatch, label] : std::array<std::pair<DispatchSize, const char *>, 2>{
                 std::pair{geometry.two_dimensional_dispatch, "2D"},
                 std::pair{geometry.c_value_dispatch, "c-value traversal"}}) {
            const auto valid = resources_validate_dispatch(dispatch, limits, label);
            if (!valid) {
                return Result<std::unique_ptr<Resources>>::failure(valid.error());
            }
        }
        geometry.c_value_initial_window_source_visits = resources_initial_window_source_visits(
            width, height, resources->window_size, geometry.c_value_segment_rows,
            geometry.c_value_segment_count);
        std::uint32_t partial_count = geometry.radix_group_count;
        geometry.pool_partial_offset = pool_partial_count;
        pool_partial_count += geometry.radix_group_count;
        while (partial_count > 1U) {
            geometry.reduction_input_counts.push_back(partial_count);
            partial_count = resources_divide_round_up(partial_count, linear_group_size);
        }
        geometry.final_partial_index =
            static_cast<std::uint32_t>(geometry.reduction_input_counts.size() & 1U);
    }

    std::uint64_t batched_radix_group_count = 0U;
    for (const ScaleGeometry &geometry : resources->geometry) {
        batched_radix_group_count += geometry.radix_group_count;
    }
    if (batched_radix_group_count > std::numeric_limits<std::uint32_t>::max()) {
        return Result<std::unique_ptr<Resources>>::failure(
            {ErrorCode::unsupported_format,
             "CAMBI batched radix group count exceeds 32-bit indexing"});
    }
    const std::uint32_t radix_group_count = static_cast<std::uint32_t>(batched_radix_group_count);
    resources->radix_histogram_uses_2d_fallback =
        radix_group_count > limits.maxComputeWorkGroupCount[0];
    resources->radix_histogram_dispatch = resources_calculate_batched_radix_dispatch(
        radix_group_count, limits.maxComputeWorkGroupCount[0]);
    const auto valid_radix_dispatch = resources_validate_dispatch(
        resources->radix_histogram_dispatch, limits, "batched radix histogram");
    if (!valid_radix_dispatch) {
        return Result<std::unique_ptr<Resources>>::failure(valid_radix_dispatch.error());
    }

    const auto image_bytes = resources_checked_size(original_count, sizeof(std::uint32_t),
                                                    limits.maxStorageBufferRange, "image");
    const auto c_value_bytes =
        resources_checked_size(packed_c_value_count, sizeof(std::uint32_t),
                               limits.maxStorageBufferRange, "packed C-value");
    const auto radix_count_bytes =
        resources_checked_size(radix_bin_count * scale_count, sizeof(std::uint32_t),
                               limits.maxStorageBufferRange, "radix count");
    // The pool partials ping-pong buffers give every scale a disjoint
    // partition (geometry.pool_partial_offset) so the five reduction chains
    // can be issued as lockstep rounds without cross-scale hazards.
    const auto pool_bytes = resources_checked_size(pool_partial_count, sizeof(PoolPartial),
                                                   limits.maxStorageBufferRange, "pool partial");
    const auto state_bytes =
        resources_checked_size(selection_state_word_count * scale_count, sizeof(std::uint32_t),
                               limits.maxStorageBufferRange, "selection state");
    const auto result_bytes =
        resources_checked_size(6U, sizeof(U64Words), limits.maxStorageBufferRange, "result");
    if (!image_bytes || !c_value_bytes || !radix_count_bytes || !pool_bytes || !state_bytes ||
        !result_bytes) {
        const Error error = !image_bytes         ? image_bytes.error()
                            : !c_value_bytes     ? c_value_bytes.error()
                            : !radix_count_bytes ? radix_count_bytes.error()
                            : !pool_bytes        ? pool_bytes.error()
                            : !state_bytes       ? state_bytes.error()
                                                 : result_bytes.error();
        return Result<std::unique_ptr<Resources>>::failure(error);
    }

    const auto create_into = [&](std::unique_ptr<Buffer> &destination,
                                 VkDeviceSize size) -> Result<void> {
        auto buffer = resources_create_storage_buffer(device, size);
        if (!buffer) return Result<void>::failure(buffer.error());
        destination = std::move(buffer).value();
        return Result<void>::success();
    };
    for (auto *buffers : {&resources->image, &resources->mask}) {
        for (auto &buffer : *buffers) {
            const auto created = create_into(buffer, image_bytes.value());
            if (!created) {
                return Result<std::unique_ptr<Resources>>::failure(created.error());
            }
        }
    }
    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        const ScaleGeometry &geometry = resources->geometry[scale];
        const auto mode_output_bytes = resources_checked_size(
            static_cast<std::uint64_t>(geometry.width) * geometry.height, sizeof(std::uint32_t),
            limits.maxStorageBufferRange, "mode output");
        if (!mode_output_bytes) {
            return Result<std::unique_ptr<Resources>>::failure(mode_output_bytes.error());
        }
        const auto created = create_into(resources->mode_output[scale], mode_output_bytes.value());
        if (!created) {
            return Result<std::unique_ptr<Resources>>::failure(created.error());
        }
    }
    auto c_value_buffer = resources_create_storage_buffer(
        device, c_value_bytes.value(),
        diagnostics_mode == DetailedDiagnosticsMode::c_value_maps ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                                  : 0U);
    if (!c_value_buffer) {
        return Result<std::unique_ptr<Resources>>::failure(c_value_buffer.error());
    }
    resources->c_values = std::move(c_value_buffer).value();
    if (diagnostics_mode == DetailedDiagnosticsMode::c_value_maps) {
        for (std::uint32_t scale_index = 0U; scale_index < scale_count; ++scale_index) {
            const VkDeviceSize diagnostics_size =
                static_cast<VkDeviceSize>(resources->geometry[scale_index].value_count) *
                sizeof(std::uint32_t);
            auto diagnostics_buffer = resources_create_diagnostics_buffer(device, diagnostics_size);
            if (!diagnostics_buffer) {
                return Result<std::unique_ptr<Resources>>::failure(diagnostics_buffer.error());
            }
            resources->c_value_diagnostics_readback[scale_index] =
                std::move(diagnostics_buffer).value();
        }
    }
    auto created = create_into(resources->radix_counts, radix_count_bytes.value());
    if (!created) {
        return Result<std::unique_ptr<Resources>>::failure(created.error());
    }
    created = create_into(resources->selection_state, state_bytes.value());
    if (!created) return Result<std::unique_ptr<Resources>>::failure(created.error());
    for (auto &buffer : resources->pool_partials) {
        created = create_into(buffer, pool_bytes.value());
        if (!created) return Result<std::unique_ptr<Resources>>::failure(created.error());
    }
    auto results_buffer = Buffer::create(device, gpu::host_visible_storage(result_bytes.value()));
    if (!results_buffer) {
        return Result<std::unique_ptr<Resources>>::failure(results_buffer.error());
    }
    resources->results = std::move(results_buffer).value();
    if (!created) return Result<std::unique_ptr<Resources>>::failure(created.error());

    auto preprocess = ComputePipeline::create(device, gpu::generated::cambi_preprocess_spv, 2U,
                                              sizeof(PreprocessParameters));
    if (!preprocess) return Result<std::unique_ptr<Resources>>::failure(preprocess.error());
    resources->preprocess_pipeline = std::move(preprocess).value();

    auto mask =
        ComputePipeline::create(device, gpu::generated::cambi_mask_spv, 2U, sizeof(MaskParameters));
    if (!mask) return Result<std::unique_ptr<Resources>>::failure(mask.error());
    resources->mask_pipeline = std::move(mask).value();
    const std::array mask_bindings{BufferBinding{resources->image[0].get()},
                                   BufferBinding{resources->mask[0].get()}};
    auto updated = resources->mask_pipeline->update_buffers(0U, mask_bindings);
    if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());

    auto decimate =
        ComputePipeline::create(device, gpu::generated::cambi_decimate_spv, 4U,
                                sizeof(DecimateParameters), std::max(decimate_set_index, 1U));
    if (!decimate) return Result<std::unique_ptr<Resources>>::failure(decimate.error());
    resources->decimate_pipeline = std::move(decimate).value();

    auto mode = ComputePipeline::create(device, gpu::generated::cambi_mode_spv, 5U,
                                        sizeof(ModeParameters), scale_count);
    if (!mode) return Result<std::unique_ptr<Resources>>::failure(mode.error());
    resources->mode_pipeline = std::move(mode).value();

    std::span<const std::uint8_t> c_value_spirv = gpu::generated::cambi_c_values_rows_spv;
    if (use_uint16_histogram) {
        c_value_spirv = gpu::generated::cambi_c_values_rows_u16_spv;
    }
    auto c_value =
        ComputePipeline::create(device, c_value_spirv, 3U, sizeof(CValueParameters), scale_count);
    if (!c_value) return Result<std::unique_ptr<Resources>>::failure(c_value.error());
    resources->c_value_pipeline = std::move(c_value).value();

    for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
        const auto &geometry = resources->geometry[scale];
        if (geometry.decimate) {
            const std::uint32_t output_image = geometry.image_input_index;
            const std::uint32_t input_image = 1U - output_image;
            // Only the initial high-resolution downsample remains standalone.
            const std::uint32_t output_mask = geometry.mask_index;
            const std::uint32_t input_mask = 1U - output_mask;
            const std::array decimate_bindings{BufferBinding{resources->image[input_image].get()},
                                               BufferBinding{resources->image[output_image].get()},
                                               BufferBinding{resources->mask[input_mask].get()},
                                               BufferBinding{resources->mask[output_mask].get()}};
            updated = resources->decimate_pipeline->update_buffers(geometry.decimate_set,
                                                                   decimate_bindings);
            if (!updated) {
                return Result<std::unique_ptr<Resources>>::failure(updated.error());
            }
        }
        const std::array mode_bindings{
            BufferBinding{resources->image[geometry.image_input_index].get()},
            BufferBinding{resources->mode_output[scale].get()},
            BufferBinding{resources->mask[geometry.mask_index].get()},
            BufferBinding{resources->image[1U - geometry.image_input_index].get()},
            BufferBinding{resources->mask[1U - geometry.mask_index].get()}};
        updated = resources->mode_pipeline->update_buffers(geometry.mode_set, mode_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
        const std::array c_value_bindings{BufferBinding{resources->mode_output[scale].get()},
                                          BufferBinding{resources->c_values.get()},
                                          BufferBinding{resources->selection_state.get()}};
        updated =
            resources->c_value_pipeline->update_buffers(geometry.c_value_set, c_value_bindings);
        if (!updated) {
            return Result<std::unique_ptr<Resources>>::failure(updated.error());
        }
    }

    auto radix_init = ComputePipeline::create(device, gpu::generated::cambi_radix_init_spv, 2U,
                                              sizeof(RadixInitParameters));
    if (!radix_init) {
        return Result<std::unique_ptr<Resources>>::failure(radix_init.error());
    }
    resources->radix_init_pipeline = std::move(radix_init).value();
    const std::array radix_init_bindings{BufferBinding{resources->selection_state.get()},
                                         BufferBinding{resources->radix_counts.get()}};
    updated = resources->radix_init_pipeline->update_buffers(0U, radix_init_bindings);
    if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());

    auto radix_histogram =
        resources->radix_histogram_uses_2d_fallback
            ? ComputePipeline::create(device, gpu::generated::cambi_radix_histogram_2d_spv, 4U,
                                      sizeof(RadixHistogram2DParameters))
            : ComputePipeline::create(device, gpu::generated::cambi_radix_histogram_spv, 4U,
                                      sizeof(RadixHistogramParameters));
    if (!radix_histogram) {
        return Result<std::unique_ptr<Resources>>::failure(radix_histogram.error());
    }
    resources->radix_histogram_pipeline = std::move(radix_histogram).value();
    const std::array radix_histogram_bindings{BufferBinding{resources->c_values.get()},
                                              BufferBinding{resources->selection_state.get()},
                                              BufferBinding{resources->radix_counts.get()},
                                              BufferBinding{resources->pool_partials[0].get()}};
    updated = resources->radix_histogram_pipeline->update_buffers(0U, radix_histogram_bindings);
    if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());

    auto radix_update = ComputePipeline::create(device, gpu::generated::cambi_radix_update_spv, 3U,
                                                sizeof(RadixUpdateParameters));
    if (!radix_update) {
        return Result<std::unique_ptr<Resources>>::failure(radix_update.error());
    }
    resources->radix_update_pipeline = std::move(radix_update).value();
    const std::array radix_update_bindings{BufferBinding{resources->radix_counts.get()},
                                           BufferBinding{resources->selection_state.get()},
                                           BufferBinding{resources->pool_partials[0].get()}};
    updated = resources->radix_update_pipeline->update_buffers(0U, radix_update_bindings);
    if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());

    auto pool_reduce = ComputePipeline::create(device, gpu::generated::cambi_pool_reduce_spv, 2U,
                                               sizeof(PoolReduceParameters), 2U);
    if (!pool_reduce) {
        return Result<std::unique_ptr<Resources>>::failure(pool_reduce.error());
    }
    resources->pool_reduce_pipeline = std::move(pool_reduce).value();
    for (std::uint32_t index = 0; index < 2U; ++index) {
        const std::array bindings{BufferBinding{resources->pool_partials[index].get()},
                                  BufferBinding{resources->pool_partials[1U - index].get()}};
        updated = resources->pool_reduce_pipeline->update_buffers(index, bindings);
        if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    auto pool_finalize = ComputePipeline::create(device, gpu::generated::cambi_pool_finalize_spv,
                                                 3U, sizeof(PoolFinalizeParameters), 2U);
    if (!pool_finalize) {
        return Result<std::unique_ptr<Resources>>::failure(pool_finalize.error());
    }
    resources->pool_finalize_pipeline = std::move(pool_finalize).value();
    for (std::uint32_t index = 0; index < 2U; ++index) {
        const std::array bindings{BufferBinding{resources->pool_partials[index].get()},
                                  BufferBinding{resources->selection_state.get()},
                                  BufferBinding{resources->results.get()}};
        updated = resources->pool_finalize_pipeline->update_buffers(index, bindings);
        if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());
    }

    auto combine = ComputePipeline::create(device, gpu::generated::cambi_combine_spv, 1U,
                                           sizeof(CombineParameters));
    if (!combine) return Result<std::unique_ptr<Resources>>::failure(combine.error());
    resources->combine_pipeline = std::move(combine).value();
    const std::array result_binding{BufferBinding{resources->results.get()}};
    updated = resources->combine_pipeline->update_buffers(0U, result_binding);
    if (!updated) return Result<std::unique_ptr<Resources>>::failure(updated.error());

    auto commands = CommandContext::create(device);
    if (!commands) return Result<std::unique_ptr<Resources>>::failure(commands.error());
    resources->commands = std::move(commands).value();
    auto prefix_commands = CommandContext::create(device);
    if (!prefix_commands) {
        return Result<std::unique_ptr<Resources>>::failure(prefix_commands.error());
    }
    resources->prefix_commands = std::move(prefix_commands).value();

    const auto gpu_timestamp_queries = resources_initialize_gpu_timestamp_queries(*resources);
    if (!gpu_timestamp_queries) {
        return Result<std::unique_ptr<Resources>>::failure(gpu_timestamp_queries.error());
    }
    return Result<std::unique_ptr<Resources>>::success(std::move(resources));
}

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi::vulkan {
namespace {

using gpu::BufferBinding;
using gpu::calculate_timestamp_duration;
using gpu::CommandContext;
using gpu::CommandRecordingGuard;
using gpu::DeviceView;
using gpu::DispatchSize;
using gpu::TimestampQueryPool;

using CambiGpuStage = GpuStage;

[[nodiscard]] std::uint16_t vulkan_spatial_mask_index(std::uint32_t width, std::uint32_t height,
                                                      std::uint16_t filter_size = 7U) noexcept {
    const std::uint32_t shifted_area = (width >> 6U) * (height >> 6U);
    const int logarithm =
        shifted_area == 0U ? 0 : static_cast<int>(std::bit_width(shifted_area - 1U));
    const int value = static_cast<int>(filter_size) * filter_size + 3 * (logarithm - 11) - 1;
    return static_cast<std::uint16_t>(std::max(value, 0) >> 1);
}

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
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(value.low) |
                                     (static_cast<std::uint64_t>(value.high) << 32U));
}

[[nodiscard]] Result<void> vulkan_validate_luma(const video::GpuPlaneView &plane,
                                                const FrameDescriptor &descriptor) {
    if (!video::is_compatible_gpu_plane(plane, descriptor.width, descriptor.height,
                                        descriptor.bit_depth)) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "CAMBI received an incompatible GPU luma plane"});
    }
    return Result<void>::success();
}

class vulkan_CambiGpuTimestampRecording final {
  public:
    vulkan_CambiGpuTimestampRecording(TimestampQueryPool *query_pool,
                                      VkCommandBuffer command_buffer,
                                      std::span<const CambiGpuStage> interval_stages,
                                      std::size_t next_interval_index = 0U,
                                      bool recording_valid = false) noexcept
        : query_pool_(query_pool), command_buffer_(command_buffer),
          interval_stages_(interval_stages), next_interval_index_(next_interval_index),
          recording_valid_(recording_valid) {}

    void reset() {
        if (query_pool_ == nullptr) return;
        const auto reset_result = query_pool_->reset_queries(
            command_buffer_, 0U, static_cast<std::uint32_t>(interval_stages_.size() * 2U));
        recording_valid_ = static_cast<bool>(reset_result);
    }

    void write_stage_start(CambiGpuStage stage) {
        if (!recording_valid_) return;
        const bool expected_stage_is_available = !interval_open_ &&
                                                 next_interval_index_ < interval_stages_.size() &&
                                                 interval_stages_[next_interval_index_] == stage;
        if (!expected_stage_is_available) {
            recording_valid_ = false;
            return;
        }
        write_timestamp(static_cast<std::uint32_t>(next_interval_index_ * 2U));
        if (recording_valid_) interval_open_ = true;
    }

    void write_stage_end(CambiGpuStage stage) {
        if (!recording_valid_) return;
        const bool expected_stage_is_open = interval_open_ &&
                                            next_interval_index_ < interval_stages_.size() &&
                                            interval_stages_[next_interval_index_] == stage;
        if (!expected_stage_is_open) {
            recording_valid_ = false;
            return;
        }
        write_timestamp(static_cast<std::uint32_t>(next_interval_index_ * 2U + 1U));
        if (recording_valid_) {
            interval_open_ = false;
            ++next_interval_index_;
        }
    }

    [[nodiscard]] bool query_pool_available() const noexcept { return query_pool_ != nullptr; }

    [[nodiscard]] bool complete() const noexcept {
        return recording_valid_ && !interval_open_ &&
               next_interval_index_ == interval_stages_.size();
    }

    [[nodiscard]] bool complete_through(std::size_t one_past_interval_index) const noexcept {
        return recording_valid_ && !interval_open_ &&
               next_interval_index_ == one_past_interval_index;
    }

  private:
    void write_timestamp(std::uint32_t query_index) {
        const auto write_result = query_pool_->write_timestamp(
            command_buffer_, query_index, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        // Timestamp failures are diagnostic-only unless result collection
        // later reports device loss.
        if (!write_result) recording_valid_ = false;
    }

    TimestampQueryPool *query_pool_{};
    VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
    std::span<const CambiGpuStage> interval_stages_;
    std::size_t next_interval_index_{};
    bool interval_open_{};
    bool recording_valid_{};
};

void vulkan_record_gpu_timestamp_frame_failure(Telemetry &telemetry) noexcept {
    telemetry.record_parent_failure();
    for (std::uint8_t stage_index = 0U;
         stage_index < static_cast<std::uint8_t>(CambiGpuStage::count); ++stage_index) {
        telemetry.record_stage_failure(static_cast<CambiGpuStage>(stage_index));
    }
    for (std::uint32_t scale_index = 0U; scale_index < scale_count; ++scale_index) {
        telemetry.record_c_value_scale_failure(scale_index);
    }
}

[[nodiscard]] Result<void>
vulkan_collect_gpu_timestamp_results(Resources &resources, bool query_pool_available,
                                     bool recording_valid,
                                     std::chrono::nanoseconds submission_host_elapsed) {
    Telemetry *const telemetry = resources.telemetry;
    if (telemetry == nullptr || !telemetry->enabled()) {
        return Result<void>::success();
    }

    telemetry->record_submission(query_pool_available);
    if (!query_pool_available) {
        return Result<void>::success();
    }
    if (!recording_valid || resources.gpu_timestamp_query_pool == nullptr) {
        vulkan_record_gpu_timestamp_frame_failure(*telemetry);
        return Result<void>::success();
    }

    TimestampQueryPool &query_pool = *resources.gpu_timestamp_query_pool;
    const std::uint32_t query_count =
        static_cast<std::uint32_t>(resources.gpu_timestamp_interval_stages.size() * 2U);
    const auto query_results_result = query_pool.read_results(0U, query_count);
    if (!query_results_result) {
        const Error &read_error = query_results_result.error();
        vulkan_record_gpu_timestamp_frame_failure(*telemetry);
        if (read_error.code == ErrorCode::device_lost) {
            return Result<void>::failure(read_error);
        }
        return Result<void>::success();
    }

    const auto &query_results = query_results_result.value();
    constexpr std::size_t split_boundary_query_count = 4U;
    if (query_results.size() != static_cast<std::size_t>(query_count) ||
        query_results.size() < split_boundary_query_count) {
        vulkan_record_gpu_timestamp_frame_failure(*telemetry);
        return Result<void>::success();
    }

    const long double maximum_elapsed_nanoseconds =
        static_cast<long double>(submission_host_elapsed.count());
    const auto latency_span_duration =
        calculate_timestamp_duration(query_results.front(), query_results.back(),
                                     query_pool.capabilities(), maximum_elapsed_nanoseconds);
    const auto prefix_duration =
        calculate_timestamp_duration(query_results[0U], query_results[1U],
                                     query_pool.capabilities(), maximum_elapsed_nanoseconds);
    const auto prefix_to_suffix_gap_duration =
        calculate_timestamp_duration(query_results[1U], query_results[2U],
                                     query_pool.capabilities(), maximum_elapsed_nanoseconds);
    const auto suffix_duration =
        calculate_timestamp_duration(query_results[2U], query_results.back(),
                                     query_pool.capabilities(), maximum_elapsed_nanoseconds);
    const bool split_parent_durations_are_available = latency_span_duration && prefix_duration &&
                                                      prefix_to_suffix_gap_duration &&
                                                      suffix_duration;
    if (split_parent_durations_are_available) {
        const long double active_duration_nanoseconds =
            prefix_duration.value().nanoseconds + suffix_duration.value().nanoseconds;
        telemetry->record_parent_success(active_duration_nanoseconds,
                                         latency_span_duration.value().nanoseconds,
                                         prefix_to_suffix_gap_duration.value().nanoseconds);
        telemetry->record_timeline(query_results[0U].value, query_results[1U].value,
                                   query_pool.capabilities().timestamp_valid_bits,
                                   query_pool.capabilities().timestamp_period_nanoseconds,
                                   crossvmaf::gpu::TimelineSegment::prefix);
        telemetry->record_timeline(query_results[2U].value, query_results.back().value,
                                   query_pool.capabilities().timestamp_valid_bits,
                                   query_pool.capabilities().timestamp_period_nanoseconds,
                                   crossvmaf::gpu::TimelineSegment::suffix);
    } else {
        telemetry->record_parent_failure();
    }

    constexpr std::size_t stage_count = static_cast<std::size_t>(CambiGpuStage::count);
    std::array<long double, stage_count> stage_totals{};
    std::array<bool, stage_count> stage_samples_valid{};
    stage_samples_valid.fill(true);
    // Per-scale c-value attribution exists only while diagnostics keeps the
    // dispatches serialized. In the grouped production shape the five scales
    // are co-resident inside one interval, so they are reported as skipped
    // rather than fabricated from an interval that does not bound them.
    const bool per_scale_c_value_intervals =
        resources.diagnostics_mode == DetailedDiagnosticsMode::c_value_maps;
    std::uint32_t c_value_scale_index = 0U;
    for (std::size_t interval_index = 0U;
         interval_index < resources.gpu_timestamp_interval_stages.size(); ++interval_index) {
        const CambiGpuStage stage = resources.gpu_timestamp_interval_stages[interval_index];
        const std::size_t stage_index = static_cast<std::size_t>(stage);
        const std::size_t start_query_index = interval_index * 2U;
        const auto interval_duration = calculate_timestamp_duration(
            query_results[start_query_index], query_results[start_query_index + 1U],
            query_pool.capabilities(), maximum_elapsed_nanoseconds);
        if (interval_duration) {
            stage_totals[stage_index] += interval_duration.value().nanoseconds;
        } else {
            stage_samples_valid[stage_index] = false;
        }
        if (stage == CambiGpuStage::c_value && per_scale_c_value_intervals) {
            if (c_value_scale_index < scale_count) {
                if (interval_duration) {
                    telemetry->record_c_value_scale_success(c_value_scale_index,
                                                            interval_duration.value().nanoseconds);
                } else {
                    telemetry->record_c_value_scale_failure(c_value_scale_index);
                }
            }
            ++c_value_scale_index;
        }
    }

    if (!per_scale_c_value_intervals) {
        for (; c_value_scale_index < scale_count; ++c_value_scale_index) {
            telemetry->record_c_value_scale_skip(c_value_scale_index);
        }
    } else if (c_value_scale_index != scale_count) {
        stage_samples_valid[static_cast<std::size_t>(CambiGpuStage::c_value)] = false;
        for (; c_value_scale_index < scale_count; ++c_value_scale_index) {
            telemetry->record_c_value_scale_failure(c_value_scale_index);
        }
    }

    for (std::size_t stage_index = 0U; stage_index < stage_count; ++stage_index) {
        const CambiGpuStage stage = static_cast<CambiGpuStage>(stage_index);
        if (stage_samples_valid[stage_index]) {
            telemetry->record_stage_success(stage, stage_totals[stage_index]);
        } else {
            telemetry->record_stage_failure(stage);
        }
    }
    return Result<void>::success();
}

[[nodiscard]] Result<void> vulkan_prepare_preprocess_input(Resources &resources,
                                                           const video::GpuFrameView &distorted) {
    const auto luma_validation = vulkan_validate_luma(distorted.planes[0], resources.descriptor);
    if (!luma_validation) {
        return Result<void>::failure(luma_validation.error());
    }

    const std::array preprocess_bindings{BufferBinding{distorted.planes[0].buffer,
                                                       distorted.planes[0].offset_bytes,
                                                       distorted.planes[0].size_bytes},
                                         BufferBinding{resources.image[0].get()}};
    return resources.preprocess_pipeline->update_buffers(0U, preprocess_bindings);
}

[[nodiscard]] Result<void> vulkan_record_preprocess_stage(
    Resources &resources, CommandContext &commands, VkCommandBuffer command_buffer,
    vulkan_CambiGpuTimestampRecording &timestamp_recording, const video::GpuFrameView &distorted) {
    commands.memory_barrier(VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    timestamp_recording.write_stage_start(CambiGpuStage::preprocess);

    const PreprocessParameters preprocess_parameters{
        resources.descriptor.width, resources.descriptor.height, distorted.planes[0].stride_bytes,
        resources.descriptor.bit_depth};
    const auto recorded = resources.preprocess_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(preprocess_parameters),
        {vulkan_divide_round_up(resources.descriptor.width, tile_size),
         vulkan_divide_round_up(resources.descriptor.height, tile_size), 1U});
    if (!recorded) return Result<void>::failure(recorded.error());

    commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT);
    timestamp_recording.write_stage_end(CambiGpuStage::preprocess);
    return Result<void>::success();
}

[[nodiscard]] Result<void>
vulkan_record_analysis_stages(Resources &resources, CommandContext &commands,
                              VkCommandBuffer command_buffer,
                              vulkan_CambiGpuTimestampRecording &timestamp_recording) {
    const MaskParameters mask_parameters{
        resources.descriptor.width, resources.descriptor.height,
        vulkan_spatial_mask_index(resources.descriptor.width, resources.descriptor.height)};
    timestamp_recording.write_stage_start(CambiGpuStage::mask);
    auto recorded = resources.mask_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(mask_parameters),
        {vulkan_divide_round_up(resources.descriptor.width, tile_size),
         vulkan_divide_round_up(resources.descriptor.height, tile_size), 1U});
    if (!recorded) return Result<void>::failure(recorded.error());
    commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT);
    timestamp_recording.write_stage_end(CambiGpuStage::mask);

    // radix_init touches only selection_state and radix_counts, which nothing
    // before the first radix histogram pass reads or writes. Issuing it here
    // lets it ride the dense decimate/mode section instead of owning a
    // serialized point (and its enclosing barriers) right before the pass.
    // The chain's later barriers still order it ahead of the histogram pass.
    RadixInitParameters init_parameters{};
    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        init_parameters.top_counts[scale] = resources.geometry[scale].top_count;
    }
    recorded = resources.radix_init_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(init_parameters), {scale_count, 1U, 1U});
    if (!recorded) return Result<void>::failure(recorded.error());

    std::uint32_t previous_width = resources.descriptor.width;
    for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
        const auto &geometry = resources.geometry[scale];
        if (geometry.decimate) {
            timestamp_recording.write_stage_start(CambiGpuStage::decimate);
            const DecimateParameters parameters{previous_width, geometry.width, geometry.height};
            recorded = resources.decimate_pipeline->record(command_buffer, geometry.decimate_set,
                                                           vulkan_bytes_of(parameters),
                                                           geometry.two_dimensional_dispatch);
            if (!recorded) return Result<void>::failure(recorded.error());
            commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            timestamp_recording.write_stage_end(CambiGpuStage::decimate);
        }
        previous_width = geometry.width;

        const ModeParameters mode_parameters{geometry.width, geometry.height,
                                             scale + 1U < scale_count ? 1U : 0U};
        timestamp_recording.write_stage_start(CambiGpuStage::mode);
        recorded = resources.mode_pipeline->record(command_buffer, geometry.mode_set,
                                                   vulkan_bytes_of(mode_parameters),
                                                   geometry.two_dimensional_dispatch);
        if (!recorded) return Result<void>::failure(recorded.error());
        commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT);
        timestamp_recording.write_stage_end(CambiGpuStage::mode);

        // Diagnostics mode reads each scale's c-values back individually, so it
        // keeps the serialized per-scale shape. Production dispatches the whole
        // group after the loop instead.
        if (resources.diagnostics_mode == DetailedDiagnosticsMode::c_value_maps) {
            const CValueParameters c_value_parameters{
                geometry.width,          geometry.height,
                resources.window_size,   geometry.c_value_segment_rows,
                geometry.c_value_offset, scale * selection_state_word_count};
            timestamp_recording.write_stage_start(CambiGpuStage::c_value);
            recorded = resources.c_value_pipeline->record(command_buffer, geometry.c_value_set,
                                                          vulkan_bytes_of(c_value_parameters),
                                                          geometry.c_value_dispatch);
            if (!recorded) return Result<void>::failure(recorded.error());
            commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            timestamp_recording.write_stage_end(CambiGpuStage::c_value);
        }

        if (resources.diagnostics_mode == DetailedDiagnosticsMode::c_value_maps) {
            gpu::Buffer *const diagnostics_buffer =
                resources.c_value_diagnostics_readback[scale].get();
            if (diagnostics_buffer == nullptr) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state, "CAMBI C-value diagnostics buffer is unavailable"});
            }
            commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_ACCESS_TRANSFER_READ_BIT);
            commands.copy_buffer(
                *resources.c_values, *diagnostics_buffer,
                static_cast<VkDeviceSize>(geometry.c_value_offset) * sizeof(std::uint32_t), 0U,
                static_cast<VkDeviceSize>(geometry.value_count) * sizeof(std::uint32_t));
            commands.memory_barrier(VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_TRANSFER_READ_BIT,
                                    VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        }
    }

    // Every scale now reads its own mode output and writes a disjoint slice of
    // c_values, so the five dispatches carry no mutual dependency. Issuing them
    // as one barrier-free group lets the small scales, which alone occupy far
    // too few workgroups to fill the device, run alongside the large ones.
    if (resources.diagnostics_mode != DetailedDiagnosticsMode::c_value_maps) {
        timestamp_recording.write_stage_start(CambiGpuStage::c_value);
        for (std::uint32_t scale = 0; scale < scale_count; ++scale) {
            const auto &geometry = resources.geometry[scale];
            const CValueParameters c_value_parameters{
                geometry.width,          geometry.height,
                resources.window_size,   geometry.c_value_segment_rows,
                geometry.c_value_offset, scale * selection_state_word_count};
            recorded = resources.c_value_pipeline->record(command_buffer, geometry.c_value_set,
                                                          vulkan_bytes_of(c_value_parameters),
                                                          geometry.c_value_dispatch);
            if (!recorded) return Result<void>::failure(recorded.error());
        }
        commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT);
        timestamp_recording.write_stage_end(CambiGpuStage::c_value);
    }

    RadixHistogramParameters histogram_parameters{};
    std::uint32_t radix_group_count = 0U;
    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        const ScaleGeometry &geometry = resources.geometry[scale];
        histogram_parameters.value_offsets[scale] = geometry.c_value_offset;
        histogram_parameters.value_counts[scale] = geometry.value_count;
        histogram_parameters.group_offsets[scale] = radix_group_count;
        histogram_parameters.partial_offsets[scale] = geometry.pool_partial_offset;
        radix_group_count += geometry.radix_group_count;
    }

    timestamp_recording.write_stage_start(CambiGpuStage::radix);
    for (std::uint32_t pass = 0U; pass < radix_pass_count; ++pass) {
        const std::uint32_t digit_shift = 32U - radix_bits * (pass + 1U);
        histogram_parameters.digit_shift = digit_shift;
        if (resources.radix_histogram_uses_2d_fallback) {
            const RadixHistogram2DParameters fallback_parameters{
                histogram_parameters.value_offsets,
                histogram_parameters.value_counts,
                histogram_parameters.group_offsets,
                histogram_parameters.partial_offsets,
                digit_shift,
                resources.radix_histogram_dispatch.x,
                radix_group_count,
            };
            recorded = resources.radix_histogram_pipeline->record(
                command_buffer, 0U, vulkan_bytes_of(fallback_parameters),
                resources.radix_histogram_dispatch);
        } else {
            recorded = resources.radix_histogram_pipeline->record(
                command_buffer, 0U, vulkan_bytes_of(histogram_parameters),
                resources.radix_histogram_dispatch);
        }
        if (!recorded) return Result<void>::failure(recorded.error());
        commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT);

        const RadixUpdateParameters update_parameters{digit_shift,
                                                      histogram_parameters.partial_offsets};
        recorded = resources.radix_update_pipeline->record(
            command_buffer, 0U, vulkan_bytes_of(update_parameters), {scale_count, 1U, 1U});
        if (!recorded) return Result<void>::failure(recorded.error());
        commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT);
    }
    timestamp_recording.write_stage_end(CambiGpuStage::radix);

    // The final radix histogram pass writes exact per-workgroup sum/count
    // partials for values above the selected threshold. The update pass adds
    // the bins above the selected final digit to each scale's first partial,
    // so pooling continues directly with the grouped reduction rounds.
    std::uint32_t reduction_round_count = 0U;
    for (const ScaleGeometry &geometry : resources.geometry) {
        reduction_round_count = std::max<std::uint32_t>(
            reduction_round_count,
            static_cast<std::uint32_t>(geometry.reduction_input_counts.size()));
    }

    timestamp_recording.write_stage_start(CambiGpuStage::pool_reduce);
    for (std::uint32_t round = 0U; round < reduction_round_count; ++round) {
        for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
            const ScaleGeometry &geometry = resources.geometry[scale];
            if (round >= geometry.reduction_input_counts.size()) continue;
            const PoolReduceParameters reduce_parameters{geometry.reduction_input_counts[round],
                                                         geometry.pool_partial_offset};
            const DispatchSize dispatch{
                vulkan_divide_round_up(reduce_parameters.input_count, linear_group_size), 1U, 1U};
            recorded = resources.pool_reduce_pipeline->record(
                command_buffer, round & 1U, vulkan_bytes_of(reduce_parameters), dispatch);
            if (!recorded) return Result<void>::failure(recorded.error());
        }
        commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_ACCESS_SHADER_READ_BIT);
    }

    for (std::uint32_t scale = 0U; scale < scale_count; ++scale) {
        const ScaleGeometry &geometry = resources.geometry[scale];
        const PoolFinalizeParameters finalize_parameters{scale, geometry.pool_partial_offset};
        recorded = resources.pool_finalize_pipeline->record(
            command_buffer, geometry.final_partial_index, vulkan_bytes_of(finalize_parameters),
            {1U, 1U, 1U});
        if (!recorded) return Result<void>::failure(recorded.error());
    }
    commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                            VK_ACCESS_SHADER_READ_BIT);
    timestamp_recording.write_stage_end(CambiGpuStage::pool_reduce);

    const CombineParameters combine_parameters{resources.window_size,
                                               vulkan_split_i64(resources.config.maximum.raw())};
    timestamp_recording.write_stage_start(CambiGpuStage::combine);
    recorded = resources.combine_pipeline->record(
        command_buffer, 0U, vulkan_bytes_of(combine_parameters), {1U, 1U, 1U});
    if (!recorded) return Result<void>::failure(recorded.error());
    timestamp_recording.write_stage_end(CambiGpuStage::combine);

    if (resources.diagnostics_mode == DetailedDiagnosticsMode::c_value_maps) {
        commands.memory_barrier(
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_HOST_READ_BIT);
    } else {
        commands.memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    }
    return Result<void>::success();
}

} // namespace

Executor::Executor(std::unique_ptr<Resources> resources) noexcept
    : resources_(std::move(resources)) {
}
Executor::~Executor() = default;

Result<std::unique_ptr<Executor>> Executor::create(DeviceView device, FrameDescriptor descriptor,
                                                   Config config, Telemetry *telemetry,
                                                   DetailedDiagnosticsMode diagnostics_mode) {
    if (!device.complete()) {
        return Result<std::unique_ptr<Executor>>::failure(
            {ErrorCode::backend_error, "CAMBI Vulkan executor received an incomplete device"});
    }
    const auto descriptor_validation = video::validate_frame_layout(descriptor);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<Executor>>::failure(descriptor_validation.error());
    }
    const auto metric_limit = metrics::cambi::validate_input(descriptor);
    if (!metric_limit) {
        return Result<std::unique_ptr<Executor>>::failure(metric_limit.error());
    }
    auto resources = Resources::create(device, descriptor, config, telemetry, diagnostics_mode);
    if (!resources) {
        return Result<std::unique_ptr<Executor>>::failure(resources.error());
    }
    return Result<std::unique_ptr<Executor>>::success(
        std::unique_ptr<Executor>(new Executor(std::move(resources).value())));
}

Result<DetailedResult> Executor::compute_detailed(const video::GpuFrameView &distorted) {
    const auto recorded = record(distorted);
    if (!recorded) {
        return Result<DetailedResult>::failure(recorded.error());
    }
    const auto retirement = resources_->commands->submit_finished_and_wait();
    if (!retirement) {
        // CommandContext only returns to retired when submission never
        // reached vkQueueSubmit. Clear CAMBI's matching outer pending flag in
        // that retry-safe case; submitted/failed commands remain fail-stop.
        cancel();
        return Result<DetailedResult>::failure(retirement.error());
    }
    result_state_.mark_submission_observed();
    return readback(retirement.value().host_elapsed);
}

Result<VkCommandBuffer> Executor::record(const video::GpuFrameView &distorted) {
    if (result_state_.pending()) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "CAMBI already has a recorded result pending retirement"});
    }
    const auto prepared = vulkan_prepare_preprocess_input(*resources_, distorted);
    if (!prepared) {
        return Result<VkCommandBuffer>::failure(prepared.error());
    }

    auto begun = resources_->commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    vulkan_CambiGpuTimestampRecording timestamp_recording(
        resources_->gpu_timestamp_query_pool.get(), command_buffer,
        resources_->gpu_timestamp_interval_stages);
    timestamp_recording.reset();
    const auto preprocess_recorded = vulkan_record_preprocess_stage(
        *resources_, *resources_->commands, command_buffer, timestamp_recording, distorted);
    if (!preprocess_recorded) {
        return Result<VkCommandBuffer>::failure(preprocess_recorded.error());
    }

    const auto analysis_recorded = vulkan_record_analysis_stages(
        *resources_, *resources_->commands, command_buffer, timestamp_recording);
    if (!analysis_recorded) {
        return Result<VkCommandBuffer>::failure(analysis_recorded.error());
    }

    const auto finished = resources_->commands->finish();
    if (!finished) {
        return Result<VkCommandBuffer>::failure(finished.error());
    }
    recording_guard.release();
    result_state_.mark_recorded();
    pending_timestamp_query_pool_available_ = timestamp_recording.query_pool_available();
    pending_timestamp_recording_valid_ = timestamp_recording.complete();
    return finished;
}

Result<VkCommandBuffer> Executor::record_prefix(const video::GpuFrameView &distorted) {
    if (prefix_state_.pending() || result_state_.pending()) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "CAMBI already has a recorded prefix or suffix pending"});
    }
    const auto prepared = vulkan_prepare_preprocess_input(*resources_, distorted);
    if (!prepared) {
        return Result<VkCommandBuffer>::failure(prepared.error());
    }

    auto begun = resources_->prefix_commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->prefix_commands);
    vulkan_CambiGpuTimestampRecording timestamp_recording(
        resources_->gpu_timestamp_query_pool.get(), command_buffer,
        resources_->gpu_timestamp_interval_stages);
    timestamp_recording.reset();
    const auto preprocess_recorded = vulkan_record_preprocess_stage(
        *resources_, *resources_->prefix_commands, command_buffer, timestamp_recording, distorted);
    if (!preprocess_recorded) {
        return Result<VkCommandBuffer>::failure(preprocess_recorded.error());
    }

    const auto finished = resources_->prefix_commands->finish();
    if (!finished) {
        return Result<VkCommandBuffer>::failure(finished.error());
    }
    recording_guard.release();
    prefix_state_.mark_recorded();
    pending_timestamp_query_pool_available_ = timestamp_recording.query_pool_available();
    pending_prefix_timestamp_recording_valid_ = timestamp_recording.complete_through(1U);
    pending_prefix_submission_host_elapsed_ = std::chrono::nanoseconds::zero();
    return finished;
}

Result<VkCommandBuffer> Executor::record_suffix() {
    if (!prefix_state_.pending() || result_state_.pending()) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state,
             "CAMBI suffix requires a pending prefix and no suffix in flight"});
    }
    auto begun = resources_->commands->begin();
    if (!begun) return Result<VkCommandBuffer>::failure(begun.error());
    const VkCommandBuffer command_buffer = begun.value();
    CommandRecordingGuard recording_guard(*resources_->commands);
    vulkan_CambiGpuTimestampRecording timestamp_recording(
        resources_->gpu_timestamp_query_pool.get(), command_buffer,
        resources_->gpu_timestamp_interval_stages, 1U, pending_prefix_timestamp_recording_valid_);
    resources_->commands->memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                         VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

    const auto analysis_recorded = vulkan_record_analysis_stages(
        *resources_, *resources_->commands, command_buffer, timestamp_recording);
    if (!analysis_recorded) {
        return Result<VkCommandBuffer>::failure(analysis_recorded.error());
    }

    const auto finished = resources_->commands->finish();
    if (!finished) {
        return Result<VkCommandBuffer>::failure(finished.error());
    }
    recording_guard.release();
    result_state_.mark_recorded();
    pending_timestamp_query_pool_available_ = timestamp_recording.query_pool_available();
    pending_timestamp_recording_valid_ = timestamp_recording.complete();
    return finished;
}

Result<void> Executor::mark_submitted(gpu::SubmissionTicket ticket) {
    if (!result_state_.pending()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "CAMBI has no recorded result to submit"});
    }
    const auto command_submitted = resources_->commands->mark_submitted(ticket);
    if (command_submitted) result_state_.mark_submission_observed();
    return command_submitted;
}

Result<void> Executor::mark_prefix_submitted(gpu::SubmissionTicket ticket) {
    if (!prefix_state_.pending()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "CAMBI has no recorded prefix to submit"});
    }
    const auto command_submitted = resources_->prefix_commands->mark_submitted(ticket);
    if (command_submitted) prefix_state_.mark_submission_observed();
    return command_submitted;
}

Result<void> Executor::retire_prefix(gpu::SubmissionTicket ticket,
                                     std::chrono::nanoseconds submission_host_elapsed) {
    const auto command_retired = resources_->prefix_commands->mark_retired(ticket);
    if (!command_retired) return command_retired;
    pending_prefix_submission_host_elapsed_ = submission_host_elapsed;
    prefix_state_.clear();
    return Result<void>::success();
}

Result<DetailedResult> Executor::retire(gpu::SubmissionTicket ticket,
                                        std::chrono::nanoseconds submission_host_elapsed) {
    const auto command_retired = resources_->commands->mark_retired(ticket);
    if (!command_retired) {
        return Result<DetailedResult>::failure(command_retired.error());
    }
    return readback(submission_host_elapsed);
}

void Executor::cancel() noexcept {
    resources_->commands->cancel();
    result_state_.rollback_after_cancel(resources_->commands->state());
    if (resources_->prefix_commands != nullptr) {
        resources_->prefix_commands->cancel();
        prefix_state_.rollback_after_cancel(resources_->prefix_commands->state());
    }
    if (!result_state_.pending()) {
        pending_timestamp_query_pool_available_ = false;
        pending_prefix_timestamp_recording_valid_ = false;
        pending_timestamp_recording_valid_ = false;
        pending_prefix_submission_host_elapsed_ = std::chrono::nanoseconds::zero();
    }
}

Result<DetailedResult> Executor::readback(std::chrono::nanoseconds submission_host_elapsed) {
    if (!result_state_.pending() ||
        resources_->commands->state() != gpu::CommandBufferState::retired) {
        return Result<DetailedResult>::failure(
            {ErrorCode::invalid_state, "CAMBI result is not ready for host readback"});
    }
    const std::chrono::nanoseconds maximum_duration = std::chrono::nanoseconds::max();
    const std::chrono::nanoseconds timestamp_upper_bound =
        pending_prefix_submission_host_elapsed_ > maximum_duration - submission_host_elapsed
            ? maximum_duration
            : pending_prefix_submission_host_elapsed_ + submission_host_elapsed;
    if (resources_->telemetry != nullptr) {
        resources_->telemetry->record_submission_latency(timestamp_upper_bound);
    }
    const auto timestamp_result = vulkan_collect_gpu_timestamp_results(
        *resources_, pending_timestamp_query_pool_available_, pending_timestamp_recording_valid_,
        timestamp_upper_bound);
    if (!timestamp_result) {
        return Result<DetailedResult>::failure(timestamp_result.error());
    }
    std::array<U64Words, 6> raw_results{};
    const auto read = resources_->results->read(std::as_writable_bytes(std::span(raw_results)));
    if (!read) return Result<DetailedResult>::failure(read.error());
    DetailedResult output{};
    for (std::size_t scale = 0; scale < output.scales.size(); ++scale) {
        output.scales[scale] = numeric::Q32_32::from_raw(vulkan_join_i64(raw_results[scale]));
    }
    output.value = Q32Value{vulkan_join_i64(raw_results[5U])};
    output.window_size = resources_->window_size;
    output.high_res_speedup = resources_->high_res_speedup;
    if (resources_->diagnostics_mode == DetailedDiagnosticsMode::c_value_maps) {
        output.diagnostics.emplace();
        for (std::uint32_t scale_index = 0U; scale_index < scale_count; ++scale_index) {
            const ScaleGeometry &geometry = resources_->geometry[scale_index];
            CValueMap &c_value_map = output.diagnostics->c_value_maps[scale_index];
            c_value_map.width = geometry.width;
            c_value_map.height = geometry.height;
            c_value_map.values.resize(geometry.value_count);
            gpu::Buffer *const diagnostics_buffer =
                resources_->c_value_diagnostics_readback[scale_index].get();
            if (diagnostics_buffer == nullptr) {
                return Result<DetailedResult>::failure(
                    {ErrorCode::invalid_state, "CAMBI C-value diagnostics buffer is unavailable"});
            }
            const auto diagnostics_read =
                diagnostics_buffer->read(std::as_writable_bytes(std::span(c_value_map.values)));
            if (!diagnostics_read) {
                return Result<DetailedResult>::failure(diagnostics_read.error());
            }
        }
    }
    result_state_.clear();
    pending_timestamp_query_pool_available_ = false;
    pending_prefix_timestamp_recording_valid_ = false;
    pending_timestamp_recording_valid_ = false;
    pending_prefix_submission_host_elapsed_ = std::chrono::nanoseconds::zero();
    return Result<DetailedResult>::success(output);
}

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi::vulkan {
namespace {

constexpr std::uint32_t telemetry_latency_span_member_id = 0U;
constexpr std::uint32_t telemetry_prefix_to_suffix_gap_member_id = 1U;

[[nodiscard]] constexpr std::uint32_t telemetry_stage_id(GpuStage stage) noexcept {
    return static_cast<std::uint32_t>(stage);
}

} // namespace

const char *gpu_stage_name(GpuStage stage) noexcept {
    switch (stage) {
    case GpuStage::preprocess:
        return "preprocess";
    case GpuStage::mask:
        return "mask";
    case GpuStage::decimate:
        return "decimate";
    case GpuStage::mode:
        return "mode";
    case GpuStage::c_value:
        return "c_value";
    case GpuStage::radix:
        return "radix";
    case GpuStage::pool_reduce:
        return "pool_reduce";
    case GpuStage::combine:
        return "combine";
    case GpuStage::count:
        break;
    }
    return "unknown";
}

Telemetry::Telemetry(gpu::TelemetryCollector &collector)
    : collector_(&collector), operation_(collector.register_gpu_operation()),
      parent_detail_series_(collector.register_duration_series(2U)),
      stage_series_(collector.register_duration_series(gpu_stage_count)),
      c_value_scale_series_(collector.register_duration_series(scale_count)) {
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

void Telemetry::record_submission(bool timestamp_available) noexcept {
    collector_->record_gpu_submission(operation_, timestamp_available);
}

void Telemetry::record_parent_success(long double active_duration_nanoseconds,
                                      long double latency_span_nanoseconds,
                                      long double prefix_to_suffix_gap_nanoseconds) noexcept {
    const bool all_durations_are_valid =
        std::isfinite(active_duration_nanoseconds) && active_duration_nanoseconds >= 0.0L &&
        std::isfinite(latency_span_nanoseconds) && latency_span_nanoseconds >= 0.0L &&
        std::isfinite(prefix_to_suffix_gap_nanoseconds) && prefix_to_suffix_gap_nanoseconds >= 0.0L;
    if (!all_durations_are_valid) {
        record_parent_failure();
        return;
    }

    collector_->record_gpu_parent(operation_, gpu::TelemetrySampleOutcome::success,
                                  active_duration_nanoseconds);
    collector_->record_duration(parent_detail_series_, telemetry_latency_span_member_id,
                                gpu::TelemetrySampleOutcome::success, latency_span_nanoseconds);
    collector_->record_duration(parent_detail_series_, telemetry_prefix_to_suffix_gap_member_id,
                                gpu::TelemetrySampleOutcome::success,
                                prefix_to_suffix_gap_nanoseconds);
}

void Telemetry::record_parent_failure() noexcept {
    collector_->record_gpu_parent(operation_, gpu::TelemetrySampleOutcome::failure);
    collector_->record_duration(parent_detail_series_, telemetry_latency_span_member_id,
                                gpu::TelemetrySampleOutcome::failure);
    collector_->record_duration(parent_detail_series_, telemetry_prefix_to_suffix_gap_member_id,
                                gpu::TelemetrySampleOutcome::failure);
}

void Telemetry::record_parent_skip() noexcept {
    collector_->record_gpu_parent(operation_, gpu::TelemetrySampleOutcome::skipped);
    collector_->record_duration(parent_detail_series_, telemetry_latency_span_member_id,
                                gpu::TelemetrySampleOutcome::skipped);
    collector_->record_duration(parent_detail_series_, telemetry_prefix_to_suffix_gap_member_id,
                                gpu::TelemetrySampleOutcome::skipped);
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

void Telemetry::configure_c_value_scale(std::uint32_t scale_index,
                                        const CValueScaleConfiguration &configuration) noexcept {
    if (!enabled() || !is_valid_scale(scale_index)) return;
    c_value_scale_configurations_[scale_index] = configuration;
    c_value_scale_configurations_[scale_index].configured = true;
}

void Telemetry::record_c_value_scale_success(std::uint32_t scale_index,
                                             long double duration_nanoseconds) noexcept {
    if (!is_valid_scale(scale_index)) return;
    collector_->record_duration(c_value_scale_series_, scale_index,
                                gpu::TelemetrySampleOutcome::success, duration_nanoseconds);
}

void Telemetry::record_c_value_scale_failure(std::uint32_t scale_index) noexcept {
    if (!is_valid_scale(scale_index)) return;
    collector_->record_duration(c_value_scale_series_, scale_index,
                                gpu::TelemetrySampleOutcome::failure);
}

void Telemetry::record_c_value_scale_skip(std::uint32_t scale_index) noexcept {
    if (!is_valid_scale(scale_index)) return;
    collector_->record_duration(c_value_scale_series_, scale_index,
                                gpu::TelemetrySampleOutcome::skipped);
}

const gpu::GpuOperationSnapshot &Telemetry::operation() const noexcept {
    return collector_->gpu_operation(operation_);
}

const gpu::GpuDurationStatistics &Telemetry::latency_span_statistics() const noexcept {
    return collector_->duration_statistics(parent_detail_series_, telemetry_latency_span_member_id);
}

const gpu::GpuDurationStatistics &Telemetry::prefix_to_suffix_gap_statistics() const noexcept {
    return collector_->duration_statistics(parent_detail_series_,
                                           telemetry_prefix_to_suffix_gap_member_id);
}

const gpu::GpuDurationStatistics &Telemetry::stage_statistics(GpuStage stage) const noexcept {
    if (!is_valid_stage(stage)) {
        static constexpr gpu::GpuDurationStatistics empty_statistics{};
        return empty_statistics;
    }
    return collector_->duration_statistics(stage_series_, telemetry_stage_id(stage));
}

const CValueScaleConfiguration &
Telemetry::c_value_scale_configuration(std::uint32_t scale_index) const noexcept {
    static constexpr CValueScaleConfiguration empty_configuration{};
    if (!is_valid_scale(scale_index)) return empty_configuration;
    return c_value_scale_configurations_[scale_index];
}

const gpu::GpuDurationStatistics &
Telemetry::c_value_scale_statistics(std::uint32_t scale_index) const noexcept {
    if (!is_valid_scale(scale_index)) {
        static constexpr gpu::GpuDurationStatistics empty_statistics{};
        return empty_statistics;
    }
    return collector_->duration_statistics(c_value_scale_series_, scale_index);
}

TelemetrySnapshot Telemetry::snapshot() const {
    TelemetrySnapshot result{};
    result.operation_handle = operation_;
    result.operation = operation();
    result.latency_span = latency_span_statistics();
    result.prefix_to_suffix_gap = prefix_to_suffix_gap_statistics();
    result.c_value_scale_configurations = c_value_scale_configurations_;
    for (std::size_t stage_index = 0U; stage_index < gpu_stage_count; ++stage_index) {
        result.stages[stage_index] = stage_statistics(static_cast<GpuStage>(stage_index));
    }
    for (std::uint32_t scale_index = 0U; scale_index < scale_count; ++scale_index) {
        result.c_value_scale_statistics[scale_index] = c_value_scale_statistics(scale_index);
    }
    return result;
}

bool Telemetry::is_valid_stage(GpuStage stage) noexcept {
    return telemetry_stage_id(stage) < gpu_stage_count;
}

bool Telemetry::is_valid_scale(std::uint32_t scale_index) noexcept {
    return scale_index < scale_count;
}

} // namespace crossvmaf::metrics::cambi::vulkan

namespace crossvmaf::metrics::cambi {
class Vulkan::Impl final {
  public:
    gpu::DeviceView device;
    FrameDescriptor descriptor;
    Config config;
    vulkan::Telemetry *telemetry{};
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
    return Result<std::unique_ptr<Vulkan>>::success(
        std::unique_ptr<Vulkan>(new Vulkan(std::move(impl))));
}
vulkan::Executor &Vulkan::slot(std::size_t index) noexcept {
    return *impl_->slots[index];
}
Result<void> Vulkan::append_slot() {
    auto created =
        vulkan::Executor::create(impl_->device, impl_->descriptor, impl_->config, impl_->telemetry);
    if (!created) return Result<void>::failure(created.error());
    impl_->slots.push_back(std::move(created).value());
    return Result<void>::success();
}
} // namespace crossvmaf::metrics::cambi
