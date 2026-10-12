#include "gpu/timestamp.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include "gpu/error.hpp"
#include <cstddef>
#include <vector>
#include <chrono>

namespace crossvmaf::gpu {
namespace {

constexpr std::size_t telemetry_collector_maximum_timeline_event_count = 50'000U;

void telemetry_collector_increment_counter(std::uint64_t &counter) noexcept {
    if (counter != std::numeric_limits<std::uint64_t>::max()) {
        ++counter;
    }
}

void telemetry_collector_record_host_duration_sample(HostDurationStatistics &statistics,
                                                     std::chrono::nanoseconds elapsed) noexcept {
    if (statistics.call_count == 0U) {
        statistics.minimum = elapsed;
        statistics.maximum = elapsed;
    } else {
        statistics.minimum = std::min(statistics.minimum, elapsed);
        statistics.maximum = std::max(statistics.maximum, elapsed);
    }
    telemetry_collector_increment_counter(statistics.call_count);
    statistics.total += elapsed;
}

void telemetry_collector_record_gpu_duration_success(GpuDurationStatistics &statistics,
                                                     long double duration_nanoseconds) noexcept {
    const bool duration_is_valid =
        std::isfinite(duration_nanoseconds) && duration_nanoseconds >= 0.0L;
    if (!duration_is_valid || statistics.call_count == std::numeric_limits<std::uint64_t>::max()) {
        telemetry_collector_increment_counter(statistics.failed_call_count);
        return;
    }

    const long double normalized_duration_nanoseconds =
        duration_nanoseconds == 0.0L ? 0.0L : duration_nanoseconds;
    const long double updated_total_nanoseconds =
        statistics.total_nanoseconds + normalized_duration_nanoseconds;
    if (!std::isfinite(updated_total_nanoseconds)) {
        telemetry_collector_increment_counter(statistics.failed_call_count);
        return;
    }

    if (statistics.call_count == 0U) {
        statistics.minimum_nanoseconds = normalized_duration_nanoseconds;
        statistics.maximum_nanoseconds = normalized_duration_nanoseconds;
    } else {
        statistics.minimum_nanoseconds =
            std::min(statistics.minimum_nanoseconds, normalized_duration_nanoseconds);
        statistics.maximum_nanoseconds =
            std::max(statistics.maximum_nanoseconds, normalized_duration_nanoseconds);
    }
    telemetry_collector_increment_counter(statistics.call_count);
    statistics.total_nanoseconds = updated_total_nanoseconds;
}

void telemetry_collector_record_gpu_duration_sample(GpuDurationStatistics &statistics,
                                                    TelemetrySampleOutcome outcome,
                                                    long double duration_nanoseconds) noexcept {
    switch (outcome) {
    case TelemetrySampleOutcome::success:
        telemetry_collector_record_gpu_duration_success(statistics, duration_nanoseconds);
        return;
    case TelemetrySampleOutcome::failure:
        telemetry_collector_increment_counter(statistics.failed_call_count);
        return;
    case TelemetrySampleOutcome::skipped:
        telemetry_collector_increment_counter(statistics.skipped_call_count);
        return;
    }
}

[[nodiscard]] std::optional<std::uint64_t>
telemetry_collector_unwrap_timestamp(std::uint64_t timestamp, std::uint64_t high_watermark,
                                     std::uint64_t modulus, std::uint64_t mask) noexcept {
    const std::uint64_t raw_timestamp = timestamp & mask;
    if (modulus == 0U) return raw_timestamp;

    const std::uint64_t high_watermark_raw = high_watermark & mask;
    const std::uint64_t epoch_start = high_watermark - high_watermark_raw;
    if (raw_timestamp > std::numeric_limits<std::uint64_t>::max() - epoch_start) {
        return std::nullopt;
    }

    std::uint64_t candidate = epoch_start + raw_timestamp;
    const std::uint64_t half_modulus = modulus / 2U;
    if (candidate < high_watermark && high_watermark - candidate > half_modulus) {
        if (candidate > std::numeric_limits<std::uint64_t>::max() - modulus) {
            return std::nullopt;
        }
        candidate += modulus;
    } else if (candidate > high_watermark && candidate - high_watermark > half_modulus &&
               candidate >= modulus) {
        candidate -= modulus;
    }
    return candidate;
}

} // namespace

TelemetryCollector::TelemetryCollector(bool enabled, std::size_t host_stage_count)
    : enabled_(enabled), host_stage_statistics_(host_stage_count) {
}

GpuOperationHandle TelemetryCollector::register_gpu_operation() {
    if (gpu_operations_.size() >= invalid_telemetry_handle) {
        return {};
    }
    const GpuOperationHandle handle{static_cast<std::uint32_t>(gpu_operations_.size())};
    gpu_operations_.emplace_back();
    return handle;
}

DurationSeriesHandle TelemetryCollector::register_duration_series(std::size_t member_count) {
    if (duration_series_.size() >= invalid_telemetry_handle) {
        return {};
    }
    const DurationSeriesHandle handle{static_cast<std::uint32_t>(duration_series_.size())};
    duration_series_.push_back(
        DurationSeriesSnapshot{std::vector<GpuDurationStatistics>(member_count)});
    return handle;
}

WaveDurationSeriesHandle
TelemetryCollector::register_wave_duration_series(std::size_t member_count) {
    if (wave_duration_series_.size() >= invalid_telemetry_handle) {
        return {};
    }
    const WaveDurationSeriesHandle handle{static_cast<std::uint32_t>(wave_duration_series_.size())};
    wave_duration_series_.push_back(
        WaveDurationSeriesSnapshot{std::vector<GpuWaveDurationStatistics>(member_count)});
    return handle;
}

void TelemetryCollector::record_host_duration(
    std::uint32_t stage_id, std::chrono::steady_clock::time_point started_at,
    std::chrono::steady_clock::time_point finished_at) noexcept {
    if (!enabled_ || stage_id >= host_stage_statistics_.size() || finished_at < started_at) {
        return;
    }

    establish_timeline_origin(started_at);
    if (host_timeline_events_.size() < telemetry_collector_maximum_timeline_event_count) {
        try {
            host_timeline_events_.push_back({stage_id, started_at, finished_at});
        } catch (...) {
            host_timeline_truncated_ = true;
        }
    } else {
        host_timeline_truncated_ = true;
    }

    telemetry_collector_record_host_duration_sample(
        host_stage_statistics_[stage_id],
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished_at - started_at));
}

void TelemetryCollector::configure_input_buffer_memory(
    const InputBufferMemoryConfiguration &configuration) noexcept {
    if (!enabled_) return;

    InputBufferMemoryPlacement &placement = input_buffer_memory_placement_;
    if (placement.buffer_count == 0U) {
        static_cast<InputBufferMemoryConfiguration &>(placement) = configuration;
    } else {
        const auto &current = static_cast<const InputBufferMemoryConfiguration &>(placement);
        placement.uniform_across_buffers =
            placement.uniform_across_buffers &&
            current.memory_type_index == configuration.memory_type_index &&
            current.memory_heap_index == configuration.memory_heap_index &&
            current.property_flags == configuration.property_flags &&
            current.non_coherent_atom_size == configuration.non_coherent_atom_size &&
            current.host_visible == configuration.host_visible &&
            current.host_coherent == configuration.host_coherent &&
            current.device_local == configuration.device_local;
    }
    telemetry_collector_increment_counter(placement.buffer_count);
}

void TelemetryCollector::configure_vulkan_runtime(
    std::size_t requested_token_depth, std::size_t effective_token_depth,
    std::size_t spatial_input_lease_depth, std::size_t reference_luma_lease_depth,
    std::size_t maximum_pending_submissions, std::uint64_t depth_scaled_buffer_bytes) noexcept {
    if (!enabled_) return;
    vulkan_runtime_configuration_ = {
        true,
        static_cast<std::uint64_t>(requested_token_depth),
        static_cast<std::uint64_t>(effective_token_depth),
        static_cast<std::uint64_t>(spatial_input_lease_depth),
        static_cast<std::uint64_t>(reference_luma_lease_depth),
        static_cast<std::uint64_t>(maximum_pending_submissions),
        depth_scaled_buffer_bytes,
    };
}

void TelemetryCollector::configure_input_lease_runtime(
    const InputLeaseRuntimeStatistics &statistics) noexcept {
    if (!enabled_) return;
    input_lease_runtime_statistics_ = statistics;
    input_lease_runtime_statistics_.configured = true;
}

void TelemetryCollector::record_upload_bytes(std::uint64_t byte_count) noexcept {
    if (!enabled_) return;
    UploadByteStatistics &statistics = upload_byte_statistics_;
    if (statistics.frame_count == 0U) {
        statistics.bytes_per_frame = byte_count;
    } else if (statistics.bytes_per_frame != byte_count) {
        statistics.uniform_bytes_per_frame = false;
    }
    if (statistics.total_bytes <= std::numeric_limits<std::uint64_t>::max() - byte_count) {
        statistics.total_bytes += byte_count;
    } else {
        statistics.total_bytes = std::numeric_limits<std::uint64_t>::max();
    }
    telemetry_collector_increment_counter(statistics.frame_count);
}

void TelemetryCollector::record_pending_submissions(std::size_t pending_count) noexcept {
    if (!enabled_) return;
    const std::uint64_t normalized_count = static_cast<std::uint64_t>(pending_count);
    const auto observation_time = std::chrono::steady_clock::now();
    record_pending_submission_timeline(observation_time, normalized_count);

    if (!submission_observation_active_ && normalized_count > 0U) {
        submission_observation_active_ = true;
        submission_observation_started_at_ = observation_time;
        last_pending_transition_at_ = observation_time;
        telemetry_collector_increment_counter(submission_overview_.pending_count_transition_count);
    } else if (submission_observation_active_ &&
               normalized_count != submission_overview_.current_pending) {
        accumulate_submission_observation(observation_time);
        telemetry_collector_increment_counter(submission_overview_.pending_count_transition_count);
    }

    submission_overview_.current_pending = normalized_count;
    submission_overview_.maximum_pending_observed =
        std::max(submission_overview_.maximum_pending_observed, normalized_count);
}

void TelemetryCollector::configure_gpu_timestamps(
    GpuOperationHandle operation, const GpuTimestampConfiguration &configuration) noexcept {
    if (!enabled_ || operation.value >= gpu_operations_.size()) return;

    GpuTimestampMetadata &metadata = gpu_operations_[operation.value].timestamp_metadata;
    metadata.configured = true;
    metadata.device_supported = metadata.device_supported || configuration.device_supported;
    metadata.query_pool_available =
        metadata.query_pool_available || configuration.query_pool_available;
    if (configuration.queue_family_index.has_value()) {
        metadata.queue_family_index = configuration.queue_family_index;
    }
    if (configuration.timestamp_valid_bits.has_value()) {
        metadata.timestamp_valid_bits = configuration.timestamp_valid_bits;
    }
    if (configuration.timestamp_period_nanoseconds.has_value() &&
        std::isfinite(*configuration.timestamp_period_nanoseconds) &&
        *configuration.timestamp_period_nanoseconds > 0.0L) {
        metadata.timestamp_period_nanoseconds = configuration.timestamp_period_nanoseconds;
    }

    try {
        if (metadata.query_pool_available || configuration.unavailable_reason.empty()) {
            metadata.unavailable_reason.clear();
        } else {
            metadata.unavailable_reason.assign(configuration.unavailable_reason.data(),
                                               configuration.unavailable_reason.size());
        }
    } catch (...) {
        metadata.unavailable_reason.clear();
    }
}

void TelemetryCollector::record_gpu_submission(GpuOperationHandle operation,
                                               bool timestamp_available) noexcept {
    if (!enabled_ || operation.value >= gpu_operations_.size()) return;
    GpuOperationStatistics &statistics = gpu_operations_[operation.value].operation_statistics;
    telemetry_collector_increment_counter(statistics.logical_call_count);
    telemetry_collector_increment_counter(statistics.submitted_call_count);
    if (!timestamp_available) {
        telemetry_collector_increment_counter(statistics.timestamp_unavailable_call_count);
    }
}

void TelemetryCollector::record_gpu_parent(GpuOperationHandle operation,
                                           TelemetrySampleOutcome outcome,
                                           long double duration_nanoseconds) noexcept {
    if (!enabled_ || operation.value >= gpu_operations_.size()) return;
    GpuOperationSnapshot &operation_snapshot = gpu_operations_[operation.value];
    if (outcome == TelemetrySampleOutcome::skipped) {
        telemetry_collector_increment_counter(
            operation_snapshot.operation_statistics.logical_call_count);
    }
    telemetry_collector_record_gpu_duration_sample(operation_snapshot.parent_statistics, outcome,
                                                   duration_nanoseconds);
}

void TelemetryCollector::record_gpu_submission_latency(GpuOperationHandle operation,
                                                       std::chrono::nanoseconds duration) noexcept {
    if (!enabled_ || operation.value >= gpu_operations_.size() || duration.count() < 0) {
        return;
    }
    telemetry_collector_record_host_duration_sample(
        gpu_operations_[operation.value].submission_latency_statistics, duration);
}

void TelemetryCollector::record_gpu_timeline(GpuOperationHandle operation, std::uint32_t kind_id,
                                             std::uint64_t start_timestamp,
                                             std::uint64_t end_timestamp,
                                             std::uint32_t timestamp_valid_bits,
                                             double timestamp_period_nanoseconds) noexcept {
    if (!enabled_ || operation.value >= gpu_operations_.size()) return;
    if (timestamp_valid_bits == 0U || timestamp_valid_bits > 64U ||
        !std::isfinite(timestamp_period_nanoseconds) || timestamp_period_nanoseconds <= 0.0) {
        telemetry_collector_increment_counter(gpu_timeline_rejected_event_count_);
        return;
    }
    if (gpu_timeline_origin_recorded_ &&
        (gpu_timeline_valid_bits_ != timestamp_valid_bits ||
         gpu_timeline_period_nanoseconds_ != timestamp_period_nanoseconds)) {
        telemetry_collector_increment_counter(gpu_timeline_rejected_event_count_);
        return;
    }

    const std::uint64_t modulus =
        timestamp_valid_bits == 64U ? 0U : (std::uint64_t{1} << timestamp_valid_bits);
    const std::uint64_t mask =
        modulus == 0U ? std::numeric_limits<std::uint64_t>::max() : modulus - 1U;
    const std::uint64_t raw_start = start_timestamp & mask;
    if (!gpu_timeline_origin_recorded_) {
        gpu_timeline_origin_recorded_ = true;
        gpu_timeline_valid_bits_ = timestamp_valid_bits;
        gpu_timeline_period_nanoseconds_ = timestamp_period_nanoseconds;
        gpu_timeline_origin_timestamp_ = raw_start;
        gpu_timeline_high_watermark_timestamp_ = raw_start;
    }

    const std::optional<std::uint64_t> unwrapped_start = telemetry_collector_unwrap_timestamp(
        start_timestamp, gpu_timeline_high_watermark_timestamp_, modulus, mask);
    if (!unwrapped_start.has_value()) {
        telemetry_collector_increment_counter(gpu_timeline_rejected_event_count_);
        return;
    }
    const std::uint64_t high_watermark_after_start =
        std::max(gpu_timeline_high_watermark_timestamp_, *unwrapped_start);
    const std::optional<std::uint64_t> unwrapped_end = telemetry_collector_unwrap_timestamp(
        end_timestamp, high_watermark_after_start, modulus, mask);
    if (!unwrapped_end.has_value() || *unwrapped_end < *unwrapped_start) {
        telemetry_collector_increment_counter(gpu_timeline_rejected_event_count_);
        return;
    }
    gpu_timeline_high_watermark_timestamp_ = std::max(high_watermark_after_start, *unwrapped_end);
    if (gpu_timeline_events_.size() >= telemetry_collector_maximum_timeline_event_count) {
        gpu_timeline_truncated_ = true;
        return;
    }

    try {
        const long double origin = static_cast<long double>(gpu_timeline_origin_timestamp_);
        const long double period = static_cast<long double>(gpu_timeline_period_nanoseconds_);
        gpu_timeline_events_.push_back({
            operation,
            kind_id,
            (static_cast<long double>(*unwrapped_start) - origin) * period,
            (static_cast<long double>(*unwrapped_end) - origin) * period,
        });
    } catch (...) {
        gpu_timeline_truncated_ = true;
    }
}

void TelemetryCollector::record_duration(DurationSeriesHandle series, std::uint32_t member_id,
                                         TelemetrySampleOutcome outcome,
                                         long double duration_nanoseconds) noexcept {
    if (!enabled_ || series.value >= duration_series_.size() ||
        member_id >= duration_series_[series.value].members.size()) {
        return;
    }
    telemetry_collector_record_gpu_duration_sample(
        duration_series_[series.value].members[member_id], outcome, duration_nanoseconds);
}

void TelemetryCollector::record_wave_duration(WaveDurationSeriesHandle series,
                                              std::uint32_t member_id,
                                              TelemetrySampleOutcome outcome,
                                              long double wave_nanoseconds,
                                              std::uint32_t task_count) noexcept {
    if (!enabled_ || series.value >= wave_duration_series_.size() ||
        member_id >= wave_duration_series_[series.value].members.size()) {
        return;
    }
    GpuWaveDurationStatistics &statistics = wave_duration_series_[series.value].members[member_id];
    if (outcome != TelemetrySampleOutcome::success || task_count == 0U ||
        !std::isfinite(wave_nanoseconds) || wave_nanoseconds < 0.0L) {
        if (outcome != TelemetrySampleOutcome::skipped) {
            telemetry_collector_increment_counter(statistics.failed_wave_count);
        }
        return;
    }

    const long double updated_total = statistics.total_wave_nanoseconds + wave_nanoseconds;
    const long double updated_amortized = statistics.total_amortized_nanoseconds +
                                          wave_nanoseconds / static_cast<long double>(task_count);
    if (!std::isfinite(updated_total) || !std::isfinite(updated_amortized) ||
        statistics.task_count > std::numeric_limits<std::uint64_t>::max() - task_count) {
        telemetry_collector_increment_counter(statistics.failed_wave_count);
        return;
    }

    telemetry_collector_increment_counter(statistics.wave_count);
    statistics.task_count += task_count;
    statistics.total_wave_nanoseconds = updated_total;
    statistics.total_amortized_nanoseconds = updated_amortized;
    if (statistics.wave_count == 1U) {
        statistics.minimum_wave_nanoseconds = wave_nanoseconds;
        statistics.maximum_wave_nanoseconds = wave_nanoseconds;
    } else {
        statistics.minimum_wave_nanoseconds =
            std::min(statistics.minimum_wave_nanoseconds, wave_nanoseconds);
        statistics.maximum_wave_nanoseconds =
            std::max(statistics.maximum_wave_nanoseconds, wave_nanoseconds);
    }
}

const HostDurationStatistics &
TelemetryCollector::host_statistics(std::uint32_t stage_id) const noexcept {
    static constexpr HostDurationStatistics empty_statistics{};
    if (stage_id >= host_stage_statistics_.size()) return empty_statistics;
    return host_stage_statistics_[stage_id];
}

const GpuOperationSnapshot &
TelemetryCollector::gpu_operation(GpuOperationHandle operation) const noexcept {
    static const GpuOperationSnapshot empty_operation{};
    if (operation.value >= gpu_operations_.size()) return empty_operation;
    return gpu_operations_[operation.value];
}

const GpuDurationStatistics &
TelemetryCollector::duration_statistics(DurationSeriesHandle series,
                                        std::uint32_t member_id) const noexcept {
    static constexpr GpuDurationStatistics empty_statistics{};
    if (series.value >= duration_series_.size() ||
        member_id >= duration_series_[series.value].members.size()) {
        return empty_statistics;
    }
    return duration_series_[series.value].members[member_id];
}

const GpuWaveDurationStatistics &
TelemetryCollector::wave_statistics(WaveDurationSeriesHandle series,
                                    std::uint32_t member_id) const noexcept {
    static constexpr GpuWaveDurationStatistics empty_statistics{};
    if (series.value >= wave_duration_series_.size() ||
        member_id >= wave_duration_series_[series.value].members.size()) {
        return empty_statistics;
    }
    return wave_duration_series_[series.value].members[member_id];
}

TelemetrySnapshot TelemetryCollector::snapshot() {
    finalize_submission_observation();
    return {
        host_stage_statistics_,
        timeline_origin_,
        timeline_origin_recorded_,
        host_timeline_events_,
        pending_submission_timeline_events_,
        host_timeline_truncated_,
        pending_submission_timeline_truncated_,
        gpu_operations_,
        duration_series_,
        wave_duration_series_,
        gpu_timeline_events_,
        static_cast<long double>(gpu_timeline_period_nanoseconds_),
        gpu_timeline_rejected_event_count_,
        gpu_timeline_origin_recorded_,
        gpu_timeline_truncated_,
        input_buffer_memory_placement_,
        vulkan_runtime_configuration_,
        input_lease_runtime_statistics_,
        upload_byte_statistics_,
        submission_overview_,
    };
}

void TelemetryCollector::establish_timeline_origin(
    std::chrono::steady_clock::time_point observation) noexcept {
    if (!timeline_origin_recorded_) {
        timeline_origin_ = observation;
        timeline_origin_recorded_ = true;
    }
}

void TelemetryCollector::record_pending_submission_timeline(
    std::chrono::steady_clock::time_point observed_at, std::uint64_t pending_count) noexcept {
    establish_timeline_origin(observed_at);
    if (pending_submission_timeline_events_.size() >=
        telemetry_collector_maximum_timeline_event_count) {
        pending_submission_timeline_truncated_ = true;
        return;
    }
    try {
        pending_submission_timeline_events_.push_back({observed_at, pending_count});
    } catch (...) {
        pending_submission_timeline_truncated_ = true;
    }
}

void TelemetryCollector::accumulate_submission_observation(
    std::chrono::steady_clock::time_point observation_time) noexcept {
    if (!submission_observation_active_) return;
    const std::chrono::nanoseconds elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        observation_time - last_pending_transition_at_);
    if (submission_overview_.current_pending == 0U) {
        submission_overview_.zero_pending_duration += elapsed;
    }
    pending_submission_nanoseconds_ +=
        static_cast<long double>(elapsed.count()) *
        static_cast<long double>(submission_overview_.current_pending);
    last_pending_transition_at_ = observation_time;
}

void TelemetryCollector::finalize_submission_observation() noexcept {
    if (!submission_observation_active_) return;
    const auto observation_time = std::chrono::steady_clock::now();
    accumulate_submission_observation(observation_time);
    submission_overview_.observation_duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(observation_time -
                                                             submission_observation_started_at_);
    if (submission_overview_.observation_duration.count() > 0) {
        submission_overview_.average_pending =
            pending_submission_nanoseconds_ /
            static_cast<long double>(submission_overview_.observation_duration.count());
    }
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {
namespace {

constexpr std::size_t timestamp_words_per_timestamp_query_result = 2U;

[[nodiscard]] bool timestamp_has_timestamp_query_functions(const VolkDeviceTable &table) noexcept {
    return table.vkCreateQueryPool != nullptr && table.vkDestroyQueryPool != nullptr &&
           table.vkCmdResetQueryPool != nullptr && table.vkCmdWriteTimestamp != nullptr &&
           table.vkGetQueryPoolResults != nullptr;
}

} // namespace

Result<TimestampQueryCapabilities> query_timestamp_capabilities(DeviceView device) {
    if (device.physical_device == VK_NULL_HANDLE) {
        return Result<TimestampQueryCapabilities>::failure(
            {ErrorCode::backend_error,
             "timestamp capability query received no Vulkan physical device"});
    }
    if (vkGetPhysicalDeviceQueueFamilyProperties == nullptr ||
        vkGetPhysicalDeviceProperties == nullptr) {
        return Result<TimestampQueryCapabilities>::failure(
            {ErrorCode::backend_error,
             "timestamp capability query requires loaded Vulkan instance functions"});
    }

    std::uint32_t queue_family_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(device.physical_device, &queue_family_count, nullptr);
    if (device.queue_family >= queue_family_count) {
        return Result<TimestampQueryCapabilities>::failure(
            {ErrorCode::invalid_argument,
             "timestamp capability query received an invalid Vulkan queue family"});
    }

    std::vector<VkQueueFamilyProperties> queue_family_properties(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device.physical_device, &queue_family_count,
                                             queue_family_properties.data());
    if (device.queue_family >= queue_family_count) {
        return Result<TimestampQueryCapabilities>::failure(
            {ErrorCode::backend_error,
             "Vulkan queue family properties changed during timestamp capability query"});
    }

    VkPhysicalDeviceProperties physical_device_properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &physical_device_properties);

    TimestampQueryCapabilities capabilities{};
    capabilities.timestamp_valid_bits =
        queue_family_properties[device.queue_family].timestampValidBits;
    capabilities.timestamp_period_nanoseconds =
        static_cast<double>(physical_device_properties.limits.timestampPeriod);
    return Result<TimestampQueryCapabilities>::success(capabilities);
}

Result<TimestampDuration>
calculate_timestamp_duration(const TimestampQueryResult &start, const TimestampQueryResult &end,
                             std::uint32_t timestamp_valid_bits,
                             long double timestamp_period_nanoseconds,
                             std::optional<long double> maximum_elapsed_nanoseconds) {
    constexpr std::uint32_t timestamp_storage_bits =
        static_cast<std::uint32_t>(std::numeric_limits<std::uint64_t>::digits);
    if (timestamp_valid_bits == 0U || timestamp_valid_bits > timestamp_storage_bits) {
        return Result<TimestampDuration>::failure(
            {ErrorCode::invalid_argument, "timestamp valid bit count must be between 1 and 64"});
    }
    if (!std::isfinite(timestamp_period_nanoseconds) || timestamp_period_nanoseconds <= 0.0L) {
        return Result<TimestampDuration>::failure(
            {ErrorCode::invalid_argument, "timestamp period must be finite and greater than zero"});
    }
    if (maximum_elapsed_nanoseconds.has_value()) {
        if (!std::isfinite(*maximum_elapsed_nanoseconds) || *maximum_elapsed_nanoseconds < 0.0L) {
            return Result<TimestampDuration>::failure(
                {ErrorCode::invalid_argument,
                 "maximum timestamp elapsed time must be finite and non-negative"});
        }

        const long double counter_wrap_period_nanoseconds =
            std::ldexp(timestamp_period_nanoseconds, static_cast<int>(timestamp_valid_bits));
        if (std::isfinite(counter_wrap_period_nanoseconds) &&
            *maximum_elapsed_nanoseconds >= counter_wrap_period_nanoseconds) {
            return Result<TimestampDuration>::failure(
                {ErrorCode::invalid_state,
                 "timestamp duration is ambiguous because the host measurement window spans at "
                 "least one full counter period"});
        }
    }
    if (!start.available() || !end.available()) {
        return Result<TimestampDuration>::failure(
            {ErrorCode::invalid_state, "timestamp duration requires two available query results"});
    }

    const std::uint64_t valid_bits_mask = timestamp_valid_bits == timestamp_storage_bits
                                              ? std::numeric_limits<std::uint64_t>::max()
                                              : (std::uint64_t{1} << timestamp_valid_bits) - 1U;
    const std::uint64_t start_ticks = start.value & valid_bits_mask;
    const std::uint64_t end_ticks = end.value & valid_bits_mask;
    const bool wrapped = end_ticks < start_ticks;

    const std::uint64_t elapsed_ticks =
        wrapped ? (valid_bits_mask - start_ticks) + end_ticks + 1U : end_ticks - start_ticks;
    const long double elapsed_nanoseconds =
        static_cast<long double>(elapsed_ticks) * timestamp_period_nanoseconds;
    if (!std::isfinite(elapsed_nanoseconds)) {
        return Result<TimestampDuration>::failure(
            {ErrorCode::invalid_argument,
             "timestamp duration exceeds the representable numeric range"});
    }
    if (maximum_elapsed_nanoseconds.has_value() &&
        elapsed_nanoseconds > *maximum_elapsed_nanoseconds) {
        return Result<TimestampDuration>::failure(
            {ErrorCode::invalid_state,
             "timestamp duration is inconsistent with the host submission window"});
    }

    return Result<TimestampDuration>::success({elapsed_ticks, elapsed_nanoseconds, wrapped});
}

Result<TimestampDuration>
calculate_timestamp_duration(const TimestampQueryResult &start, const TimestampQueryResult &end,
                             const TimestampQueryCapabilities &capabilities,
                             std::optional<long double> maximum_elapsed_nanoseconds) {
    return calculate_timestamp_duration(
        start, end, capabilities.timestamp_valid_bits,
        static_cast<long double>(capabilities.timestamp_period_nanoseconds),
        maximum_elapsed_nanoseconds);
}

TimestampQueryPool::TimestampQueryPool(DeviceView device, std::uint32_t query_count,
                                       TimestampQueryCapabilities capabilities) noexcept
    : device_(device), query_count_(query_count), capabilities_(capabilities) {
}

Result<std::unique_ptr<TimestampQueryPool>> TimestampQueryPool::create(DeviceView device,
                                                                       std::uint32_t query_count) {
    if (query_count == 0U) {
        return Result<std::unique_ptr<TimestampQueryPool>>::failure(
            {ErrorCode::invalid_argument, "Vulkan timestamp query pool count must be non-zero"});
    }
    if (device.physical_device == VK_NULL_HANDLE || device.device == VK_NULL_HANDLE ||
        device.table == nullptr || device.resource_lifetime == nullptr) {
        return Result<std::unique_ptr<TimestampQueryPool>>::failure(
            {ErrorCode::backend_error,
             "timestamp query pool creation received an incomplete Vulkan device"});
    }

    auto capabilities_result = query_timestamp_capabilities(device);
    if (!capabilities_result) {
        return Result<std::unique_ptr<TimestampQueryPool>>::failure(capabilities_result.error());
    }
    const TimestampQueryCapabilities capabilities = capabilities_result.value();
    auto query_pool = std::unique_ptr<TimestampQueryPool>(
        new TimestampQueryPool(device, query_count, capabilities));
    if (!capabilities.supported()) {
        return Result<std::unique_ptr<TimestampQueryPool>>::success(std::move(query_pool));
    }
    if (!timestamp_has_timestamp_query_functions(*device.table)) {
        return Result<std::unique_ptr<TimestampQueryPool>>::failure(
            {ErrorCode::backend_error,
             "timestamp query pool requires unavailable Vulkan device functions"});
    }

    VkQueryPoolCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    create_info.queryCount = query_count;
    const VkResult create_result = device.table->vkCreateQueryPool(
        device.device, &create_info, nullptr, &query_pool->query_pool_);
    if (create_result != VK_SUCCESS) {
        return Result<std::unique_ptr<TimestampQueryPool>>::failure(
            make_error("vkCreateQueryPool", create_result));
    }
    return Result<std::unique_ptr<TimestampQueryPool>>::success(std::move(query_pool));
}

TimestampQueryPool::~TimestampQueryPool() {
    if (query_pool_ != VK_NULL_HANDLE && device_.resource_lifetime->safe_to_destroy()) {
        device_.table->vkDestroyQueryPool(device_.device, query_pool_, nullptr);
    }
    query_pool_ = VK_NULL_HANDLE;
}

Result<void> TimestampQueryPool::validate_supported() const {
    if (!supported()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan queue family does not support timestamp queries"});
    }
    if (query_pool_ == VK_NULL_HANDLE) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan timestamp query pool is not initialized"});
    }
    return Result<void>::success();
}

Result<void> TimestampQueryPool::validate_query_range(std::uint32_t first_query,
                                                      std::uint32_t query_count) const {
    if (query_count == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "Vulkan timestamp query range must be non-empty"});
    }
    if (first_query >= query_count_ || query_count > query_count_ - first_query) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "Vulkan timestamp query range exceeds the query pool"});
    }
    return Result<void>::success();
}

Result<void> TimestampQueryPool::reset_queries(VkCommandBuffer command_buffer,
                                               std::uint32_t first_query,
                                               std::uint32_t query_count) const {
    const auto support_result = validate_supported();
    if (!support_result) return support_result;
    if (command_buffer == VK_NULL_HANDLE) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "timestamp query reset requires a Vulkan command buffer"});
    }
    const auto range_result = validate_query_range(first_query, query_count);
    if (!range_result) return range_result;

    device_.table->vkCmdResetQueryPool(command_buffer, query_pool_, first_query, query_count);
    return Result<void>::success();
}

Result<void> TimestampQueryPool::write_timestamp(VkCommandBuffer command_buffer,
                                                 std::uint32_t query_index,
                                                 VkPipelineStageFlagBits pipeline_stage) const {
    const auto support_result = validate_supported();
    if (!support_result) return support_result;
    if (command_buffer == VK_NULL_HANDLE) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "timestamp write requires a Vulkan command buffer"});
    }
    if (pipeline_stage == static_cast<VkPipelineStageFlagBits>(0)) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "timestamp write requires a non-zero Vulkan pipeline stage"});
    }
    const auto range_result = validate_query_range(query_index, 1U);
    if (!range_result) return range_result;

    device_.table->vkCmdWriteTimestamp(command_buffer, pipeline_stage, query_pool_, query_index);
    return Result<void>::success();
}

Result<std::vector<TimestampQueryResult>>
TimestampQueryPool::read_results(std::uint32_t first_query, std::uint32_t query_count) const {
    const auto support_result = validate_supported();
    if (!support_result) {
        return Result<std::vector<TimestampQueryResult>>::failure(support_result.error());
    }
    const auto range_result = validate_query_range(first_query, query_count);
    if (!range_result) {
        return Result<std::vector<TimestampQueryResult>>::failure(range_result.error());
    }

    constexpr std::size_t bytes_per_timestamp_query_result =
        sizeof(std::uint64_t) * timestamp_words_per_timestamp_query_result;
    if (static_cast<std::size_t>(query_count) >
        std::numeric_limits<std::size_t>::max() / bytes_per_timestamp_query_result) {
        return Result<std::vector<TimestampQueryResult>>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan timestamp query read exceeds the host address space"});
    }
    const std::size_t result_word_count =
        static_cast<std::size_t>(query_count) * timestamp_words_per_timestamp_query_result;
    std::vector<std::uint64_t> interleaved_results(result_word_count, 0U);

    constexpr VkQueryResultFlags result_flags =
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT;
    constexpr VkDeviceSize result_stride = bytes_per_timestamp_query_result;
    const VkResult read_result = device_.table->vkGetQueryPoolResults(
        device_.device, query_pool_, first_query, query_count,
        interleaved_results.size() * sizeof(std::uint64_t), interleaved_results.data(),
        result_stride, result_flags);
    // Without WAIT_BIT, Vulkan reports VK_NOT_READY when at least one query is
    // unavailable. WITH_AVAILABILITY still writes each availability field, so
    // preserve that useful state and reserve Result failure for VkResult
    // errors.
    if (read_result != VK_SUCCESS && read_result != VK_NOT_READY) {
        return Result<std::vector<TimestampQueryResult>>::failure(
            make_error("vkGetQueryPoolResults", read_result));
    }

    std::vector<TimestampQueryResult> results;
    results.reserve(query_count);
    for (std::uint32_t query_offset = 0U; query_offset < query_count; ++query_offset) {
        const std::size_t value_offset =
            static_cast<std::size_t>(query_offset) * timestamp_words_per_timestamp_query_result;
        const std::uint64_t availability = interleaved_results[value_offset + 1U];
        const std::uint64_t value = availability == 0U ? 0U : interleaved_results[value_offset];
        results.push_back({value, availability});
    }
    return Result<std::vector<TimestampQueryResult>>::success(std::move(results));
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {
namespace {

constexpr std::uint32_t gpu_timestamp_interval_start_query_index = 0U;
constexpr std::uint32_t gpu_timestamp_interval_end_query_index = 1U;
constexpr std::uint32_t gpu_timestamp_interval_timestamp_pair_query_count = 2U;

[[nodiscard]] GpuTimestampConfiguration
gpu_timestamp_interval_make_configuration(DeviceView device,
                                          const TimestampQueryCapabilities &capabilities) noexcept {
    GpuTimestampConfiguration configuration{};
    configuration.queue_family_index = device.queue_family;
    configuration.timestamp_valid_bits = capabilities.timestamp_valid_bits;
    configuration.timestamp_period_nanoseconds =
        static_cast<long double>(capabilities.timestamp_period_nanoseconds);
    configuration.device_supported = capabilities.supported();
    return configuration;
}

} // namespace

GpuTimestampPair::GpuTimestampPair(TelemetryCollector *telemetry_collector,
                                   GpuOperationHandle operation,
                                   VkPipelineStageFlagBits timestamp_stage) noexcept
    : telemetry_collector_(telemetry_collector), operation_(operation),
      timestamp_stage_(timestamp_stage) {
}

Result<std::unique_ptr<GpuTimestampPair>>
GpuTimestampPair::create(DeviceView device, TelemetryCollector *telemetry_collector,
                         GpuOperationHandle operation, VkPipelineStageFlagBits timestamp_stage) {
    auto timestamp = std::unique_ptr<GpuTimestampPair>(
        new GpuTimestampPair(telemetry_collector, operation, timestamp_stage));
    if (telemetry_collector == nullptr || !telemetry_collector->enabled() || !operation) {
        return Result<std::unique_ptr<GpuTimestampPair>>::success(std::move(timestamp));
    }

    const auto capabilities_result = query_timestamp_capabilities(device);
    if (!capabilities_result) {
        GpuTimestampConfiguration configuration{};
        configuration.queue_family_index = device.queue_family;
        configuration.unavailable_reason = capabilities_result.error().message;
        telemetry_collector->configure_gpu_timestamps(operation, configuration);
        if (capabilities_result.error().code == ErrorCode::device_lost) {
            return Result<std::unique_ptr<GpuTimestampPair>>::failure(capabilities_result.error());
        }
        return Result<std::unique_ptr<GpuTimestampPair>>::success(std::move(timestamp));
    }

    GpuTimestampConfiguration configuration =
        gpu_timestamp_interval_make_configuration(device, capabilities_result.value());
    if (!capabilities_result.value().supported()) {
        configuration.unavailable_reason =
            "selected Vulkan queue family reports timestampValidBits == 0";
        telemetry_collector->configure_gpu_timestamps(operation, configuration);
        return Result<std::unique_ptr<GpuTimestampPair>>::success(std::move(timestamp));
    }

    auto query_pool_result =
        TimestampQueryPool::create(device, gpu_timestamp_interval_timestamp_pair_query_count);
    if (!query_pool_result) {
        configuration.unavailable_reason = query_pool_result.error().message;
        telemetry_collector->configure_gpu_timestamps(operation, configuration);
        if (query_pool_result.error().code == ErrorCode::device_lost) {
            return Result<std::unique_ptr<GpuTimestampPair>>::failure(query_pool_result.error());
        }
        return Result<std::unique_ptr<GpuTimestampPair>>::success(std::move(timestamp));
    }

    timestamp->query_pool_ = std::move(query_pool_result).value();
    configuration =
        gpu_timestamp_interval_make_configuration(device, timestamp->query_pool_->capabilities());
    configuration.query_pool_available = timestamp->query_pool_->supported();
    if (!configuration.query_pool_available) {
        configuration.unavailable_reason =
            "selected Vulkan queue family reports timestampValidBits == 0";
        timestamp->query_pool_.reset();
    }
    telemetry_collector->configure_gpu_timestamps(operation, configuration);
    return Result<std::unique_ptr<GpuTimestampPair>>::success(std::move(timestamp));
}

GpuTimestampRecordingState
GpuTimestampPair::begin_recording(VkCommandBuffer command_buffer) const noexcept {
    GpuTimestampRecordingState recording_state{};
    recording_state.query_pool_available = query_pool_ != nullptr;
    if (query_pool_ == nullptr) return recording_state;

    const auto reset_result = query_pool_->reset_queries(
        command_buffer, 0U, gpu_timestamp_interval_timestamp_pair_query_count);
    recording_state.recording_valid = static_cast<bool>(reset_result);
    return recording_state;
}

void GpuTimestampPair::write_start(VkCommandBuffer command_buffer,
                                   GpuTimestampRecordingState &recording_state) const noexcept {
    write_timestamp(command_buffer, gpu_timestamp_interval_start_query_index, recording_state);
}

void GpuTimestampPair::write_end(VkCommandBuffer command_buffer,
                                 GpuTimestampRecordingState &recording_state) const noexcept {
    write_timestamp(command_buffer, gpu_timestamp_interval_end_query_index, recording_state);
}

void GpuTimestampPair::write_timestamp(VkCommandBuffer command_buffer, std::uint32_t query_index,
                                       GpuTimestampRecordingState &recording_state) const noexcept {
    if (!recording_state.recording_valid || query_pool_ == nullptr) return;
    const auto write_result =
        query_pool_->write_timestamp(command_buffer, query_index, timestamp_stage_);
    if (!write_result) recording_state.recording_valid = false;
}

Result<void> GpuTimestampPair::collect(const GpuTimestampRecordingState &recording_state,
                                       std::chrono::nanoseconds submission_host_elapsed,
                                       std::uint32_t timeline_kind_id) const {
    if (telemetry_collector_ == nullptr || !telemetry_collector_->enabled()) {
        return Result<void>::success();
    }
    telemetry_collector_->record_gpu_submission(operation_, recording_state.query_pool_available);
    if (!recording_state.query_pool_available) {
        return Result<void>::success();
    }
    if (!recording_state.recording_valid || query_pool_ == nullptr) {
        telemetry_collector_->record_gpu_parent(operation_, TelemetrySampleOutcome::failure);
        return Result<void>::success();
    }

    const auto query_results_result =
        query_pool_->read_results(0U, gpu_timestamp_interval_timestamp_pair_query_count);
    if (!query_results_result) {
        telemetry_collector_->record_gpu_parent(operation_, TelemetrySampleOutcome::failure);
        if (query_results_result.error().code == ErrorCode::device_lost) {
            return Result<void>::failure(query_results_result.error());
        }
        return Result<void>::success();
    }
    const auto &query_results = query_results_result.value();
    if (query_results.size() != gpu_timestamp_interval_timestamp_pair_query_count) {
        telemetry_collector_->record_gpu_parent(operation_, TelemetrySampleOutcome::failure);
        return Result<void>::success();
    }

    const long double maximum_elapsed_nanoseconds =
        std::chrono::duration<long double, std::nano>(submission_host_elapsed).count();
    const auto duration_result =
        calculate_timestamp_duration(query_results[gpu_timestamp_interval_start_query_index],
                                     query_results[gpu_timestamp_interval_end_query_index],
                                     query_pool_->capabilities(), maximum_elapsed_nanoseconds);
    if (!duration_result) {
        telemetry_collector_->record_gpu_parent(operation_, TelemetrySampleOutcome::failure);
        return Result<void>::success();
    }

    telemetry_collector_->record_gpu_parent(operation_, TelemetrySampleOutcome::success,
                                            duration_result.value().nanoseconds);
    telemetry_collector_->record_gpu_timeline(
        operation_, timeline_kind_id, query_results[gpu_timestamp_interval_start_query_index].value,
        query_results[gpu_timestamp_interval_end_query_index].value,
        query_pool_->capabilities().timestamp_valid_bits,
        query_pool_->capabilities().timestamp_period_nanoseconds);
    return Result<void>::success();
}

void GpuTimestampPair::record_skip() const noexcept {
    if (telemetry_collector_ == nullptr) return;
    telemetry_collector_->record_gpu_parent(operation_, TelemetrySampleOutcome::skipped);
}

GpuTimestampIntervals::GpuTimestampIntervals(TelemetryCollector *telemetry_collector,
                                             DurationSeriesHandle interval_series,
                                             std::uint32_t interval_count) noexcept
    : telemetry_collector_(telemetry_collector), interval_series_(interval_series),
      interval_count_(interval_count), query_count_(interval_count + 1U) {
}

Result<std::unique_ptr<GpuTimestampIntervals>>
GpuTimestampIntervals::create(DeviceView device, TelemetryCollector *telemetry_collector,
                              GpuOperationHandle operation, DurationSeriesHandle interval_series,
                              std::uint32_t interval_count, bool intervals_enabled) {
    if (interval_count == std::numeric_limits<std::uint32_t>::max()) {
        return Result<std::unique_ptr<GpuTimestampIntervals>>::failure(
            {ErrorCode::invalid_argument,
             "GPU timestamp interval count exceeds the query index range"});
    }
    auto timestamp = std::unique_ptr<GpuTimestampIntervals>(
        new GpuTimestampIntervals(telemetry_collector, interval_series, interval_count));
    auto parent_timestamp_result = GpuTimestampPair::create(device, telemetry_collector, operation);
    if (!parent_timestamp_result) {
        return Result<std::unique_ptr<GpuTimestampIntervals>>::failure(
            parent_timestamp_result.error());
    }
    timestamp->parent_timestamp_ = std::move(parent_timestamp_result).value();

    if (telemetry_collector == nullptr || !telemetry_collector->enabled() || !intervals_enabled ||
        interval_count == 0U || !interval_series) {
        return Result<std::unique_ptr<GpuTimestampIntervals>>::success(std::move(timestamp));
    }

    const auto capabilities_result = query_timestamp_capabilities(device);
    if (!capabilities_result) {
        GpuTimestampConfiguration configuration{};
        configuration.queue_family_index = device.queue_family;
        configuration.unavailable_reason = capabilities_result.error().message;
        telemetry_collector->configure_gpu_timestamps(operation, configuration);
        if (capabilities_result.error().code == ErrorCode::device_lost) {
            return Result<std::unique_ptr<GpuTimestampIntervals>>::failure(
                capabilities_result.error());
        }
        return Result<std::unique_ptr<GpuTimestampIntervals>>::success(std::move(timestamp));
    }

    GpuTimestampConfiguration configuration =
        gpu_timestamp_interval_make_configuration(device, capabilities_result.value());
    if (!capabilities_result.value().supported()) {
        configuration.unavailable_reason =
            "selected Vulkan queue family reports timestampValidBits == 0";
        telemetry_collector->configure_gpu_timestamps(operation, configuration);
        return Result<std::unique_ptr<GpuTimestampIntervals>>::success(std::move(timestamp));
    }

    auto query_pool_result = TimestampQueryPool::create(device, timestamp->query_count_);
    if (!query_pool_result) {
        configuration.unavailable_reason = query_pool_result.error().message;
        telemetry_collector->configure_gpu_timestamps(operation, configuration);
        if (query_pool_result.error().code == ErrorCode::device_lost) {
            return Result<std::unique_ptr<GpuTimestampIntervals>>::failure(
                query_pool_result.error());
        }
        return Result<std::unique_ptr<GpuTimestampIntervals>>::success(std::move(timestamp));
    }

    timestamp->query_pool_ = std::move(query_pool_result).value();
    configuration =
        gpu_timestamp_interval_make_configuration(device, timestamp->query_pool_->capabilities());
    configuration.query_pool_available = timestamp->query_pool_->supported();
    telemetry_collector->configure_gpu_timestamps(operation, configuration);
    return Result<std::unique_ptr<GpuTimestampIntervals>>::success(std::move(timestamp));
}

GpuTimestampIntervalRecordingState
GpuTimestampIntervals::begin_recording(VkCommandBuffer command_buffer) const noexcept {
    GpuTimestampIntervalRecordingState recording_state{};
    recording_state.parent = parent_timestamp_->begin_recording(command_buffer);
    recording_state.intervals.query_pool_available = query_pool_ != nullptr;
    if (query_pool_ == nullptr) return recording_state;

    const auto reset_result = query_pool_->reset_queries(command_buffer, 0U, query_count_);
    recording_state.intervals.recording_valid = static_cast<bool>(reset_result);
    return recording_state;
}

void GpuTimestampIntervals::write_boundary(
    VkCommandBuffer command_buffer, std::uint32_t boundary_index,
    GpuTimestampIntervalRecordingState &recording_state) const noexcept {
    if (boundary_index == 0U) {
        parent_timestamp_->write_start(command_buffer, recording_state.parent);
    }
    if (boundary_index + 1U == query_count_) {
        parent_timestamp_->write_end(command_buffer, recording_state.parent);
    }
    if (!recording_state.intervals.recording_valid || query_pool_ == nullptr ||
        boundary_index >= query_count_) {
        recording_state.intervals.recording_valid = false;
        return;
    }
    const auto write_result = query_pool_->write_timestamp(command_buffer, boundary_index,
                                                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    if (!write_result) recording_state.intervals.recording_valid = false;
}

Result<void>
GpuTimestampIntervals::collect(const GpuTimestampIntervalRecordingState &recording_state,
                               std::chrono::nanoseconds submission_host_elapsed,
                               std::uint32_t timeline_kind_id) const {
    if (telemetry_collector_ == nullptr || !telemetry_collector_->enabled()) {
        return Result<void>::success();
    }
    const auto parent_result = parent_timestamp_->collect(
        recording_state.parent, submission_host_elapsed, timeline_kind_id);
    if (!parent_result) return parent_result;
    if (!recording_state.intervals.query_pool_available) {
        return Result<void>::success();
    }
    if (!recording_state.intervals.recording_valid || query_pool_ == nullptr) {
        record_all_interval_failures();
        return Result<void>::success();
    }

    const auto query_results_result = query_pool_->read_results(0U, query_count_);
    if (!query_results_result) {
        record_all_interval_failures();
        if (query_results_result.error().code == ErrorCode::device_lost) {
            return Result<void>::failure(query_results_result.error());
        }
        return Result<void>::success();
    }
    const auto &query_results = query_results_result.value();
    if (query_results.size() != query_count_) {
        record_all_interval_failures();
        return Result<void>::success();
    }

    const long double maximum_elapsed_nanoseconds =
        std::chrono::duration<long double, std::nano>(submission_host_elapsed).count();
    for (std::uint32_t interval_index = 0U; interval_index < interval_count_; ++interval_index) {
        const auto duration_result = calculate_timestamp_duration(
            query_results[interval_index], query_results[interval_index + 1U],
            query_pool_->capabilities(), maximum_elapsed_nanoseconds);
        telemetry_collector_->record_duration(
            interval_series_, interval_index,
            duration_result ? TelemetrySampleOutcome::success : TelemetrySampleOutcome::failure,
            duration_result ? duration_result.value().nanoseconds : 0.0L);
    }
    return Result<void>::success();
}

void GpuTimestampIntervals::write_parent_start(
    VkCommandBuffer command_buffer,
    GpuTimestampIntervalRecordingState &recording_state) const noexcept {
    parent_timestamp_->write_start(command_buffer, recording_state.parent);
}

void GpuTimestampIntervals::write_parent_end(
    VkCommandBuffer command_buffer,
    GpuTimestampIntervalRecordingState &recording_state) const noexcept {
    parent_timestamp_->write_end(command_buffer, recording_state.parent);
}

Result<void>
GpuTimestampIntervals::collect_parent(const GpuTimestampIntervalRecordingState &recording_state,
                                      std::chrono::nanoseconds submission_host_elapsed,
                                      std::uint32_t timeline_kind_id) const {
    return parent_timestamp_->collect(recording_state.parent, submission_host_elapsed,
                                      timeline_kind_id);
}

void GpuTimestampIntervals::record_all_interval_failures() const noexcept {
    for (std::uint32_t interval_index = 0U; interval_index < interval_count_; ++interval_index) {
        telemetry_collector_->record_duration(interval_series_, interval_index,
                                              TelemetrySampleOutcome::failure);
    }
}

} // namespace crossvmaf::gpu
