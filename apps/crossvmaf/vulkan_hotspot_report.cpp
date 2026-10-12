#include "vulkan_hotspot_report.hpp"

#include "vulkan_hotspot_analysis.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace crossvmaf::cli {
namespace {

namespace adm3 = metrics::adm3::vulkan;
namespace cambi = metrics::cambi::vulkan;
namespace gpu = gpu;
namespace speed_chroma =
    metrics::speed_chroma::vulkan;
namespace vulkan_telemetry = crossvmaf::evaluation::telemetry;

using Json = nlohmann::ordered_json;
using Snapshot = vulkan_telemetry::VulkanHotspotSnapshot;

constexpr long double nanoseconds_per_millisecond = 1'000'000.0L;

using hotspot::provider_child_stages;
using hotspot::feature_record_submit_child_stages;
using hotspot::feature_retirement_child_stages;

[[nodiscard]] double round_six(long double value) noexcept {
    constexpr long double precision = 1'000'000.0L;
    return static_cast<double>(std::round(value * precision) / precision);
}

[[nodiscard]] double milliseconds(
    std::chrono::nanoseconds duration) noexcept {
    return round_six(
        static_cast<long double>(duration.count()) /
        nanoseconds_per_millisecond);
}

[[nodiscard]] double milliseconds(long double nanoseconds) noexcept {
    return round_six(nanoseconds / nanoseconds_per_millisecond);
}

[[nodiscard]] Json optional_number(
    bool available, long double value) {
    if (!available || !std::isfinite(value)) return nullptr;
    return round_six(value);
}

[[nodiscard]] Json optional_milliseconds(
    bool available, long double nanoseconds) {
    if (!available || !std::isfinite(nanoseconds)) return nullptr;
    return milliseconds(nanoseconds);
}

[[nodiscard]] Json optional_u32(
    const std::optional<std::uint32_t>& value) {
    return value.has_value() ? Json(*value) : Json(nullptr);
}

[[nodiscard]] Json host_statistics_json(
    const gpu::HostDurationStatistics& statistics) {
    const bool has_samples = statistics.call_count != 0U;
    const long double average_nanoseconds = has_samples
        ? static_cast<long double>(statistics.total.count()) /
            static_cast<long double>(statistics.call_count)
        : 0.0L;
    return {
        {"calls", statistics.call_count},
        {"total_ms", milliseconds(statistics.total)},
        {"average_ms", optional_milliseconds(
            has_samples, average_nanoseconds)},
        {"minimum_ms", milliseconds(statistics.minimum)},
        {"maximum_ms", milliseconds(statistics.maximum)},
    };
}

[[nodiscard]] Json gpu_duration_json(
    const gpu::GpuDurationStatistics& statistics) {
    const bool has_samples = statistics.call_count != 0U;
    const long double average_nanoseconds = has_samples
        ? statistics.total_nanoseconds /
            static_cast<long double>(statistics.call_count)
        : 0.0L;
    return {
        {"calls", statistics.call_count},
        {"failed_calls", statistics.failed_call_count},
        {"skipped_calls", statistics.skipped_call_count},
        {"total_ms", optional_milliseconds(
            has_samples, statistics.total_nanoseconds)},
        {"average_ms", optional_milliseconds(
            has_samples, average_nanoseconds)},
        {"minimum_ms", optional_milliseconds(
            has_samples, statistics.minimum_nanoseconds)},
        {"maximum_ms", optional_milliseconds(
            has_samples, statistics.maximum_nanoseconds)},
    };
}

template <typename Dispatch>
[[nodiscard]] Json dispatch_json(const Dispatch& dispatch) {
    return {{"x", dispatch.x}, {"y", dispatch.y}, {"z", dispatch.z}};
}

[[nodiscard]] const gpu::HostDurationStatistics& host_statistics(
    const Snapshot& snapshot,
    vulkan_telemetry::SessionStage stage) noexcept {
    static constexpr gpu::HostDurationStatistics empty_statistics{};
    const std::size_t stage_index = static_cast<std::size_t>(stage);
    return stage_index < snapshot.runtime.host_stage_statistics.size()
        ? snapshot.runtime.host_stage_statistics[stage_index]
        : empty_statistics;
}

template <std::size_t StageCount>
[[nodiscard]] std::chrono::nanoseconds add_host_stage_totals(
    const Snapshot& snapshot,
    const std::array<vulkan_telemetry::SessionStage, StageCount>& stages) {
    std::chrono::nanoseconds total{};
    for (const vulkan_telemetry::SessionStage stage : stages) {
        total += host_statistics(snapshot, stage).total;
    }
    return total;
}

[[nodiscard]] Json host_stage_json(
    const Snapshot& snapshot,
    vulkan_telemetry::SessionStage stage,
    std::chrono::nanoseconds parent_total) {
    const gpu::HostDurationStatistics& statistics =
        host_statistics(snapshot, stage);
    const double total_milliseconds = milliseconds(statistics.total);
    const double parent_milliseconds = milliseconds(parent_total);
    return {
        {"name", vulkan_telemetry::session_stage_name(stage)},
        {"calls", statistics.call_count},
        {"total_ms", total_milliseconds},
        {"average_ms", statistics.call_count != 0U
            ? round_six(
                static_cast<long double>(statistics.total.count()) /
                static_cast<long double>(statistics.call_count) /
                nanoseconds_per_millisecond)
            : 0.0},
        {"minimum_ms", milliseconds(statistics.minimum)},
        {"maximum_ms", milliseconds(statistics.maximum)},
        {"percent_of_parent", parent_milliseconds > 0.0
            ? round_six(
                static_cast<long double>(total_milliseconds) * 100.0L /
                static_cast<long double>(parent_milliseconds))
            : 0.0},
    };
}

template <std::size_t StageCount>
[[nodiscard]] Json host_breakdown_json(
    const Snapshot& snapshot,
    const std::array<vulkan_telemetry::SessionStage, StageCount>& stages,
    std::chrono::nanoseconds parent_total,
    std::string_view timing_semantics) {
    const std::chrono::nanoseconds children_total =
        add_host_stage_totals(snapshot, stages);
    const std::chrono::nanoseconds residual =
        parent_total > children_total
        ? parent_total - children_total
        : std::chrono::nanoseconds{};
    Json stage_reports = Json::array();
    for (const vulkan_telemetry::SessionStage stage : stages) {
        stage_reports.push_back(
            host_stage_json(snapshot, stage, parent_total));
    }
    return {
        {"clock_domain", "host_steady_clock"},
        {"timing_semantics", timing_semantics},
        {"parent_total_ms", milliseconds(parent_total)},
        {"children_total_ms", milliseconds(children_total)},
        {"residual_ms", milliseconds(residual)},
        {"stages", std::move(stage_reports)},
    };
}

[[nodiscard]] Json timestamp_fields(
    const gpu::GpuOperationSnapshot& operation) {
    const gpu::GpuTimestampMetadata& metadata =
        operation.timestamp_metadata;
    const bool available = metadata.query_pool_available &&
        operation.parent_statistics.call_count != 0U;
    return {
        {"supported", metadata.device_supported},
        {"available", available},
        {"configured", metadata.configured},
        {"query_pool_available", metadata.query_pool_available},
        {"unavailable_reason", metadata.unavailable_reason.empty()
            ? Json(nullptr)
            : Json(metadata.unavailable_reason)},
        {"clock_domain", "vulkan_device_timestamp"},
        {"timestamp_stage", "compute_shader"},
        {"queue_family_index", optional_u32(metadata.queue_family_index)},
        {"timestamp_valid_bits", optional_u32(
            metadata.timestamp_valid_bits)},
        {"timestamp_period_ns",
         metadata.timestamp_period_nanoseconds.has_value()
            ? Json(static_cast<double>(
                  *metadata.timestamp_period_nanoseconds))
            : Json(nullptr)},
    };
}

[[nodiscard]] Json operations_json(
    const gpu::GpuOperationSnapshot& operation) {
    const auto& operation_statistics = operation.operation_statistics;
    const auto& parent_statistics = operation.parent_statistics;
    const bool timestamp_coverage_complete =
        operation_statistics.submitted_call_count != 0U &&
        operation_statistics.timestamp_unavailable_call_count == 0U &&
        parent_statistics.call_count ==
            operation_statistics.submitted_call_count;
    return {
        {"logical_calls", operation_statistics.logical_call_count},
        {"submitted_calls", operation_statistics.submitted_call_count},
        {"timestamp_sampled_calls", parent_statistics.call_count},
        {"timestamp_unavailable_calls",
         operation_statistics.timestamp_unavailable_call_count},
        {"skipped_no_gpu_work", parent_statistics.skipped_call_count},
        {"timestamp_coverage_complete", timestamp_coverage_complete},
    };
}

[[nodiscard]] Json host_gpu_comparison_json(
    const gpu::GpuOperationSnapshot& operation,
    bool split_submission) {
    const auto& host = operation.submission_latency_statistics;
    const auto& parent = operation.parent_statistics;
    const bool comparison_available = host.call_count != 0U &&
        host.call_count == parent.call_count &&
        parent.failed_call_count == 0U && host.total.count() > 0;
    const long double host_nanoseconds =
        static_cast<long double>(host.total.count());
    const long double residual_nanoseconds = comparison_available
        ? std::max(
            host_nanoseconds - parent.total_nanoseconds, 0.0L)
        : 0.0L;
    const long double gpu_share = comparison_available
        ? parent.total_nanoseconds / host_nanoseconds * 100.0L
        : 0.0L;
    return {
        {"clock_domain", "host_steady_clock"},
        {"timing_semantics", split_submission
            ? "sum of prefix and suffix vkQueueSubmit-to-fence-observation "
              "durations; includes queue waiting, GPU execution, and delayed "
              "host polling"
            : "vkQueueSubmit entry through caller-thread fence observation; "
              "includes queue waiting, GPU execution, and delayed host polling"},
        {"statistics", host_statistics_json(host)},
        {"gpu_comparison_available", comparison_available},
        {"gpu_execution_share_percent", optional_number(
            comparison_available, gpu_share)},
        {"non_gpu_or_observation_residual_ms", optional_milliseconds(
            comparison_available, residual_nanoseconds)},
    };
}

[[nodiscard]] Json common_feature_json(
    std::string_view name,
    std::string_view sample_scope,
    const gpu::GpuOperationSnapshot& operation,
    bool split_submission = false) {
    Json report = timestamp_fields(operation);
    report["name"] = name;
    report["sample_scope"] = sample_scope;
    report["operations"] = operations_json(operation);
    report["parent"] = gpu_duration_json(operation.parent_statistics);
    report["host_submission_to_fence_observation"] =
        host_gpu_comparison_json(operation, split_submission);
    return report;
}

[[nodiscard]] Json stage_json(
    std::string name,
    const gpu::GpuDurationStatistics& statistics,
    const gpu::GpuDurationStatistics& parent,
    bool derived_metrics_available) {
    Json report = gpu_duration_json(statistics);
    report["name"] = std::move(name);
    const bool percentage_available = derived_metrics_available &&
        statistics.call_count != 0U && parent.total_nanoseconds > 0.0L;
    report["percent_of_gpu_parent"] = optional_number(
        percentage_available,
        percentage_available
            ? statistics.total_nanoseconds /
                parent.total_nanoseconds * 100.0L
            : 0.0L);
    return report;
}

[[nodiscard]] Json cambi_feature_json(
    const Snapshot& snapshot, const hotspot::FeatureStageAnalysis& analysis) {
    const auto& feature = snapshot.cambi;
    const auto& parent = feature.operation.parent_statistics;
    const bool derived_available = analysis.available;
    const long double children_total = analysis.children_nanoseconds;

    Json report = common_feature_json(
        "cambi",
        "submitted_prefix_and_suffix_command_buffers",
        feature.operation,
        true);
    report["parent_timing_semantics"] =
        "sum of disjoint prefix and suffix GPU intervals; excludes the "
        "prefix-to-suffix gap";
    report["latency_span_timing_semantics"] =
        "preprocess start through combine end; includes the prefix-to-suffix "
        "gap and must not be treated as CAMBI active GPU work";
    report["latency_span"] = gpu_duration_json(feature.latency_span);
    report["prefix_to_suffix_gap_timing_semantics"] =
        "preprocess end through mask start; may contain other feature "
        "execution, queue waiting, and scheduling gaps";
    report["prefix_to_suffix_gap"] =
        gpu_duration_json(feature.prefix_to_suffix_gap);
    report["derived_metrics_available"] = derived_available;
    report["children_total_ms"] = optional_milliseconds(
        derived_available, children_total);
    report["residual_ms"] = optional_milliseconds(
        derived_available,
        analysis.residual_nanoseconds);

    Json stages = Json::array();
    for (std::size_t stage_index = 0U;
         stage_index < cambi::gpu_stage_count; ++stage_index) {
        stages.push_back(stage_json(
            cambi::gpu_stage_name(
                static_cast<cambi::GpuStage>(stage_index)),
            feature.stages[stage_index], parent, derived_available));
    }
    report["stages"] = std::move(stages);

    Json scales = Json::array();
    for (std::uint32_t scale_index = 0U;
         scale_index < cambi::scale_count; ++scale_index) {
        const auto& configuration =
            feature.c_value_scale_configurations[scale_index];
        Json scale_report = gpu_duration_json(
            feature.c_value_scale_statistics[scale_index]);
        scale_report["scale"] = scale_index;
        scale_report["configuration"] = {
            {"configured", configuration.configured},
            {"width", configuration.width},
            {"height", configuration.height},
            {"window_size", configuration.window_size},
            {"segment_rows", configuration.segment_rows},
            {"segment_count", configuration.segment_count},
            {"initial_window_source_visits",
             configuration.initial_window_source_visits},
            {"dispatch", {
                {"x", configuration.dispatch_x},
                {"y", configuration.dispatch_y},
                {"z", configuration.dispatch_z},
            }},
        };
        scales.push_back(std::move(scale_report));
    }
    report["derived"] = {
        {"c_value_scale_breakdown", {
            {"clock_domain", "vulkan_device_timestamp"},
            {"timestamp_stage", "compute_shader"},
            {"aggregate_stage_name", "c_value"},
            {"scales", std::move(scales)},
        }},
    };
    return report;
}

[[nodiscard]] Json speed_execution_json(
    const speed_chroma::GpuExecutionConfiguration& configuration) {
    return {
        {"configured", configuration.configured},
        {"model_options", {
            {"kernel_scale_raw_q32_32",
             configuration.kernel_scale_raw_q32_32},
            {"prescale_raw_q32_32", configuration.prescale_raw_q32_32},
            {"prescale_method",
             configuration.prescale_method ==
                     speed_chroma::PrescaleMethod::bilinear
                ? "bilinear"
                : "nearest"},
            {"sigma_nn_raw_q32_32", configuration.sigma_nn_raw_q32_32},
            {"nn_floor_raw_q32_32", configuration.nn_floor_raw_q32_32},
            {"weight_variance_mode", configuration.weight_variance_mode},
            {"maximum_raw_q32_32", configuration.maximum_raw_q32_32},
        }},
        {"geometry", {
            {"source", {
                {"width", configuration.source_width},
                {"height", configuration.source_height},
            }},
            {"scaled", {
                {"width", configuration.scaled_width},
                {"height", configuration.scaled_height},
            }},
            {"operating", {
                {"width", configuration.operating_width},
                {"height", configuration.operating_height},
            }},
            {"truncated", {
                {"width", configuration.truncated_width},
                {"height", configuration.truncated_height},
            }},
            {"block_count", configuration.block_count},
            {"covariance_sample_count",
             configuration.covariance_sample_count},
        }},
        {"dispatch", {
            {"source", dispatch_json(configuration.source_dispatch)},
            {"scaled", dispatch_json(configuration.scaled_dispatch)},
            {"operating", dispatch_json(configuration.operating_dispatch)},
            {"covariance", dispatch_json(
                configuration.covariance_dispatch)},
            {"score_blocks", dispatch_json(
                configuration.score_block_dispatch)},
        }},
    };
}

[[nodiscard]] Json speed_feature_json(
    const Snapshot& snapshot, const hotspot::FeatureStageAnalysis& analysis) {
    const auto& feature = snapshot.speed_chroma;
    const auto& parent = feature.operation.parent_statistics;
    const bool derived_available = analysis.available;
    const long double children_total = analysis.children_nanoseconds;

    Json report = common_feature_json(
        "speed_chroma",
        "submitted_frame_command_buffer",
        feature.operation);
    report["derived_metrics_available"] = derived_available;
    report["children_total_ms"] = optional_milliseconds(
        derived_available, children_total);
    report["residual_ms"] = optional_milliseconds(
        derived_available,
        analysis.residual_nanoseconds);

    Json stages = Json::array();
    for (std::size_t stage_index = 0U;
         stage_index < speed_chroma::gpu_stage_count; ++stage_index) {
        stages.push_back(stage_json(
            speed_chroma::gpu_stage_name(
                static_cast<speed_chroma::GpuStage>(stage_index)),
            feature.stages[stage_index], parent, derived_available));
    }
    report["stages"] = std::move(stages);

    Json compact_factor = gpu_duration_json(feature.compact_factor);
    compact_factor["name"] = "compact_factor";
    compact_factor["composition"] = Json::array({
        "compact_tridiagonal", "compact_regularity"});
    compact_factor["clock_domain"] = "vulkan_device_timestamp";
    compact_factor["percent_of_gpu_parent"] = optional_number(
        derived_available && feature.compact_factor.call_count != 0U &&
            parent.total_nanoseconds > 0.0L,
        parent.total_nanoseconds > 0.0L
            ? feature.compact_factor.total_nanoseconds /
                parent.total_nanoseconds * 100.0L
            : 0.0L);

    constexpr std::array wave_stages{
        speed_chroma::GpuStage::compact_tridiagonal,
        speed_chroma::GpuStage::compact_regularity,
        speed_chroma::GpuStage::block_statistics,
        speed_chroma::GpuStage::finalize,
    };
    Json wave_suffix = Json::array();
    for (const speed_chroma::GpuStage stage : wave_stages) {
        const auto& statistics =
            feature.wave_stages[static_cast<std::size_t>(stage)];
        const bool has_samples = statistics.wave_count != 0U;
        wave_suffix.push_back({
            {"name", speed_chroma::gpu_stage_name(stage)},
            {"clock_domain", "vulkan_device_timestamp"},
            {"aggregation", "per_wave"},
            {"waves", statistics.wave_count},
            {"failed_waves", statistics.failed_wave_count},
            {"tasks", statistics.task_count},
            {"total_wave_ms", optional_milliseconds(
                has_samples, statistics.total_wave_nanoseconds)},
            {"average_wave_ms", optional_milliseconds(
                has_samples,
                has_samples
                    ? statistics.total_wave_nanoseconds /
                        static_cast<long double>(statistics.wave_count)
                    : 0.0L)},
            {"minimum_wave_ms", optional_milliseconds(
                has_samples, statistics.minimum_wave_nanoseconds)},
            {"maximum_wave_ms", optional_milliseconds(
                has_samples, statistics.maximum_wave_nanoseconds)},
            {"average_amortized_ms", optional_milliseconds(
                has_samples,
                has_samples
                    ? statistics.total_amortized_nanoseconds /
                        static_cast<long double>(statistics.wave_count)
                    : 0.0L)},
        });
    }
    report["derived"] = {
        {"compact_factor", std::move(compact_factor)},
        {"wave_suffix", std::move(wave_suffix)},
    };
    report["execution_configuration"] = speed_execution_json(
        feature.execution_configuration);
    return report;
}

[[nodiscard]] Json adm3_feature_json(
    const Snapshot& snapshot, const hotspot::FeatureStageAnalysis& analysis) {
    const auto& feature = snapshot.adm3;
    const auto& parent = feature.operation.parent_statistics;
    const bool derived_available = analysis.available;
    const long double children_total = analysis.children_nanoseconds;

    Json report = common_feature_json(
        "adm3", "submitted_frame_command_buffer", feature.operation);
    report["derived_metrics_available"] = derived_available;
    report["children_total_ms"] = optional_milliseconds(
        derived_available, children_total);
    report["residual_ms"] = optional_milliseconds(
        derived_available,
        analysis.residual_nanoseconds);
    report["stage_aggregation"] =
        "adjacent non-overlapping command-buffer timestamp intervals";
    Json stages = Json::array();
    for (std::uint32_t stage_id = 0U;
         stage_id < adm3::gpu_stage_count; ++stage_id) {
        stages.push_back(stage_json(
            adm3::gpu_stage_name(stage_id),
            feature.stages[stage_id], parent, derived_available));
    }
    report["stages"] = std::move(stages);
    return report;
}

[[nodiscard]] Json motion_feature_json(const Snapshot& snapshot) {
    return common_feature_json(
        "motion3",
        "submitted_push_or_flush_command_buffer",
        snapshot.motion3.operation);
}

[[nodiscard]] std::string feature_name_for_operation(
    const Snapshot& snapshot,
    gpu::GpuOperationHandle operation) {
    if (operation.value ==
        snapshot.input_staging_operation_handle.value) {
        return "input_staging_copy";
    }
    if (operation.value == snapshot.cambi.operation_handle.value) {
        return "cambi";
    }
    if (operation.value == snapshot.speed_chroma.operation_handle.value) {
        return "speed_chroma";
    }
    if (operation.value == snapshot.adm3.operation_handle.value) {
        return "adm3";
    }
    if (operation.value == snapshot.motion3.operation_handle.value) {
        return "motion3";
    }
    return "unknown";
}

[[nodiscard]] Json host_timeline_json(const Snapshot& snapshot) {
    Json events = Json::array();
    for (const gpu::HostTimelineEvent& event :
         snapshot.runtime.host_timeline_events) {
        const auto start = snapshot.runtime.timeline_origin_recorded
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                event.started_at - snapshot.runtime.timeline_origin)
            : std::chrono::nanoseconds{};
        const auto finish = snapshot.runtime.timeline_origin_recorded
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                event.finished_at - snapshot.runtime.timeline_origin)
            : std::chrono::nanoseconds{};
        events.push_back({
            {"stage", vulkan_telemetry::session_stage_name(
                static_cast<vulkan_telemetry::SessionStage>(
                    event.stage_id))},
            {"start_ms", milliseconds(start)},
            {"end_ms", milliseconds(finish)},
        });
    }
    Json pending_events = Json::array();
    for (const gpu::PendingSubmissionTimelineEvent& event :
         snapshot.runtime.pending_submission_timeline_events) {
        const auto offset = snapshot.runtime.timeline_origin_recorded
            ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                event.observed_at - snapshot.runtime.timeline_origin)
            : std::chrono::nanoseconds{};
        pending_events.push_back({
            {"at_ms", milliseconds(offset)},
            {"pending_count", event.pending_count},
        });
    }
    return {
        {"clock_domain", "host_steady_clock"},
        {"origin", "first host measurement or pending-submission observation"},
        {"host_events_truncated",
         snapshot.runtime.host_timeline_truncated},
        {"pending_depth_events_truncated",
         snapshot.runtime.pending_submission_timeline_truncated},
        {"events", std::move(events)},
        {"pending_depth_events", std::move(pending_events)},
    };
}

[[nodiscard]] Json gpu_timeline_json(const Snapshot& snapshot) {
    Json events = Json::array();
    for (const gpu::GpuTimelineEvent& event :
         snapshot.runtime.gpu_timeline_events) {
        events.push_back({
            {"feature", feature_name_for_operation(
                snapshot, event.operation)},
            {"kind", gpu::timeline_segment_name(
                static_cast<gpu::TimelineSegment>(
                    event.kind_id))},
            {"start_ms", static_cast<double>(
                event.start_nanoseconds / nanoseconds_per_millisecond)},
            {"end_ms", static_cast<double>(
                event.end_nanoseconds / nanoseconds_per_millisecond)},
        });
    }
    return {
        {"clock_domain", "vulkan_device_timestamp"},
        {"timestamp_period_ns",
         snapshot.runtime.gpu_timeline_origin_recorded
            ? Json(static_cast<double>(
                  snapshot.runtime.gpu_timeline_period_nanoseconds))
            : Json(nullptr)},
        {"events_truncated", snapshot.runtime.gpu_timeline_truncated},
        {"rejected_event_count",
         snapshot.runtime.gpu_timeline_rejected_event_count},
        {"events", std::move(events)},
    };
}

[[nodiscard]] Json optional_number(const std::optional<long double>& value) {
    return optional_number(value.has_value(), value.value_or(0.0L));
}

[[nodiscard]] Json feature_ranking_json(const hotspot::Analysis& analysis) {
    Json ranking = Json::array();
    for (const auto& entry : analysis.feature_ranking) {
        ranking.push_back({
            {"name", entry.name},
            {"average_gpu_ms", milliseconds(entry.average_nanoseconds)},
            {"timestamp_coverage_percent", optional_number(entry.coverage_percent)},
            {"timestamp_failure_percent_of_submissions", optional_number(entry.failure_percent)},
            {"timestamp_unavailable_percent_of_submissions", optional_number(entry.unavailable_percent)},
        });
    }
    return ranking;
}

template <typename StatisticsRange, typename NameFunction>
[[nodiscard]] Json stage_hotspot_json(
    std::string_view feature_name,
    const StatisticsRange& stages,
    const gpu::GpuDurationStatistics& parent,
    const hotspot::FeatureStageAnalysis& analysis,
    NameFunction name_function) {
    if (!analysis.hotspot_index) {
        return {
            {"feature", feature_name},
            {"available", false},
            {"stage", nullptr},
            {"total_ms", nullptr},
            {"percent_of_gpu_parent", nullptr},
        };
    }
    const auto index = *analysis.hotspot_index;
    return {
        {"feature", feature_name},
        {"available", true},
        {"stage", name_function(index)},
        {"total_ms", milliseconds(stages[index].total_nanoseconds)},
        {"percent_of_gpu_parent", round_six(
            stages[index].total_nanoseconds / parent.total_nanoseconds * 100.0L)},
    };
}

[[nodiscard]] Json analysis_json(const Snapshot& snapshot, const hotspot::Analysis& analysis) {
    const auto& runtime = snapshot.runtime;
    const auto& timeline = analysis.timeline;
    const auto& queue = analysis.queue;
    const auto& leases = analysis.leases;
    Json host_hotspots = Json::array();
    for (const auto& entry : analysis.host_hotspots) {
        host_hotspots.push_back({
            {"scope", entry.scope},
            {"available", entry.stage.has_value()},
            {"stage", entry.stage ? Json(vulkan_telemetry::session_stage_name(*entry.stage)) : Json(nullptr)},
            {"total_ms", entry.stage ? Json(milliseconds(entry.total)) : Json(nullptr)},
        });
    }
    Json stage_hotspots = Json::array();
    stage_hotspots.push_back(stage_hotspot_json(
        "cambi", snapshot.cambi.stages, snapshot.cambi.operation.parent_statistics,
        analysis.cambi, [](std::size_t index) {
            return std::string(cambi::gpu_stage_name(static_cast<cambi::GpuStage>(index)));
        }));
    stage_hotspots.push_back(stage_hotspot_json(
        "speed_chroma", snapshot.speed_chroma.stages, snapshot.speed_chroma.operation.parent_statistics,
        analysis.speed_chroma, [](std::size_t index) {
            return std::string(speed_chroma::gpu_stage_name(static_cast<speed_chroma::GpuStage>(index)));
        }));
    stage_hotspots.push_back(stage_hotspot_json(
        "adm3", snapshot.adm3.stages, snapshot.adm3.operation.parent_statistics,
        analysis.adm3, [](std::size_t index) {
            return adm3::gpu_stage_name(static_cast<std::uint32_t>(index));
        }));
    return {
        {"semantics", "derived summary calculated from the raw aggregates "
         "and timelines in this report; host and GPU clock domains remain "
         "independent"},
        {"data_quality", {
            {"report_complete", snapshot.complete},
            {"host_timeline_complete", !runtime.host_timeline_truncated},
            {"pending_depth_timeline_complete", !runtime.pending_submission_timeline_truncated},
            {"gpu_feature_timeline_complete", !runtime.gpu_timeline_truncated &&
                runtime.gpu_timeline_rejected_event_count == 0U},
        }},
        {"host", {
            {"provider_average_ms_per_push", optional_milliseconds(
                analysis.provider_average_nanoseconds.has_value(),
                analysis.provider_average_nanoseconds.value_or(0.0L))},
            {"hotspots", std::move(host_hotspots)},
        }},
        {"queue", {
            {"analysis_available", queue.available},
            {"average_pending_utilization_percent", optional_number(queue.average_utilization_percent)},
            {"peak_pending_utilization_percent", optional_number(queue.peak_utilization_percent)},
            {"zero_pending_percent", optional_number(queue.zero_pending_percent)},
        }},
        {"leases", {
            {"analysis_available", leases.available},
            {"spatial_peak_occupancy_percent", optional_number(leases.spatial_peak_percent)},
            {"reference_luma_peak_occupancy_percent", optional_number(leases.reference_luma_peak_percent)},
            {"spatial_release_completion_percent", optional_number(leases.spatial_release_percent)},
            {"reference_luma_release_completion_percent", optional_number(leases.reference_luma_release_percent)},
        }},
        {"gpu", {
            {"feature_ranking_by_average_gpu_ms", feature_ranking_json(analysis)},
            {"stage_hotspots", std::move(stage_hotspots)},
            {"feature_timeline", {
                {"analysis_available", timeline.available},
                {"valid_event_count", timeline.valid_event_count},
                {"invalid_event_count", timeline.invalid_event_count},
                {"span_ms", optional_milliseconds(timeline.available, timeline.span_nanoseconds)},
                {"union_busy_ms", optional_milliseconds(timeline.available, timeline.union_busy_nanoseconds)},
                {"summed_feature_time_ms", optional_milliseconds(timeline.available, timeline.summed_feature_nanoseconds)},
                {"extra_concurrent_feature_time_ms", optional_milliseconds(timeline.available, timeline.extra_concurrent_nanoseconds)},
                {"busy_percent_within_feature_timeline", optional_number(timeline.busy_percent)},
                {"average_active_feature_intervals_while_busy", optional_number(timeline.average_active_intervals)},
            }},
        }},
    };
}

}  // namespace

nlohmann::ordered_json make_vulkan_hotspot_report(
    const crossvmaf::evaluation::telemetry::VulkanHotspotSnapshot& snapshot) {
    const auto& runtime_snapshot = snapshot.runtime;
    const auto& runtime_configuration =
        runtime_snapshot.vulkan_runtime_configuration;
    const auto& leases = runtime_snapshot.input_lease_runtime_statistics;
    const auto& input_buffers =
        runtime_snapshot.input_buffer_memory_placement;
    const auto& uploads = runtime_snapshot.upload_byte_statistics;
    const auto& submissions = runtime_snapshot.submission_overview;

    const std::chrono::nanoseconds provider_parent_total =
        host_statistics(snapshot, vulkan_telemetry::SessionStage::provider_push)
            .total +
        host_statistics(snapshot, vulkan_telemetry::SessionStage::provider_flush)
            .total;
    const std::chrono::nanoseconds record_submit_parent_total =
        host_statistics(
            snapshot,
            vulkan_telemetry::SessionStage::feature_record_submit)
            .total;
    const std::chrono::nanoseconds retirement_parent_total =
        host_statistics(
            snapshot,
            vulkan_telemetry::SessionStage::readback_retirement)
            .total;

    const auto analysis = hotspot::analyze(snapshot);
    Json cambi_report = cambi_feature_json(snapshot, analysis.cambi);
    Json speed_report = speed_feature_json(snapshot, analysis.speed_chroma);
    Json adm3_report = adm3_feature_json(snapshot, analysis.adm3);
    Json motion_report = motion_feature_json(snapshot);
    Json input_staging_report = common_feature_json(
        "input_staging_copy",
        "submitted_transfer_command_buffer",
        snapshot.input_staging_operation);
    input_staging_report["timestamp_stage"] = "transfer";
    Json features = Json::array({
        cambi_report, speed_report, adm3_report, motion_report,
        input_staging_report});

    Json report{
        {"schema", "crossvmaf.vulkan_hotspots.v12"},
        {"compatible_with", Json::array()},
        {"complete", snapshot.complete},
        {"device", {
            {"name", snapshot.device.name},
            {"uuid", snapshot.device.uuid},
            {"vendor_id", snapshot.device.vendor_id},
            {"device_id", snapshot.device.device_id},
        }},
        {"input", {
            {"width", snapshot.descriptor.width},
            {"height", snapshot.descriptor.height},
            {"bit_depth", snapshot.descriptor.bit_depth},
            {"provider_push_calls", host_statistics(
                snapshot,
                vulkan_telemetry::SessionStage::provider_push).call_count},
            {"upload_bytes_per_frame", uploads.bytes_per_frame},
            {"upload_total_bytes", uploads.total_bytes},
            {"upload_frame_count", uploads.frame_count},
        }},
        {"vulkan_runtime", {
            {"configured", runtime_configuration.configured},
            {"requested_token_depth",
             runtime_configuration.requested_token_depth},
            {"effective_token_depth",
             runtime_configuration.effective_token_depth},
            {"input_lookahead", 0U},
            {"spatial_input_lease_depth",
             runtime_configuration.spatial_input_lease_depth},
            {"reference_luma_lease_depth",
             runtime_configuration.reference_luma_lease_depth},
            {"maximum_pending_submissions",
             runtime_configuration.maximum_pending_submissions},
            {"depth_scaled_buffer_bytes",
             runtime_configuration.depth_scaled_buffer_bytes},
            {"input_leases", {
                {"configured", leases.configured},
                {"logical_buffer_bytes", leases.logical_buffer_bytes},
                {"reservation_count", leases.reservation_count},
                {"spatial_release_count", leases.spatial_release_count},
                {"reference_luma_release_count",
                 leases.reference_luma_release_count},
                {"maximum_active_spatial_leases",
                 leases.maximum_active_spatial_leases},
                {"maximum_active_reference_luma_leases",
                 leases.maximum_active_reference_luma_leases},
                {"spatial_hold_average_ms",
                 leases.spatial_release_count != 0U
                    ? milliseconds(
                        leases.spatial_hold_total.count() /
                        static_cast<long double>(
                            leases.spatial_release_count))
                    : 0.0},
                {"reference_luma_hold_average_ms",
                 leases.reference_luma_release_count != 0U
                    ? milliseconds(
                        leases.reference_luma_hold_total.count() /
                        static_cast<long double>(
                            leases.reference_luma_release_count))
                    : 0.0},
                {"spatial_hold_maximum_ms",
                 milliseconds(leases.spatial_hold_maximum)},
                {"reference_luma_hold_maximum_ms",
                 milliseconds(leases.reference_luma_hold_maximum)},
            }},
            {"input_buffers", {
                {"buffer_count", input_buffers.buffer_count},
                {"uniform_across_buffers",
                 input_buffers.uniform_across_buffers},
                {"memory_type_index", input_buffers.memory_type_index},
                {"heap_index", input_buffers.memory_heap_index},
                {"property_flags", input_buffers.property_flags},
                {"host_visible", input_buffers.host_visible},
                {"host_coherent", input_buffers.host_coherent},
                {"device_local", input_buffers.device_local},
                {"non_coherent_atom_size",
                 input_buffers.non_coherent_atom_size},
            }},
            {"submissions", {
                {"current_pending", submissions.current_pending},
                {"maximum_pending_observed",
                 submissions.maximum_pending_observed},
                {"pending_count_transition_count",
                 submissions.pending_count_transition_count},
                {"observation_window", "first successful submission "
                 "through report finalization"},
                {"observation_duration_ms",
                 milliseconds(submissions.observation_duration)},
                {"zero_pending_duration_ms",
                 milliseconds(submissions.zero_pending_duration)},
                {"zero_pending_percent", optional_number(
                    submissions.observation_duration.count() > 0,
                    submissions.observation_duration.count() > 0
                        ? static_cast<long double>(
                              submissions.zero_pending_duration.count()) /
                            static_cast<long double>(
                              submissions.observation_duration.count()) *
                            100.0L
                        : 0.0L)},
                {"average_pending", optional_number(
                    submissions.observation_duration.count() > 0,
                    submissions.average_pending)},
            }},
        }},
        {"host_timeline", host_timeline_json(snapshot)},
        {"gpu_feature_timeline", gpu_timeline_json(snapshot)},
        {"provider_breakdown", host_breakdown_json(
            snapshot, provider_child_stages, provider_parent_total,
            "non-overlapping caller-thread host stages for asynchronous "
            "transactions; GPU execution remains outside this host clock "
            "domain")},
        {"feature_record_submit_breakdown", host_breakdown_json(
            snapshot,
            feature_record_submit_child_stages,
            record_submit_parent_total,
            "per-feature command recording and vkQueueSubmit calls nested "
            "inside the aggregate feature_record_submit stage")},
        {"feature_retirement_breakdown", host_breakdown_json(
            snapshot,
            feature_retirement_child_stages,
            retirement_parent_total,
            "per-feature readback and timestamp retirement nested inside "
            "the aggregate readback_retirement stage")},
        {"gpu_feature_breakdown", {
            {"clock_domain", "vulkan_device_timestamp"},
            {"features", std::move(features)},
        }},
        {"input_staging_gpu_copy", std::move(input_staging_report)},
        {"speed_chroma_gpu_breakdown", speed_report},
    };
    report["analysis"] = analysis_json(snapshot, analysis);
    return report;
}

} // namespace crossvmaf::cli
