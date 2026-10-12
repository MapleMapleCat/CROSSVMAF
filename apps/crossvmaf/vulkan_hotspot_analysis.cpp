#include "vulkan_hotspot_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <span>
#include <utility>

namespace crossvmaf::cli::hotspot {
namespace {

using Snapshot = evaluation::telemetry::VulkanHotspotSnapshot;

[[nodiscard]] const gpu::HostDurationStatistics& host_statistics(
    const Snapshot& snapshot,
    SessionStage stage) noexcept {
    static constexpr gpu::HostDurationStatistics empty_statistics{};
    const std::size_t stage_index = static_cast<std::size_t>(stage);
    return stage_index < snapshot.runtime.host_stage_statistics.size()
        ? snapshot.runtime.host_stage_statistics[stage_index]
        : empty_statistics;
}

[[nodiscard]] bool duration_samples_align_with_parent(
    const gpu::GpuDurationStatistics& statistics,
    std::uint64_t parent_call_count,
    bool allow_skips) noexcept {
    if (statistics.failed_call_count != 0U ||
        statistics.call_count > parent_call_count) {
        return false;
    }
    const std::uint64_t expected_skips =
        parent_call_count - statistics.call_count;
    return allow_skips
        ? statistics.skipped_call_count == expected_skips
        : statistics.call_count == parent_call_count &&
            statistics.skipped_call_count == 0U;
}

template <typename StatisticsRange>
[[nodiscard]] long double add_duration_totals(
    const StatisticsRange& statistics) noexcept {
    long double total{};
    for (const gpu::GpuDurationStatistics& member : statistics) {
        total += member.total_nanoseconds;
    }
    return total;
}

[[nodiscard]] FeatureStageAnalysis analyze_stages(
    const gpu::GpuDurationStatistics& parent,
    std::span<const gpu::GpuDurationStatistics> stages,
    bool allow_skips,
    const gpu::GpuDurationStatistics* additional_sample = nullptr) {
    FeatureStageAnalysis result;
    result.children_nanoseconds = add_duration_totals(stages);
    result.available = parent.call_count != 0U && parent.failed_call_count == 0U &&
        std::all_of(stages.begin(), stages.end(), [&](const auto& stage) {
            return duration_samples_align_with_parent(stage, parent.call_count, allow_skips);
        }) && (!additional_sample || duration_samples_align_with_parent(
            *additional_sample, parent.call_count, allow_skips)) &&
        std::isfinite(result.children_nanoseconds);
    if (result.available) {
        result.residual_nanoseconds = std::max(
            parent.total_nanoseconds - result.children_nanoseconds, 0.0L);
        if (!stages.empty() && parent.total_nanoseconds > 0.0L) {
            const auto largest = std::max_element(stages.begin(), stages.end(),
                [](const auto& left, const auto& right) {
                    return left.total_nanoseconds < right.total_nanoseconds;
                });
            result.hotspot_index = static_cast<std::size_t>(largest - stages.begin());
        }
    }
    return result;
}

[[nodiscard]] TimelineAnalysis analyze_gpu_timeline(
    const Snapshot& snapshot) {
    TimelineAnalysis analysis{};
    analysis.invalid_event_count =
        snapshot.runtime.gpu_timeline_rejected_event_count;
    std::vector<std::pair<long double, long double>> intervals;
    intervals.reserve(snapshot.runtime.gpu_timeline_events.size());
    for (const gpu::GpuTimelineEvent& event :
         snapshot.runtime.gpu_timeline_events) {
        const bool valid = std::isfinite(event.start_nanoseconds) &&
            std::isfinite(event.end_nanoseconds) &&
            event.start_nanoseconds >= 0.0L &&
            event.end_nanoseconds >= event.start_nanoseconds;
        if (!valid) {
            ++analysis.invalid_event_count;
            continue;
        }
        ++analysis.valid_event_count;
        intervals.emplace_back(
            event.start_nanoseconds, event.end_nanoseconds);
        analysis.summed_feature_nanoseconds +=
            event.end_nanoseconds - event.start_nanoseconds;
    }
    if (intervals.empty() ||
        snapshot.runtime.gpu_timeline_truncated ||
        analysis.invalid_event_count != 0U) {
        return analysis;
    }
    std::sort(intervals.begin(), intervals.end());
    const long double first_start = intervals.front().first;
    long double merged_start = intervals.front().first;
    long double merged_end = intervals.front().second;
    long double final_end = merged_end;
    for (std::size_t interval_index = 1U;
         interval_index < intervals.size(); ++interval_index) {
        const auto [start, end] = intervals[interval_index];
        final_end = std::max(final_end, end);
        if (start <= merged_end) {
            merged_end = std::max(merged_end, end);
            continue;
        }
        analysis.union_busy_nanoseconds += merged_end - merged_start;
        merged_start = start;
        merged_end = end;
    }
    analysis.union_busy_nanoseconds += merged_end - merged_start;
    analysis.span_nanoseconds = final_end - first_start;
    analysis.available = true;
    analysis.extra_concurrent_nanoseconds = std::max(
        analysis.summed_feature_nanoseconds - analysis.union_busy_nanoseconds, 0.0L);
    if (analysis.span_nanoseconds > 0.0L) {
        analysis.busy_percent = analysis.union_busy_nanoseconds /
            analysis.span_nanoseconds * 100.0L;
    }
    if (analysis.union_busy_nanoseconds > 0.0L) {
        analysis.average_active_intervals = analysis.summed_feature_nanoseconds /
            analysis.union_busy_nanoseconds;
    }
    return analysis;
}

[[nodiscard]] std::optional<long double> percent(
    bool available, long double numerator, long double denominator) {
    if (!available || denominator <= 0.0L) return std::nullopt;
    return numerator / denominator * 100.0L;
}

[[nodiscard]] std::vector<FeatureRanking> rank_features(const Snapshot& snapshot) {
    struct Entry {
        std::string_view name;
        const gpu::GpuOperationSnapshot* operation;
    };
    std::vector<Entry> entries{
        {"cambi", &snapshot.cambi.operation},
        {"speed_chroma", &snapshot.speed_chroma.operation},
        {"adm3", &snapshot.adm3.operation},
        {"motion3", &snapshot.motion3.operation},
        {"input_staging_copy", &snapshot.input_staging_operation},
    };
    std::erase_if(entries, [](const Entry& entry) {
        return entry.operation->parent_statistics.call_count == 0U;
    });
    const auto average = [](const Entry& entry) {
        return entry.operation->parent_statistics.total_nanoseconds /
            static_cast<long double>(entry.operation->parent_statistics.call_count);
    };
    std::sort(entries.begin(), entries.end(), [&](const Entry& left, const Entry& right) {
        return average(left) > average(right);
    });
    std::vector<FeatureRanking> ranking;
    for (const Entry& entry : entries) {
        const auto& parent = entry.operation->parent_statistics;
        const auto& operations = entry.operation->operation_statistics;
        const long double submitted = static_cast<long double>(operations.submitted_call_count);
        ranking.push_back({entry.name, average(entry),
            percent(true, static_cast<long double>(parent.call_count), submitted),
            percent(true, static_cast<long double>(parent.failed_call_count), submitted),
            percent(true, static_cast<long double>(operations.timestamp_unavailable_call_count), submitted)});
    }
    return ranking;
}

[[nodiscard]] HostHotspot host_hotspot(
    const Snapshot& snapshot, std::string_view scope,
    std::span<const SessionStage> stages, std::chrono::nanoseconds parent_total) {
    HostHotspot result{scope, std::nullopt, {}};
    const auto largest = std::max_element(stages.begin(), stages.end(),
        [&snapshot](auto left, auto right) {
            return host_statistics(snapshot, left).total < host_statistics(snapshot, right).total;
        });
    if (parent_total.count() > 0 && largest != stages.end()) {
        result.stage = *largest;
        result.total = host_statistics(snapshot, *largest).total;
    }
    return result;
}

} // namespace

Analysis analyze(const Snapshot& snapshot) {
    Analysis result;
    result.cambi = analyze_stages(
        snapshot.cambi.operation.parent_statistics, snapshot.cambi.stages, false);
    result.speed_chroma = analyze_stages(
        snapshot.speed_chroma.operation.parent_statistics, snapshot.speed_chroma.stages,
        true, &snapshot.speed_chroma.compact_factor);
    result.adm3 = analyze_stages(
        snapshot.adm3.operation.parent_statistics, snapshot.adm3.stages, false);
    result.timeline = analyze_gpu_timeline(snapshot);
    result.feature_ranking = rank_features(snapshot);

    const auto& push = host_statistics(snapshot, SessionStage::provider_push);
    const auto provider_total = push.total + host_statistics(snapshot, SessionStage::provider_flush).total;
    if (push.call_count != 0U) {
        result.provider_average_nanoseconds = static_cast<long double>(provider_total.count()) /
            static_cast<long double>(push.call_count);
    }
    result.host_hotspots = {
        host_hotspot(snapshot, "provider", provider_child_stages, provider_total),
        host_hotspot(snapshot, "feature_record_submit", feature_record_submit_child_stages,
            host_statistics(snapshot, SessionStage::feature_record_submit).total),
        host_hotspot(snapshot, "feature_retirement", feature_retirement_child_stages,
            host_statistics(snapshot, SessionStage::readback_retirement).total),
    };

    const auto& configuration = snapshot.runtime.vulkan_runtime_configuration;
    const auto& submission = snapshot.runtime.submission_overview;
    const bool has_observation = submission.observation_duration.count() > 0;
    const auto capacity = static_cast<long double>(configuration.maximum_pending_submissions);
    result.queue.available = has_observation && capacity > 0.0L;
    result.queue.average_utilization_percent = percent(has_observation, submission.average_pending, capacity);
    result.queue.peak_utilization_percent = percent(has_observation,
        static_cast<long double>(submission.maximum_pending_observed), capacity);
    result.queue.zero_pending_percent = percent(has_observation,
        static_cast<long double>(submission.zero_pending_duration.count()),
        static_cast<long double>(submission.observation_duration.count()));

    const auto& leases = snapshot.runtime.input_lease_runtime_statistics;
    result.leases.available = leases.configured;
    result.leases.spatial_peak_percent = percent(leases.configured,
        static_cast<long double>(leases.maximum_active_spatial_leases),
        static_cast<long double>(configuration.spatial_input_lease_depth));
    result.leases.reference_luma_peak_percent = percent(leases.configured,
        static_cast<long double>(leases.maximum_active_reference_luma_leases),
        static_cast<long double>(configuration.reference_luma_lease_depth));
    result.leases.spatial_release_percent = percent(leases.configured,
        static_cast<long double>(leases.spatial_release_count),
        static_cast<long double>(leases.reservation_count));
    result.leases.reference_luma_release_percent = percent(leases.configured,
        static_cast<long double>(leases.reference_luma_release_count),
        static_cast<long double>(leases.reservation_count));
    return result;
}

} // namespace crossvmaf::cli::hotspot
