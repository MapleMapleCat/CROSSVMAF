#pragma once

#include "evaluation/session_telemetry.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace crossvmaf::cli::hotspot {

using evaluation::telemetry::SessionStage;

inline constexpr std::array provider_child_stages{
    SessionStage::input_commit, SessionStage::feature_record_submit,
    SessionStage::nonblocking_retire, SessionStage::readback_retirement,
    SessionStage::flush_drain,
};
inline constexpr std::array feature_record_submit_child_stages{
    SessionStage::cambi_record, SessionStage::cambi_submit,
    SessionStage::speed_chroma_record, SessionStage::speed_chroma_submit,
    SessionStage::adm3_record, SessionStage::adm3_submit,
    SessionStage::motion3_record, SessionStage::motion3_submit,
};
inline constexpr std::array feature_retirement_child_stages{
    SessionStage::cambi_retire, SessionStage::speed_chroma_retire,
    SessionStage::adm3_retire, SessionStage::motion3_retire,
};

// Analysis uses unrounded values in the original clock domains. JSON field
// names, unit conversion, rounding, and unavailable-value encoding belong to
// the report serializer; analysis never consumes serialized report fields.
struct FeatureStageAnalysis final {
    bool available{};
    long double children_nanoseconds{};
    long double residual_nanoseconds{};
    std::optional<std::size_t> hotspot_index;
};

struct TimelineAnalysis final {
    bool available{};
    std::uint64_t valid_event_count{};
    std::uint64_t invalid_event_count{};
    long double span_nanoseconds{};
    long double union_busy_nanoseconds{};
    long double summed_feature_nanoseconds{};
    long double extra_concurrent_nanoseconds{};
    std::optional<long double> busy_percent;
    std::optional<long double> average_active_intervals;
};

struct FeatureRanking final {
    std::string_view name;
    long double average_nanoseconds{};
    std::optional<long double> coverage_percent;
    std::optional<long double> failure_percent;
    std::optional<long double> unavailable_percent;
};

struct HostHotspot final {
    std::string_view scope;
    std::optional<SessionStage> stage;
    std::chrono::nanoseconds total{};
};

struct QueueAnalysis final {
    bool available{};
    std::optional<long double> average_utilization_percent;
    std::optional<long double> peak_utilization_percent;
    std::optional<long double> zero_pending_percent;
};

struct LeaseAnalysis final {
    bool available{};
    std::optional<long double> spatial_peak_percent;
    std::optional<long double> reference_luma_peak_percent;
    std::optional<long double> spatial_release_percent;
    std::optional<long double> reference_luma_release_percent;
};

struct Analysis final {
    FeatureStageAnalysis cambi;
    FeatureStageAnalysis speed_chroma;
    FeatureStageAnalysis adm3;
    TimelineAnalysis timeline;
    std::vector<FeatureRanking> feature_ranking;
    std::array<HostHotspot, 3U> host_hotspots;
    std::optional<long double> provider_average_nanoseconds;
    QueueAnalysis queue;
    LeaseAnalysis leases;
};

[[nodiscard]] Analysis analyze(
    const evaluation::telemetry::VulkanHotspotSnapshot& snapshot);

} // namespace crossvmaf::cli::hotspot
