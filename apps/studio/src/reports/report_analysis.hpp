#pragma once

// Unified report analysis (P1). Pure domain logic extracted from the Slint
// report model: statistics, low-score segments, and series decimation return
// numbers, times, and ranges — never colors or SVG paths. One policy version
// covers quantile rules and thresholds so UI, exports, and future consumers
// share identical results.

#include "types.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace crossvmaf::studio {

struct AnalysisPolicy final {
    static constexpr std::string_view version{"1.0"};

    double low_score_threshold{90.0};
    std::size_t merge_distance_frames{30U};
    std::size_t maximum_listed_segments{6U};
    double visually_lossless_threshold{93.0};
};

struct SeriesStatistics final {
    double mean{};
    double minimum{};
    double harmonic_mean{};
    double standard_deviation{};
    double percentile_5{};
    double median{};
    std::size_t minimum_index{};
};

struct LowScoreSegment final {
    std::size_t start{};
    std::size_t end{};
    std::size_t minimum_index{};
    // Media time on the reference timeline when frame timings are valid.
    std::optional<double> start_seconds;
    std::optional<double> end_seconds;
    std::optional<double> minimum_seconds;
};

struct ReportAnalysis final {
    SeriesStatistics statistics;
    std::vector<LowScoreSegment> segments;
    // Segments ordered by lowest minimum score, capped by policy.
    std::vector<LowScoreSegment> top_segments;
    std::size_t visually_lossless_count{};
    double visually_lossless_fraction{};
    std::string_view policy_version{AnalysisPolicy::version};
};

[[nodiscard]] std::vector<double> report_scores(
    const EvaluationReport& report);

[[nodiscard]] SeriesStatistics calculate_statistics(
    std::span<const double> values);

// Frames at or above the policy's visually lossless threshold.
[[nodiscard]] std::size_t count_visually_lossless(
    std::span<const double> values,
    const AnalysisPolicy& policy = {});

[[nodiscard]] ReportAnalysis analyze_report(
    const EvaluationReport& report,
    const AnalysisPolicy& policy = {});

// Min–max decimation preserving valleys; returns (frame_index, value) pairs.
[[nodiscard]] std::vector<std::pair<std::size_t, double>> decimate_series(
    std::span<const double> values,
    std::size_t maximum_points);

[[nodiscard]] const FrameResult* get_frame(
    const EvaluationReport& report,
    std::uint64_t frame_index) noexcept;

// Media time of a frame when its timing is valid or non-monotonic (missing
// timings yield nullopt rather than a synthesized time).
[[nodiscard]] std::optional<double> frame_media_time_seconds(
    const FrameResult& frame) noexcept;

[[nodiscard]] int qualityBandKind(double value) noexcept;

} // namespace crossvmaf::studio
