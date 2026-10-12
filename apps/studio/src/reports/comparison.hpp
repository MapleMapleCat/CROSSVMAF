#pragma once

// Strict multi-track comparison (P1). Validation compares the facts that
// make two evaluations comparable — reference identity, spec, timeline,
// model, and score-affecting configuration — and reports the reason when
// they are not. Results sort by bit rate with an explicit baseline and
// label the origin of every rate figure.

#include "reports/report_analysis.hpp"
#include "types.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace crossvmaf::studio {

struct ComparisonIssue final {
    // Stable codes: different_reference, different_model, different_spec,
    // different_frame_count, different_timeline,
    // duplicate_report, insufficient_reports; comparison_selection.cpp adds
    // empty_report and too_many_reports.
    std::string reason_code;
    std::string message;
    std::string report_id;
};

struct ComparisonValidation final {
    bool comparable{};
    std::vector<ComparisonIssue> issues;
    // Reports that passed validation against the baseline, in input order.
    std::vector<std::size_t> eligible_indices;
};

// The first report is the reference point every other candidate is
// validated against (typically the currently displayed report).
[[nodiscard]] ComparisonValidation validate_comparison(
    std::span<const std::shared_ptr<const EvaluationReport>> reports);

struct ComparisonOptions final {
    std::size_t maximum_reports{4U};
    // Explicit baseline report id; empty keeps the first eligible report.
    std::string baseline_report_id;
};

struct ComparisonRow final {
    std::string report_id;
    std::string display_name;
    double mean{};
    double minimum{};
    double percentile_5{};
    double delta_vs_baseline{};
    bool is_baseline{};
    // Coded video average bit rate when the media snapshot carries one.
    std::optional<std::uint64_t> average_bit_rate;
    BitRateSource bit_rate_source{BitRateSource::video_stream};
    // Distorted file size snapshot from the stored media identity.
    std::optional<std::uint64_t> file_size_bytes;
};

struct ComparisonResult final {
    ComparisonValidation validation;
    std::string baseline_report_id;
    // Sorted by average bit rate ascending; rows without a bit rate keep
    // their input order at the end.
    std::vector<ComparisonRow> rows;
    std::vector<std::string> warnings;
};

[[nodiscard]] ComparisonResult build_comparison(
    std::span<const std::shared_ptr<const EvaluationReport>> reports,
    const ComparisonOptions& options = {});

// Facts worth stating in words next to a comparison. Row indices refer to
// ComparisonResult::rows.
struct ComparisonHighlights final {
    std::optional<std::size_t> highest_mean;
    // Every row has a bit rate, so no unknown rate could be lower than the
    // one reported below.
    bool all_rates_known{};
    // Lowest bit rate whose mean reaches the visually lossless threshold;
    // only when all rates are known.
    std::optional<std::size_t> lowest_visually_lossless_rate;
    // Row whose mean is furthest below the baseline, when any is below it.
    std::optional<std::size_t> largest_drop;
    // Frame where the curves of the compared reports are furthest apart.
    std::optional<std::size_t> widest_gap_frame;
    double widest_gap{};
};

// `reports` are the compared reports, in any order; they supply the
// per-frame scores for the widest gap.
[[nodiscard]] ComparisonHighlights summarize_comparison(
    const ComparisonResult& comparison,
    std::span<const std::shared_ptr<const EvaluationReport>> reports,
    const AnalysisPolicy& policy = {});

}  // namespace crossvmaf::studio
