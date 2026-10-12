#pragma once

// Slint-facing report view model. All domain data lives in types.hpp and all
// analysis in report_analysis.cpp / comparison.cpp; this layer only converts
// domain results into text, colors, and chart geometry for the UI.

#include "app.h"

#include "presentation/chart_data_builder.hpp"
#include "reports/comparison.hpp"
#include "reports/comparison_selection.hpp"
#include "reports/comparison_session.hpp"
#include "reports/report_analysis.hpp"
#include "types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <vector>

namespace crossvmaf::studio {

class ReportPresenter final {
public:
    ReportPresenter();

    void clearReport(const VM& view_model);
    void setReport(std::shared_ptr<const EvaluationReport> report);
    void applyReport(const VM& view_model) const;
    // Live curve from the controller's incrementally accumulated scores.
    void applyLive(
        const VM& view_model,
        std::span<const double> scores,
        const ProgressSnapshot& progress) const;
    void applyHover(
        const VM& view_model,
        double pointer_x,
        double pointer_y,
        double chart_width,
        double chart_height) const;

    // Hidden curves stay hidden while their report remains selected.
    void setComparison(const ComparisonSession& session, ComparisonResult comparison);
    // Progress of the videos still being evaluated; the compared reports
    // are unchanged.
    void setComparisonPending(std::vector<ComparisonSession::PendingTrack> pending);
    void toggleComparisonSeries(std::size_t track);
    void applyComparison(const VM& view_model) const;
    // Only the evaluations table, e.g. while a video's progress changes.
    void applyComparisonTracks(const VM& view_model) const;
    void applyComparisonHover(
        const VM& view_model, double pointer_x, double chart_width) const;

    [[nodiscard]] double reportX(std::size_t frame_index) const;
    [[nodiscard]] double reportY(double value) const;
    [[nodiscard]] const std::shared_ptr<const EvaluationReport>& report()
        const noexcept {
        return report_;
    }

private:
    [[nodiscard]] bool comparisonTrackVisible(std::size_t track) const;
    [[nodiscard]] std::size_t comparisonBaselineTrack() const;
    // Tracks in bit-rate order when comparable, selection order otherwise.
    [[nodiscard]] std::vector<std::size_t> comparisonDisplayOrder() const;
    [[nodiscard]] ChartGeometry comparisonGeometry() const;

    std::shared_ptr<const EvaluationReport> report_;
    ReportAnalysis analysis_;
    std::vector<ComparisonSelection::Entry> comparison_entries_;
    // Listed after the compared reports; CmpTrack.track continues their
    // numbering.
    std::vector<ComparisonSession::PendingTrack> comparison_pending_;
    std::filesystem::path comparison_reference_;
    ComparisonResult comparison_;
    // Per selected report, in selection order.
    std::vector<std::vector<double>> comparison_scores_;
    std::vector<SeriesStatistics> comparison_statistics_;
    std::set<std::string> hidden_comparison_ids_;
    double report_y_minimum_{50.0};
};

} // namespace crossvmaf::studio
