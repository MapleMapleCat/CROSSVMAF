#include "reports/comparison_selection.hpp"

#include "i18n/translator.hpp"

#include <algorithm>
#include <array>
#include <utility>

namespace crossvmaf::studio {

std::vector<std::shared_ptr<const EvaluationReport>>
ComparisonSelection::reports() const {
    std::vector<std::shared_ptr<const EvaluationReport>> reports;
    reports.reserve(entries_.size());
    for (const Entry& entry : entries_) reports.push_back(entry.report);
    return reports;
}

bool ComparisonSelection::contains(std::string_view report_id) const noexcept {
    return std::any_of(entries_.begin(), entries_.end(),
        [report_id](const Entry& entry) {
            return entry.report->report_id == report_id;
        });
}

const EvaluationReport* ComparisonSelection::anchor() const noexcept {
    return entries_.empty() ? nullptr : entries_.front().report.get();
}

void ComparisonSelection::reset(
    std::shared_ptr<const EvaluationReport> anchor,
    std::span<const std::shared_ptr<const EvaluationReport>> related) {
    clear();
    if (add(std::move(anchor)).has_value()) return;
    for (const auto& report : related) {
        if (full()) break;
        static_cast<void>(add(report));
    }
}

void ComparisonSelection::clear() noexcept {
    entries_.clear();
    baseline_report_id_.clear();
}

std::optional<ComparisonIssue> ComparisonSelection::add(
    std::shared_ptr<const EvaluationReport> report) {
    if (!report || report->frames.empty()) {
        return ComparisonIssue{
            "empty_report", i18n::tr("The report has no frame scores"),
            report ? report->report_id : std::string{}};
    }
    if (contains(report->report_id)) {
        return ComparisonIssue{
            "duplicate_report",
            i18n::tr("The same report was added to the comparison twice"),
            report->report_id};
    }
    if (full()) {
        return ComparisonIssue{
            "too_many_reports",
            i18n::trn("Up to {0} evaluation can be compared at once",
                      "Up to {0} evaluations can be compared at once",
                      maximum_reports, std::to_string(maximum_reports)),
            report->report_id};
    }
    if (!entries_.empty()) {
        const std::array<std::shared_ptr<const EvaluationReport>, 2U> pair{
            entries_.front().report, report};
        ComparisonValidation validation = validate_comparison(pair);
        if (!validation.comparable) {
            const auto issue = std::find_if(
                validation.issues.begin(), validation.issues.end(),
                [&report](const ComparisonIssue& candidate) {
                    return candidate.report_id == report->report_id;
                });
            return std::move(issue != validation.issues.end()
                                 ? *issue
                                 : validation.issues.front());
        }
    }
    std::size_t slot = 0U;
    while (std::any_of(entries_.begin(), entries_.end(),
                       [slot](const Entry& entry) {
                           return entry.slot == slot;
                       })) {
        ++slot;
    }
    entries_.push_back({std::move(report), slot});
    return std::nullopt;
}

void ComparisonSelection::remove(std::string_view report_id) {
    // The id may live in the report being removed; keep a copy.
    const std::string removed(report_id);
    std::erase_if(entries_, [&removed](const Entry& entry) {
        return entry.report->report_id == removed;
    });
    if (baseline_report_id_ == removed) baseline_report_id_.clear();
}

void ComparisonSelection::set_baseline(std::string_view report_id) {
    baseline_report_id_ =
        contains(report_id) ? std::string(report_id) : std::string{};
}

ComparisonResult ComparisonSelection::build() const {
    ComparisonOptions options;
    options.maximum_reports = maximum_reports;
    options.baseline_report_id = baseline_report_id_;
    return build_comparison(reports(), options);
}

}  // namespace crossvmaf::studio
