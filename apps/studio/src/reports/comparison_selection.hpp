#pragma once

// The evaluations shown on the Compare screen. Every report is validated
// against the first one when it is added, so the selection is always
// comparable. Each report keeps its color slot while it stays selected:
// removing or hiding one report never repaints the others.

#include "reports/comparison.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crossvmaf::studio {

class ComparisonSelection final {
public:
    static constexpr std::size_t maximum_reports{4U};

    struct Entry final {
        std::shared_ptr<const EvaluationReport> report;
        // Unique within the selection, below maximum_reports.
        std::size_t slot{};
    };

    [[nodiscard]] const std::vector<Entry>& entries() const noexcept {
        return entries_;
    }
    [[nodiscard]] std::vector<std::shared_ptr<const EvaluationReport>>
    reports() const;
    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] bool full() const noexcept {
        return entries_.size() >= maximum_reports;
    }
    [[nodiscard]] bool contains(std::string_view report_id) const noexcept;
    // The report every other one was validated against.
    [[nodiscard]] const EvaluationReport* anchor() const noexcept;
    // Empty while the first report is the baseline.
    [[nodiscard]] const std::string& baseline_report_id() const noexcept {
        return baseline_report_id_;
    }

    // Starts over with `anchor`, then adds each comparable report of
    // `related`, in order, while there is room.
    void reset(std::shared_ptr<const EvaluationReport> anchor,
               std::span<const std::shared_ptr<const EvaluationReport>>
                   related = {});
    void clear() noexcept;
    // Adds the report when it is comparable with the selection and there is
    // room; otherwise returns why it was not added.
    [[nodiscard]] std::optional<ComparisonIssue> add(
        std::shared_ptr<const EvaluationReport> report);
    void remove(std::string_view report_id);
    // An id outside the selection resets the baseline to the first report.
    void set_baseline(std::string_view report_id);

    [[nodiscard]] ComparisonResult build() const;

private:
    std::vector<Entry> entries_;
    std::string baseline_report_id_;
};

}  // namespace crossvmaf::studio
