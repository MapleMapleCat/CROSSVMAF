#pragma once

// Persistent report library (P1). Reports are stored as one versioned JSON
// document per report plus a lightweight index for fast listing. Writes go
// through a temporary file + rename so a failed write never leaves half a
// successful record.

#include "types.hpp"

#include <crossvmaf/error.hpp>

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace crossvmaf::studio {

struct ReportSummary final {
    std::string report_id;
    std::string distorted_name;
    std::string reference_name;
    double pooled_score{};
    std::int64_t completed_at_utc_seconds{};
    std::uint64_t frame_count{};
    ModelVariant model{default_model};
    SceneId scene{SceneId::tv_standard};
};

// The Compare screen's comparison, kept so it survives a restart. Only
// finished reports are kept; evaluations still running are not.
struct SavedComparison final {
    std::filesystem::path reference_path;
    SceneId scene{SceneId::tv_standard};
    std::vector<std::string> report_ids;
    std::string baseline_report_id;
};

struct ReportListPage final {
    std::vector<ReportSummary> items;
    // Present when more items remain; pass as the next list_reports cursor.
    std::optional<std::size_t> next_cursor;
    // Index entries whose backing file was unreadable at list time.
    std::vector<std::string> corrupt_report_ids;
};

class ReportStore final {
public:
    explicit ReportStore(std::filesystem::path directory);

    // Saves the report and returns its stable id. The report's own id is
    // kept when it is non-empty and unused.
    [[nodiscard]] Result<std::string> save_report(
        const EvaluationReport& report);
    [[nodiscard]] Result<ReportListPage> list_reports(
        std::size_t cursor,
        std::size_t limit) const;
    [[nodiscard]] Result<EvaluationReport> load_report(
        const std::string& report_id) const;
    [[nodiscard]] Result<void> delete_report(const std::string& report_id);
    // Imports an external versioned report JSON; assigns a fresh id when the
    // embedded one is missing or already taken.
    [[nodiscard]] Result<std::string> import_report(
        const std::filesystem::path& path);

    [[nodiscard]] Result<void> save_comparison(const SavedComparison& comparison) const;
    // Empty when nothing was saved or the file cannot be read.
    [[nodiscard]] std::optional<SavedComparison> load_comparison() const;

    [[nodiscard]] const std::filesystem::path& directory() const noexcept {
        return directory_;
    }

private:
    [[nodiscard]] std::filesystem::path report_path(
        const std::string& report_id) const;
    [[nodiscard]] std::filesystem::path index_path() const;
    [[nodiscard]] std::filesystem::path comparison_path() const;
    [[nodiscard]] Result<std::vector<ReportSummary>> read_index() const;
    [[nodiscard]] Result<void> write_index(
        const std::vector<ReportSummary>& entries) const;
    [[nodiscard]] std::string generate_report_id() const;

    std::filesystem::path directory_;
};

}  // namespace crossvmaf::studio
