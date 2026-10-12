#pragma once

#include "reports/comparison.hpp"
#include "types.hpp"

#include <crossvmaf/error.hpp>

#include <filesystem>

namespace crossvmaf::studio {

enum class ReportFormat {
    json,
    csv,
    pdf,
};

// JSON output is the shared versioned DTO (report_dto.cpp); CSV is the flat
// engineering table with raw Q32.32 feature columns and frame timing. PDF is
// the localized customer report with embedded fonts and vector charts.
[[nodiscard]] Result<void> writeEvaluationReport(
    const std::filesystem::path& destination,
    ReportFormat format,
    const EvaluationReport& report);

[[nodiscard]] Result<void> writeComparisonReport(
    const std::filesystem::path& destination,
    const ComparisonResult& comparison);

}  // namespace crossvmaf::studio
