#pragma once

#include "reports/evaluation_report.hpp"
#include <crossvmaf/error.hpp>
#include <filesystem>

namespace crossvmaf::studio {

// Self-contained, vector PDF in the current Studio language. Reads only the
// report snapshot; neither source videos nor system fonts are required.
[[nodiscard]] Result<void> writeEvaluationPdf(
    const std::filesystem::path& destination, const EvaluationReport& report);

} // namespace crossvmaf::studio
