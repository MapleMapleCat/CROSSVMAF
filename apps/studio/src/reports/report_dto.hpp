#pragma once

// Versioned report DTO serialization shared by the report store and the
// JSON export. One schema serves both so a persisted report and an exported
// report carry identical data: schema_version, actual model, math mode,
// media metadata/timeline snapshot, pooling configuration, and analysis
// rule version.

#include "types.hpp"

#include <nlohmann/json.hpp>

#include <string_view>

namespace crossvmaf::studio {

inline constexpr std::string_view report_schema_version{"1.0.0"};

[[nodiscard]] nlohmann::json evaluation_report_to_json(
    const EvaluationReport& report);

// Accepts schema major version 1 only; older unversioned exports and newer
// majors fail with a clear unsupported_format error.
[[nodiscard]] Result<EvaluationReport> evaluation_report_from_json(
    const nlohmann::json& document);

}  // namespace crossvmaf::studio
