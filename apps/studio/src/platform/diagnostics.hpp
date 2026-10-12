#pragma once

// Task diagnostics and support-bundle export (P1). Packages the version,
// device, resolved configuration, error, and stream statistics of one task
// into a single JSON document the user can attach to a report.

#include "types.hpp"

#include <nlohmann/json.hpp>

#include <crossvmaf/error.hpp>

#include <filesystem>

namespace crossvmaf::studio {

[[nodiscard]] nlohmann::json build_task_diagnostics(
    const TaskSnapshot& task);

[[nodiscard]] Result<void> export_support_bundle(
    const nlohmann::json& diagnostics,
    const std::filesystem::path& destination);

}  // namespace crossvmaf::studio
