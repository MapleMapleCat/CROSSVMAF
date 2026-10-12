#pragma once

#include <filesystem>
#include <optional>
#include <string_view>
#include <vector>
#include <crossvmaf/error.hpp>

namespace crossvmaf::studio {

[[nodiscard]] std::optional<std::filesystem::path> chooseVideoFile();
// Several videos at once; empty when the dialog was cancelled.
[[nodiscard]] std::vector<std::filesystem::path> chooseVideoFiles();
[[nodiscard]] Result<void> openDirectory(const std::filesystem::path& directory);

[[nodiscard]] std::optional<std::filesystem::path> chooseReportDestination(
    std::string_view suggested_stem,
    std::string_view extension,
    std::string_view format_name);

} // namespace crossvmaf::studio
