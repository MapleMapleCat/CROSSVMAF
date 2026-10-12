#pragma once

#include <filesystem>

namespace crossvmaf::conformance {

[[nodiscard]] int run_speed_chroma_probe(
    const std::filesystem::path& output_directory);

}  // namespace crossvmaf::conformance
