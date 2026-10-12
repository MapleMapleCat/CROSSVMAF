#pragma once

#include "evaluation/evaluation_request.hpp"

#include <crossvmaf/frame.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace crossvmaf::studio {

[[nodiscard]] std::string formatDecimal(double value, int decimal_places = 1);
[[nodiscard]] std::string formatInteger(double value);
[[nodiscard]] std::string formatByteSize(std::uintmax_t byte_count);
[[nodiscard]] std::string formatBitRate(std::uint64_t bits_per_second);
[[nodiscard]] std::string formatMediaTime(double seconds);
[[nodiscard]] std::string fileDisplayName(const std::filesystem::path& path);
[[nodiscard]] std::string pairDisplayName(const EvaluationRequest& request);
[[nodiscard]] std::string qualityBandName(int kind);

[[nodiscard]] std::string formatTimestamp(std::chrono::system_clock::time_point time_point);
[[nodiscard]] std::string compactFrameLabel(std::size_t frame_index);
[[nodiscard]] std::string pixelFormatName(PixelFormat pixel_format);

} // namespace crossvmaf::studio
