#include "presentation/display_format.hpp"

#include "i18n/translator.hpp"

#include <cmath>
#include <cstdio>
#include <ctime>

namespace crossvmaf::studio {

namespace {

[[nodiscard]] std::string withThousandsSeparators(std::string value) {
    const std::size_t decimal_position = value.find('.');
    std::size_t integer_length = decimal_position == std::string::npos
        ? value.size()
        : decimal_position;
    for (std::size_t position = integer_length; position > 3U;) {
        position -= 3U;
        value.insert(position, ",");
    }
    return value;
}

[[nodiscard]] std::string pathToUtf8(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

} // namespace

[[nodiscard]] std::string formatTimestamp(
    std::chrono::system_clock::time_point time_point) {
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(time_point);
    std::tm local_time{};
#ifdef _WIN32
    localtime_s(&local_time, &raw_time);
#else
    localtime_r(&raw_time, &local_time);
#endif
    char buffer[32]{};
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M", &local_time);
    return buffer;
}

[[nodiscard]] std::string compactFrameLabel(std::size_t frame_index) {
    if (frame_index >= 1'000'000U) {
        return formatDecimal(static_cast<double>(frame_index) / 1'000'000.0, 1) + "M";
    }
    if (frame_index >= 1'000U) {
        return formatDecimal(static_cast<double>(frame_index) / 1'000.0, 1) + "k";
    }
    return std::to_string(frame_index);
}

[[nodiscard]] std::string pixelFormatName(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::yuv420p: return "YUV 4:2:0";
    case PixelFormat::yuv422p: return "YUV 4:2:2";
    case PixelFormat::yuv444p: return "YUV 4:4:4";
    }
    return "YUV";
}

std::string formatDecimal(double value, int decimal_places) {
    char buffer[64]{};
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimal_places, value);
    return withThousandsSeparators(buffer);
}

std::string formatInteger(double value) {
    char buffer[64]{};
    std::snprintf(buffer, sizeof(buffer), "%.0f", std::floor(value + 0.5));
    return withThousandsSeparators(buffer);
}

std::string formatByteSize(std::uintmax_t byte_count) {
    constexpr double bytes_per_kibibyte = 1024.0;
    constexpr double bytes_per_mebibyte = bytes_per_kibibyte * 1024.0;
    constexpr double bytes_per_gibibyte = bytes_per_mebibyte * 1024.0;
    if (byte_count >= static_cast<std::uintmax_t>(bytes_per_gibibyte)) {
        return formatDecimal(
            static_cast<double>(byte_count) / bytes_per_gibibyte, 2) + " GiB";
    }
    if (byte_count >= static_cast<std::uintmax_t>(bytes_per_mebibyte)) {
        return formatDecimal(
            static_cast<double>(byte_count) / bytes_per_mebibyte, 1) + " MiB";
    }
    if (byte_count >= static_cast<std::uintmax_t>(bytes_per_kibibyte)) {
        return formatDecimal(
            static_cast<double>(byte_count) / bytes_per_kibibyte, 1) + " KiB";
    }
    return std::to_string(byte_count) + " B";
}

std::string formatBitRate(std::uint64_t bits_per_second) {
    constexpr double bits_per_megabit = 1.0e6;
    constexpr double bits_per_kilobit = 1.0e3;
    if (bits_per_second >= static_cast<std::uint64_t>(bits_per_megabit)) {
        return formatDecimal(
            static_cast<double>(bits_per_second) / bits_per_megabit, 2) +
            " Mbps";
    }
    if (bits_per_second >= static_cast<std::uint64_t>(bits_per_kilobit)) {
        return formatDecimal(
            static_cast<double>(bits_per_second) / bits_per_kilobit, 0) +
            " kbps";
    }
    return std::to_string(bits_per_second) + " bps";
}

std::string formatMediaTime(double seconds) {
    if (seconds < 0.0) seconds = 0.0;
    const auto total_seconds = static_cast<std::uint64_t>(seconds);
    const double fraction = seconds - static_cast<double>(total_seconds);
    const std::uint64_t hours = total_seconds / 3600U;
    const std::uint64_t minutes = (total_seconds % 3600U) / 60U;
    const std::uint64_t secs = total_seconds % 60U;
    char buffer[40]{};
    if (hours > 0U) {
        std::snprintf(
            buffer, sizeof(buffer), "%llu:%02llu:%04.1f",
            static_cast<unsigned long long>(hours),
            static_cast<unsigned long long>(minutes),
            static_cast<double>(secs) + fraction);
    } else {
        std::snprintf(
            buffer, sizeof(buffer), "%02llu:%04.1f",
            static_cast<unsigned long long>(minutes),
            static_cast<double>(secs) + fraction);
    }
    return buffer;
}

std::string fileDisplayName(const std::filesystem::path& path) {
    const std::filesystem::path filename = path.filename();
    return pathToUtf8(filename.empty() ? path : filename);
}

std::string pairDisplayName(const EvaluationRequest& request) {
    return fileDisplayName(request.distorted_path) + " ↔ " +
        fileDisplayName(request.reference_path);
}

std::string qualityBandName(int kind) {
    switch (kind) {
    // English uses the usual VMAF band names; zh_CN describes visibility.
    case 0: return i18n::tr("Excellent");
    case 1: return i18n::tr("Good");
    case 2: return i18n::tr("Fair");
    default: return i18n::tr("Poor");
    }
}

} // namespace crossvmaf::studio
