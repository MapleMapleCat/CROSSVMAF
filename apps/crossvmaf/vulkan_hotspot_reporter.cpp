#include "vulkan_hotspot_reporter.hpp"
#include "vulkan_hotspot_report.hpp"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string_view>
#include <utility>

namespace crossvmaf::cli {
namespace {

constexpr std::string_view output_environment_variable =
    "CROSSVMAF_VULKAN_HOTSPOT_OUTPUT";

[[nodiscard]] std::string read_output_path_from_environment() {
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t length = 0U;
    const int status = _dupenv_s(
        &value, &length, output_environment_variable.data());
    if (status != 0 || value == nullptr) {
        std::free(value);
        return {};
    }
    std::string output_path(value);
    std::free(value);
    return output_path;
#else
    const char* const value = std::getenv(
        output_environment_variable.data());
    return value == nullptr ? std::string{} : std::string(value);
#endif
}

}  // namespace

std::shared_ptr<VulkanHotspotReporter>
VulkanHotspotReporter::create_from_environment() {
    std::string output_path = read_output_path_from_environment();
    if (output_path.empty() || output_path == "0") return nullptr;
    return std::shared_ptr<VulkanHotspotReporter>(
        new VulkanHotspotReporter(std::move(output_path)));
}

VulkanHotspotReporter::VulkanHotspotReporter(
    std::string output_path) noexcept
    : output_path_(std::move(output_path)) {}

void VulkanHotspotReporter::publish(
    crossvmaf::evaluation::telemetry::VulkanHotspotSnapshot snapshot) noexcept {
    if (report_attempted_) return;
    report_attempted_ = true;
    try {
        const std::string report_text =
            make_vulkan_hotspot_report(snapshot).dump(2) + '\n';
        if (output_path_ == "1" || output_path_ == "-" ||
            output_path_ == "stderr") {
            const std::size_t bytes_written = std::fwrite(
                report_text.data(), 1U, report_text.size(), stderr);
            if (bytes_written != report_text.size()) {
                std::fprintf(
                    stderr,
                    "[crossvmaf-hotspot] failed to write Vulkan hotspot "
                    "report to stderr\n");
            }
            return;
        }

        std::ofstream output(
            output_path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            std::fprintf(
                stderr,
                "[crossvmaf-hotspot] failed to open Vulkan hotspot output: "
                "%s\n",
                output_path_.c_str());
            return;
        }
        output.write(
            report_text.data(),
            static_cast<std::streamsize>(report_text.size()));
        output.flush();
        if (!output) {
            std::fprintf(
                stderr,
                "[crossvmaf-hotspot] failed to write Vulkan hotspot output: "
                "%s\n",
                output_path_.c_str());
        }
    } catch (...) {
        std::fprintf(
            stderr,
            "[crossvmaf-hotspot] failed to serialize Vulkan hotspot report\n");
    }
}

}  // namespace crossvmaf::cli
