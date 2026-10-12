#pragma once

#include "evaluation/session_telemetry.hpp"

#include <memory>
#include <string>

namespace crossvmaf::cli {

class VulkanHotspotReporter final
    : public crossvmaf::evaluation::telemetry::SnapshotSink {
public:
    [[nodiscard]] static std::shared_ptr<VulkanHotspotReporter>
    create_from_environment();

    VulkanHotspotReporter(const VulkanHotspotReporter&) = delete;
    VulkanHotspotReporter& operator=(const VulkanHotspotReporter&) = delete;
    ~VulkanHotspotReporter() override = default;

    void publish(
        crossvmaf::evaluation::telemetry::VulkanHotspotSnapshot snapshot)
        noexcept override;

private:
    explicit VulkanHotspotReporter(std::string output_path) noexcept;

    std::string output_path_;
    bool report_attempted_{};
};

}  // namespace crossvmaf::cli
