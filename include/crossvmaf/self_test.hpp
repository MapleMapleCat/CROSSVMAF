#pragma once

// Standalone device self-test: exercises device creation and the core's
// integer GPU pipeline check without requiring video input files.

#include "crossvmaf/device.hpp"
#include "crossvmaf/error.hpp"

#include <optional>
#include <string>
#include <vector>

namespace crossvmaf {

struct SelfTestCheck final {
    std::string name;
    bool passed{};
    std::string detail;
};

struct DeviceSelfTestResult final {
    bool passed{};
    // Valid when device enumeration/creation got far enough to identify a
    // device; default-initialized otherwise.
    DeviceInfo device;
    double elapsed_milliseconds{};
    std::vector<SelfTestCheck> checks;
    // Raw failure from the device/session layer when passed is false.
    std::optional<Error> error;
};

// Runs the integer GPU pipeline self-test on the selected device (an empty
// uuid keeps automatic selection). A failed self-test is data, not an
// exception: passed is false and error carries the raw failure.
[[nodiscard]] DeviceSelfTestResult run_device_self_test(
    const DeviceSelector& device);

}  // namespace crossvmaf
