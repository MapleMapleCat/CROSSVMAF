#include "crossvmaf/crossvmaf.hpp"
#include "speed_chroma_probe.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    if (argc == 3 &&
        std::string_view(argv[1]) == "--speed-chroma-probe") {
        return crossvmaf::conformance::run_speed_chroma_probe(
            std::filesystem::path(argv[2]));
    }
    const auto result = crossvmaf::run_device_self_test({});
    if (!result.passed) {
        std::cerr << "backend_self_test=fail\nerror="
                  << (result.error ? result.error->message : "self-test failed") << '\n';
        return 5;
    }

    const auto& device = result.device;
    std::cout << "crossvmaf_conformance_complete=false\n"
              << "implemented_gate=integer_backend_self_test\n"
              << "backend_self_test=pass\n"
              << "device_name=" << device.name << '\n'
              << "device_uuid=" << device.uuid << '\n'
              << "vendor_id=" << device.vendor_id << '\n'
              << "device_id=" << device.device_id << '\n'
              << "api_version_raw=" << device.api_version << '\n'
              << "driver_version_raw=" << device.driver_version << '\n'
              << "initialization_ms=" << std::fixed << std::setprecision(3)
              << result.elapsed_milliseconds << '\n';
    return 0;
}
