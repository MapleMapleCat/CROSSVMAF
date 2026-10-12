#include "measurement.hpp"
#include "options.hpp"

#include "crossvmaf/crossvmaf.hpp"

#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace {

void print_version(std::uint32_t version) {
    std::cout << ((version >> 22U) & 0x7fU) << '.'
              << ((version >> 12U) & 0x3ffU) << '.'
              << (version & 0xfffU);
}

int run_self_test(std::string uuid) {
    crossvmaf::DeviceSelector selector;
    selector.uuid = std::move(uuid);
    const auto result = crossvmaf::run_device_self_test(selector);
    if (!result.passed) {
        if (result.error) std::cerr << result.error->message << '\n';
        return 5;
    }

    const crossvmaf::DeviceInfo& device = result.device;
    std::cout << "integer GPU self-test: pass\n"
              << "device: " << device.name << '\n'
              << "uuid: " << device.uuid << '\n'
              << "api: ";
    print_version(device.api_version);
    std::cout << "\ndriver_raw: " << device.driver_version << '\n'
              << "initialization_ms: " << std::fixed
              << std::setprecision(3) << result.elapsed_milliseconds << '\n';
    return 0;
}

int run(int argc, char** argv) try {
    if (argc == 1) {
        std::cout << crossvmaf::cli::usage();
        return 0;
    }
    if (argc == 2 &&
        (std::string_view(argv[1]) == "--help" ||
         std::string_view(argv[1]) == "-h")) {
        std::cout << crossvmaf::cli::usage();
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--list-decode-backends") {
        for (const auto& backend : crossvmaf::enumerate_decode_backends())
            std::cout << crossvmaf::to_string(backend.backend) << "  " << backend.status << '\n';
        return 0;
    }
    if ((argc == 2 || argc == 4) && std::string_view(argv[1]) == "--list-decode-devices") {
        auto preference = crossvmaf::DecodeBackendPreference::automatic;
        if (argc == 4) {
            const auto parsed = crossvmaf::parse_decode_backend_preference(argv[3]);
            if (std::string_view(argv[2]) != "--decode-backend" || !parsed) { std::cerr << "invalid backend selector\n"; return 2; }
            preference = *parsed;
        }
        auto devices = crossvmaf::enumerate_decode_devices(preference);
        if (!devices) { std::cerr << devices.error().message << '\n'; return 4; }
        for (const auto& device : devices.value()) std::cout << device.id << "  " << device.name << "  candidate\n";
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--list-models") {
        for (const crossvmaf::ModelVariant model : crossvmaf::all_models()) {
            std::cout << crossvmaf::to_string(model) << '\n';
        }
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--list-devices") {
        const auto devices = crossvmaf::EvaluationStream::enumerate_devices();
        if (!devices) {
            std::cerr << devices.error().message << '\n';
            return 4;
        }
        for (const crossvmaf::DeviceInfo& device : devices.value()) {
            std::cout << device.uuid << "  " << device.name << "  "
                      << (device.discrete ? "discrete" : "integrated")
                      << "  Vulkan ";
            print_version(device.api_version);
            std::cout << '\n';
        }
        return 0;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
        return run_self_test({});
    }
    if (argc == 3 && std::string_view(argv[1]) == "--self-test") {
        return run_self_test(argv[2]);
    }

    const auto options = crossvmaf::cli::parse_options(argc, argv);
    if (!options) {
        std::cerr << options.error().message << '\n';
        return 2;
    }
    return crossvmaf::cli::run_measurement(options.value());
} catch (const std::exception& error) {
    std::cerr << "CrossVMAF failed: " << error.what() << '\n';
    return 5;
} catch (...) {
    std::cerr << "CrossVMAF failed with an unknown exception\n";
    return 5;
}

}  // namespace

#if defined(_WIN32)
namespace {

[[nodiscard]] std::string utf8_from_wide(std::wstring_view value) {
    if (value.empty()) return {};
    const int size = WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0) return {};
    std::string output(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(
        CP_UTF8, WC_ERR_INVALID_CHARS, value.data(),
        static_cast<int>(value.size()), output.data(), size,
        nullptr, nullptr);
    return output;
}

}  // namespace

int wmain(int argc, wchar_t** wide_argv) {
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
        arguments.push_back(utf8_from_wide(wide_argv[index]));
    }
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) {
        argv.push_back(argument.data());
    }
    return run(argc, argv.data());
}
#else
int main(int argc, char** argv) {
    return run(argc, argv);
}
#endif
