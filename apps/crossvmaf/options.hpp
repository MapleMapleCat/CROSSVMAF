#pragma once

#include "crossvmaf/device.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/model.hpp"
#include "crossvmaf/pooling.hpp"
#include "crossvmaf/stream_options.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string_view>

namespace crossvmaf::cli {

enum class OutputFormat {
    json,
    csv,
};

struct Options {
    std::filesystem::path reference_path;
    std::filesystem::path distorted_path;
    std::optional<ModelVariant> model;
    DeviceSelector device;
    PoolMethod pool{PoolMethod::mean};
    OutputFormat output_format{OutputFormat::json};
    std::optional<std::filesystem::path> output_path;
    DecodeMode decode_mode{DecodeMode::automatic};
    std::size_t decoded_frame_queue_capacity{4U};
    std::size_t decoder_threads_per_source{};
    bool verify_cpu{};
    bool diagnostic_speed_chroma{};
    DecodeBackendPreference decode_backend{DecodeBackendPreference::automatic};
    std::string decode_device{"same"};
    DecodeTransferPolicy decode_transfer{DecodeTransferPolicy::automatic};

};

[[nodiscard]] Result<Options> parse_options(int argc, char** argv);
[[nodiscard]] std::string_view usage() noexcept;

}  // namespace crossvmaf::cli
