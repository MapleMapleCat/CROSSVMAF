#include "options.hpp"
#include "crossvmaf/version.hpp"

#include <charconv>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace crossvmaf::cli {
namespace {

[[nodiscard]] std::filesystem::path path_from_utf8(std::string_view value) {
    const auto* const data = reinterpret_cast<const char8_t*>(value.data());
    return std::filesystem::path(std::u8string(data, data + value.size()));
}

[[nodiscard]] Result<std::uint32_t> parse_u32(
    std::string_view value,
    std::string_view option) {
    std::uint32_t parsed{};
    const auto [end, error] = std::from_chars(
        value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size()) {
        return Result<std::uint32_t>::failure(
            {ErrorCode::invalid_argument,
             "invalid integer for " + std::string(option)});
    }
    return Result<std::uint32_t>::success(parsed);
}

}  // namespace

std::string_view usage() noexcept {
    return
        "CrossVMAF " CROSSVMAF_VERSION_STRING "\n"
        "Usage:\n"
        "  crossvmaf --reference REF --distorted DIST [options]\n"
        "  crossvmaf --list-models\n"
        "  crossvmaf --list-devices\n"
        "  crossvmaf --self-test [device-uuid]\n\n"
        "Input:\n"
        "  REF and DIST must contain a supported video codec.\n"
        "  Codecs: H.264, HEVC, AV1, VP8/VP9, MPEG-1/2, MPEG-4, "
        "VC-1, FLV1, MJPEG.\n"
        "  Containers: MP4/MOV/fMP4/CMAF, Matroska/WebM, IVF, "
        "MPEG-PS/TS, AVI, ASF, FLV.\n"
        "Processing options:\n"
        "  --model NAME               Force a model (default: automatic)\n"
        "                             4K: 2160p/1.5H; smaller: 1080p/3H\n"
        "                             Constant >= 50 fps: HFR; unknown/VFR: standard\n"
        "  --device auto|UUID         Default: auto\n"
        "  --decode auto|hardware|software  Default: auto; hardware requires "
        "hardware for both inputs (no software fallback)\n"
        "  --decode-backend auto|vulkan|qsv|nvdec|vaapi  Explicit backend is strict\n"
        "  --decode-device same|ID    Default: same physical GPU as compute\n"
        "  --decode-transfer auto|gpu|host  gpu forbids host transfer\n"
        "  --list-decode-backends     Show compiled backends\n"
        "  --list-decode-devices [--decode-backend NAME]\n"
        "  --pool mean|min|harmonic   Default: mean\n"
        "  --decoded-frame-queue N    Per source, 1..64; default: 4\n"
        "  --decoder-threads N        Per source, 0..64; 0 uses pair-aware auto\n"
        "  --verify-cpu               Require GPU/CPU feature agreement\n"
        "Output options:\n"
        "  --format json|csv          Default: json\n"
        "  --diagnostic-speed-chroma  Add per-frame U/V JSON diagnostics\n"
        "  --output PATH              Default: stdout\n";
}

Result<Options> parse_options(int argc, char** argv) {
    Options options{};

    for (int argument_index = 1; argument_index < argc; ++argument_index) {
        const std::string_view argument(argv[argument_index]);
        if (argument == "--verify-cpu") {
            options.verify_cpu = true;
            continue;
        }
        if (argument == "--diagnostic-speed-chroma") {
            options.diagnostic_speed_chroma = true;
            continue;
        }
        if (argument == "--help" || argument == "-h") {
            return Result<Options>::failure(
                {ErrorCode::invalid_argument, std::string(usage())});
        }

        const bool accepts_value =
            argument == "--reference" || argument == "--distorted" ||
            argument == "--model" || argument == "--device" ||
            argument == "--decode" || argument == "--pool" ||
            argument == "--decode-backend" || argument == "--decode-device" || argument == "--decode-transfer" ||
            argument == "--decoded-frame-queue" ||
            argument == "--decoder-threads" ||
            argument == "--format" || argument == "--output";
        if (!accepts_value) {
            return Result<Options>::failure(
                {ErrorCode::invalid_argument,
                 "unknown option: " + std::string(argument)});
        }
        if (argument_index + 1 >= argc) {
            return Result<Options>::failure(
                {ErrorCode::invalid_argument,
                 "missing value for " + std::string(argument)});
        }

        const std::string_view value(argv[++argument_index]);
        if (argument == "--reference") {
            options.reference_path = path_from_utf8(value);
        } else if (argument == "--distorted") {
            options.distorted_path = path_from_utf8(value);
        } else if (argument == "--model") {
            const auto parsed = parse_model(value);
            if (!parsed) return Result<Options>::failure(parsed.error());
            options.model = parsed.value();
        } else if (argument == "--device") {
            options.device.uuid = value == "auto"
                ? std::string{}
                : std::string(value);
        } else if (argument == "--decode") {
            if (value == "auto") {
                options.decode_mode = DecodeMode::automatic;
            } else if (value == "software") {
                options.decode_mode = DecodeMode::software;
            } else if (value == "hardware") {
                options.decode_mode = DecodeMode::hardware;
            } else {
                return Result<Options>::failure(
                    {ErrorCode::invalid_argument,
                     "decode mode must be auto, hardware or software"});
            }
        } else if (argument == "--decode-backend") {
            const auto parsed = parse_decode_backend_preference(value);
            if (!parsed) return Result<Options>::failure({ErrorCode::invalid_argument, "decode backend must be auto, vulkan, qsv, nvdec or vaapi"});
            options.decode_backend = *parsed;
        } else if (argument == "--decode-device") {
            if (value.empty()) return Result<Options>::failure({ErrorCode::invalid_argument, "decode device cannot be empty"});
            options.decode_device = value;
        } else if (argument == "--decode-transfer") {
            const auto parsed = parse_decode_transfer(value);
            if (!parsed) return Result<Options>::failure({ErrorCode::invalid_argument, "decode transfer must be auto, gpu or host"});
            options.decode_transfer = *parsed;
        } else if (argument == "--pool") {
            if (value == "mean") {
                options.pool = PoolMethod::mean;
            } else if (value == "min") {
                options.pool = PoolMethod::minimum;
            } else if (value == "harmonic") {
                options.pool = PoolMethod::harmonic_mean;
            } else {
                return Result<Options>::failure(
                    {ErrorCode::invalid_argument,
                     "pool method must be mean, min, or harmonic"});
            }
        } else if (argument == "--decoded-frame-queue") {
            const auto parsed = parse_u32(value, argument);
            if (!parsed) return Result<Options>::failure(parsed.error());
            if (parsed.value() == 0U || parsed.value() > 64U) {
                return Result<Options>::failure(
                    {ErrorCode::invalid_argument,
                     "decoded frame queue capacity must be between 1 and 64"});
            }
            options.decoded_frame_queue_capacity = parsed.value();
        } else if (argument == "--decoder-threads") {
            const auto parsed = parse_u32(value, argument);
            if (!parsed) return Result<Options>::failure(parsed.error());
            if (parsed.value() > 64U) {
                return Result<Options>::failure(
                    {ErrorCode::invalid_argument,
                     "decoder thread count must be between 0 and 64"});
            }
            options.decoder_threads_per_source = parsed.value();
        } else if (argument == "--format") {
            if (value == "json") {
                options.output_format = OutputFormat::json;
            } else if (value == "csv") {
                options.output_format = OutputFormat::csv;
            } else {
                return Result<Options>::failure(
                    {ErrorCode::invalid_argument,
                     "output format must be json or csv"});
            }
        } else if (argument == "--output") {
            options.output_path = path_from_utf8(value);
        }
    }

    if (options.decode_mode == DecodeMode::software &&
        (options.decode_backend != DecodeBackendPreference::automatic || options.decode_transfer == DecodeTransferPolicy::gpu_only || options.decode_device != "same"))
        return Result<Options>::failure({ErrorCode::invalid_argument, "software decode conflicts with hardware backend/device or GPU-only transfer"});
    if (options.reference_path.empty() || options.distorted_path.empty()) {
        return Result<Options>::failure(
            {ErrorCode::invalid_argument,
             "--reference and --distorted are required"});
    }
    if (options.diagnostic_speed_chroma &&
        options.output_format != OutputFormat::json) {
        return Result<Options>::failure(
            {ErrorCode::invalid_argument,
             "--diagnostic-speed-chroma requires --format json"});
    }
    return Result<Options>::success(std::move(options));
}

}  // namespace crossvmaf::cli
