#include "app/settings.hpp"

#include "i18n/translator.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace crossvmaf::studio {
namespace {
bool valid(const StudioSettings& s) {
    return s.theme >= 0 && s.theme <= 1 &&
        s.export_format >= 0 && s.export_format <= 1 &&
        to_string(s.decode_mode) != "unknown" && to_string(s.decode_backend) != "unknown" &&
        to_string(s.decode_transfer) != "unknown" && !s.decode_device.empty() &&
        (s.decode_mode != DecodeMode::software || (s.decode_backend == DecodeBackendPreference::automatic &&
         s.decode_transfer != DecodeTransferPolicy::gpu_only && s.decode_device == "same")) &&
        s.decoder_threads_per_source <= 64 &&
        s.decoded_frame_queue_capacity >= 1 && s.decoded_frame_queue_capacity <= 64;
}
} // namespace

Result<StudioSettings> load_settings(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) && !ec)
        return Result<StudioSettings>::success({});
    std::ifstream input(path, std::ios::binary);
    if (!input) return Result<StudioSettings>::failure(
        {ErrorCode::io_error, i18n::tr("Cannot read the settings file")});
    try {
        const auto json = nlohmann::json::parse(input);
        StudioSettings settings;
        settings.theme = json.value("theme", 0);
        // Unknown tags (a language removed in a later version) follow the system.
        const auto language = json.value("language", std::string("system"));
        if (language != "system") settings.language = i18n::parse_language_tag(language);
        settings.device_uuid = json.value("device_uuid", std::string{});
        settings.export_format = json.value("export_format", 0);
        const auto mode = json.value("decode_mode", std::string("auto"));
        if (mode != "auto" && mode != "hardware" && mode != "software")
            throw std::runtime_error("invalid decode mode");
        settings.decode_mode = mode == "hardware" ? DecodeMode::hardware :
            mode == "software" ? DecodeMode::software : DecodeMode::automatic;
        const auto backend = parse_decode_backend_preference(json.value("decode_backend", std::string("auto")));
        const auto transfer = parse_decode_transfer(json.value("decode_transfer", std::string("auto")));
        if (!backend || !transfer) throw std::runtime_error("invalid decode selection");
        settings.decode_backend = *backend;
        settings.decode_transfer = *transfer;
        settings.decode_device = json.value("decode_device", std::string("same"));
        // Read signed values first so negative JSON values cannot wrap to size_t.
        const int threads = json.value("decoder_threads_per_source", 0);
        const int capacity = json.value("decoded_frame_queue_capacity", 4);
        if (threads < 0 || threads > 64 || capacity < 1 || capacity > 64)
            throw std::runtime_error("invalid decode limits");
        settings.decoder_threads_per_source = static_cast<std::size_t>(threads);
        settings.decoded_frame_queue_capacity = static_cast<std::size_t>(capacity);
        const auto policy = json.value("queue_failure_policy", std::string("pause"));
        if (policy != "pause" && policy != "continue")
            throw std::runtime_error("invalid failure policy");
        settings.failure_policy = policy == "continue" ?
            QueueFailurePolicy::continue_on_failure : QueueFailurePolicy::pause_on_failure;
        if (!valid(settings)) throw std::runtime_error("invalid settings");
        return Result<StudioSettings>::success(std::move(settings));
    } catch (const std::exception&) {
        return Result<StudioSettings>::failure(
            {ErrorCode::invalid_argument,
             i18n::tr("The settings file is invalid; defaults are in use")});
    }
}

Result<void> save_settings(const std::filesystem::path& path, const StudioSettings& settings) {
    if (!valid(settings)) return Result<void>::failure(
        {ErrorCode::invalid_argument, i18n::tr("A setting is out of range")});
    std::error_code ec;
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) return Result<void>::failure(
        {ErrorCode::io_error, i18n::tr("Cannot create the settings folder")});
    const nlohmann::json json{
        {"theme", settings.theme}, {"device_uuid", settings.device_uuid},
        {"language", settings.language.has_value()
            ? std::string(i18n::language_info(*settings.language).tag)
            : std::string("system")},
        {"export_format", settings.export_format},
        {"decode_mode", to_string(settings.decode_mode)},
        {"decode_backend", to_string(settings.decode_backend)},
        {"decode_transfer", to_string(settings.decode_transfer)},
        {"decode_device", settings.decode_device},
        {"decoder_threads_per_source", settings.decoder_threads_per_source},
        {"decoded_frame_queue_capacity", settings.decoded_frame_queue_capacity},
        {"queue_failure_policy", settings.failure_policy == QueueFailurePolicy::pause_on_failure ? "pause" : "continue"},
    };
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output << json.dump(2) << '\n';
        output.flush();
        if (!output) return Result<void>::failure(
            {ErrorCode::io_error, i18n::tr("Cannot write the settings file")});
        output.close();
        if (!output) return Result<void>::failure(
            {ErrorCode::io_error, i18n::tr("Cannot save the settings file")});
    }
#ifdef _WIN32
    const bool replaced = MoveFileExW(temporary.c_str(), path.c_str(),
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::filesystem::rename(temporary, path, ec);
    const bool replaced = !ec;
#endif
    if (!replaced) {
        std::filesystem::remove(temporary, ec);
        return Result<void>::failure({ErrorCode::io_error,
            i18n::tr("Cannot replace the settings file; the previous settings were kept")});
    }
    return Result<void>::success();
}
} // namespace crossvmaf::studio
