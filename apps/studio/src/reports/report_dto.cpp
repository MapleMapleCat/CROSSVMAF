#include "reports/report_dto.hpp"

#include "i18n/translator.hpp"
#include "reports/report_analysis.hpp"

#include <chrono>
#include <exception>
#include <string>
#include <utility>

namespace crossvmaf::studio {
namespace {

[[nodiscard]] std::string path_to_utf8(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

[[nodiscard]] std::string_view pixel_format_name(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::yuv420p: return "yuv420p";
    case PixelFormat::yuv422p: return "yuv422p";
    case PixelFormat::yuv444p: return "yuv444p";
    }
    return "unknown";
}

[[nodiscard]] PixelFormat parse_pixel_format(std::string_view name) {
    if (name == "yuv422p") return PixelFormat::yuv422p;
    if (name == "yuv444p") return PixelFormat::yuv444p;
    return PixelFormat::yuv420p;
}

[[nodiscard]] std::string_view codec_name(StreamVideoCodec codec) {
    switch (codec) {
    case StreamVideoCodec::h264: return "h264";
    case StreamVideoCodec::hevc: return "hevc";
    case StreamVideoCodec::av1: return "av1";
    case StreamVideoCodec::vp8: return "vp8";
    case StreamVideoCodec::vp9: return "vp9";
    case StreamVideoCodec::mpeg1: return "mpeg1";
    case StreamVideoCodec::mpeg2: return "mpeg2";
    case StreamVideoCodec::mpeg4: return "mpeg4";
    case StreamVideoCodec::vc1: return "vc1";
    case StreamVideoCodec::flv1: return "flv1";
    case StreamVideoCodec::mjpeg: return "mjpeg";
    }
    return "unknown";
}

[[nodiscard]] StreamVideoCodec parse_codec(std::string_view name) {
    if (name == "hevc") return StreamVideoCodec::hevc;
    if (name == "av1") return StreamVideoCodec::av1;
    if (name == "vp8") return StreamVideoCodec::vp8;
    if (name == "vp9") return StreamVideoCodec::vp9;
    if (name == "mpeg1") return StreamVideoCodec::mpeg1;
    if (name == "mpeg2") return StreamVideoCodec::mpeg2;
    if (name == "mpeg4") return StreamVideoCodec::mpeg4;
    if (name == "vc1") return StreamVideoCodec::vc1;
    if (name == "flv1") return StreamVideoCodec::flv1;
    if (name == "mjpeg") return StreamVideoCodec::mjpeg;
    return StreamVideoCodec::h264;
}

[[nodiscard]] std::string_view decode_backend_name(
    StreamDecodeBackend backend) {
    return to_string(backend);
}

[[nodiscard]] StreamDecodeBackend parse_decode_backend(
    std::string_view name) {
    for (auto backend : {StreamDecodeBackend::software, StreamDecodeBackend::vulkan_video,
                         StreamDecodeBackend::qsv, StreamDecodeBackend::nvdec, StreamDecodeBackend::vaapi})
        if (to_string(backend) == name) return backend;
    throw std::runtime_error("unknown decode backend in report");
}

[[nodiscard]] std::string_view frame_rate_mode_name(FrameRateMode mode) {
    switch (mode) {
    case FrameRateMode::constant: return "constant";
    case FrameRateMode::variable: return "variable";
    case FrameRateMode::unknown: return "unknown";
    }
    return "unknown";
}

[[nodiscard]] FrameRateMode parse_frame_rate_mode(std::string_view name) {
    if (name == "constant") return FrameRateMode::constant;
    if (name == "variable") return FrameRateMode::variable;
    return FrameRateMode::unknown;
}

[[nodiscard]] std::string_view bit_rate_source_name(BitRateSource source) {
    switch (source) {
    case BitRateSource::video_stream: return "video_stream";
    case BitRateSource::container_overall: return "container_overall";
    case BitRateSource::file_estimate: return "file_estimate";
    }
    return "video_stream";
}

[[nodiscard]] BitRateSource parse_bit_rate_source(std::string_view name) {
    if (name == "container_overall") return BitRateSource::container_overall;
    if (name == "file_estimate") return BitRateSource::file_estimate;
    return BitRateSource::video_stream;
}

[[nodiscard]] std::string_view scene_id_name(SceneId scene) {
    switch (scene) {
    case SceneId::tv_standard: return "tv_standard";
    case SceneId::tv_4k_close: return "tv_4k_close";
    case SceneId::tv_4k_standard: return "tv_4k_standard";
    case SceneId::cinema_far: return "cinema_far";
    }
    return "tv_standard";
}

[[nodiscard]] SceneId parse_scene_id(std::string_view name) {
    if (name == "tv_4k_close") return SceneId::tv_4k_close;
    if (name == "tv_4k_standard") return SceneId::tv_4k_standard;
    if (name == "cinema_far") return SceneId::cinema_far;
    return SceneId::tv_standard;
}

[[nodiscard]] nlohmann::json rational_json(const Rational& value) {
    return {
        {"num", value.numerator},
        {"den", value.denominator},
    };
}

[[nodiscard]] Rational rational_from_json(const nlohmann::json& document) {
    Rational value;
    if (!document.is_object()) return value;
    value.numerator = document.value("num", std::int64_t{});
    value.denominator = document.value("den", std::int64_t{});
    return value;
}

[[nodiscard]] nlohmann::json q32_json(Q32Value value) {
    return {
        {"raw_q32_32", value.raw_q32_32},
        {"decimal", value.value()},
    };
}

[[nodiscard]] Q32Value q32_from_json(const nlohmann::json& document) {
    Q32Value value;
    if (document.is_object()) {
        value.raw_q32_32 = document.value("raw_q32_32", std::int64_t{});
    }
    return value;
}

[[nodiscard]] nlohmann::json descriptor_json(
    const FrameDescriptor& descriptor) {
    return {
        {"width", descriptor.width},
        {"height", descriptor.height},
        {"bit_depth", descriptor.bit_depth},
        {"pixel_format", pixel_format_name(descriptor.pixel_format)},
    };
}

[[nodiscard]] FrameDescriptor descriptor_from_json(
    const nlohmann::json& document) {
    FrameDescriptor descriptor;
    descriptor.width = document.value("width", std::uint32_t{});
    descriptor.height = document.value("height", std::uint32_t{});
    descriptor.bit_depth = static_cast<std::uint8_t>(
        document.value("bit_depth", 8U));
    descriptor.pixel_format = parse_pixel_format(
        document.value("pixel_format", std::string_view{"yuv420p"}));
    return descriptor;
}

[[nodiscard]] nlohmann::json file_identity_json(const FileIdentity& identity) {
    return {
        {"path", path_to_utf8(identity.path)},
        {"size_bytes", identity.size_bytes},
        {"modification_time_utc_seconds",
         identity.modification_time_utc_seconds},
        {"captured", identity.captured},
    };
}

[[nodiscard]] FileIdentity file_identity_from_json(
    const nlohmann::json& document) {
    FileIdentity identity;
    if (!document.is_object()) return identity;
    identity.path = std::filesystem::path(
        document.value("path", std::string{}));
    identity.size_bytes = document.value("size_bytes", std::uint64_t{});
    identity.modification_time_utc_seconds =
        document.value("modification_time_utc_seconds", std::int64_t{});
    identity.captured = document.value("captured", false);
    return identity;
}

template <typename T>
[[nodiscard]] nlohmann::json optional_json(
    const std::optional<T>& value) {
    return value.has_value() ? nlohmann::json(*value)
                             : nlohmann::json(nullptr);
}

[[nodiscard]] nlohmann::json video_metadata_json(
    const VideoMetadata& metadata) {
    nlohmann::json document;
    document["identity"] = file_identity_json(metadata.identity);
    document["codec"] = codec_name(metadata.codec);
    document["container_format"] = metadata.container_format;
    document["video_stream_index"] = metadata.video_stream_index;
    document["descriptor"] = metadata.descriptor.has_value()
        ? descriptor_json(*metadata.descriptor)
        : nlohmann::json(nullptr);
    document["descriptor_confirmed"] = metadata.descriptor_confirmed;
    document["time_base"] = rational_json(metadata.time_base);
    document["frame_rate"] = {
        {"average", rational_json(metadata.frame_rate.average)},
        {"nominal", rational_json(metadata.frame_rate.nominal)},
        {"mode", frame_rate_mode_name(metadata.frame_rate.mode)},
    };
    document["duration_seconds"] = optional_json(metadata.duration_seconds);
    document["frame_count"] = optional_json(metadata.frame_count);
    document["frame_count_estimated"] = metadata.frame_count_estimated;
    document["average_bit_rate"] = optional_json(metadata.average_bit_rate);
    document["bit_rate_source"] =
        bit_rate_source_name(metadata.bit_rate_source);
    return document;
}

[[nodiscard]] VideoMetadata video_metadata_from_json(
    const nlohmann::json& document) {
    VideoMetadata metadata;
    if (!document.is_object()) return metadata;
    metadata.identity = file_identity_from_json(document.value(
        "identity", nlohmann::json::object()));
    metadata.codec = parse_codec(
        document.value("codec", std::string_view{"unknown"}));
    metadata.container_format =
        document.value("container_format", std::string{});
    metadata.video_stream_index = document.value("video_stream_index", -1);
    if (document.contains("descriptor") &&
        document["descriptor"].is_object()) {
        metadata.descriptor = descriptor_from_json(document["descriptor"]);
    }
    metadata.descriptor_confirmed =
        document.value("descriptor_confirmed", false);
    metadata.time_base =
        rational_from_json(document.value("time_base", nlohmann::json()));
    if (document.contains("frame_rate")) {
        const nlohmann::json& frame_rate = document["frame_rate"];
        metadata.frame_rate.average =
            rational_from_json(frame_rate.value("average", nlohmann::json()));
        metadata.frame_rate.nominal =
            rational_from_json(frame_rate.value("nominal", nlohmann::json()));
        metadata.frame_rate.mode = parse_frame_rate_mode(
            frame_rate.value("mode", std::string_view{"unknown"}));
    }
    if (document.contains("duration_seconds") &&
        document["duration_seconds"].is_number()) {
        metadata.duration_seconds =
            document["duration_seconds"].get<double>();
    }
    if (document.contains("frame_count") &&
        document["frame_count"].is_number()) {
        metadata.frame_count = document["frame_count"].get<std::uint64_t>();
    }
    metadata.frame_count_estimated =
        document.value("frame_count_estimated", false);
    if (document.contains("average_bit_rate") &&
        document["average_bit_rate"].is_number()) {
        metadata.average_bit_rate =
            document["average_bit_rate"].get<std::uint64_t>();
    }
    metadata.bit_rate_source = parse_bit_rate_source(
        document.value("bit_rate_source", std::string_view{"video_stream"}));
    return metadata;
}

[[nodiscard]] nlohmann::json timing_json(const FrameTiming& timing) {
    std::string_view status = "missing";
    if (timing.status == FrameTimingStatus::valid) status = "valid";
    if (timing.status == FrameTimingStatus::non_monotonic) {
        status = "non_monotonic";
    }
    return {
        {"status", status},
        {"presentation_timestamp", timing.presentation_timestamp},
        {"time_base", rational_json(timing.time_base)},
        {"media_time_seconds", timing.media_time_seconds},
        {"duration_seconds", optional_json(timing.duration_seconds)},
    };
}

[[nodiscard]] FrameTiming timing_from_json(const nlohmann::json& document) {
    FrameTiming timing;
    if (!document.is_object()) return timing;
    const std::string status = document.value("status", std::string{});
    timing.status = status == "valid"
        ? FrameTimingStatus::valid
        : status == "non_monotonic"
            ? FrameTimingStatus::non_monotonic
            : FrameTimingStatus::missing;
    timing.presentation_timestamp =
        document.value("presentation_timestamp", std::int64_t{});
    timing.time_base =
        rational_from_json(document.value("time_base", nlohmann::json()));
    timing.media_time_seconds = document.value("media_time_seconds", 0.0);
    if (document.contains("duration_seconds") &&
        document["duration_seconds"].is_number()) {
        timing.duration_seconds = document["duration_seconds"].get<double>();
    }
    return timing;
}

[[nodiscard]] nlohmann::json input_statistics_json(
    const StreamInputStatistics& statistics) {
    return {
        {"codec", codec_name(statistics.codec)},
        {"decode_backend", decode_backend_name(statistics.decode_backend)},
            {"decode_device", statistics.decode.device_id},
            {"decode_storage", to_string(statistics.decode.native_storage)},
            {"decode_transfer", to_string(statistics.decode.transfer_path)},
            {"decode_attempts", statistics.decode.attempts},
            {"host_transfer_bytes", statistics.decode.host_transfer_bytes},
            {"gpu_copy_bytes", statistics.decode.gpu_copy_bytes},
            {"transfer_nanoseconds", statistics.decode.transfer_nanoseconds},
        {"packets_read", statistics.packets_read},
        {"compressed_bytes", statistics.compressed_bytes},
        {"decoded_frames", statistics.decoded_frames},
        {"decoder_threads", statistics.decoder_threads},
        {"decode_active_ms",
         static_cast<double>(statistics.decode_active_nanoseconds) / 1.0e6},
        {"output_queue_blocked_ms",
         static_cast<double>(statistics.output_queue_blocked_nanoseconds) /
             1.0e6},
    };
}

[[nodiscard]] StreamInputStatistics input_statistics_from_json(
    const nlohmann::json& document) {
    StreamInputStatistics statistics;
    if (!document.is_object()) return statistics;
    statistics.codec = parse_codec(
        document.value("codec", std::string_view{"unknown"}));
    statistics.decode_backend = parse_decode_backend(
        document.value("decode_backend", std::string_view{"software"}));
    statistics.decode.device_id = document.value("decode_device", std::string{});
    statistics.decode.attempts = document.value("decode_attempts", std::vector<std::string>{});
    statistics.decode.host_transfer_bytes = document.value("host_transfer_bytes", std::uint64_t{});
    statistics.decode.gpu_copy_bytes = document.value("gpu_copy_bytes", std::uint64_t{});
    statistics.decode.transfer_nanoseconds = document.value("transfer_nanoseconds", std::uint64_t{});
    const auto storage = document.value("decode_storage", statistics.decode_backend == StreamDecodeBackend::vulkan_video ? std::string("vulkan") : std::string("host"));
    bool storage_valid = false;
    for (auto value : {DecodeStorage::host, DecodeStorage::vulkan, DecodeStorage::qsv_surface, DecodeStorage::cuda, DecodeStorage::vaapi})
        if (to_string(value) == storage) { statistics.decode.native_storage = value; storage_valid = true; }
    const auto path = document.value("decode_transfer", statistics.decode_backend == StreamDecodeBackend::vulkan_video ? std::string("native") : std::string("host"));
    bool path_valid = false;
    for (auto value : {DecodeTransferPath::host, DecodeTransferPath::native, DecodeTransferPath::mapped, DecodeTransferPath::gpu_copy})
        if (to_string(value) == path) { statistics.decode.transfer_path = value; path_valid = true; }
    if (!path_valid || !storage_valid) throw std::runtime_error("unknown decode storage or transfer path");
    statistics.packets_read = document.value("packets_read", std::uint64_t{});
    statistics.compressed_bytes =
        document.value("compressed_bytes", std::uint64_t{});
    statistics.decoded_frames =
        document.value("decoded_frames", std::uint64_t{});
    statistics.decoder_threads =
        document.value("decoder_threads", std::size_t{});
    statistics.decode_active_nanoseconds = static_cast<std::uint64_t>(
        document.value("decode_active_ms", 0.0) * 1.0e6);
    statistics.output_queue_blocked_nanoseconds =
        static_cast<std::uint64_t>(
            document.value("output_queue_blocked_ms", 0.0) * 1.0e6);
    return statistics;
}

[[nodiscard]] std::int64_t to_unix_seconds(
    std::chrono::system_clock::time_point time_point) {
    return std::chrono::duration_cast<std::chrono::seconds>(
        time_point.time_since_epoch()).count();
}

}  // namespace

nlohmann::json evaluation_report_to_json(const EvaluationReport& report) {
    nlohmann::json document;
    document["schema_version"] = std::string(report_schema_version);
    document["crossvmaf_version"] = report.producer_version;
    document["report_id"] = report.report_id;
    document["source"] = {
        {"reference", path_to_utf8(report.request.reference_path)},
        {"distorted", path_to_utf8(report.request.distorted_path)},
        {"scene_id", scene_id_name(report.request.scene)},
        {"scene_policy_version", report.request.scene_policy_version},
        {"model_decision_reason", report.request.model_decision_reason},
        {"reference_identity",
         file_identity_json(report.request.reference_identity)},
        {"distorted_identity",
         file_identity_json(report.request.distorted_identity)},
    };
    document["config"] = {
        {"model", to_string(report.model)},
        {"device_uuid", report.request.device.uuid},
        {"decode_mode", to_string(report.request.decode_mode)},
        {"decode_backend", to_string(report.request.decode_backend)},
        {"decode_device", report.request.decode_device},
        {"decode_transfer", to_string(report.request.decode_transfer)},
        {"decoder_threads_per_source", report.request.decoder_threads_per_source},
        {"decoded_frame_queue_capacity", report.request.decoded_frame_queue_capacity},
    };
    document["device"] = {
        {"name", report.device.name},
        {"uuid", report.device.uuid},
        {"vendor_id", report.device.vendor_id},
        {"device_id", report.device.device_id},
        {"api_version_raw", report.device.api_version},
        {"driver_version_raw", report.device.driver_version},
        {"discrete", report.device.discrete},
        {"device_local_heap_bytes", report.device.device_local_heap_bytes},
    };
    document["media"] = {
        {"reference", video_metadata_json(report.media.reference)},
        {"distorted", video_metadata_json(report.media.distorted)},
    };
    document["input"] = {
        {"width", report.descriptor.width},
        {"height", report.descriptor.height},
        {"pixel_format", pixel_format_name(report.descriptor.pixel_format)},
        {"bit_depth", report.descriptor.bit_depth},
        {"frame_count", report.frames.size()},
    };
    document["timing"] = {
        {"initialization_ms", report.initialization_milliseconds},
        {"processing_ms", report.processing_milliseconds},
        {"end_to_end_ms", report.end_to_end_milliseconds},
        {"frames_per_second", report.frames_per_second},
        {"completed_at_utc_seconds", to_unix_seconds(report.completed_at)},
    };
    document["pool"] = {
        {"method", "mean"},
        {"score", report.pooled_score},
    };
    document["analysis"] = {
        {"rules_version", std::string(AnalysisPolicy::version)},
    };
    document["execution"] = {
        {"frame_count_hint",
         optional_json(report.stream_statistics.frame_count_hint)},
        {"decoded_frame_count",
         report.stream_statistics.decoded_frame_count},
        {"cancellation_requested",
         report.stream_statistics.cancellation_requested},
        {"reference",
         input_statistics_json(report.stream_statistics.reference)},
        {"distorted",
         input_statistics_json(report.stream_statistics.distorted)},
    };
    document["frames"] = nlohmann::json::array();
    for (const FrameResult& frame : report.frames) {
        document["frames"].push_back({
            {"index", frame.index},
            {"features",
             {
                 {"cambi", q32_json(frame.features.cambi)},
                 {"speed_chroma_uv", q32_json(frame.features.speed_chroma_uv)},
                 {"adm3", q32_json(frame.features.adm3)},
                 {"motion3", q32_json(frame.features.motion3)},
             }},
            {"vmaf", frame.vmaf},
            {"speed_chroma",
             {
                 {"u", q32_json(frame.speed_chroma.u)},
                 {"v", q32_json(frame.speed_chroma.v)},
                 {"uv", q32_json(frame.speed_chroma.uv)},
                 {"u_singular", frame.speed_chroma.u_singular},
                 {"v_singular", frame.speed_chroma.v_singular},
             }},
            {"frame_timing", timing_json(frame.timing)},
        });
    }
    return document;
}

Result<EvaluationReport> evaluation_report_from_json(
    const nlohmann::json& document) {
    try {
        if (!document.is_object()) {
            return Result<EvaluationReport>::failure(
                {ErrorCode::unsupported_format,
                 i18n::tr("The report file is not a valid JSON object")});
        }
        const std::string schema_version =
            document.value("schema_version", std::string{});
        if (schema_version.empty()) {
            return Result<EvaluationReport>::failure(
                {ErrorCode::unsupported_format,
                 i18n::tr("The report file has no schema_version (it may be an older "
                          "export format) and cannot be read")});
        }
        if (schema_version.front() != '1') {
            return Result<EvaluationReport>::failure(
                {ErrorCode::unsupported_format,
                 i18n::tr("Report schema major version {} is not supported; 1.x is required",
                          schema_version)});
        }

        EvaluationReport report;
        report.producer_version = document.value("crossvmaf_version", std::string{});
        report.report_id = document.value("report_id", std::string{});
        if (document.contains("source")) {
            const nlohmann::json& source = document["source"];
            report.request.reference_path = std::filesystem::path(
                source.value("reference", std::string{}));
            report.request.distorted_path = std::filesystem::path(
                source.value("distorted", std::string{}));
            report.request.scene = parse_scene_id(
                source.value("scene_id", std::string_view{"tv_standard"}));
            report.request.scene_policy_version =
                source.value("scene_policy_version", std::string{});
            report.request.model_decision_reason =
                source.value("model_decision_reason", std::string{});
            report.request.reference_identity = file_identity_from_json(
                source.value("reference_identity", nlohmann::json()));
            report.request.distorted_identity = file_identity_from_json(
                source.value("distorted_identity", nlohmann::json()));
        }
        if (document.contains("config")) {
            const nlohmann::json& config = document["config"];
            const auto model =
                parse_model(config.value("model", std::string_view{}));
            if (model) report.model = model.value();
            report.request.device.uuid =
                config.value("device_uuid", std::string{});
            const auto decode_mode = config.value("decode_mode", std::string("auto"));
            const int threads = config.value("decoder_threads_per_source", 0);
            const int capacity = config.value("decoded_frame_queue_capacity", 4);
            if ((decode_mode != "auto" && decode_mode != "hardware" && decode_mode != "software") ||
                threads < 0 || threads > 64 || capacity < 1 || capacity > 64) {
                return Result<EvaluationReport>::failure(
                    {ErrorCode::unsupported_format,
                     i18n::tr("The report contains an invalid decoding configuration")});
            }
            report.request.decode_mode = decode_mode == "hardware" ? DecodeMode::hardware :
                decode_mode == "software" ? DecodeMode::software : DecodeMode::automatic;
            const auto backend = parse_decode_backend_preference(config.value("decode_backend", std::string("auto")));
            const auto transfer = parse_decode_transfer(config.value("decode_transfer", std::string("auto")));
            if (!backend || !transfer) throw std::runtime_error("invalid report decode configuration");
            report.request.decode_backend = *backend;
            report.request.decode_transfer = *transfer;
            report.request.decode_device = config.value("decode_device", std::string("same"));
            if (report.request.decode_device.empty() || (report.request.decode_mode == DecodeMode::software &&
                (report.request.decode_backend != DecodeBackendPreference::automatic ||
                 report.request.decode_transfer == DecodeTransferPolicy::gpu_only || report.request.decode_device != "same")))
                throw std::runtime_error("conflicting decode options in report");
            report.request.decoder_threads_per_source = static_cast<std::size_t>(threads);
            report.request.decoded_frame_queue_capacity = static_cast<std::size_t>(capacity);
        }
        report.request.model = report.model;
        if (document.contains("device")) {
            const nlohmann::json& device = document["device"];
            report.device.name = device.value("name", std::string{});
            report.device.uuid = device.value("uuid", std::string{});
            report.device.vendor_id =
                device.value("vendor_id", std::uint32_t{});
            report.device.device_id =
                device.value("device_id", std::uint32_t{});
            report.device.api_version =
                device.value("api_version_raw", std::uint32_t{});
            report.device.driver_version =
                device.value("driver_version_raw", std::uint32_t{});
            report.device.discrete = device.value("discrete", false);
            report.device.device_local_heap_bytes =
                device.value("device_local_heap_bytes", std::uint64_t{});
        }
        if (document.contains("media")) {
            report.media.reference = video_metadata_from_json(
                document["media"].value("reference", nlohmann::json()));
            report.media.distorted = video_metadata_from_json(
                document["media"].value("distorted", nlohmann::json()));
        }
        if (document.contains("input")) {
            report.descriptor = descriptor_from_json(document["input"]);
        }
        if (document.contains("timing")) {
            const nlohmann::json& timing = document["timing"];
            report.initialization_milliseconds =
                timing.value("initialization_ms", 0.0);
            report.processing_milliseconds =
                timing.value("processing_ms", 0.0);
            report.end_to_end_milliseconds =
                timing.value("end_to_end_ms", 0.0);
            report.frames_per_second =
                timing.value("frames_per_second", 0.0);
            report.completed_at = std::chrono::system_clock::time_point(
                std::chrono::seconds(
                    timing.value("completed_at_utc_seconds",
                                 std::int64_t{})));
        }
        if (document.contains("pool")) {
            report.pooled_score = document["pool"].value("score", 0.0);
        }
        if (document.contains("execution")) {
            const nlohmann::json& execution = document["execution"];
            if (execution.contains("frame_count_hint") &&
                execution["frame_count_hint"].is_number()) {
                report.stream_statistics.frame_count_hint =
                    execution["frame_count_hint"].get<std::uint64_t>();
            }
            report.stream_statistics.decoded_frame_count =
                execution.value("decoded_frame_count", std::uint64_t{});
            report.stream_statistics.cancellation_requested =
                execution.value("cancellation_requested", false);
            report.stream_statistics.reference = input_statistics_from_json(
                execution.value("reference", nlohmann::json()));
            report.stream_statistics.distorted = input_statistics_from_json(
                execution.value("distorted", nlohmann::json()));
        }
        if (document.contains("frames") && document["frames"].is_array()) {
            report.frames.reserve(document["frames"].size());
            for (const nlohmann::json& frame_json : document["frames"]) {
                FrameResult frame;
                frame.index = frame_json.value("index", std::uint64_t{});
                if (frame_json.contains("features")) {
                    const nlohmann::json& features = frame_json["features"];
                    frame.features.cambi =
                        q32_from_json(features.value("cambi", nlohmann::json()));
                    frame.features.speed_chroma_uv = q32_from_json(
                        features.value("speed_chroma_uv", nlohmann::json()));
                    frame.features.adm3 =
                        q32_from_json(features.value("adm3", nlohmann::json()));
                    frame.features.motion3 = q32_from_json(
                        features.value("motion3", nlohmann::json()));
                }
                frame.vmaf = frame_json.value("vmaf", 0.0);
                if (frame_json.contains("speed_chroma")) {
                    const nlohmann::json& speed_chroma =
                        frame_json["speed_chroma"];
                    frame.speed_chroma.u =
                        q32_from_json(speed_chroma.value("u", nlohmann::json()));
                    frame.speed_chroma.v =
                        q32_from_json(speed_chroma.value("v", nlohmann::json()));
                    frame.speed_chroma.uv = q32_from_json(
                        speed_chroma.value("uv", nlohmann::json()));
                    frame.speed_chroma.u_singular =
                        speed_chroma.value("u_singular", false);
                    frame.speed_chroma.v_singular =
                        speed_chroma.value("v_singular", false);
                }
                frame.timing = timing_from_json(
                    frame_json.value("frame_timing", nlohmann::json()));
                report.frames.push_back(frame);
            }
        }
        if (report.frames.empty()) {
            return Result<EvaluationReport>::failure(
                {ErrorCode::unsupported_format,
                 i18n::tr("The report file contains no frame data")});
        }
        return Result<EvaluationReport>::success(std::move(report));
    } catch (const std::exception& exception) {
        return Result<EvaluationReport>::failure(
            {ErrorCode::unsupported_format,
             i18n::tr("Failed to parse the report file: {}", exception.what())});
    }
}

}  // namespace crossvmaf::studio
