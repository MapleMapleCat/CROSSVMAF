#include "output.hpp"

#include <nlohmann/json.hpp>

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

namespace crossvmaf::cli {
namespace {

[[nodiscard]] std::string_view pool_name(PoolMethod method) noexcept {
    switch (method) {
    case PoolMethod::mean:
        return "mean";
    case PoolMethod::minimum:
        return "min";
    case PoolMethod::harmonic_mean:
        return "harmonic";
    }
    return "unknown";
}

[[nodiscard]] std::string_view pixel_format_name(PixelFormat format) noexcept {
    switch (format) {
    case PixelFormat::yuv420p:
        return "yuv420p";
    case PixelFormat::yuv422p:
        return "yuv422p";
    case PixelFormat::yuv444p:
        return "yuv444p";
    }
    return "unknown";
}

[[nodiscard]] std::string_view video_codec_name(
    StreamVideoCodec codec) noexcept {
    switch (codec) {
    case StreamVideoCodec::h264:
        return "h264";
    case StreamVideoCodec::hevc:
        return "hevc";
    case StreamVideoCodec::av1:
        return "av1";
    case StreamVideoCodec::vp8:
        return "vp8";
    case StreamVideoCodec::vp9:
        return "vp9";
    case StreamVideoCodec::mpeg1:
        return "mpeg1";
    case StreamVideoCodec::mpeg2:
        return "mpeg2";
    case StreamVideoCodec::mpeg4:
        return "mpeg4";
    case StreamVideoCodec::vc1:
        return "vc1";
    case StreamVideoCodec::flv1:
        return "flv1";
    case StreamVideoCodec::mjpeg:
        return "mjpeg";
    }
    return "unknown";
}

[[nodiscard]] std::string_view decode_backend_name(
    StreamDecodeBackend backend) noexcept {
    return to_string(backend);
}

[[nodiscard]] std::string_view hardware_pixel_format_name(
    StreamHardwarePixelFormat format) noexcept {
    switch (format) {
    case StreamHardwarePixelFormat::none:
        return "none";
    case StreamHardwarePixelFormat::nv12:
        return "nv12";
    case StreamHardwarePixelFormat::p010:
        return "p010";
    }
    return "none";
}

[[nodiscard]] nlohmann::json q32_json(Q32Value value) {
    return {
        {"raw_q32_32", value.raw_q32_32},
        {"decimal", value.decimal()},
    };
}

[[nodiscard]] std::string csv_quote(std::string_view value) {
    std::string output;
    output.reserve(value.size() + 2U);
    output.push_back('"');
    for (const char character : value) {
        if (character == '"') output.push_back('"');
        output.push_back(character);
    }
    output.push_back('"');
    return output;
}

void write_csv(std::ostream& output, const RunReport& report) {
    output << "device_name,device_uuid,vendor_id,device_id,model,width,height,"
              "pixel_format,bit_depth,frame_index,cambi_raw_q32_32,cambi_decimal,"
              "speed_chroma_uv_raw_q32_32,speed_chroma_uv_decimal,adm3_raw_q32_32,"
              "adm3_decimal,motion3_raw_q32_32,motion3_decimal,vmaf,"
              "pool_method,pool_score,"
              "warmup_ms,initialization_ms,processing_ms,end_to_end_ms,frames_per_second,"
              "cpu_verified,decoded_frame_queue_capacity,"
              "decoder_threads_per_source,input_lease_capacity,"
              "reference_codec,reference_decode_backend,"
              "reference_hardware_pixel_format,reference_packets_read,"
              "reference_compressed_bytes,reference_decoded_frames,"
              "distorted_codec,distorted_decode_backend,"
              "distorted_hardware_pixel_format,distorted_packets_read,"
              "distorted_compressed_bytes,distorted_decoded_frames,requested_decode_backend,decode_device,decode_transfer,reference_transfer,distorted_transfer,reference_host_transfer_bytes,distorted_host_transfer_bytes,reference_gpu_copy_bytes,distorted_gpu_copy_bytes\n";
    output << std::fixed << std::setprecision(3);
    for (const auto& frame : report.frames) {
        output << csv_quote(report.device.name) << ','
               << csv_quote(report.device.uuid) << ',' << report.device.vendor_id
               << ',' << report.device.device_id << ','
               << csv_quote(to_string(report.model)) << ','
               << report.descriptor.width << ',' << report.descriptor.height << ','
               << pixel_format_name(report.descriptor.pixel_format) << ','
               << static_cast<unsigned>(report.descriptor.bit_depth) << ','
               << frame.index << ',' << frame.features.cambi.raw_q32_32 << ','
               << frame.features.cambi.decimal() << ','
               << frame.features.speed_chroma_uv.raw_q32_32 << ','
               << frame.features.speed_chroma_uv.decimal() << ','
               << frame.features.adm3.raw_q32_32 << ','
               << frame.features.adm3.decimal() << ','
               << frame.features.motion3.raw_q32_32 << ','
               << frame.features.motion3.decimal() << ','
               << std::setprecision(9) << frame.vmaf << ','
               << pool_name(report.pool_method) << ','
               << report.pooled_score << ','
               << std::setprecision(3)
               << report.warmup_ms << ',' << report.initialization_ms << ','
               << report.processing_ms << ',' << report.end_to_end_ms << ','
               << report.frames_per_second << ','
               << (report.cpu_verified ? "true" : "false") << ','
               << report.decoded_frame_queue_capacity << ','
               << report.decoder_threads_per_source << ','
               << report.stream_statistics.input_lease_capacity << ','
               << video_codec_name(report.stream_statistics.reference.codec)
               << ','
               << decode_backend_name(
                      report.stream_statistics.reference.decode_backend)
               << ','
               << hardware_pixel_format_name(
                      report.stream_statistics.reference.hardware_pixel_format)
               << ','
               << report.stream_statistics.reference.packets_read << ','
               << report.stream_statistics.reference.compressed_bytes << ','
               << report.stream_statistics.reference.decoded_frames << ','
               << video_codec_name(report.stream_statistics.distorted.codec)
               << ','
               << decode_backend_name(
                      report.stream_statistics.distorted.decode_backend)
               << ','
               << hardware_pixel_format_name(
                      report.stream_statistics.distorted.hardware_pixel_format)
               << ','
               << report.stream_statistics.distorted.packets_read << ','
               << report.stream_statistics.distorted.compressed_bytes << ','
               << report.stream_statistics.distorted.decoded_frames << ','
               << to_string(report.decode_backend) << ',' << csv_quote(report.decode_device) << ',' << to_string(report.decode_transfer) << ','
               << to_string(report.stream_statistics.reference.decode.transfer_path) << ','
               << to_string(report.stream_statistics.distorted.decode.transfer_path) << ','
               << report.stream_statistics.reference.decode.host_transfer_bytes << ','
               << report.stream_statistics.distorted.decode.host_transfer_bytes << ','
               << report.stream_statistics.reference.decode.gpu_copy_bytes << ','
               << report.stream_statistics.distorted.decode.gpu_copy_bytes
               << '\n';
    }
}

void write_json(std::ostream& output, const RunReport& report) {
    nlohmann::json document;
    document["decode_request"] = {
        {"backend", to_string(report.decode_backend)}, {"device", report.decode_device},
        {"transfer", to_string(report.decode_transfer)}};
    document["device"] = {
        {"name", report.device.name},
        {"uuid", report.device.uuid},
        {"vendor_id", report.device.vendor_id},
        {"device_id", report.device.device_id},
        {"api_version_raw", report.device.api_version},
        {"driver_version_raw", report.device.driver_version},
        {"discrete", report.device.discrete},
        {"memory_budget_available", report.device.memory_budget_available},
        {"host_visible_device_local", report.device.host_visible_device_local},
        {"device_local_heap_bytes", report.device.device_local_heap_bytes},
    };
    document["model"] = to_string(report.model);
    document["input"] = {
        {"width", report.descriptor.width},
        {"height", report.descriptor.height},
        {"pixel_format", pixel_format_name(report.descriptor.pixel_format)},
        {"bit_depth", report.descriptor.bit_depth},
        {"frame_count", report.frames.size()},
    };
    document["timing"] = {
        {"warmup_ms", report.warmup_ms},
        {"initialization_ms", report.initialization_ms},
        {"processing_ms", report.processing_ms},
        {"end_to_end_ms", report.end_to_end_ms},
        {"frames_per_second", report.frames_per_second},
    };
    document["cpu_verified"] = report.cpu_verified;
    const auto input_statistics_json = [](const StreamInputStatistics& input) {
        const double decode_active_milliseconds =
            static_cast<double>(input.decode_active_nanoseconds) / 1.0e6;
        const double queue_blocked_milliseconds =
            static_cast<double>(input.output_queue_blocked_nanoseconds) /
            1.0e6;
        return nlohmann::json{
            {"codec", video_codec_name(input.codec)},
            {"decode_backend", decode_backend_name(input.decode_backend)},
            {"decode_device", input.decode.device_id},
            {"decode_storage", to_string(input.decode.native_storage)},
            {"decode_transfer", to_string(input.decode.transfer_path)},
            {"decode_attempts", input.decode.attempts},
            {"host_transfer_bytes", input.decode.host_transfer_bytes},
            {"gpu_copy_bytes", input.decode.gpu_copy_bytes},
            {"transfer_nanoseconds", input.decode.transfer_nanoseconds},
            {"hardware_pixel_format",
             hardware_pixel_format_name(input.hardware_pixel_format)},
            {"packets_read", input.packets_read},
            {"compressed_bytes", input.compressed_bytes},
            {"decoded_frames", input.decoded_frames},
            {"decode_operations", input.decode_operation_count},
            {"decode_active_ms", decode_active_milliseconds},
            {"decode_average_ms_per_frame",
             input.decoded_frames != 0U
                 ? decode_active_milliseconds /
                       static_cast<double>(input.decoded_frames)
                 : 0.0},
            {"output_queue_blocked_waits",
             input.output_queue_blocked_wait_count},
            {"output_queue_blocked_ms", queue_blocked_milliseconds},
            {"queue_capacity", input.queue_capacity},
            {"queued_frames", input.queued_frames},
            {"decoder_threads", input.decoder_threads},
            {"end_of_stream", input.end_of_stream},
            {"cancellation_requested", input.cancellation_requested},
        };
    };
    const double frame_pair_copy_milliseconds =
        static_cast<double>(
            report.stream_statistics.frame_pair_copy_nanoseconds) /
        1.0e6;
    constexpr double bytes_per_gibibyte =
        1024.0 * 1024.0 * 1024.0;
    const double frame_pair_copy_gibibytes_per_second =
        report.stream_statistics.frame_pair_copy_nanoseconds != 0U
        ? static_cast<double>(
              report.stream_statistics.frame_pair_copy_bytes) *
              1.0e9 /
              static_cast<double>(
                  report.stream_statistics.frame_pair_copy_nanoseconds) /
              bytes_per_gibibyte
        : 0.0;
    const double staging_input_wait_milliseconds =
        static_cast<double>(
            report.stream_statistics.staging_input_wait_nanoseconds) /
        1.0e6;
    const double staging_grant_wait_milliseconds =
        static_cast<double>(
            report.stream_statistics.staging_grant_wait_nanoseconds) /
        1.0e6;
    const double coordinator_wait_milliseconds =
        static_cast<double>(
            report.stream_statistics.coordinator_wait_nanoseconds) /
        1.0e6;
    document["execution"] = {
        {"configuration",
         {
             {"decoded_frame_queue_capacity",
              report.decoded_frame_queue_capacity},
             {"decoder_threads_per_source",
              report.decoder_threads_per_source},
         }},
        {"decoded_input",
         {
             {"frame_count_hint",
              report.stream_statistics.frame_count_hint.has_value()
                  ? nlohmann::json(*report.stream_statistics.frame_count_hint)
                  : nlohmann::json(nullptr)},
             {"decoded_frame_count",
              report.stream_statistics.decoded_frame_count},
             {"scheduled_frames", report.stream_statistics.scheduled_frames},
             {"acquired_frames", report.stream_statistics.acquired_frames},
             {"record_capacity", report.stream_statistics.record_capacity},
             {"owned_pairs", report.stream_statistics.owned_pairs},
             {"ready_lead_frames",
              report.stream_statistics.ready_lead_frames},
             {"frame_pair_copy_calls",
              report.stream_statistics.frame_pair_copy_calls},
             {"frame_pair_copy_bytes",
              report.stream_statistics.frame_pair_copy_bytes},
             {"frame_pair_copy_ms", frame_pair_copy_milliseconds},
             {"frame_pair_copy_average_ms",
              report.stream_statistics.frame_pair_copy_calls != 0U
                  ? frame_pair_copy_milliseconds /
                        static_cast<double>(
                            report.stream_statistics.frame_pair_copy_calls)
                  : 0.0},
             {"frame_pair_copy_gib_per_second",
              frame_pair_copy_gibibytes_per_second},
             {"staging_input_wait_ms", staging_input_wait_milliseconds},
             {"staging_input_waits",
              report.stream_statistics.staging_input_wait_count},
             {"staging_grant_wait_ms", staging_grant_wait_milliseconds},
             {"staging_grant_waits",
              report.stream_statistics.staging_grant_wait_count},
             {"coordinator_wait_ms", coordinator_wait_milliseconds},
             {"coordinator_waits",
              report.stream_statistics.coordinator_wait_count},
             {"input_lease_capacity",
              report.stream_statistics.input_lease_capacity},
             {"scheduling_complete",
              report.stream_statistics.scheduling_complete},
             {"cancellation_requested",
              report.stream_statistics.cancellation_requested},
             {"reference",
              input_statistics_json(report.stream_statistics.reference)},
             {"distorted",
              input_statistics_json(report.stream_statistics.distorted)},
         }},
    };
    document["pool"] = {
        {"method", pool_name(report.pool_method)},
        {"score", report.pooled_score},
    };
    document["frames"] = nlohmann::json::array();
    for (const auto& frame : report.frames) {
        nlohmann::json frame_json{
            {"index", frame.index},
            {"features",
             {
                 {"cambi", q32_json(frame.features.cambi)},
                 {"speed_chroma_uv",
                  q32_json(frame.features.speed_chroma_uv)},
                 {"adm3", q32_json(frame.features.adm3)},
                 {"motion3", q32_json(frame.features.motion3)},
             }},
            {"vmaf", frame.vmaf},
        };
        if (report.diagnostic_speed_chroma) {
            // Emitted results have completed successfully; singularity comes
            // directly from the backend's per-channel detailed result.
            const auto channel_json = [](Q32Value value, bool singular) {
                return nlohmann::json{
                    {"raw_q32_32", value.raw_q32_32},
                    {"decimal", value.decimal()},
                    {"valid", true},
                    {"status", singular ? "singular" : "computed"},
                    {"singular", singular},
                };
            };
            frame_json["speed_chroma"] = {
                {"u", channel_json(frame.speed_chroma.u,
                                    frame.speed_chroma.u_singular)},
                {"v", channel_json(frame.speed_chroma.v,
                                    frame.speed_chroma.v_singular)},
                {"combined", q32_json(frame.speed_chroma.uv)},
            };
        }
        document["frames"].push_back(std::move(frame_json));
    }
    output << document.dump(2) << '\n';
}

}  // namespace

Result<void> write_report(std::ostream& output, OutputFormat format,
                          const RunReport& report) try {
    if (format == OutputFormat::json) {
        write_json(output, report);
    } else {
        write_csv(output, report);
    }
    output.flush();
    if (!output) {
        return Result<void>::failure(
            {ErrorCode::io_error, "failed to write CrossVMAF output"});
    }
    return Result<void>::success();
} catch (const std::exception& exception) {
    return Result<void>::failure({ErrorCode::io_error,
        "failed to write CrossVMAF output: " + std::string(exception.what())});
}

}  // namespace crossvmaf::cli
