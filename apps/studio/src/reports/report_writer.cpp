#include "reports/report_writer.hpp"

#include "i18n/translator.hpp"
#include "reports/report_dto.hpp"
#include "reports/report_pdf.hpp"

#include <nlohmann/json.hpp>

#include <fstream>
#include <iomanip>
#include <ostream>
#include <string>
#include <string_view>

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

[[nodiscard]] std::string_view bit_rate_source_name(BitRateSource source) {
    switch (source) {
    case BitRateSource::video_stream: return "video_stream";
    case BitRateSource::container_overall: return "container_overall";
    case BitRateSource::file_estimate: return "file_estimate";
    }
    return "video_stream";
}

[[nodiscard]] std::string csv_quote(std::string_view value) {
    std::string quoted;
    quoted.reserve(value.size() + 2U);
    quoted.push_back('"');
    for (const char character : value) {
        if (character == '"') quoted.push_back('"');
        quoted.push_back(character);
    }
    quoted.push_back('"');
    return quoted;
}

[[nodiscard]] std::string_view timing_status_name(FrameTimingStatus status) {
    switch (status) {
    case FrameTimingStatus::valid: return "valid";
    case FrameTimingStatus::non_monotonic: return "non_monotonic";
    case FrameTimingStatus::missing: return "missing";
    }
    return "missing";
}

void write_csv(std::ostream& output, const EvaluationReport& report) {
    output << "reference,distorted,device_name,device_uuid,model,width,height,"
              "pixel_format,bit_depth,frame_index,"
              "cambi_q32_32,speed_chroma_uv_q32_32,adm3_q32_32,motion3_q32_32,"
              "cambi,speed_chroma_uv,adm3,motion3,vmaf,"
              "pts,time_base_num,time_base_den,media_time_seconds,"
              "timing_status,pool_score,end_to_end_ms,frames_per_second,"
              "decode_mode,decoder_threads_per_source,decoded_frame_queue_capacity,"
              "reference_decode_backend,distorted_decode_backend,requested_decode_backend,decode_device,decode_transfer,reference_transfer,distorted_transfer\n";
    output << std::fixed;
    for (const FrameResult& frame : report.frames) {
        output << csv_quote(path_to_utf8(report.request.reference_path)) << ','
               << csv_quote(path_to_utf8(report.request.distorted_path)) << ','
               << csv_quote(report.device.name) << ','
               << csv_quote(report.device.uuid) << ','
               << csv_quote(to_string(report.model)) << ','
               << report.descriptor.width << ','
               << report.descriptor.height << ','
               << pixel_format_name(report.descriptor.pixel_format) << ','
               << static_cast<unsigned>(report.descriptor.bit_depth) << ','
               << frame.index << ','
               << frame.features.cambi.raw_q32_32 << ','
               << frame.features.speed_chroma_uv.raw_q32_32 << ','
               << frame.features.adm3.raw_q32_32 << ','
               << frame.features.motion3.raw_q32_32 << ','
               << std::setprecision(9)
               << frame.features.cambi.value() << ','
               << frame.features.speed_chroma_uv.value() << ','
               << frame.features.adm3.value() << ','
               << frame.features.motion3.value() << ','
               << frame.vmaf << ','
               << frame.timing.presentation_timestamp << ','
               << frame.timing.time_base.numerator << ','
               << frame.timing.time_base.denominator << ','
               << frame.timing.media_time_seconds << ','
               << timing_status_name(frame.timing.status) << ','
               << report.pooled_score << ','
               << std::setprecision(3)
               << report.end_to_end_milliseconds << ','
               << report.frames_per_second << ','
               << to_string(report.request.decode_mode) << ','
               << report.request.decoder_threads_per_source << ','
               << report.request.decoded_frame_queue_capacity << ','
               << (to_string(report.stream_statistics.reference.decode_backend)) << ','
               << to_string(report.stream_statistics.distorted.decode_backend) << ','
               << to_string(report.request.decode_backend) << ',' << csv_quote(report.request.decode_device) << ','
               << to_string(report.request.decode_transfer) << ','
               << to_string(report.stream_statistics.reference.decode.transfer_path) << ','
               << to_string(report.stream_statistics.distorted.decode.transfer_path) << '\n';
    }
}

}  // namespace

Result<void> writeEvaluationReport(
    const std::filesystem::path& destination,
    ReportFormat format,
    const EvaluationReport& report) try {
    if (format == ReportFormat::pdf) return writeEvaluationPdf(destination, report);
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        return Result<void>::failure({
            ErrorCode::io_error,
            i18n::tr("Cannot open the export file: {}", path_to_utf8(destination))});
    }
    if (format == ReportFormat::json) {
        output << evaluation_report_to_json(report).dump(2) << '\n';
    } else {
        write_csv(output, report);
    }
    output.close();
    if (!output) {
        return Result<void>::failure({
            ErrorCode::io_error,
            i18n::tr("Failed to write the export file: {}", path_to_utf8(destination))});
    }
    return Result<void>::success();
} catch (const std::exception& exception) {
    return Result<void>::failure({ErrorCode::io_error,
        std::string("writing output: ") + exception.what()});
}

Result<void> writeComparisonReport(
    const std::filesystem::path& destination,
    const ComparisonResult& comparison) try {
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        return Result<void>::failure({
            ErrorCode::io_error,
            i18n::tr("Cannot open the comparison export file: {}", path_to_utf8(destination))});
    }
    output << "report_id,distorted,is_baseline,mean_vmaf,minimum_vmaf,"
              "percentile_5_vmaf,delta_vs_baseline,bit_rate_bps,"
              "bit_rate_source,file_size_bytes\n";
    output << std::fixed << std::setprecision(9);
    for (const ComparisonRow& row : comparison.rows) {
        output << csv_quote(row.report_id) << ','
               << csv_quote(row.display_name) << ','
               << (row.is_baseline ? "1" : "0") << ','
               << row.mean << ','
               << row.minimum << ','
               << row.percentile_5 << ','
               << row.delta_vs_baseline << ',';
        if (row.average_bit_rate.has_value()) {
            output << *row.average_bit_rate;
        }
        output << ','
               << (row.average_bit_rate.has_value()
                       ? bit_rate_source_name(row.bit_rate_source)
                       : std::string_view{})
               << ',';
        if (row.file_size_bytes.has_value()) {
            output << *row.file_size_bytes;
        }
        output << '\n';
    }
    output.close();
    if (!output) {
        return Result<void>::failure({
            ErrorCode::io_error,
            i18n::tr("Failed to write the comparison export file: {}", path_to_utf8(destination))});
    }
    return Result<void>::success();
} catch (const std::exception& exception) {
    return Result<void>::failure({ErrorCode::io_error,
        std::string("writing output: ") + exception.what()});
}

}  // namespace crossvmaf::studio
