#include "hotspot_profiler.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string_view>

namespace crossvmaf::cli {
namespace {

constexpr std::string_view output_environment_variable =
    "CROSSVMAF_CLI_HOTSPOT_OUTPUT";

[[nodiscard]] std::string read_output_path_from_environment() {
#if defined(_WIN32)
    char* value = nullptr;
    std::size_t length = 0U;
    const int status = _dupenv_s(
        &value, &length, output_environment_variable.data());
    if (status != 0 || value == nullptr) {
        std::free(value);
        return {};
    }
    std::string output_path(value);
    std::free(value);
    return output_path;
#else
    const char* const value = std::getenv(output_environment_variable.data());
    return value == nullptr ? std::string{} : std::string(value);
#endif
}

[[nodiscard]] double milliseconds(
    std::chrono::nanoseconds duration) noexcept {
    return std::chrono::duration<double, std::milli>(duration).count();
}

[[nodiscard]] std::chrono::nanoseconds percentile_duration(
    std::vector<std::chrono::nanoseconds> samples,
    std::size_t percentile_numerator) {
    if (samples.empty()) return {};
    std::sort(samples.begin(), samples.end());
    constexpr std::size_t percentile_denominator = 100U;
    const std::size_t nearest_rank =
        (samples.size() * percentile_numerator +
         percentile_denominator - 1U) /
        percentile_denominator;
    return samples[std::max<std::size_t>(nearest_rank, 1U) - 1U];
}

[[nodiscard]] bool is_recordable_stage(CliHotspotStage stage) noexcept {
    return static_cast<std::size_t>(stage) <
        static_cast<std::size_t>(CliHotspotStage::count);
}

[[nodiscard]] const char* stream_codec_name(
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

[[nodiscard]] const char* decode_backend_name(
    StreamDecodeBackend backend) noexcept {
    return backend == StreamDecodeBackend::vulkan_video
        ? "vulkan_video"
        : "software";
}

[[nodiscard]] const char* hardware_pixel_format_name(
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

[[nodiscard]] nlohmann::json stream_input_statistics_json(
    const StreamInputStatistics& statistics) {
    const double decode_active_milliseconds = milliseconds(
        std::chrono::nanoseconds(statistics.decode_active_nanoseconds));
    const double queue_blocked_milliseconds = milliseconds(
        std::chrono::nanoseconds(
            statistics.output_queue_blocked_nanoseconds));
    return {
        {"codec", stream_codec_name(statistics.codec)},
        {"decode_backend", decode_backend_name(statistics.decode_backend)},
        {"hardware_pixel_format",
         hardware_pixel_format_name(statistics.hardware_pixel_format)},
        {"packets_read", statistics.packets_read},
        {"compressed_bytes", statistics.compressed_bytes},
        {"decoded_frames", statistics.decoded_frames},
        {"decode_operations", statistics.decode_operation_count},
        {"decode_active_ms", decode_active_milliseconds},
        {"decode_average_ms_per_frame",
         statistics.decoded_frames != 0U
             ? decode_active_milliseconds /
                   static_cast<double>(statistics.decoded_frames)
             : 0.0},
        {"output_queue_blocked_waits",
         statistics.output_queue_blocked_wait_count},
        {"output_queue_blocked_ms", queue_blocked_milliseconds},
        {"queue_capacity", statistics.queue_capacity},
        {"queued_frames", statistics.queued_frames},
        {"decoder_threads", statistics.decoder_threads},
        {"end_of_stream", statistics.end_of_stream},
        {"cancellation_requested", statistics.cancellation_requested},
    };
}

[[nodiscard]] nlohmann::json decoded_input_json(
    const EvaluationStreamStatistics& statistics) {
    const double copy_milliseconds = milliseconds(
        std::chrono::nanoseconds(statistics.frame_pair_copy_nanoseconds));
    const double staging_input_wait_milliseconds = milliseconds(
        std::chrono::nanoseconds(
            statistics.staging_input_wait_nanoseconds));
    const double staging_grant_wait_milliseconds = milliseconds(
        std::chrono::nanoseconds(
            statistics.staging_grant_wait_nanoseconds));
    const double coordinator_wait_milliseconds = milliseconds(
        std::chrono::nanoseconds(
            statistics.coordinator_wait_nanoseconds));
    constexpr double bytes_per_gibibyte =
        1024.0 * 1024.0 * 1024.0;
    const double copy_gibibytes_per_second =
        statistics.frame_pair_copy_nanoseconds != 0U
        ? static_cast<double>(statistics.frame_pair_copy_bytes) * 1.0e9 /
              static_cast<double>(statistics.frame_pair_copy_nanoseconds) /
              bytes_per_gibibyte
        : 0.0;
    return {
        {"frame_count_hint",
         statistics.frame_count_hint.has_value()
             ? nlohmann::json(*statistics.frame_count_hint)
             : nlohmann::json(nullptr)},
        {"decoded_frame_count", statistics.decoded_frame_count},
        {"scheduled_frames", statistics.scheduled_frames},
        {"acquired_frames", statistics.acquired_frames},
        {"record_capacity", statistics.record_capacity},
        {"owned_pairs", statistics.owned_pairs},
        {"ready_lead_frames", statistics.ready_lead_frames},
        {"frame_pair_copy_calls", statistics.frame_pair_copy_calls},
        {"frame_pair_copy_bytes", statistics.frame_pair_copy_bytes},
        {"frame_pair_copy_ms", copy_milliseconds},
        {"frame_pair_copy_average_ms",
         statistics.frame_pair_copy_calls != 0U
             ? copy_milliseconds /
                   static_cast<double>(statistics.frame_pair_copy_calls)
             : 0.0},
        {"frame_pair_copy_gib_per_second",
         copy_gibibytes_per_second},
        {"staging_input_wait_ms", staging_input_wait_milliseconds},
        {"staging_input_waits", statistics.staging_input_wait_count},
        {"staging_grant_wait_ms", staging_grant_wait_milliseconds},
        {"staging_grant_waits", statistics.staging_grant_wait_count},
        {"coordinator_wait_ms", coordinator_wait_milliseconds},
        {"coordinator_waits", statistics.coordinator_wait_count},
        {"input_lease_capacity", statistics.input_lease_capacity},
        {"scheduling_complete", statistics.scheduling_complete},
        {"cancellation_requested", statistics.cancellation_requested},
        {"reference", stream_input_statistics_json(statistics.reference)},
        {"distorted", stream_input_statistics_json(statistics.distorted)},
    };
}

}  // namespace

const char* cli_hotspot_stage_name(CliHotspotStage stage) noexcept {
    switch (stage) {
    case CliHotspotStage::input_open:
        return "input_open";
    case CliHotspotStage::warmup_stream_open:
        return "warmup_stream_open";
    case CliHotspotStage::warmup_stream_next:
        return "warmup_stream_next";
    case CliHotspotStage::warmup_stream_finish:
        return "warmup_stream_finish";
    case CliHotspotStage::stream_open:
        return "stream_open";
    case CliHotspotStage::stream_next:
        return "stream_next";
    case CliHotspotStage::stream_finish:
        return "stream_finish";
    case CliHotspotStage::reference_open:
        return "reference_open";
    case CliHotspotStage::frame_read:
        return "frame_read";
    case CliHotspotStage::reference_push:
        return "reference_push";
    case CliHotspotStage::reference_finish:
        return "reference_finish";
    case CliHotspotStage::verification:
        return "verification";
    case CliHotspotStage::result_collection:
        return "result_collection";
    case CliHotspotStage::pooling:
        return "pooling";
    case CliHotspotStage::output_open:
        return "output_open";
    case CliHotspotStage::report_write:
        return "report_write";
    case CliHotspotStage::count:
        break;
    }
    return "unknown";
}

CliHotspotMeasurement::CliHotspotMeasurement(
    CliHotspotProfiler* profiler, CliHotspotStage stage) noexcept
    : profiler_(profiler), stage_(stage) {
    if (profiler_ != nullptr) {
        start_ = profiler_->begin_measurement(stage_);
    }
}

CliHotspotMeasurement::CliHotspotMeasurement(
    CliHotspotMeasurement&& other) noexcept
    : profiler_(std::exchange(other.profiler_, nullptr)),
      stage_(other.stage_),
      start_(other.start_) {}

CliHotspotMeasurement::~CliHotspotMeasurement() {
    if (profiler_ != nullptr) {
        profiler_->finish_measurement(stage_, start_);
    }
}

CliHotspotProfiler CliHotspotProfiler::create_from_environment() {
    std::string output_path = read_output_path_from_environment();
    const bool enabled = !output_path.empty() && output_path != "0";
    return CliHotspotProfiler(enabled, std::move(output_path));
}

CliHotspotProfiler::CliHotspotProfiler(
    bool enabled, std::string output_path) noexcept
    : enabled_(enabled),
      output_path_(std::move(output_path)),
      run_start_(enabled ? std::chrono::steady_clock::now()
                         : std::chrono::steady_clock::time_point{}) {}

CliHotspotProfiler::~CliHotspotProfiler() {
    if (enabled_ && !report_attempted_) {
        write_report(false, 0U, 0U, nullptr);
    }
}

CliHotspotMeasurement CliHotspotProfiler::measure(
    CliHotspotStage stage) noexcept {
    const bool should_measure = enabled_ && is_recordable_stage(stage);
    return CliHotspotMeasurement(should_measure ? this : nullptr, stage);
}

CliHotspotStatistics CliHotspotProfiler::statistics(
    CliHotspotStage stage) const noexcept {
    if (!is_recordable_stage(stage)) return {};
    std::lock_guard lock(statistics_mutex_);
    return statistics_[static_cast<std::size_t>(stage)];
}

void CliHotspotProfiler::record_stream_statistics(
    const EvaluationStreamStatistics& statistics) noexcept {
    if (!enabled_) return;
    std::lock_guard lock(statistics_mutex_);
    stream_statistics_ = statistics;
    stream_statistics_recorded_ = true;
}

std::chrono::steady_clock::time_point
CliHotspotProfiler::begin_measurement(CliHotspotStage) noexcept {
    return std::chrono::steady_clock::now();
}

void CliHotspotProfiler::finish_measurement(
    CliHotspotStage stage,
    std::chrono::steady_clock::time_point started_at) noexcept {
    if (!is_recordable_stage(stage)) return;
    const std::chrono::nanoseconds elapsed =
        std::chrono::steady_clock::now() - started_at;
    std::lock_guard lock(statistics_mutex_);
    CliHotspotStatistics& statistics =
        statistics_[static_cast<std::size_t>(stage)];
    if (statistics.call_count == 0U) {
        statistics.minimum = elapsed;
        statistics.maximum = elapsed;
    } else {
        statistics.minimum = std::min(statistics.minimum, elapsed);
        statistics.maximum = std::max(statistics.maximum, elapsed);
    }
    if (statistics.call_count != (std::numeric_limits<std::uint64_t>::max)()) {
        ++statistics.call_count;
    }
    if (elapsed.count() >= 0 &&
        statistics.total.count() <=
            (std::chrono::nanoseconds::max)().count() - elapsed.count()) {
        statistics.total += elapsed;
    }
    try {
        stage_samples_[static_cast<std::size_t>(stage)].push_back(elapsed);
    } catch (...) {
        // Aggregate timings remain useful if percentile storage cannot grow.
    }
}

CliHotspotProfiler::ReportSnapshot CliHotspotProfiler::report_snapshot() {
    std::lock_guard lock(statistics_mutex_);
    return {
        statistics_, stage_samples_, stream_statistics_,
        stream_statistics_recorded_, std::chrono::steady_clock::now()};
}

void CliHotspotProfiler::complete(
    std::uint64_t input_frame_count,
    std::uint64_t emitted_frame_count,
    const DeviceInfo& device) noexcept {
    write_report(true, input_frame_count, emitted_frame_count, &device);
}

void CliHotspotProfiler::write_report(
    bool complete,
    std::uint64_t input_frame_count,
    std::uint64_t emitted_frame_count,
    const DeviceInfo* device) noexcept {
    if (!enabled_ || report_attempted_) return;
    report_attempted_ = true;

    try {
        const ReportSnapshot snapshot = report_snapshot();
        const std::chrono::nanoseconds run_wall_time =
            snapshot.captured_at - run_start_;
        const double run_wall_milliseconds = milliseconds(run_wall_time);

        nlohmann::json stages = nlohmann::json::array();
        for (std::size_t stage_index = 0U;
             stage_index < snapshot.statistics.size(); ++stage_index) {
            const CliHotspotStatistics& statistics =
                snapshot.statistics[stage_index];
            if (statistics.call_count == 0U) continue;
            const std::vector<std::chrono::nanoseconds>& samples =
                snapshot.stage_samples[stage_index];
            const double total_milliseconds = milliseconds(statistics.total);
            stages.push_back({
                {"name", cli_hotspot_stage_name(
                     static_cast<CliHotspotStage>(stage_index))},
                {"calls", statistics.call_count},
                {"total_ms", total_milliseconds},
                {"min_ms", milliseconds(statistics.minimum)},
                {"p50_ms", milliseconds(percentile_duration(samples, 50U))},
                {"p95_ms", milliseconds(percentile_duration(samples, 95U))},
                {"max_ms", milliseconds(statistics.maximum)},
                {"percent_of_run",
                 run_wall_milliseconds > 0.0
                     ? total_milliseconds * 100.0 / run_wall_milliseconds
                     : 0.0},
            });
        }

        nlohmann::json report{
            {"schema", "crossvmaf.cli_hotspots.v11"},
            {"complete", complete},
            {"timing_scope",
             "main-thread call wall time only; background worker time is not "
             "measured separately"},
            {"input_frame_count", input_frame_count},
            {"emitted_frame_count", emitted_frame_count},
            {"run_wall_ms", run_wall_milliseconds},
            {"decoded_input",
             snapshot.stream_recorded
                 ? decoded_input_json(snapshot.stream)
                 : nlohmann::json(nullptr)},
            {"device",
             device != nullptr
                 ? nlohmann::json{
                       {"name", device->name},
                       {"uuid", device->uuid},
                       {"vendor_id", device->vendor_id},
                       {"device_id", device->device_id},
                   }
                 : nlohmann::json(nullptr)},
            {"stages", std::move(stages)},
        };

        const std::string report_text = report.dump(2) + '\n';
        if (output_path_ == "1" || output_path_ == "-" ||
            output_path_ == "stderr") {
            const std::size_t bytes_written = std::fwrite(
                report_text.data(), 1U, report_text.size(), stderr);
            if (bytes_written == report_text.size()) {
                static_cast<void>(std::fflush(stderr));
            }
            return;
        }

        std::ofstream output(
            output_path_, std::ios::binary | std::ios::trunc);
        if (!output) {
            std::fprintf(
                stderr,
                "[crossvmaf-hotspot] failed to open CLI hotspot output: %s\n",
                output_path_.c_str());
            return;
        }
        output.write(
            report_text.data(),
            static_cast<std::streamsize>(report_text.size()));
        output.flush();
        if (!output) {
            std::fprintf(
                stderr,
                "[crossvmaf-hotspot] failed to write CLI hotspot output: %s\n",
                output_path_.c_str());
        }
    } catch (...) {
        std::fprintf(
            stderr,
            "[crossvmaf-hotspot] failed to serialize CLI hotspot report\n");
    }
}

}  // namespace crossvmaf::cli
