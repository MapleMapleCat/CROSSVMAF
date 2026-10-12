#include "measurement.hpp"

#include "cpu_verification.hpp"
#include "hotspot_profiler.hpp"
#include "options.hpp"
#include "output.hpp"
#include "vulkan_hotspot_reporter.hpp"

#include "evaluation/stream_opener.hpp"
#include "crossvmaf/pooling.hpp"

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

namespace crossvmaf::cli {
namespace {

[[nodiscard]] crossvmaf::StreamOptions make_stream_options(
    const crossvmaf::cli::Options& options) {
    crossvmaf::StreamOptions stream_options;
    stream_options.reference_path = options.reference_path;
    stream_options.distorted_path = options.distorted_path;
    stream_options.model = options.model;
    stream_options.device = options.device;
    stream_options.decode_mode = options.decode_mode;
    stream_options.decode_backend = options.decode_backend;
    stream_options.decode_device = options.decode_device;
    stream_options.decode_transfer = options.decode_transfer;

    stream_options.decoded_frame_queue_capacity =
        options.decoded_frame_queue_capacity;
    stream_options.decoder_threads_per_source =
        options.decoder_threads_per_source;
    return stream_options;
}

template <typename Clock>
[[nodiscard]] double milliseconds(
    const std::chrono::time_point<Clock>& start,
    const std::chrono::time_point<Clock>& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

int run_measurement(const crossvmaf::cli::Options& options) {
    auto hotspot_profiler =
        crossvmaf::cli::CliHotspotProfiler::create_from_environment();
    auto vulkan_hotspot_reporter =
        crossvmaf::cli::VulkanHotspotReporter::create_from_environment();
    const auto total_start = std::chrono::steady_clock::now();
    const auto initialization_start = std::chrono::steady_clock::now();
    auto stream = hotspot_profiler.measure_operation(
        crossvmaf::cli::CliHotspotStage::stream_open,
        [&options, &vulkan_hotspot_reporter] {
            return crossvmaf::detail::EvaluationStreamTelemetryOpener::open(
                make_stream_options(options),
                vulkan_hotspot_reporter);
        });
    const auto initialization_end = std::chrono::steady_clock::now();
    if (!stream) {
        std::cerr << stream.error().message << '\n';
        return 5;
    }
    const crossvmaf::FrameDescriptor descriptor =
        stream.value()->descriptor();
    std::unique_ptr<CpuVerification> cpu_verification;
    if (options.verify_cpu) {
        auto created = CpuVerification::create(
            descriptor, stream.value()->model(), hotspot_profiler);
        if (!created) {
            std::cerr << created.error().message << '\n';
            return 6;
        }
        cpu_verification = std::move(created).value();
    }

    std::vector<crossvmaf::FrameResult> frames;
    const auto processing_start = std::chrono::steady_clock::now();
    for (;;) {
        auto available = hotspot_profiler.measure_operation(
            crossvmaf::cli::CliHotspotStage::stream_next,
            [&stream] { return stream.value()->next(); });
        if (!available) {
            std::cerr << available.error().message << '\n';
            return 5;
        }
        if (available.value().empty()) break;
        {
            [[maybe_unused]] auto collection_measurement =
                hotspot_profiler.measure(
                    crossvmaf::cli::CliHotspotStage::result_collection);
            frames.insert(
                frames.end(), available.value().begin(),
                available.value().end());
        }
    }

    auto tail = hotspot_profiler.measure_operation(
        crossvmaf::cli::CliHotspotStage::stream_finish,
        [&stream] { return stream.value()->finish(); });
    if (!tail) {
        std::cerr << tail.error().message << '\n';
        return 5;
    }
    frames.insert(frames.end(), tail.value().begin(), tail.value().end());

    if (cpu_verification) {
        auto reference_results = cpu_verification->evaluate(options, hotspot_profiler);
        if (!reference_results) {
            std::cerr << reference_results.error().message << '\n';
            return 6;
        }
        const auto verified = hotspot_profiler.measure_operation(
            crossvmaf::cli::CliHotspotStage::verification,
            [&] {
                return verify_cpu_results(frames, reference_results.value());
            });
        if (!verified) {
            std::cerr << verified.error().message << '\n';
            return 7;
        }
    }
    const auto processing_end = std::chrono::steady_clock::now();

    if (frames.empty()) {
        std::cerr << "input contains no decoded frames\n";
        return 3;
    }
    const auto pooled = hotspot_profiler.measure_operation(
        crossvmaf::cli::CliHotspotStage::pooling,
        [&] {
            std::vector<double> scores;
            scores.reserve(frames.size());
            for (const crossvmaf::FrameResult& frame : frames) {
                scores.push_back(frame.vmaf);
            }
            return crossvmaf::pool_scores(scores, options.pool);
        });
    if (!pooled) {
        std::cerr << pooled.error().message << '\n';
        return 8;
    }

    const auto total_end = std::chrono::steady_clock::now();
    const double processing_ms = milliseconds(
        processing_start, processing_end);
    const double frames_per_second = processing_ms > 0.0
        ? static_cast<double>(frames.size()) * 1000.0 / processing_ms
        : 0.0;
    const crossvmaf::DeviceInfo device = stream.value()->device_info();
    const crossvmaf::EvaluationStreamStatistics stream_statistics =
        stream.value()->statistics();
    crossvmaf::cli::RunReport report{
        device,
        descriptor,
        stream.value()->model(),
        options.pool,
        pooled.value(),
        std::move(frames),
        0.0,
        milliseconds(initialization_start, initialization_end),
        processing_ms,
        milliseconds(total_start, total_end),
        frames_per_second,
        options.verify_cpu,
        options.diagnostic_speed_chroma,
        options.decoded_frame_queue_capacity,
        options.decoder_threads_per_source,
        stream_statistics, options.decode_backend, options.decode_device, options.decode_transfer};

    std::ofstream output_file;
    std::ostream* output = &std::cout;
    if (options.output_path.has_value()) {
        output_file.open(
            *options.output_path,
            std::ios::binary | std::ios::trunc);
        if (!output_file) {
            std::cerr << "failed to open output: "
                      << options.output_path->string() << '\n';
            return 3;
        }
        output = &output_file;
    }
    const auto wrote = hotspot_profiler.measure_operation(
        crossvmaf::cli::CliHotspotStage::report_write,
        [&] {
            return crossvmaf::cli::write_report(
                *output, options.output_format, report);
        });
    if (!wrote) {
        std::cerr << wrote.error().message << '\n';
        return 3;
    }
    if (output_file.is_open()) {
        output_file.close();
        if (!output_file) {
            std::cerr << "failed to close CrossVMAF output\n";
            return 3;
        }
    }
    hotspot_profiler.record_stream_statistics(stream_statistics);
    hotspot_profiler.complete(
        stream.value()->decoded_frame_count(),
        static_cast<std::uint64_t>(report.frames.size()),
        report.device);
    return 0;
}

} // namespace crossvmaf::cli
