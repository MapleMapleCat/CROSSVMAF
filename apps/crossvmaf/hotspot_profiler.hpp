#pragma once

#include "crossvmaf/device.hpp"
#include "crossvmaf/stream.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace crossvmaf::cli {

enum class CliHotspotStage : std::uint8_t {
    input_open,
    warmup_stream_open,
    warmup_stream_next,
    warmup_stream_finish,
    stream_open,
    stream_next,
    stream_finish,
    reference_open,
    frame_read,
    reference_push,
    reference_finish,
    verification,
    result_collection,
    pooling,
    output_open,
    report_write,
    count,
};

struct CliHotspotStatistics {
    std::uint64_t call_count{};
    std::chrono::nanoseconds total{};
    std::chrono::nanoseconds minimum{};
    std::chrono::nanoseconds maximum{};
};

class CliHotspotProfiler;

class CliHotspotMeasurement final {
public:
    CliHotspotMeasurement() noexcept = default;
    CliHotspotMeasurement(const CliHotspotMeasurement&) = delete;
    CliHotspotMeasurement& operator=(const CliHotspotMeasurement&) = delete;
    CliHotspotMeasurement(CliHotspotMeasurement&& other) noexcept;
    CliHotspotMeasurement& operator=(CliHotspotMeasurement&&) = delete;
    ~CliHotspotMeasurement();

private:
    friend class CliHotspotProfiler;

    CliHotspotMeasurement(
        CliHotspotProfiler* profiler, CliHotspotStage stage) noexcept;

    CliHotspotProfiler* profiler_{};
    CliHotspotStage stage_{};
    std::chrono::steady_clock::time_point start_{};
};

class CliHotspotProfiler final {
public:
    [[nodiscard]] static CliHotspotProfiler create_from_environment();

    CliHotspotProfiler(const CliHotspotProfiler&) = delete;
    CliHotspotProfiler& operator=(const CliHotspotProfiler&) = delete;
    ~CliHotspotProfiler();

    [[nodiscard]] bool enabled() const noexcept { return enabled_; }

    [[nodiscard]] CliHotspotMeasurement measure(
        CliHotspotStage stage) noexcept;

    template <typename Operation>
    [[nodiscard]] decltype(auto) measure_operation(
        CliHotspotStage stage, Operation&& operation) {
        [[maybe_unused]] auto measurement = measure(stage);
        return std::forward<Operation>(operation)();
    }

    [[nodiscard]] CliHotspotStatistics statistics(
        CliHotspotStage stage) const noexcept;

    void record_stream_statistics(
        const EvaluationStreamStatistics& statistics) noexcept;

    void complete(
        std::uint64_t input_frame_count,
        std::uint64_t emitted_frame_count,
        const DeviceInfo& device) noexcept;

private:
    friend class CliHotspotMeasurement;

    static constexpr std::size_t stage_count =
        static_cast<std::size_t>(CliHotspotStage::count);

    CliHotspotProfiler(bool enabled, std::string output_path) noexcept;

    [[nodiscard]] std::chrono::steady_clock::time_point begin_measurement(
        CliHotspotStage stage) noexcept;
    void finish_measurement(
        CliHotspotStage stage,
        std::chrono::steady_clock::time_point started_at) noexcept;
    void write_report(
        bool complete, std::uint64_t input_frame_count,
        std::uint64_t emitted_frame_count,
        const DeviceInfo* device) noexcept;

    struct ReportSnapshot {
        std::array<CliHotspotStatistics, stage_count> statistics;
        std::array<std::vector<std::chrono::nanoseconds>, stage_count>
            stage_samples;
        EvaluationStreamStatistics stream;
        bool stream_recorded{};
        std::chrono::steady_clock::time_point captured_at;
    };

    [[nodiscard]] ReportSnapshot report_snapshot();

    bool enabled_{};
    bool report_attempted_{};
    std::string output_path_;
    std::chrono::steady_clock::time_point run_start_{};
    mutable std::mutex statistics_mutex_;
    std::array<CliHotspotStatistics, stage_count> statistics_{};
    std::array<std::vector<std::chrono::nanoseconds>, stage_count>
        stage_samples_{};
    EvaluationStreamStatistics stream_statistics_{};
    bool stream_statistics_recorded_{};
};

[[nodiscard]] const char* cli_hotspot_stage_name(
    CliHotspotStage stage) noexcept;

}  // namespace crossvmaf::cli
