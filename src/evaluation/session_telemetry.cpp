#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "evaluation/session_telemetry.hpp"
#include <array>
#include <utility>

namespace crossvmaf::evaluation::telemetry {
namespace {

[[nodiscard]] constexpr bool session_telemetry_is_valid_stage(SessionStage stage) noexcept {
    return static_cast<std::size_t>(stage) < session_stage_count;
}

[[nodiscard]] constexpr std::uint32_t session_telemetry_stage_id(SessionStage stage) noexcept {
    return static_cast<std::uint32_t>(stage);
}

} // namespace

const char *session_stage_name(SessionStage stage) noexcept {
    switch (stage) {
    case SessionStage::provider_push:
        return "provider_push";
    case SessionStage::provider_flush:
        return "provider_flush";
    case SessionStage::input_commit:
        return "input_commit";
    case SessionStage::feature_record_submit:
        return "feature_record_submit";
    case SessionStage::cambi_record:
        return "cambi_record";
    case SessionStage::cambi_submit:
        return "cambi_submit";
    case SessionStage::speed_chroma_record:
        return "speed_chroma_record";
    case SessionStage::speed_chroma_submit:
        return "speed_chroma_submit";
    case SessionStage::adm3_record:
        return "adm3_record";
    case SessionStage::adm3_submit:
        return "adm3_submit";
    case SessionStage::motion3_record:
        return "motion3_record";
    case SessionStage::motion3_submit:
        return "motion3_submit";
    case SessionStage::nonblocking_retire:
        return "nonblocking_retire";
    case SessionStage::readback_retirement:
        return "readback_retirement";
    case SessionStage::cambi_retire:
        return "cambi_retire";
    case SessionStage::speed_chroma_retire:
        return "speed_chroma_retire";
    case SessionStage::adm3_retire:
        return "adm3_retire";
    case SessionStage::motion3_retire:
        return "motion3_retire";
    case SessionStage::flush_drain:
        return "flush_drain";
    case SessionStage::count:
        break;
    }
    return "unknown";
}

SessionMeasurement::SessionMeasurement(SessionTelemetry *telemetry, SessionStage stage) noexcept
    : telemetry_(telemetry), stage_(stage) {
    if (telemetry_ != nullptr) {
        start_ = std::chrono::steady_clock::now();
    }
}

SessionMeasurement::SessionMeasurement(SessionMeasurement &&other) noexcept
    : telemetry_(std::exchange(other.telemetry_, nullptr)), stage_(other.stage_),
      start_(other.start_) {
}

SessionMeasurement::~SessionMeasurement() {
    if (telemetry_ == nullptr) return;
    telemetry_->record(stage_, start_, std::chrono::steady_clock::now());
}

SessionTelemetry::SessionTelemetry(DeviceInfo device, FrameDescriptor descriptor,
                                   std::shared_ptr<SnapshotSink> snapshot_sink)
    : device_(std::move(device)), descriptor_(descriptor), snapshot_sink_(std::move(snapshot_sink)),
      runtime_telemetry_(snapshot_sink_ != nullptr, session_stage_count),
      input_staging_operation_(runtime_telemetry_.register_gpu_operation()),
      cambi_telemetry_(runtime_telemetry_), speed_chroma_telemetry_(runtime_telemetry_),
      adm3_telemetry_(runtime_telemetry_), motion3_telemetry_(runtime_telemetry_) {
}

bool SessionTelemetry::enabled() const noexcept {
    return runtime_telemetry_.enabled();
}

SessionMeasurement SessionTelemetry::measure(SessionStage stage) noexcept {
    return SessionMeasurement(enabled() && session_telemetry_is_valid_stage(stage) ? this : nullptr,
                              stage);
}

const gpu::HostDurationStatistics &SessionTelemetry::statistics(SessionStage stage) const noexcept {
    if (!session_telemetry_is_valid_stage(stage)) {
        static constexpr gpu::HostDurationStatistics empty_statistics{};
        return empty_statistics;
    }
    return runtime_telemetry_.host_statistics(session_telemetry_stage_id(stage));
}

gpu::TelemetryCollector &SessionTelemetry::runtime() noexcept {
    return runtime_telemetry_;
}

gpu::GpuOperationHandle SessionTelemetry::input_staging_operation_handle() const noexcept {
    return input_staging_operation_;
}

metrics::cambi::vulkan::Telemetry &SessionTelemetry::cambi() noexcept {
    return cambi_telemetry_;
}

metrics::speed_chroma::vulkan::Telemetry &SessionTelemetry::speed_chroma() noexcept {
    return speed_chroma_telemetry_;
}

metrics::adm3::vulkan::Telemetry &SessionTelemetry::adm3() noexcept {
    return adm3_telemetry_;
}

metrics::motion3::vulkan::Telemetry &SessionTelemetry::motion3() noexcept {
    return motion3_telemetry_;
}

void SessionTelemetry::finalize(bool complete) noexcept {
    if (!enabled() || finalized_) return;
    finalized_ = true;
    try {
        snapshot_sink_->publish({
            complete,
            device_,
            descriptor_,
            runtime_telemetry_.snapshot(),
            input_staging_operation_,
            runtime_telemetry_.gpu_operation(input_staging_operation_),
            cambi_telemetry_.snapshot(),
            speed_chroma_telemetry_.snapshot(),
            adm3_telemetry_.snapshot(),
            motion3_telemetry_.snapshot(),
        });
    } catch (...) {
        // Diagnostics are fail-open and must not affect evaluation cleanup.
    }
}

void SessionTelemetry::record(SessionStage stage, std::chrono::steady_clock::time_point started_at,
                              std::chrono::steady_clock::time_point finished_at) noexcept {
    if (!session_telemetry_is_valid_stage(stage)) return;
    runtime_telemetry_.record_host_duration(session_telemetry_stage_id(stage), started_at,
                                            finished_at);
}

} // namespace crossvmaf::evaluation::telemetry
