#pragma once
#include "crossvmaf/device.hpp"
#include "crossvmaf/frame.hpp"
#include "metrics/adm3/vulkan.hpp"
#include "metrics/cambi/vulkan.hpp"
#include "metrics/motion3/vulkan.hpp"
#include "metrics/speed_chroma/vulkan.hpp"
#include "gpu/timestamp.hpp"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace crossvmaf::evaluation::telemetry {

enum class SessionStage : std::uint8_t {
    provider_push,
    provider_flush,
    input_commit,
    feature_record_submit,
    cambi_record,
    cambi_submit,
    speed_chroma_record,
    speed_chroma_submit,
    adm3_record,
    adm3_submit,
    motion3_record,
    motion3_submit,
    nonblocking_retire,
    readback_retirement,
    cambi_retire,
    speed_chroma_retire,
    adm3_retire,
    motion3_retire,
    flush_drain,
    count,
};

inline constexpr std::size_t session_stage_count = static_cast<std::size_t>(SessionStage::count);

[[nodiscard]] const char *session_stage_name(SessionStage stage) noexcept;

struct VulkanHotspotSnapshot final {
    bool complete{};
    DeviceInfo device;
    FrameDescriptor descriptor;
    gpu::TelemetrySnapshot runtime;
    gpu::GpuOperationHandle input_staging_operation_handle{};
    gpu::GpuOperationSnapshot input_staging_operation;
    metrics::cambi::vulkan::TelemetrySnapshot cambi;
    metrics::speed_chroma::vulkan::TelemetrySnapshot speed_chroma;
    metrics::adm3::vulkan::TelemetrySnapshot adm3;
    metrics::motion3::vulkan::TelemetrySnapshot motion3;
};

class SnapshotSink {
  public:
    SnapshotSink() = default;
    SnapshotSink(const SnapshotSink &) = delete;
    SnapshotSink &operator=(const SnapshotSink &) = delete;
    virtual ~SnapshotSink() = default;

    virtual void publish(VulkanHotspotSnapshot snapshot) noexcept = 0;
};

#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
class SessionTelemetry;

class SessionMeasurement final {
  public:
    SessionMeasurement() noexcept = default;
    SessionMeasurement(const SessionMeasurement &) = delete;
    SessionMeasurement &operator=(const SessionMeasurement &) = delete;
    SessionMeasurement(SessionMeasurement &&other) noexcept;
    SessionMeasurement &operator=(SessionMeasurement &&) = delete;
    ~SessionMeasurement();

  private:
    friend class SessionTelemetry;

    SessionMeasurement(SessionTelemetry *telemetry, SessionStage stage) noexcept;

    SessionTelemetry *telemetry_{};
    SessionStage stage_{};
    std::chrono::steady_clock::time_point start_{};
};

class SessionTelemetry final {
  public:
    SessionTelemetry(DeviceInfo device, FrameDescriptor descriptor,
                     std::shared_ptr<SnapshotSink> snapshot_sink);
    SessionTelemetry(const SessionTelemetry &) = delete;
    SessionTelemetry &operator=(const SessionTelemetry &) = delete;
    ~SessionTelemetry() = default;

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] SessionMeasurement measure(SessionStage stage) noexcept;
    [[nodiscard]] const gpu::HostDurationStatistics &statistics(SessionStage stage) const noexcept;

    [[nodiscard]] gpu::TelemetryCollector &runtime() noexcept;
    [[nodiscard]] gpu::GpuOperationHandle input_staging_operation_handle() const noexcept;
    [[nodiscard]] metrics::cambi::vulkan::Telemetry &cambi() noexcept;
    [[nodiscard]] metrics::speed_chroma::vulkan::Telemetry &speed_chroma() noexcept;
    [[nodiscard]] metrics::adm3::vulkan::Telemetry &adm3() noexcept;
    [[nodiscard]] metrics::motion3::vulkan::Telemetry &motion3() noexcept;

    void finalize(bool complete) noexcept;

  private:
    friend class SessionMeasurement;

    void record(SessionStage stage, std::chrono::steady_clock::time_point started_at,
                std::chrono::steady_clock::time_point finished_at) noexcept;

    DeviceInfo device_;
    FrameDescriptor descriptor_;
    std::shared_ptr<SnapshotSink> snapshot_sink_;
    gpu::TelemetryCollector runtime_telemetry_;
    gpu::GpuOperationHandle input_staging_operation_{};
    metrics::cambi::vulkan::Telemetry cambi_telemetry_;
    metrics::speed_chroma::vulkan::Telemetry speed_chroma_telemetry_;
    metrics::adm3::vulkan::Telemetry adm3_telemetry_;
    metrics::motion3::vulkan::Telemetry motion3_telemetry_;
    bool finalized_{};
};
#endif

} // namespace crossvmaf::evaluation::telemetry
