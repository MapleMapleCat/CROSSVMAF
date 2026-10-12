#pragma once
#include "crossvmaf/self_test.hpp"
#include "crossvmaf/stream.hpp"
#include <memory>
#include <stop_token>

namespace crossvmaf::evaluation {
class InitializationObserver;
}

namespace crossvmaf::evaluation::telemetry {
class SnapshotSink;
}

namespace crossvmaf::detail {

class EvaluationStreamTelemetryOpener final {
  public:
    [[nodiscard]] static DeviceSelfTestResult run_self_test(const DeviceSelector &device);
    [[nodiscard]] static Result<std::unique_ptr<EvaluationStream>>
    open(StreamOptions options,
         std::shared_ptr<evaluation::telemetry::SnapshotSink> telemetry_snapshot_sink = nullptr,
         evaluation::InitializationObserver *initialization_observer = nullptr,
         std::stop_token stop_token = {});
};

} // namespace crossvmaf::detail

#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "crossvmaf/device.hpp"
#include "crossvmaf/error.hpp"
#include <vector>

namespace crossvmaf::evaluation {

[[nodiscard]] Result<std::vector<DeviceInfo>> enumerate_devices();

} // namespace crossvmaf::evaluation
#endif
