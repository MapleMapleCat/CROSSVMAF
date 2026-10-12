#pragma once

#include "evaluation/evaluation_progress.hpp"
#include "evaluation/evaluation_request.hpp"
#include "reports/evaluation_report.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <thread>

namespace crossvmaf::studio {

// Notifications execute on the worker thread. UI dispatch belongs to the caller.
// Callbacks must not throw or call start()/wait() on this runner.
struct EvaluationCallbacks final {
    std::function<void(FrameDescriptor, DeviceInfo, EvaluationStreamStatistics)> opened;
    std::function<void(LiveProgress)> progress;
    std::function<void(std::shared_ptr<const EvaluationReport>)> succeeded;
    std::function<void(std::string, bool)> failed;
};

class EvaluationRunner final {
public:
    EvaluationRunner() = default;
    ~EvaluationRunner();
    EvaluationRunner(const EvaluationRunner&) = delete;
    EvaluationRunner& operator=(const EvaluationRunner&) = delete;

    // start()/wait() are called by the owning thread; cancellation is thread-safe.
    void start(EvaluationRequest request, EvaluationCallbacks callbacks);
    void requestCancellation() noexcept;
    void wait();

private:
    void run(EvaluationRequest request, const EvaluationCallbacks& callbacks);

    std::thread thread_;
    std::atomic_bool cancellation_requested_{false};
};

} // namespace crossvmaf::studio
