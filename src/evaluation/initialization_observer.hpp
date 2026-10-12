#pragma once
#include "crossvmaf/error.hpp"
#include <cstddef>

namespace crossvmaf::gpu {
class Device;
}

namespace crossvmaf::evaluation {

enum class InitializationStage {
    frame_slot,
    input_pool,
    storage_provider,
    input_capacity,
    result_buffer,
};

// Internal per-open fault/observation seam. Never retained by a running stream
// or consulted on the frame submission path. Observers may return an error or
// throw to exercise both initialization unwind mechanisms.
class InitializationObserver {
  public:
    virtual ~InitializationObserver() = default;
    virtual void device_prepared(gpu::Device &) {}
    virtual void decoders_prepared(bool) {}
    [[nodiscard]] virtual Result<void> before(InitializationStage stage,
                                              std::size_t index = 0U) = 0;
};

[[nodiscard]] inline Result<void> observe_initialization(InitializationObserver *observer,
                                                         InitializationStage stage,
                                                         std::size_t index = 0U) {
    return observer == nullptr ? Result<void>::success() : observer->before(stage, index);
}

} // namespace crossvmaf::evaluation
