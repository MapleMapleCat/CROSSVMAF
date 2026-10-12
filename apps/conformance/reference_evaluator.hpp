#pragma once

#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/model.hpp"
#include "crossvmaf/result.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace crossvmaf::verification {

struct ReferenceEvaluatorConfig {
    FrameDescriptor frame;
    ModelVariant model{default_model};
    std::size_t maximum_batch_size{1U};
};

// Synchronous CPU oracle used by conformance and explicit verification tools.
// It consumes borrowed frame storage before push() or push_batch() returns;
// the batch path may parallelize independent feature work internally.
class ReferenceEvaluator final {
public:
    [[nodiscard]] static Result<std::unique_ptr<ReferenceEvaluator>> create(
        const ReferenceEvaluatorConfig& config);

    ~ReferenceEvaluator();
    ReferenceEvaluator(ReferenceEvaluator&&) noexcept;
    ReferenceEvaluator& operator=(ReferenceEvaluator&&) noexcept;

    ReferenceEvaluator(const ReferenceEvaluator&) = delete;
    ReferenceEvaluator& operator=(const ReferenceEvaluator&) = delete;

    [[nodiscard]] Result<std::vector<FrameResult>> push(
        const FrameView& reference, const FrameView& distorted);
    [[nodiscard]] Result<std::vector<FrameResult>> push_batch(
        std::span<const FrameView> reference_frames,
        std::span<const FrameView> distorted_frames);
    [[nodiscard]] Result<std::vector<FrameResult>> flush();

private:
    class Impl;
    explicit ReferenceEvaluator(std::unique_ptr<Impl> impl) noexcept;

    std::unique_ptr<Impl> impl_;
};

}  // namespace crossvmaf::verification
