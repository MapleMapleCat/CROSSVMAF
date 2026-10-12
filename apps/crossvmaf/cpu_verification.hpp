#pragma once

#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/model.hpp"
#include "crossvmaf/result.hpp"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace crossvmaf::verification { class ReferenceEvaluator; }

namespace crossvmaf::cli {

struct Options;
class CliHotspotProfiler;

// CPU verification owns its bounded frame storage and fresh software decode
// pass. Create before GPU processing, then evaluate after that pass completes.
class CpuVerification final {
public:
    [[nodiscard]] static Result<std::unique_ptr<CpuVerification>> create(
        const FrameDescriptor& descriptor, ModelVariant model,
        CliHotspotProfiler& hotspot_profiler);
    ~CpuVerification();

    CpuVerification(const CpuVerification&) = delete;
    CpuVerification& operator=(const CpuVerification&) = delete;

    [[nodiscard]] Result<std::vector<FrameResult>> evaluate(
        const Options& options, CliHotspotProfiler& hotspot_profiler);

private:
    CpuVerification(std::unique_ptr<verification::ReferenceEvaluator> evaluator,
                    FrameDescriptor descriptor, std::size_t batch_size);

    std::unique_ptr<verification::ReferenceEvaluator> evaluator_;
    FrameDescriptor descriptor_;
    std::size_t batch_size_;
};

// Shared comparison policy for public features and VMAF, independent of decoding.
[[nodiscard]] Result<void> verify_cpu_results(
    std::span<const FrameResult> gpu_results,
    std::span<const FrameResult> reference_results);

} // namespace crossvmaf::cli
