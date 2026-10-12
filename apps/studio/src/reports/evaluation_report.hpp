#pragma once

#include "evaluation/evaluation_request.hpp"
#include "crossvmaf/version.hpp"

#include <chrono>
#include <string>
#include <vector>

namespace crossvmaf::studio {

struct EvaluationReport final {
    // New evaluations capture this build's version; loading replaces it with
    // the recorded version (empty means unknown), never the reader's version.
    std::string producer_version{software_version};
    // Stable identifier (matches the originating task id).
    std::string report_id;
    EvaluationRequest request;
    DeviceInfo device;
    FrameDescriptor descriptor;
    ModelVariant model{default_model};
    // Media metadata snapshot of this evaluation; opening a saved report
    // never re-derives old information from the original files.
    StreamMediaMetadata media;
    EvaluationStreamStatistics stream_statistics;
    std::vector<FrameResult> frames;
    double pooled_score{};
    double initialization_milliseconds{};
    double processing_milliseconds{};
    double end_to_end_milliseconds{};
    double frames_per_second{};
    std::chrono::system_clock::time_point completed_at;
};

} // namespace crossvmaf::studio
