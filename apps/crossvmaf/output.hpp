#pragma once

#include "options.hpp"

#include "crossvmaf/result.hpp"
#include "crossvmaf/stream.hpp"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

namespace crossvmaf::cli {

struct RunReport {
    DeviceInfo device;
    FrameDescriptor descriptor;
    ModelVariant model{};
    PoolMethod pool_method{};
    double pooled_score{};
    std::vector<FrameResult> frames;
    double warmup_ms{};
    double initialization_ms{};
    double processing_ms{};
    double end_to_end_ms{};
    double frames_per_second{};
    bool cpu_verified{};
    bool diagnostic_speed_chroma{};
    std::size_t decoded_frame_queue_capacity{};
    std::size_t decoder_threads_per_source{};
    EvaluationStreamStatistics stream_statistics;
    DecodeBackendPreference decode_backend{DecodeBackendPreference::automatic};
    std::string decode_device{"same"};
    DecodeTransferPolicy decode_transfer{DecodeTransferPolicy::automatic};

};

[[nodiscard]] Result<void> write_report(std::ostream& output,
                                        OutputFormat format,
                                        const RunReport& report);

}  // namespace crossvmaf::cli
