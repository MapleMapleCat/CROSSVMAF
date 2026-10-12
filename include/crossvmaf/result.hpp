#pragma once

#include "crossvmaf/frame.hpp"
#include "crossvmaf/value.hpp"

#include <cstdint>
#include <optional>

namespace crossvmaf {

struct FeatureVector {
    Q32Value cambi;
    Q32Value speed_chroma_uv;
    Q32Value adm3;
    Q32Value motion3;
};

// Per-frame components accompany the model feature so diagnostic output uses
// the values produced by the selected execution path.
struct SpeedChromaFrameDiagnostics {
    Q32Value u;
    Q32Value v;
    Q32Value uv;
    bool u_singular{};
    bool v_singular{};
};

enum class FrameTimingStatus : std::uint8_t {
    // The input carried no usable presentation timestamp.
    missing,
    // Presentation timestamp and relative media time are known.
    valid,
    // The presentation timestamp moved backwards; the media time is still
    // reported as decoded, but consumers should not treat it as monotonic.
    non_monotonic,
};

// Media timeline position of one evaluated frame on the reference timeline,
// aligned strictly through FrameResult::index. Never synthesized from result
// delivery order or delivery time.
struct FrameTiming final {
    FrameTimingStatus status{FrameTimingStatus::missing};
    // Raw presentation timestamp in time_base units, as decoded.
    std::int64_t presentation_timestamp{};
    Rational time_base;
    // Media time relative to the first evaluated frame, in seconds.
    double media_time_seconds{};
    // Frame duration in seconds when it is derivable (constant frame rate
    // inputs); absent for variable or unknown frame rates.
    std::optional<double> duration_seconds;
};

struct FrameResult {
    std::uint64_t index{};
    FeatureVector features;
    // Model prediction remains floating-point through public delivery and
    // sequence pooling; only extractor-produced features use Q32.32.
    double vmaf{};
    SpeedChromaFrameDiagnostics speed_chroma;
    FrameTiming timing;
};

}  // namespace crossvmaf
