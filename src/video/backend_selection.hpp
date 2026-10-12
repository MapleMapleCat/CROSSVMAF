#pragma once
#include "crossvmaf/stream_options.hpp"
namespace crossvmaf::video {
[[nodiscard]] Result<void> validate_decode_selection(const StreamOptions &options);

struct DecodeCandidate final {
    StreamDecodeBackend backend{StreamDecodeBackend::software};
    // Named backends retain their existing first-frame transfer negotiation.
    // Automatically selected backends use an explicit path constrained by policy.
    DecodeTransferPolicy transfer{DecodeTransferPolicy::automatic};
};

struct DecodeCandidatePlan final {
    std::vector<DecodeCandidate> candidates;
    std::vector<std::string> skipped;
};

[[nodiscard]] DecodeCandidatePlan plan_decode_candidates(const StreamOptions &options,
                                                          const DeviceInfo &compute_device);
} // namespace crossvmaf::video
