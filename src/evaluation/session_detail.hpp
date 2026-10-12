#pragma once
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "evaluation/input_lease_pool.hpp"
#include "gpu/submission.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace crossvmaf::metrics::adm3::vulkan {
class Executor;
}

namespace crossvmaf::metrics::cambi::vulkan {
class Executor;
}

namespace crossvmaf::metrics::motion3::vulkan {
class Executor;
class SharedState;
} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::speed_chroma::vulkan {
class Executor;
}

namespace crossvmaf::evaluation::telemetry {
class SessionTelemetry;
}

namespace crossvmaf::evaluation::session_detail {

struct FrameSlot final {
    FrameSlot();
    ~FrameSlot();
    FrameSlot(FrameSlot &&) noexcept;
    FrameSlot &operator=(FrameSlot &&) noexcept;

    FrameSlot(const FrameSlot &) = delete;
    FrameSlot &operator=(const FrameSlot &) = delete;

    void begin_frame(std::uint64_t index) noexcept;

    std::size_t slot_index{};
    std::uint64_t frame_index{};
    std::size_t pending_resource_transaction_count{};
    bool in_use{};
};

} // namespace crossvmaf::evaluation::session_detail

namespace crossvmaf::evaluation::session_detail {

// The physical FrameSlot bank is created once, while this controller adjusts
// how much of that bank admission may use. A short two-slot warmup avoids
// over-pipelining tiny streams; sustained full-bank pressure then opens the
// third slot without allocating resources in push().
class FrameSlotAdmissionController final {
  public:
    static constexpr std::size_t physical_slot_count = 3U;
    static constexpr std::size_t initial_active_slot_count = 2U;
    static constexpr std::size_t warmup_submission_count = 8U;
    static constexpr std::size_t minimum_busy_observation_count = 4U;

    [[nodiscard]] std::size_t active_slot_count() const noexcept { return active_slot_count_; }

    void observe_slot_busy() noexcept {
        if (decision_complete()) return;
        ++busy_observation_count_;
        open_third_slot_if_ready();
    }

    void observe_frame_submitted() noexcept {
        if (decision_complete()) return;
        if (warmup_submissions_remaining_ > 0U) {
            --warmup_submissions_remaining_;
        }
        open_third_slot_if_ready();
    }

  private:
    [[nodiscard]] bool decision_complete() const noexcept {
        return active_slot_count_ == physical_slot_count;
    }

    void open_third_slot_if_ready() noexcept {
        if (warmup_submissions_remaining_ != 0U ||
            busy_observation_count_ < minimum_busy_observation_count) {
            return;
        }
        active_slot_count_ = physical_slot_count;
    }

    std::size_t active_slot_count_{initial_active_slot_count};
    std::size_t warmup_submissions_remaining_{warmup_submission_count};
    std::size_t busy_observation_count_{};
};

} // namespace crossvmaf::evaluation::session_detail

namespace crossvmaf::evaluation::session_detail {

inline constexpr std::size_t invalid_resource_index = std::numeric_limits<std::size_t>::max();

enum class FeatureTransactionKind : std::uint8_t {
    input_staging,
    cambi_prefix,
    cambi,
    speed_chroma_prefix,
    speed_chroma_wave,
    speed_chroma,
    adm3_prefix,
    adm3_wave,
    adm3,
    motion3,
    motion3_flush,
};

[[nodiscard]] constexpr const char *feature_transaction_name(FeatureTransactionKind kind) noexcept {
    switch (kind) {
    case FeatureTransactionKind::input_staging: return "GPU input staging";
    case FeatureTransactionKind::cambi_prefix: return "CAMBI prefix";
    case FeatureTransactionKind::cambi: return "CAMBI";
    case FeatureTransactionKind::speed_chroma_prefix: return "Speed Chroma prefix";
    case FeatureTransactionKind::speed_chroma_wave: return "Speed Chroma wave";
    case FeatureTransactionKind::speed_chroma: return "Speed Chroma";
    case FeatureTransactionKind::adm3_prefix: return "ADM3 prefix";
    case FeatureTransactionKind::adm3_wave: return "ADM3 wave";
    case FeatureTransactionKind::adm3: return "ADM3";
    case FeatureTransactionKind::motion3: return "Motion3";
    case FeatureTransactionKind::motion3_flush: return "Motion3 flush";
    }
    return "GPU operation";
}

struct PendingTransaction final {
    gpu::SubmissionTicket ticket;
    std::size_t frame_slot_index{invalid_resource_index};
    std::uint64_t frame_index{};
    FeatureTransactionKind kind{FeatureTransactionKind::cambi};
    std::uint32_t task_slot{};
    std::uint32_t task_generation{};
    std::array<evaluation::SegmentedInputLeaseConsumer, 2U> input_consumers{};
    std::uint8_t input_consumer_count{};
};

// Fixed logical feature lanes. Lane order matches the historical submission
// order so that aliasing every lane onto one physical queue preserves the
// previous single-queue execution order exactly.
inline constexpr std::size_t feature_lane_count = 4U;
inline constexpr std::size_t cambi_lane = 0U;
inline constexpr std::size_t speed_chroma_lane = 1U;
inline constexpr std::size_t adm3_lane = 2U;
inline constexpr std::size_t motion3_lane = 3U;
inline constexpr std::size_t input_staging_lane = motion3_lane;

[[nodiscard]] constexpr std::size_t feature_lane_of(FeatureTransactionKind kind) noexcept {
    switch (kind) {
    case FeatureTransactionKind::input_staging:
        return input_staging_lane;
    case FeatureTransactionKind::cambi_prefix:
    case FeatureTransactionKind::cambi:
        return cambi_lane;
    case FeatureTransactionKind::speed_chroma_prefix:
    case FeatureTransactionKind::speed_chroma_wave:
    case FeatureTransactionKind::speed_chroma:
        return speed_chroma_lane;
    case FeatureTransactionKind::adm3_prefix:
    case FeatureTransactionKind::adm3_wave:
    case FeatureTransactionKind::adm3:
        return adm3_lane;
    case FeatureTransactionKind::motion3:
    case FeatureTransactionKind::motion3_flush:
        return motion3_lane;
    }
    return 0U;
}

} // namespace crossvmaf::evaluation::session_detail
#endif
