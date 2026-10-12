#pragma once
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace crossvmaf::evaluation {

enum class FramePairSide : std::uint8_t {
    reference = 0U,
    distorted = 1U,
};

struct FrameDecodeTiming final {
    std::optional<std::chrono::steady_clock::time_point> started_at;
    std::optional<std::chrono::steady_clock::time_point> completed_at;
};

class FramePairCompletionTracker final {
  public:
    FramePairCompletionTracker() = default;
    FramePairCompletionTracker(const FramePairCompletionTracker &) = delete;
    FramePairCompletionTracker &operator=(const FramePairCompletionTracker &) = delete;

    [[nodiscard]] Result<void> reset(std::uint64_t generation, std::uint64_t frame_index);
    [[nodiscard]] Result<void> claim(FramePairSide side, std::uint64_t expected_generation);
    [[nodiscard]] Result<void> fail_unsubmitted(FramePairSide side,
                                                std::uint64_t expected_generation, Error error);
    [[nodiscard]] Result<void> mark_started(
        FramePairSide side, std::uint64_t expected_generation,
        std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now());
    [[nodiscard]] Result<void>
    complete(FramePairSide side, std::uint64_t expected_generation, Result<void> completion,
             std::chrono::steady_clock::time_point completed_at = std::chrono::steady_clock::now());

    [[nodiscard]] bool completed() const noexcept;
    [[nodiscard]] bool completed(FramePairSide side) const noexcept;
    [[nodiscard]] bool succeeded() const noexcept;
    [[nodiscard]] bool accepted(FramePairSide side) const noexcept;
    [[nodiscard]] bool quiescent() const noexcept;
    [[nodiscard]] Result<void> wait() const;
    [[nodiscard]] FrameDecodeTiming timing(FramePairSide side) const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] std::uint64_t frame_index() const noexcept;

  private:
    static constexpr std::uint8_t reference_mask = 1U << 0U;
    static constexpr std::uint8_t distorted_mask = 1U << 1U;
    static constexpr std::uint8_t all_sides_mask = reference_mask | distorted_mask;
    static constexpr std::int64_t missing_timestamp = -1;

    [[nodiscard]] static constexpr std::size_t index_for(FramePairSide side) noexcept;
    [[nodiscard]] static constexpr std::uint8_t mask_for(FramePairSide side) noexcept;
    [[nodiscard]] static std::int64_t
    to_nanoseconds(std::chrono::steady_clock::time_point time_point) noexcept;
    [[nodiscard]] static std::optional<std::chrono::steady_clock::time_point>
    from_nanoseconds(std::int64_t nanoseconds) noexcept;
    [[nodiscard]] Result<void> validate_generation(std::uint64_t expected_generation) const;

    std::atomic<std::uint64_t> generation_{};
    std::atomic<std::uint64_t> frame_index_{};
    std::atomic<std::uint8_t> accepted_mask_{};
    std::atomic<std::uint8_t> completion_claimed_mask_{};
    mutable std::atomic<std::uint8_t> completed_mask_{};
    std::atomic<std::uint8_t> error_state_{};
    std::optional<Error> first_error_;
    std::array<std::atomic<std::int64_t>, 2U> started_nanoseconds_{
        {missing_timestamp, missing_timestamp}};
    std::array<std::atomic<std::int64_t>, 2U> completed_nanoseconds_{
        {missing_timestamp, missing_timestamp}};
};

} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {

inline constexpr std::uint32_t invalid_storage_segment = std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint32_t invalid_storage_slot = std::numeric_limits<std::uint32_t>::max();

// This storage token keeps provider-specific resource handles out of the
// frame reader and adaptive prefetch controller.
struct FrameStorageHandle final {
    std::uint32_t segment_index{invalid_storage_segment};
    std::uint32_t slot_index{invalid_storage_slot};
    std::uint64_t generation{};

    [[nodiscard]] explicit operator bool() const noexcept {
        return segment_index != invalid_storage_segment && slot_index != invalid_storage_slot &&
               generation != 0U;
    }
};

struct FramePairStorageReservation final {
    enum class Mode : std::uint8_t {
        cpu_writable,
        vulkan_decoded,
    };

    FrameStorageHandle frame_pair_storage;
    FrameStorageHandle reference_luma_storage;
    WritableFrameView reference_destination;
    WritableFrameView distorted_destination;
    Mode mode{Mode::cpu_writable};
};

} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {

enum class FramePairRecordState : std::uint8_t {
    free,
    initializing,
    scheduled,
    decoding,
    ready,
    failed,
    submitted,
    retired,
    abandoned,
    recycling,
};

struct FramePairRecordSchedule final {
    std::uint64_t frame_index{};
    std::uint64_t generation{};
    FrameStorageHandle frame_pair_storage;
    FrameStorageHandle reference_luma_storage;
    WritableFrameView reference_destination;
    WritableFrameView distorted_destination;
    FramePairStorageReservation::Mode storage_mode{FramePairStorageReservation::Mode::cpu_writable};
};

class FramePairRecord final {
  public:
    FramePairRecord() = default;
    FramePairRecord(const FramePairRecord &) = delete;
    FramePairRecord &operator=(const FramePairRecord &) = delete;

    [[nodiscard]] Result<void> schedule(const FramePairRecordSchedule &schedule);
    [[nodiscard]] FramePairRecordState state() const noexcept;
    [[nodiscard]] std::uint64_t frame_index() const noexcept;
    [[nodiscard]] std::uint64_t generation() const noexcept;
    [[nodiscard]] FrameStorageHandle frame_pair_storage() const noexcept;
    [[nodiscard]] FrameStorageHandle reference_luma_storage() const noexcept;
    [[nodiscard]] FramePairStorageReservation::Mode storage_mode() const noexcept;
    [[nodiscard]] WritableFrameView destination(FramePairSide side) const noexcept;

    [[nodiscard]] Result<void> claim_decode(FramePairSide side, std::uint64_t expected_generation);
    [[nodiscard]] Result<void>
    fail_unsubmitted_decode(FramePairSide side, std::uint64_t expected_generation, Error error);
    [[nodiscard]] Result<void> mark_decode_started(
        FramePairSide side, std::uint64_t expected_generation,
        std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now());
    [[nodiscard]] Result<void> complete_decode(
        FramePairSide side, std::uint64_t expected_generation, Result<void> completion,
        std::chrono::steady_clock::time_point completed_at = std::chrono::steady_clock::now());

    [[nodiscard]] bool decodes_completed() const noexcept;
    [[nodiscard]] bool decode_accepted(FramePairSide side) const noexcept;
    // The owner must have joined the producer before sealing interrupted work.
    [[nodiscard]] Result<void> seal_cancelled_decodes(std::uint64_t expected_generation);
    [[nodiscard]] bool ready() const noexcept;
    [[nodiscard]] Result<void> wait_for_decodes() const;
    [[nodiscard]] FrameDecodeTiming decode_timing(FramePairSide side) const noexcept;

    [[nodiscard]] Result<void> mark_submitted(std::uint64_t expected_generation);
    [[nodiscard]] Result<void> mark_retired(std::uint64_t expected_generation);
    [[nodiscard]] Result<void> abandon(std::uint64_t expected_generation);
    [[nodiscard]] Result<void> recycle(std::uint64_t expected_generation);

  private:
    void synchronize_decode_completion_state() const noexcept;
    [[nodiscard]] Result<void> validate_generation(std::uint64_t expected_generation) const;

    mutable std::atomic<FramePairRecordState> state_{FramePairRecordState::free};
    std::atomic<std::uint64_t> generation_{};
    std::uint64_t frame_index_{};
    FrameStorageHandle frame_pair_storage_;
    FrameStorageHandle reference_luma_storage_;
    FramePairStorageReservation::Mode storage_mode_{
        FramePairStorageReservation::Mode::cpu_writable};
    WritableFrameView reference_destination_;
    WritableFrameView distorted_destination_;
    FramePairCompletionTracker completion_tracker_;
};

} // namespace crossvmaf::evaluation
#endif
