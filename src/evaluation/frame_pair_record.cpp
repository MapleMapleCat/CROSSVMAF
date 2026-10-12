#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "evaluation/frame_pair_record.hpp"
#include <utility>

namespace crossvmaf::evaluation {

Result<void> FramePairCompletionTracker::reset(std::uint64_t generation,
                                               std::uint64_t frame_index) {
    if (generation == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "frame completion generation must be non-zero"});
    }
    const std::uint64_t previous_generation = generation_.load(std::memory_order_relaxed);
    if (generation <= previous_generation) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame completion generations must increase monotonically"});
    }

    first_error_.reset();
    error_state_.store(0U, std::memory_order_relaxed);
    completion_claimed_mask_.store(0U, std::memory_order_relaxed);
    completed_mask_.store(0U, std::memory_order_relaxed);
    accepted_mask_.store(0U, std::memory_order_relaxed);
    for (std::atomic<std::int64_t> &started_nanoseconds : started_nanoseconds_) {
        started_nanoseconds.store(missing_timestamp, std::memory_order_relaxed);
    }
    for (std::atomic<std::int64_t> &completed_nanoseconds : completed_nanoseconds_) {
        completed_nanoseconds.store(missing_timestamp, std::memory_order_relaxed);
    }
    frame_index_.store(frame_index, std::memory_order_relaxed);
    generation_.store(generation, std::memory_order_release);
    return Result<void>::success();
}

Result<void> FramePairCompletionTracker::claim(FramePairSide side,
                                               std::uint64_t expected_generation) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;

    const std::uint8_t side_mask = mask_for(side);
    const std::uint8_t previous_mask =
        accepted_mask_.fetch_or(side_mask, std::memory_order_acq_rel);
    if ((previous_mask & side_mask) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame-pair side was claimed more than once"});
    }
    return Result<void>::success();
}

Result<void> FramePairCompletionTracker::fail_unsubmitted(FramePairSide side,
                                                          std::uint64_t expected_generation,
                                                          Error error) {
    const auto claim_result = claim(side, expected_generation);
    if (!claim_result) return claim_result;
    return complete(side, expected_generation, Result<void>::failure(std::move(error)));
}

Result<void>
FramePairCompletionTracker::mark_started(FramePairSide side, std::uint64_t expected_generation,
                                         std::chrono::steady_clock::time_point started_at) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;

    const std::uint8_t side_mask = mask_for(side);
    if ((accepted_mask_.load(std::memory_order_acquire) & side_mask) == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "an unclaimed frame decode cannot be marked as started"});
    }
    if ((completed_mask_.load(std::memory_order_acquire) & side_mask) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "a completed frame decode cannot be marked as started"});
    }

    std::int64_t expected_timestamp = missing_timestamp;
    if (!started_nanoseconds_[index_for(side)].compare_exchange_strong(
            expected_timestamp, to_nanoseconds(started_at), std::memory_order_release,
            std::memory_order_relaxed)) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame-pair side was marked as started more than once"});
    }
    return Result<void>::success();
}

Result<void>
FramePairCompletionTracker::complete(FramePairSide side, std::uint64_t expected_generation,
                                     Result<void> completion,
                                     std::chrono::steady_clock::time_point completed_at) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;

    const std::uint8_t side_mask = mask_for(side);
    if ((accepted_mask_.load(std::memory_order_acquire) & side_mask) == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "an unclaimed frame decode cannot be completed"});
    }

    const std::uint8_t previous_claimed_mask =
        completion_claimed_mask_.fetch_or(side_mask, std::memory_order_acq_rel);
    if ((previous_claimed_mask & side_mask) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame-pair side completed more than once"});
    }

    completed_nanoseconds_[index_for(side)].store(to_nanoseconds(completed_at),
                                                  std::memory_order_relaxed);
    if (!completion) {
        std::uint8_t expected_error_state = 0U;
        if (error_state_.compare_exchange_strong(
                expected_error_state, 1U, std::memory_order_acq_rel, std::memory_order_acquire)) {
            first_error_ = std::move(completion).error();
            if (first_error_->input == ErrorInput::none) {
                first_error_->input = side == FramePairSide::reference
                    ? ErrorInput::reference : ErrorInput::distorted;
            }
            if (!first_error_->frame_index) first_error_->frame_index = frame_index_.load(std::memory_order_relaxed);
            error_state_.store(2U, std::memory_order_release);
        }
    }

    completed_mask_.fetch_or(side_mask, std::memory_order_acq_rel);
    completed_mask_.notify_all();
    return Result<void>::success();
}

bool FramePairCompletionTracker::completed() const noexcept {
    return completed_mask_.load(std::memory_order_acquire) == all_sides_mask;
}

bool FramePairCompletionTracker::completed(FramePairSide side) const noexcept {
    return (completed_mask_.load(std::memory_order_acquire) & mask_for(side)) != 0U;
}

bool FramePairCompletionTracker::succeeded() const noexcept {
    return completed() && error_state_.load(std::memory_order_acquire) == 0U;
}

bool FramePairCompletionTracker::accepted(FramePairSide side) const noexcept {
    return (accepted_mask_.load(std::memory_order_acquire) & mask_for(side)) != 0U;
}

bool FramePairCompletionTracker::quiescent() const noexcept {
    return accepted_mask_.load(std::memory_order_acquire) ==
           completed_mask_.load(std::memory_order_acquire);
}

Result<void> FramePairCompletionTracker::wait() const {
    std::uint8_t observed_mask = completed_mask_.load(std::memory_order_acquire);
    while (observed_mask != all_sides_mask) {
        completed_mask_.wait(observed_mask, std::memory_order_acquire);
        observed_mask = completed_mask_.load(std::memory_order_acquire);
    }

    // Acquiring both completion bits also acquires the first-error publication.
    if (error_state_.load(std::memory_order_acquire) == 2U) {
        return Result<void>::failure(*first_error_);
    }
    return Result<void>::success();
}

FrameDecodeTiming FramePairCompletionTracker::timing(FramePairSide side) const noexcept {
    const std::int64_t started_nanoseconds =
        started_nanoseconds_[index_for(side)].load(std::memory_order_acquire);
    const std::int64_t completed_nanoseconds =
        completed_nanoseconds_[index_for(side)].load(std::memory_order_acquire);
    return {from_nanoseconds(started_nanoseconds), from_nanoseconds(completed_nanoseconds)};
}

std::uint64_t FramePairCompletionTracker::generation() const noexcept {
    return generation_.load(std::memory_order_acquire);
}

std::uint64_t FramePairCompletionTracker::frame_index() const noexcept {
    return frame_index_.load(std::memory_order_acquire);
}

constexpr std::size_t FramePairCompletionTracker::index_for(FramePairSide side) noexcept {
    return side == FramePairSide::reference ? 0U : 1U;
}

constexpr std::uint8_t FramePairCompletionTracker::mask_for(FramePairSide side) noexcept {
    return side == FramePairSide::reference ? reference_mask : distorted_mask;
}

std::int64_t FramePairCompletionTracker::to_nanoseconds(
    std::chrono::steady_clock::time_point time_point) noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(time_point.time_since_epoch())
        .count();
}

std::optional<std::chrono::steady_clock::time_point>
FramePairCompletionTracker::from_nanoseconds(std::int64_t nanoseconds) noexcept {
    if (nanoseconds == missing_timestamp) return std::nullopt;
    return std::chrono::steady_clock::time_point(
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::nanoseconds(nanoseconds)));
}

Result<void>
FramePairCompletionTracker::validate_generation(std::uint64_t expected_generation) const {
    if (expected_generation == 0U ||
        generation_.load(std::memory_order_acquire) != expected_generation) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame completion generation does not match its tracker"});
    }
    return Result<void>::success();
}

} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {

Result<void> FramePairRecord::schedule(const FramePairRecordSchedule &schedule) {
    if (schedule.generation == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "frame-pair record generation must be non-zero"});
    }
    if (schedule.reference_destination.index != schedule.frame_index ||
        schedule.distorted_destination.index != schedule.frame_index) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "frame-pair destinations must match the scheduled index"});
    }
    if (schedule.storage_mode == FramePairStorageReservation::Mode::cpu_writable) {
        for (const WritablePlaneView &plane : schedule.reference_destination.planes) {
            if (plane.data == nullptr) {
                return Result<void>::failure({ErrorCode::invalid_argument,
                                              "all reference destination planes must be present"});
            }
        }
        for (const WritablePlaneView &plane : schedule.distorted_destination.planes) {
            if (plane.data == nullptr) {
                return Result<void>::failure({ErrorCode::invalid_argument,
                                              "all distorted destination planes must be present"});
            }
        }
    }

    FramePairRecordState expected_state = FramePairRecordState::free;
    if (!state_.compare_exchange_strong(expected_state, FramePairRecordState::initializing,
                                        std::memory_order_acq_rel, std::memory_order_acquire)) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame-pair record must be free before scheduling"});
    }

    const std::uint64_t previous_generation = generation_.load(std::memory_order_relaxed);
    if (schedule.generation <= previous_generation) {
        state_.store(FramePairRecordState::free, std::memory_order_release);
        return Result<void>::failure({ErrorCode::invalid_state,
                                      "frame-pair record generations must increase monotonically"});
    }

    const auto tracker_reset = completion_tracker_.reset(schedule.generation, schedule.frame_index);
    if (!tracker_reset) {
        state_.store(FramePairRecordState::free, std::memory_order_release);
        return tracker_reset;
    }

    frame_index_ = schedule.frame_index;
    frame_pair_storage_ = schedule.frame_pair_storage;
    reference_luma_storage_ = schedule.reference_luma_storage;
    storage_mode_ = schedule.storage_mode;
    reference_destination_ = schedule.reference_destination;
    distorted_destination_ = schedule.distorted_destination;
    generation_.store(schedule.generation, std::memory_order_relaxed);
    state_.store(FramePairRecordState::scheduled, std::memory_order_release);
    return Result<void>::success();
}

FramePairRecordState FramePairRecord::state() const noexcept {
    synchronize_decode_completion_state();
    return state_.load(std::memory_order_acquire);
}

std::uint64_t FramePairRecord::frame_index() const noexcept {
    return frame_index_;
}

std::uint64_t FramePairRecord::generation() const noexcept {
    return generation_.load(std::memory_order_acquire);
}

FrameStorageHandle FramePairRecord::frame_pair_storage() const noexcept {
    return frame_pair_storage_;
}

FrameStorageHandle FramePairRecord::reference_luma_storage() const noexcept {
    return reference_luma_storage_;
}

FramePairStorageReservation::Mode FramePairRecord::storage_mode() const noexcept {
    return storage_mode_;
}

WritableFrameView FramePairRecord::destination(FramePairSide side) const noexcept {
    return side == FramePairSide::reference ? reference_destination_ : distorted_destination_;
}

Result<void> FramePairRecord::claim_decode(FramePairSide side, std::uint64_t expected_generation) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;

    FramePairRecordState current_state = state_.load(std::memory_order_acquire);
    if (current_state == FramePairRecordState::scheduled) {
        state_.compare_exchange_strong(current_state, FramePairRecordState::decoding,
                                       std::memory_order_acq_rel, std::memory_order_acquire);
    }
    if (current_state != FramePairRecordState::scheduled &&
        current_state != FramePairRecordState::decoding) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame decoding requires a scheduled frame-pair record"});
    }
    return completion_tracker_.claim(side, expected_generation);
}

Result<void> FramePairRecord::fail_unsubmitted_decode(FramePairSide side,
                                                      std::uint64_t expected_generation,
                                                      Error error) {
    const auto claim_result = claim_decode(side, expected_generation);
    if (!claim_result) return claim_result;
    const auto completion_result = completion_tracker_.complete(
        side, expected_generation, Result<void>::failure(std::move(error)));
    synchronize_decode_completion_state();
    return completion_result;
}

Result<void>
FramePairRecord::mark_decode_started(FramePairSide side, std::uint64_t expected_generation,
                                     std::chrono::steady_clock::time_point started_at) {
    return completion_tracker_.mark_started(side, expected_generation, started_at);
}

Result<void> FramePairRecord::complete_decode(FramePairSide side, std::uint64_t expected_generation,
                                              Result<void> completion,
                                              std::chrono::steady_clock::time_point completed_at) {
    const auto completion_result = completion_tracker_.complete(
        side, expected_generation, std::move(completion), completed_at);
    synchronize_decode_completion_state();
    return completion_result;
}

bool FramePairRecord::decodes_completed() const noexcept {
    synchronize_decode_completion_state();
    return completion_tracker_.completed();
}

bool FramePairRecord::decode_accepted(FramePairSide side) const noexcept {
    return completion_tracker_.accepted(side);
}

Result<void> FramePairRecord::seal_cancelled_decodes(std::uint64_t expected_generation) {
    const auto validated = validate_generation(expected_generation);
    if (!validated) return validated;
    for (const auto side : {FramePairSide::reference, FramePairSide::distorted}) {
        if (completion_tracker_.completed(side)) continue;
        if (!completion_tracker_.accepted(side)) {
            const auto claimed = claim_decode(side, expected_generation);
            if (!claimed) return claimed;
        }
        const auto completed = complete_decode(
            side, expected_generation, Result<void>::failure({ErrorCode::cancelled, {}}));
        if (!completed) return completed;
    }
    return Result<void>::success();
}

bool FramePairRecord::ready() const noexcept {
    return state() == FramePairRecordState::ready;
}

Result<void> FramePairRecord::wait_for_decodes() const {
    const auto decode_result = completion_tracker_.wait();
    synchronize_decode_completion_state();
    if (!decode_result) return decode_result;
    if (state_.load(std::memory_order_acquire) != FramePairRecordState::ready) {
        return Result<void>::failure({ErrorCode::invalid_state,
                                      "frame decoding completed without entering the ready state"});
    }
    return Result<void>::success();
}

FrameDecodeTiming FramePairRecord::decode_timing(FramePairSide side) const noexcept {
    return completion_tracker_.timing(side);
}

Result<void> FramePairRecord::mark_submitted(std::uint64_t expected_generation) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;
    synchronize_decode_completion_state();

    FramePairRecordState expected_state = FramePairRecordState::ready;
    if (!state_.compare_exchange_strong(expected_state, FramePairRecordState::submitted,
                                        std::memory_order_acq_rel, std::memory_order_acquire)) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "only a ready frame-pair record can be submitted"});
    }
    return Result<void>::success();
}

Result<void> FramePairRecord::mark_retired(std::uint64_t expected_generation) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;

    FramePairRecordState expected_state = FramePairRecordState::submitted;
    if (!state_.compare_exchange_strong(expected_state, FramePairRecordState::retired,
                                        std::memory_order_acq_rel, std::memory_order_acquire)) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "only a submitted frame-pair record can be retired"});
    }
    return Result<void>::success();
}

Result<void> FramePairRecord::abandon(std::uint64_t expected_generation) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;
    synchronize_decode_completion_state();

    if (!completion_tracker_.quiescent()) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "frame-pair record cannot be abandoned while decoding is active"});
    }

    FramePairRecordState current_state = state_.load(std::memory_order_acquire);
    for (;;) {
        const bool state_can_be_abandoned = current_state == FramePairRecordState::scheduled ||
                                            current_state == FramePairRecordState::decoding ||
                                            current_state == FramePairRecordState::ready ||
                                            current_state == FramePairRecordState::failed;
        if (!state_can_be_abandoned) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "frame-pair record is not in an abandonable state"});
        }
        if (state_.compare_exchange_weak(current_state, FramePairRecordState::abandoned,
                                         std::memory_order_acq_rel, std::memory_order_acquire)) {
            return Result<void>::success();
        }
    }
}

Result<void> FramePairRecord::recycle(std::uint64_t expected_generation) {
    const auto generation_validation = validate_generation(expected_generation);
    if (!generation_validation) return generation_validation;

    FramePairRecordState current_state = state_.load(std::memory_order_acquire);
    for (;;) {
        const bool state_can_be_recycled = current_state == FramePairRecordState::retired ||
                                           current_state == FramePairRecordState::abandoned;
        if (!state_can_be_recycled) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "only retired or abandoned records can be recycled"});
        }
        if (state_.compare_exchange_weak(current_state, FramePairRecordState::recycling,
                                         std::memory_order_acq_rel, std::memory_order_acquire)) {
            break;
        }
    }

    frame_pair_storage_ = {};
    reference_luma_storage_ = {};
    storage_mode_ = FramePairStorageReservation::Mode::cpu_writable;
    reference_destination_ = {};
    distorted_destination_ = {};
    state_.store(FramePairRecordState::free, std::memory_order_release);
    return Result<void>::success();
}

void FramePairRecord::synchronize_decode_completion_state() const noexcept {
    if (!completion_tracker_.completed()) return;

    FramePairRecordState current_state = state_.load(std::memory_order_acquire);
    while (current_state == FramePairRecordState::scheduled ||
           current_state == FramePairRecordState::decoding) {
        const FramePairRecordState completed_state = completion_tracker_.succeeded()
                                                         ? FramePairRecordState::ready
                                                         : FramePairRecordState::failed;
        if (state_.compare_exchange_weak(current_state, completed_state, std::memory_order_acq_rel,
                                         std::memory_order_acquire)) {
            return;
        }
    }
}

Result<void> FramePairRecord::validate_generation(std::uint64_t expected_generation) const {
    if (expected_generation == 0U ||
        generation_.load(std::memory_order_acquire) != expected_generation) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "frame-pair generation does not match the record"});
    }
    return Result<void>::success();
}

} // namespace crossvmaf::evaluation
