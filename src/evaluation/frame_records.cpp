#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "evaluation/frame_records.hpp"
#include <memory>
#include <utility>
#include <vector>

namespace crossvmaf::evaluation {
namespace {

[[nodiscard]] scoring::Predictor
frame_records_create_predictor_or_throw(ModelVariant model_variant) {
    auto predictor = scoring::Predictor::create(model_variant);
    if (!predictor) {
        throw std::invalid_argument(predictor.error().message);
    }
    return std::move(predictor).value();
}

} // namespace

FrameRecords::FrameRecords(ModelVariant model, std::size_t maximum_pending_frames,
                           PredictionFunction prediction)
    : predictor_(frame_records_create_predictor_or_throw(model)),
      prediction_(prediction ? prediction : +[](const scoring::Predictor &predictor,
                                                const FeatureVector &features) {
          return predictor.predict(features);
      }),
      ring_capacity_(maximum_pending_frames),
      pending_ring_(std::make_unique<FrameRecord[]>(maximum_pending_frames)) {
    if (maximum_pending_frames == 0U) {
        throw std::invalid_argument("ordered result capacity must be greater than zero");
    }
    prediction_worker_ = std::thread(&FrameRecords::prediction_worker_main, this);
}

FrameRecords::~FrameRecords() {
    cancel();
}

bool FrameRecords::has_registration_capacity() const noexcept {
    return !finished_ && pending_count_ < ring_capacity_;
}

FrameRecords::FrameRecord &FrameRecords::pending_at_offset(std::size_t offset) noexcept {
    return pending_ring_[(pending_head_ + offset) % ring_capacity_];
}

const FrameRecords::FrameRecord &
FrameRecords::pending_at_offset(std::size_t offset) const noexcept {
    return pending_ring_[(pending_head_ + offset) % ring_capacity_];
}

Result<void> FrameRecords::register_frame(std::uint64_t index, FrameTiming timing) {
    if (finished_) {
        return Result<void>::failure({ErrorCode::invalid_state,
                                      "ordered result buffer cannot register frames after finish"});
    }
    if (index != next_registered_) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "ordered result frame indices must be contiguous and start at zero"});
    }
    if (!has_registration_capacity()) {
        return Result<void>::failure(
            {ErrorCode::backend_error,
             "model prediction backlog exhausted ordered result capacity"});
    }
    if (pending_count_ == 0U) first_pending_ = index;
    FrameRecord &pending = pending_at_offset(pending_count_);
    std::destroy_at(&pending);
    std::construct_at(&pending);
    pending.index = index;
    pending.timing = timing;
    ++pending_count_;
    ++next_registered_;
    return Result<void>::success();
}

Result<void> FrameRecords::apply_completions(const CompletionBatch &completions) {
    for (const SpatialFeature &spatial : completions.spatial) {
        const auto assigned = set_spatial(spatial.index, spatial.cambi, spatial.speed_chroma_uv,
                                          spatial.adm3, spatial.speed_chroma);
        if (!assigned) return assigned;
    }
    for (const MotionFeature &motion : completions.motion) {
        const auto assigned = set_motion(motion.index, motion.value);
        if (!assigned) return assigned;
    }
    return Result<void>::success();
}

Result<FrameRecords::FrameRecord *> FrameRecords::find_frame(std::uint64_t index) {
    if (pending_count_ == 0U || index < first_pending_ || index >= next_registered_) {
        return Result<FrameRecord *>::failure(
            {ErrorCode::invalid_argument, "feature result does not refer to a pending frame"});
    }
    const std::uint64_t offset = index - first_pending_;
    if (offset >= pending_count_) {
        return Result<FrameRecord *>::failure(
            {ErrorCode::invalid_argument,
             "feature result refers to a frame that was already emitted"});
    }
    return Result<FrameRecord *>::success(&pending_at_offset(static_cast<std::size_t>(offset)));
}

Result<void> FrameRecords::set_spatial(std::uint64_t index, Q32Value cambi,
                                       Q32Value speed_chroma_uv, Q32Value adm3,
                                       SpeedChromaFrameDiagnostics speed_chroma) {
    auto pending = find_frame(index);
    if (!pending) return Result<void>::failure(pending.error());
    FrameRecord *const frame = pending.value();
    if (frame->cambi_value || frame->speed_chroma_value || frame->adm3_value ||
        frame->speed_chroma_diagnostics) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "spatial features may only be assigned once per frame"});
    }
    if (speed_chroma_uv.raw_q32_32 != speed_chroma.uv.raw_q32_32) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "Speed Chroma UV feature and diagnostic raw values must match"});
    }
    frame->cambi_value = cambi;
    frame->speed_chroma_value = speed_chroma_uv;
    frame->adm3_value = adm3;
    frame->speed_chroma_diagnostics = speed_chroma;
    return Result<void>::success();
}

Result<void> FrameRecords::set_motion(std::uint64_t index, Q32Value motion3) {
    auto pending = find_frame(index);
    if (!pending) return Result<void>::failure(pending.error());
    FrameRecord *const frame = pending.value();
    if (frame->motion_ready) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Motion3 may only be assigned once per frame"});
    }
    frame->motion3 = motion3;
    frame->motion_ready = true;
    return Result<void>::success();
}

Result<void> FrameRecords::dispatch_ready_prefix() {
    bool queued_prediction = false;
    {
        std::lock_guard lock(prediction_mutex_);
        if (prediction_worker_failed_) return Result<void>::failure(prediction_failure());
        if (prediction_cancelled_) {
            return Result<void>::failure(
                {ErrorCode::cancelled, "model prediction worker has been cancelled"});
        }
        if (prediction_input_closed_) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "model prediction worker cannot accept frames after finish"});
        }
        if (prediction_worker_stopped_) {
            return Result<void>::failure(
                {ErrorCode::backend_error, "model prediction worker stopped unexpectedly"});
        }

        for (std::size_t pending_offset = 0U; pending_offset < pending_count_; ++pending_offset) {
            FrameRecord &pending = pending_at_offset(pending_offset);
            if (!pending.complete()) break;
            if (pending.prediction_queued) continue;

            pending.prediction_queued = true;
            ++dispatched_prediction_count_;
            queued_prediction = true;
        }
    }
    if (queued_prediction) prediction_condition_.notify_one();
    return Result<void>::success();
}

Result<void> FrameRecords::wait_for_dispatched_predictions(std::uint64_t target_count) {
    std::unique_lock lock(prediction_mutex_);
    prediction_condition_.wait(lock, [this, target_count] {
        return completed_prediction_count_ >= target_count || prediction_cancelled_ ||
               prediction_worker_failed_ || prediction_worker_stopped_;
    });
    if (prediction_worker_failed_) return Result<void>::failure(prediction_failure());
    if (prediction_cancelled_) {
        return Result<void>::failure(
            {ErrorCode::cancelled, "model prediction worker was cancelled while waiting"});
    }
    if (completed_prediction_count_ < target_count) {
        return Result<void>::failure(
            {ErrorCode::backend_error,
             "model prediction worker stopped before completing its backlog"});
    }
    return Result<void>::success();
}

Result<std::vector<FrameResult>> FrameRecords::collect_completed_predictions() {
    std::uint64_t completed_prefix = 0U;
    {
        std::lock_guard lock(prediction_mutex_);
        if (prediction_worker_failed_) {
            return Result<std::vector<FrameResult>>::failure(
                prediction_failure());
        }
        completed_prefix = completed_prediction_count_;
    }

    // Completed records are immutable until this caller retires them. The
    // worker only touches the following prefix, so output needs no queue swap
    // or lock held across allocation and public result construction.
    std::vector<FrameResult> output;
    output.reserve(static_cast<std::size_t>(completed_prefix - first_pending_));
    while (first_pending_ < completed_prefix) {
        if (pending_count_ == 0U || !pending_at_offset(0U).prediction_queued ||
            pending_at_offset(0U).index != first_pending_ ||
            !pending_at_offset(0U).prediction_result.has_value()) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state,
                 "model prediction results violated ordered frame delivery"});
        }

        FrameRecord &pending = pending_at_offset(0U);
        const Result<double> &prediction = *pending.prediction_result;
        if (!prediction) {
            return Result<std::vector<FrameResult>>::failure(prediction.error());
        }
        output.push_back({
            pending.index,
            {*pending.cambi_value, *pending.speed_chroma_value, *pending.adm3_value,
             pending.motion3},
            prediction.value(),
            *pending.speed_chroma_diagnostics,
            pending.timing,
        });
        std::destroy_at(&pending_ring_[pending_head_]);
        std::construct_at(&pending_ring_[pending_head_]);
        pending_head_ = (pending_head_ + 1U) % ring_capacity_;
        --pending_count_;
        ++first_pending_;
    }
    return Result<std::vector<FrameResult>>::success(std::move(output));
}

Result<std::vector<FrameResult>> FrameRecords::drain() {
    const auto dispatched = dispatch_ready_prefix();
    if (!dispatched) {
        return Result<std::vector<FrameResult>>::failure(dispatched.error());
    }
    return collect_completed_predictions();
}

Result<std::vector<FrameResult>> FrameRecords::wait_for_ready() {
    const auto dispatched = dispatch_ready_prefix();
    if (!dispatched) {
        return Result<std::vector<FrameResult>>::failure(dispatched.error());
    }
    const auto completed = wait_for_dispatched_predictions(dispatched_prediction_count_);
    if (!completed) {
        return Result<std::vector<FrameResult>>::failure(completed.error());
    }
    return collect_completed_predictions();
}

Result<std::vector<FrameResult>> FrameRecords::finish() {
    {
        std::lock_guard lock(prediction_mutex_);
        if (prediction_worker_failed_)
            return Result<std::vector<FrameResult>>::failure(prediction_failure());
    }
    if (finished_) {
        return Result<std::vector<FrameResult>>::failure(
            {ErrorCode::invalid_state, "ordered result buffer may only be finished once"});
    }
    for (std::size_t pending_offset = 0U; pending_offset < pending_count_; ++pending_offset) {
        if (!pending_at_offset(pending_offset).complete()) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state,
                 "flush completed with one or more incomplete feature frames"});
        }
    }

    const auto dispatched = dispatch_ready_prefix();
    if (!dispatched) {
        return Result<std::vector<FrameResult>>::failure(dispatched.error());
    }
    {
        std::lock_guard lock(prediction_mutex_);
        prediction_input_closed_ = true;
    }
    prediction_condition_.notify_all();

    const auto completed = wait_for_dispatched_predictions(dispatched_prediction_count_);
    if (prediction_worker_.joinable()) prediction_worker_.join();
    if (!completed) {
        return Result<std::vector<FrameResult>>::failure(completed.error());
    }

    auto output = collect_completed_predictions();
    if (!output) return output;
    if (pending_count_ != 0U) {
        return Result<std::vector<FrameResult>>::failure(
            {ErrorCode::invalid_state,
             "model prediction worker left one or more frames undelivered"});
    }
    finished_ = true;
    return output;
}

void FrameRecords::prediction_worker_main() noexcept {
    try {
        for (;;) {
            FrameRecord *pending_frame = nullptr;
            {
                std::unique_lock lock(prediction_mutex_);
                prediction_condition_.wait(lock, [this] {
                    return prediction_cancelled_ ||
                           completed_prediction_count_ < dispatched_prediction_count_ ||
                           prediction_input_closed_;
                });
                if (prediction_cancelled_) break;
                if (completed_prediction_count_ == dispatched_prediction_count_) {
                    break;
                }
                const std::size_t slot_index =
                    static_cast<std::size_t>(completed_prediction_count_ % ring_capacity_);
                pending_frame = &pending_ring_[slot_index];
                prediction_frame_index_ = pending_frame->index;
            }

            // Dispatch seals the features, and collection cannot recycle this
            // slot before its prediction is published. GPU coordination can
            // continue on other records while the predictor runs unlocked.
            auto prediction = prediction_(predictor_, {
                *pending_frame->cambi_value,
                *pending_frame->speed_chroma_value,
                *pending_frame->adm3_value,
                pending_frame->motion3,
            });

            {
                std::lock_guard lock(prediction_mutex_);
                if (!prediction) {
                    prediction_error_.emplace(std::move(prediction).error());
                    prediction_worker_failed_ = true;
                    break;
                }
                if (prediction_cancelled_) break;
                pending_frame->prediction_result.emplace(std::move(prediction));
                ++completed_prediction_count_;
            }
            prediction_condition_.notify_all();
        }
    } catch (...) {
        std::lock_guard lock(prediction_mutex_);
        prediction_exception_ = std::current_exception();
        prediction_worker_failed_ = true;
    }

    {
        std::lock_guard lock(prediction_mutex_);
        prediction_worker_stopped_ = true;
    }
    prediction_condition_.notify_all();
}

Error FrameRecords::prediction_failure() const {
    if (prediction_error_) return prediction_error_->with_context(
        "model prediction", ErrorInput::none, prediction_frame_index_);
    try {
        if (prediction_exception_) std::rethrow_exception(prediction_exception_);
    } catch (const std::exception &exception) {
        return Error{ErrorCode::backend_error, exception.what()}.with_context(
            "model prediction", ErrorInput::none, prediction_frame_index_);
    } catch (...) {
        return Error{ErrorCode::backend_error, "unknown exception"}.with_context(
            "model prediction", ErrorInput::none, prediction_frame_index_);
    }
    return Error{ErrorCode::backend_error, "worker stopped unexpectedly"}.with_context(
        "model prediction", ErrorInput::none, prediction_frame_index_);
}

void FrameRecords::cancel() noexcept {
    {
        std::lock_guard lock(prediction_mutex_);
        prediction_input_closed_ = true;
        prediction_cancelled_ = true;
    }
    prediction_condition_.notify_all();
    if (prediction_worker_.joinable()) prediction_worker_.join();
}

bool FrameRecords::all_spatial_ready() const noexcept {
    for (std::size_t offset = 0U; offset < pending_count_; ++offset) {
        if (!pending_at_offset(offset).spatial_complete()) return false;
    }
    return true;
}

} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {
Result<void> FrameRecords::abandon_latest(std::uint64_t index) {
    if (pending_count_ == 0U || index != next_registered_ - 1U ||
        pending_at_offset(pending_count_ - 1U).prediction_queued) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "only the latest unsubmitted frame reservation can be abandoned"});
    }
    FrameRecord &pending = pending_at_offset(pending_count_ - 1U);
    std::destroy_at(&pending);
    std::construct_at(&pending);
    --pending_count_;
    --next_registered_;
    return Result<void>::success();
}
} // namespace crossvmaf::evaluation
