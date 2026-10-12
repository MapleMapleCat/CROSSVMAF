#pragma once
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "evaluation/frame_pair_record.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/model.hpp"
#include "crossvmaf/result.hpp"
#include "gpu/submission.hpp"
#include "scoring/predictor.hpp"
#include "video/decoder.hpp"
#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

namespace crossvmaf::evaluation {

struct MotionFeature {
    std::uint64_t index{};
    Q32Value value{};
};

struct SpatialFeature {
    std::uint64_t index{};
    Q32Value cambi{};
    Q32Value speed_chroma_uv{};
    Q32Value adm3{};
    SpeedChromaFrameDiagnostics speed_chroma{};
};

struct CompletionBatch {
    std::vector<SpatialFeature> spatial;
    std::vector<MotionFeature> motion;
};

} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {

// Joins immediate spatial features with delayed Motion3 results while
// preserving strictly ordered public output in bounded storage.
// One caller owns registration, feature assignment, and output collection.
// Dispatched feature values stay immutable until their record is retired;
// only the private prediction outcome is written by the worker.
class FrameRecords final {
  public:
    using PredictionFunction = Result<double> (*)(const scoring::Predictor &, const FeatureVector &);
    FrameRecords(ModelVariant model, std::size_t maximum_pending_frames,
                 PredictionFunction prediction = nullptr);
    ~FrameRecords();

    FrameRecords(const FrameRecords &) = delete;
    FrameRecords &operator=(const FrameRecords &) = delete;
    FrameRecords(FrameRecords &&) = delete;
    FrameRecords &operator=(FrameRecords &&) = delete;

    [[nodiscard]] bool has_registration_capacity() const noexcept;
    [[nodiscard]] Result<void> register_frame(std::uint64_t index, FrameTiming timing = {});
    [[nodiscard]] Result<void> apply_completions(const CompletionBatch &completions);
    [[nodiscard]] Result<void> set_spatial(std::uint64_t index, Q32Value cambi,
                                           Q32Value speed_chroma_uv, Q32Value adm3,
                                           SpeedChromaFrameDiagnostics speed_chroma);
    [[nodiscard]] Result<void> set_motion(std::uint64_t index, Q32Value motion3);

    // Production callers poll without waiting so Vulkan submission remains
    // independent of model prediction. Verification can wait for the complete
    // prefix it just produced, while finish waits for every queued prediction.
    [[nodiscard]] Result<std::vector<FrameResult>> drain();
    [[nodiscard]] Result<std::vector<FrameResult>> wait_for_ready();
    [[nodiscard]] Result<std::vector<FrameResult>> finish();
    void cancel() noexcept;

    struct FrameRecord {
        std::uint64_t index{};
        FrameTiming timing;
        FramePairRecord record;
        video::VideoTimestamp decoded_timestamp;
        video::Frame hardware_reference;
        video::Frame hardware_distorted;
        bool staging_owns_storage{};
        bool acquired_by_caller{};
        FrameStorageHandle spatial_input;
        FrameStorageHandle reference_luma_input;
        std::array<gpu::SubmissionTicket, 4U> submissions;
        std::optional<Q32Value> cambi_value;
        std::optional<Q32Value> speed_chroma_value;
        std::optional<Q32Value> adm3_value;
        std::optional<SpeedChromaFrameDiagnostics> speed_chroma_diagnostics;
        Q32Value motion3;
        bool motion_ready{};
        bool prediction_queued{};

        [[nodiscard]] bool spatial_complete() const noexcept {
            return cambi_value && speed_chroma_value && adm3_value && speed_chroma_diagnostics;
        }

        [[nodiscard]] bool complete() const noexcept {
            return spatial_complete() && motion_ready && !staging_owns_storage &&
                   !acquired_by_caller;
        }

      private:
        friend class FrameRecords;

        // Published by the prediction worker under prediction_mutex_. The
        // record cannot be reused until the caller consumes this outcome.
        std::optional<Result<double>> prediction_result;
    };

    [[nodiscard]] Result<FrameRecord *> find_frame(std::uint64_t index);
    [[nodiscard]] bool all_spatial_ready() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return ring_capacity_; }
    [[nodiscard]] FrameRecord &slot(std::size_t index) noexcept { return pending_ring_[index]; }
    [[nodiscard]] const FrameRecord &slot(std::size_t index) const noexcept {
        return pending_ring_[index];
    }
    [[nodiscard]] Result<void> abandon_latest(std::uint64_t index);

  private:
    [[nodiscard]] FrameRecord &pending_at_offset(std::size_t offset) noexcept;
    [[nodiscard]] const FrameRecord &pending_at_offset(std::size_t offset) const noexcept;
    [[nodiscard]] Result<void> dispatch_ready_prefix();
    [[nodiscard]] Result<void> wait_for_dispatched_predictions(std::uint64_t target_count);
    [[nodiscard]] Result<std::vector<FrameResult>> collect_completed_predictions();
    void prediction_worker_main() noexcept;
    [[nodiscard]] Error prediction_failure() const;

    scoring::Predictor predictor_;
    PredictionFunction prediction_;
    std::uint64_t next_registered_{};
    std::uint64_t first_pending_{};
    std::size_t ring_capacity_{};
    std::unique_ptr<FrameRecord[]> pending_ring_;
    std::size_t pending_head_{};
    std::size_t pending_count_{};

    std::mutex prediction_mutex_;
    std::condition_variable prediction_condition_;
    // Absolute ordered prefixes identify work in the fixed ring; prediction
    // never owns a separate queue of frame features or completed results.
    std::uint64_t dispatched_prediction_count_{};
    std::uint64_t completed_prediction_count_{};
    bool prediction_input_closed_{};
    bool prediction_cancelled_{};
    bool prediction_worker_failed_{};
    std::optional<Error> prediction_error_;
    std::exception_ptr prediction_exception_;
    std::uint64_t prediction_frame_index_{};
    bool prediction_worker_stopped_{};
    bool finished_{};
    std::thread prediction_worker_;
};

} // namespace crossvmaf::evaluation
#endif
