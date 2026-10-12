#include "video/frame.hpp"
#include "reference_evaluator.hpp"

#include "metrics/adm3/reference.hpp"
#include "metrics/cambi/reference.hpp"
#include "metrics/motion3/reference.hpp"
#include "metrics/speed_chroma/reference.hpp"
#include "scoring/model.hpp"
#include "scoring/predictor.hpp"
#include <deque>
#include <optional>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <future>
#include <iterator>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace crossvmaf::verification {
namespace {

namespace adm3 = metrics::adm3::reference;
namespace cambi = metrics::cambi::reference;
namespace motion3 = metrics::motion3::reference;
namespace speed_chroma = metrics::speed_chroma::reference;

class ReferenceResultQueue final {
public:
    ReferenceResultQueue(ModelVariant model, std::size_t capacity)
        : predictor_(make_predictor(model)), capacity_(capacity) {}

    [[nodiscard]] Result<void> register_frame(std::uint64_t index) {
        if (frames_.size() == capacity_) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "reference lookahead capacity exceeded"});
        }
        PendingFrame frame;
        frame.index = index;
        frames_.push_back(frame);
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> set_spatial(
        std::uint64_t index, Q32Value cambi, Q32Value speed_chroma_uv,
        Q32Value adm3, SpeedChromaFrameDiagnostics diagnostics) {
        auto* frame = find_frame(index);
        if (frame == nullptr) return missing_frame();
        frame->features.cambi = cambi;
        frame->features.speed_chroma_uv = speed_chroma_uv;
        frame->features.adm3 = adm3;
        frame->diagnostics = diagnostics;
        frame->spatial_ready = true;
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> set_motion(std::uint64_t index, Q32Value motion) {
        auto* frame = find_frame(index);
        if (frame == nullptr) return missing_frame();
        frame->features.motion3 = motion;
        frame->motion_ready = true;
        return Result<void>::success();
    }

    [[nodiscard]] Result<std::vector<FrameResult>> wait_for_ready() {
        std::vector<FrameResult> results;
        while (!frames_.empty() && frames_.front().spatial_ready &&
               frames_.front().motion_ready) {
            const auto& frame = frames_.front();
            auto score = predictor_.predict(frame.features);
            if (!score) return Result<std::vector<FrameResult>>::failure(score.error());
            FrameResult result;
            result.index = frame.index;
            result.features = frame.features;
            result.vmaf = score.value();
            result.speed_chroma = frame.diagnostics;
            results.push_back(result);
            frames_.pop_front();
        }
        return Result<std::vector<FrameResult>>::success(std::move(results));
    }

    void cancel() noexcept { frames_.clear(); }

    [[nodiscard]] Result<std::vector<FrameResult>> finish() {
        auto results = wait_for_ready();
        if (!results) return results;
        if (!frames_.empty()) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state, "reference Motion3 tail is incomplete"});
        }
        return results;
    }

private:
    struct PendingFrame {
        std::uint64_t index{};
        FeatureVector features;
        SpeedChromaFrameDiagnostics diagnostics;
        bool spatial_ready{};
        bool motion_ready{};
    };

    static scoring::Predictor make_predictor(ModelVariant model) {
        auto predictor = scoring::Predictor::create(model);
        if (!predictor) throw std::runtime_error(predictor.error().message);
        return std::move(predictor).value();
    }

    PendingFrame* find_frame(std::uint64_t index) {
        for (auto& frame : frames_) {
            if (frame.index == index) return &frame;
        }
        return nullptr;
    }

    static Result<void> missing_frame() {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "reference result has no registered frame"});
    }

    scoring::Predictor predictor_;
    std::size_t capacity_{};
    std::deque<PendingFrame> frames_;
};

}  // namespace

class ReferenceEvaluator::Impl final {
public:
    Impl(ReferenceEvaluatorConfig config,
         scoring::ModelSnapshot model_snapshot)
        : config_(std::move(config)),
          cambi_(config_.frame, model_snapshot.cambi),
          adm3_(config_.frame, model_snapshot.adm3),
          motion3_(config_.frame, model_snapshot.motion3.five_frame_window),
          result_buffer_(config_.model, 4U) {
        speed_chroma_workers_.reserve(config_.maximum_batch_size);
        for (std::size_t worker_index = 0U;
             worker_index < config_.maximum_batch_size; ++worker_index) {
            speed_chroma_workers_.push_back(
                std::make_unique<speed_chroma::Reference>(
                    config_.frame, model_snapshot.speed_chroma));
        }
    }

    [[nodiscard]] Result<std::vector<FrameResult>> push(
        const FrameView& reference, const FrameView& distorted) {
        const auto validation = validate_frame_pair(
            reference, distorted, next_frame_index_);
        if (!validation) {
            return Result<std::vector<FrameResult>>::failure(
                validation.error());
        }

        auto cambi_result = cambi_.compute(distorted);
        if (!cambi_result) return fail(cambi_result.error());
        auto speed_chroma_result =
            speed_chroma_workers_.front()->compute_detailed(
                reference, distorted);
        if (!speed_chroma_result) return fail(speed_chroma_result.error());
        auto adm3_result = adm3_.compute(reference, distorted);
        if (!adm3_result) return fail(adm3_result.error());
        auto motion_results = motion3_.push(reference);
        if (!motion_results) return fail(motion_results.error());

        return commit_frame(
            reference.index,
            cambi_result.value(),
            speed_chroma_result.value(),
            adm3_result.value(),
            motion_results.value());
    }

    [[nodiscard]] Result<std::vector<FrameResult>> push_batch(
        std::span<const FrameView> reference_frames,
        std::span<const FrameView> distorted_frames) {
        if (reference_frames.size() != distorted_frames.size()) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_argument,
                 "reference evaluator batch requires matching frame counts"});
        }
        if (reference_frames.empty()) {
            return Result<std::vector<FrameResult>>::success({});
        }
        if (reference_frames.size() == 1U) {
            return push(reference_frames.front(), distorted_frames.front());
        }
        if (reference_frames.size() > config_.maximum_batch_size) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_argument,
                 "reference evaluator batch exceeds configured capacity"});
        }
        for (std::size_t frame_offset = 0U;
             frame_offset < reference_frames.size(); ++frame_offset) {
            const auto validation = validate_frame_pair(
                reference_frames[frame_offset],
                distorted_frames[frame_offset],
                next_frame_index_ +
                    static_cast<std::uint64_t>(frame_offset));
            if (!validation) {
                return Result<std::vector<FrameResult>>::failure(
                    validation.error());
            }
        }

        try {
            auto cambi_future = std::async(
                std::launch::async,
                [this, distorted_frames] {
                    return compute_cambi_batch(distorted_frames);
                });
            auto speed_chroma_future = std::async(
                std::launch::async,
                [this, reference_frames, distorted_frames] {
                    return compute_speed_chroma_batch(
                        reference_frames, distorted_frames);
                });
            auto adm3_future = std::async(
                std::launch::async,
                [this, reference_frames, distorted_frames] {
                    return compute_adm3_batch(
                        reference_frames, distorted_frames);
                });

            auto motion_results = compute_motion_batch(reference_frames);
            auto cambi_results = cambi_future.get();
            auto speed_chroma_results = speed_chroma_future.get();
            auto adm3_results = adm3_future.get();
            if (!motion_results) return fail(motion_results.error());
            if (!cambi_results) return fail(cambi_results.error());
            if (!speed_chroma_results) {
                return fail(speed_chroma_results.error());
            }
            if (!adm3_results) return fail(adm3_results.error());

            std::vector<FrameResult> output;
            output.reserve(reference_frames.size());
            for (std::size_t frame_offset = 0U;
                 frame_offset < reference_frames.size(); ++frame_offset) {
                auto committed = commit_frame(
                    reference_frames[frame_offset].index,
                    cambi_results.value()[frame_offset],
                    speed_chroma_results.value()[frame_offset],
                    adm3_results.value()[frame_offset],
                    motion_results.value()[frame_offset]);
                if (!committed) return committed;
                output.insert(
                    output.end(),
                    std::make_move_iterator(committed.value().begin()),
                    std::make_move_iterator(committed.value().end()));
            }
            return Result<std::vector<FrameResult>>::success(
                std::move(output));
        } catch (const std::bad_alloc&) {
            return fail(
                {ErrorCode::backend_error,
                 "reference evaluator batch allocation failed"});
        } catch (const std::exception& exception) {
            return fail(
                {ErrorCode::backend_error,
                 "reference evaluator batch worker failed: " +
                     std::string(exception.what())});
        } catch (...) {
            return fail(
                {ErrorCode::backend_error,
                 "reference evaluator batch worker failed unexpectedly"});
        }
    }

    [[nodiscard]] Result<std::vector<FrameResult>> flush() {
        if (flushed_) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state,
                 "reference evaluator flush may only be called once"});
        }
        flushed_ = true;
        if (failed_) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state,
                 "reference evaluator cannot flush after failure"});
        }

        auto motion_results = motion3_.flush();
        if (!motion_results) return fail(motion_results.error());
        for (const motion3::MotionScore& motion_result :
             motion_results.value()) {
            const auto motion_assigned = result_buffer_.set_motion(
                motion_result.index, motion_result.value);
            if (!motion_assigned) return fail(motion_assigned.error());
        }

        auto output = result_buffer_.finish();
        if (!output) return fail(output.error());
        return output;
    }

private:
    [[nodiscard]] Result<void> validate_frame_pair(
        const FrameView& reference,
        const FrameView& distorted,
        std::uint64_t expected_frame_index) const {
        if (flushed_) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "reference evaluator cannot accept frames after flush"});
        }
        if (failed_) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "reference evaluator cannot continue after failure"});
        }

        const auto reference_validation =
            video::validate_plane_views(config_.frame, reference);
        if (!reference_validation) return reference_validation;
        const auto distorted_validation =
            video::validate_plane_views(config_.frame, distorted);
        if (!distorted_validation) return distorted_validation;
        if (reference.index != distorted.index ||
            reference.index != expected_frame_index) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "reference evaluator requires matching contiguous frame indices"});
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<std::vector<Q32Value>> compute_cambi_batch(
        std::span<const FrameView> distorted_frames) {
        std::vector<Q32Value> results;
        results.reserve(distorted_frames.size());
        for (const FrameView& distorted_frame : distorted_frames) {
            auto result = cambi_.compute(distorted_frame);
            if (!result) {
                return Result<std::vector<Q32Value>>::failure(
                    result.error());
            }
            results.push_back(result.value());
        }
        return Result<std::vector<Q32Value>>::success(std::move(results));
    }

    [[nodiscard]] Result<std::vector<speed_chroma::DetailedResult>>
    compute_speed_chroma_batch(
        std::span<const FrameView> reference_frames,
        std::span<const FrameView> distorted_frames) {
        std::vector<speed_chroma::DetailedResult> results(
            reference_frames.size());
        const std::size_t worker_count = std::min(
            reference_frames.size(), speed_chroma_workers_.size());
        std::vector<std::future<Result<void>>> worker_futures;
        worker_futures.reserve(worker_count);
        for (std::size_t worker_index = 0U;
             worker_index < worker_count; ++worker_index) {
            worker_futures.push_back(std::async(
                std::launch::async,
                [this, reference_frames, distorted_frames, &results,
                 worker_index, worker_count] {
                    for (std::size_t frame_offset = worker_index;
                         frame_offset < reference_frames.size();
                         frame_offset += worker_count) {
                        auto result = speed_chroma_workers_[worker_index]
                                          ->compute_detailed(
                                              reference_frames[frame_offset],
                                              distorted_frames[frame_offset]);
                        if (!result) {
                            return Result<void>::failure(result.error());
                        }
                        results[frame_offset] = result.value();
                    }
                    return Result<void>::success();
                }));
        }
        for (std::future<Result<void>>& worker_future : worker_futures) {
            auto worker_result = worker_future.get();
            if (!worker_result) {
                return Result<std::vector<speed_chroma::DetailedResult>>::failure(
                    worker_result.error());
            }
        }
        return Result<std::vector<speed_chroma::DetailedResult>>::success(
            std::move(results));
    }

    [[nodiscard]] Result<std::vector<Q32Value>> compute_adm3_batch(
        std::span<const FrameView> reference_frames,
        std::span<const FrameView> distorted_frames) {
        std::vector<Q32Value> results;
        results.reserve(reference_frames.size());
        for (std::size_t frame_offset = 0U;
             frame_offset < reference_frames.size(); ++frame_offset) {
            auto result = adm3_.compute(
                reference_frames[frame_offset],
                distorted_frames[frame_offset]);
            if (!result) {
                return Result<std::vector<Q32Value>>::failure(
                    result.error());
            }
            results.push_back(result.value());
        }
        return Result<std::vector<Q32Value>>::success(std::move(results));
    }

    [[nodiscard]] Result<std::vector<std::vector<motion3::MotionScore>>>
    compute_motion_batch(std::span<const FrameView> reference_frames) {
        std::vector<std::vector<motion3::MotionScore>> results;
        results.reserve(reference_frames.size());
        for (const FrameView& reference_frame : reference_frames) {
            auto result = motion3_.push(reference_frame);
            if (!result) {
                return Result<
                    std::vector<std::vector<motion3::MotionScore>>>::failure(
                        result.error());
            }
            results.push_back(std::move(result).value());
        }
        return Result<
            std::vector<std::vector<motion3::MotionScore>>>::success(
                std::move(results));
    }

    [[nodiscard]] Result<std::vector<FrameResult>> commit_frame(
        std::uint64_t frame_index,
        Q32Value cambi_result,
        const speed_chroma::DetailedResult& speed_chroma_result,
        Q32Value adm3_result,
        const std::vector<motion3::MotionScore>& motion_results) {
        const auto registered = result_buffer_.register_frame(frame_index);
        if (!registered) return fail(registered.error());

        const SpeedChromaFrameDiagnostics speed_chroma_diagnostics{
            speed_chroma_result.u,
            speed_chroma_result.v,
            speed_chroma_result.uv,
            speed_chroma_result.u_singular,
            speed_chroma_result.v_singular,
        };
        const auto spatial_assigned = result_buffer_.set_spatial(
            frame_index, cambi_result, speed_chroma_result.uv,
            adm3_result, speed_chroma_diagnostics);
        if (!spatial_assigned) return fail(spatial_assigned.error());

        for (const motion3::MotionScore& motion_result : motion_results) {
            const auto motion_assigned = result_buffer_.set_motion(
                motion_result.index, motion_result.value);
            if (!motion_assigned) return fail(motion_assigned.error());
        }

        ++next_frame_index_;
        auto output = result_buffer_.wait_for_ready();
        if (!output) return fail(output.error());
        return output;
    }
    [[nodiscard]] Result<std::vector<FrameResult>> fail(Error error) {
        failed_ = true;
        result_buffer_.cancel();
        return Result<std::vector<FrameResult>>::failure(std::move(error));
    }

    ReferenceEvaluatorConfig config_;
    cambi::Reference cambi_;
    std::vector<std::unique_ptr<speed_chroma::Reference>>
        speed_chroma_workers_;
    adm3::Reference adm3_;
    motion3::Reference motion3_;
    ReferenceResultQueue result_buffer_;
    std::uint64_t next_frame_index_{};
    bool flushed_{};
    bool failed_{};
};

Result<std::unique_ptr<ReferenceEvaluator>> ReferenceEvaluator::create(
    const ReferenceEvaluatorConfig& config) {
    const auto metric_limit = metrics::cambi::validate_input(config.frame);
    if (!metric_limit) return Result<std::unique_ptr<ReferenceEvaluator>>::failure(metric_limit.error());
    const auto descriptor_validation = video::validate_frame_layout(config.frame);
    if (!descriptor_validation) {
        return Result<std::unique_ptr<ReferenceEvaluator>>::failure(
            descriptor_validation.error());
    }
    if (config.maximum_batch_size == 0U) {
        return Result<std::unique_ptr<ReferenceEvaluator>>::failure(
            {ErrorCode::invalid_argument,
             "reference evaluator batch size must be greater than zero"});
    }
    auto model_snapshot = scoring::make_model_snapshot(config.model);
    if (!model_snapshot) {
        return Result<std::unique_ptr<ReferenceEvaluator>>::failure(
            model_snapshot.error());
    }
    try {
        auto impl = std::make_unique<Impl>(config, model_snapshot.value());
        return Result<std::unique_ptr<ReferenceEvaluator>>::success(
            std::unique_ptr<ReferenceEvaluator>(
                new ReferenceEvaluator(std::move(impl))));
    } catch (const std::bad_alloc&) {
        return Result<std::unique_ptr<ReferenceEvaluator>>::failure(
            {ErrorCode::backend_error,
             "reference evaluator workspace allocation failed"});
    } catch (const std::length_error&) {
        return Result<std::unique_ptr<ReferenceEvaluator>>::failure(
            {ErrorCode::unsupported_format,
             "reference evaluator workspace geometry exceeds container limits"});
    }
}

ReferenceEvaluator::ReferenceEvaluator(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ReferenceEvaluator::~ReferenceEvaluator() = default;
ReferenceEvaluator::ReferenceEvaluator(ReferenceEvaluator&&) noexcept = default;
ReferenceEvaluator& ReferenceEvaluator::operator=(
    ReferenceEvaluator&&) noexcept = default;

Result<std::vector<FrameResult>> ReferenceEvaluator::push(
    const FrameView& reference, const FrameView& distorted) {
    return impl_->push(reference, distorted);
}

Result<std::vector<FrameResult>> ReferenceEvaluator::push_batch(
    std::span<const FrameView> reference_frames,
    std::span<const FrameView> distorted_frames) {
    return impl_->push_batch(reference_frames, distorted_frames);
}

Result<std::vector<FrameResult>> ReferenceEvaluator::flush() {
    return impl_->flush();
}

}  // namespace crossvmaf::verification
