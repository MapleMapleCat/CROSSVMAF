#include "evaluation/evaluation_runner.hpp"
#include "evaluation/scene_policy.hpp"

#include "i18n/translator.hpp"

#include <crossvmaf/pooling.hpp>
#include <crossvmaf/stream_options.hpp>

#include <chrono>
#include <optional>
#include <utility>
#include <vector>

namespace crossvmaf::studio {

namespace {

using SteadyClock = std::chrono::steady_clock;
constexpr std::chrono::milliseconds progress_publish_interval{150};

[[nodiscard]] double elapsedMilliseconds(
    SteadyClock::time_point start,
    SteadyClock::time_point end = SteadyClock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

} // namespace

EvaluationRunner::~EvaluationRunner() {
    requestCancellation();
    wait();
}

void EvaluationRunner::start(EvaluationRequest request, EvaluationCallbacks callbacks) {
    wait();
    cancellation_requested_.store(false, std::memory_order_release);
    thread_ = std::thread(
        [this, request = std::move(request), callbacks = std::move(callbacks)]() mutable {
            try {
                run(std::move(request), callbacks);
            } catch (const std::exception& exception) {
                try { if (callbacks.failed) callbacks.failed(exception.what(), false); } catch (...) {}
            } catch (...) {
                try { if (callbacks.failed) callbacks.failed("Unknown evaluation exception", false); } catch (...) {}
            }
        });
}

void EvaluationRunner::requestCancellation() noexcept {
    cancellation_requested_.store(true, std::memory_order_release);
}

void EvaluationRunner::wait() {
    if (thread_.joinable()) thread_.join();
}

void EvaluationRunner::run(EvaluationRequest request, const EvaluationCallbacks& callbacks) {
    const auto total_start = SteadyClock::now();
    const auto initialization_start = SteadyClock::now();
    StreamOptions options;
    options.reference_path = request.reference_path;
    options.distorted_path = request.distorted_path;
    options.device = request.device;
    options.decode_mode = request.decode_mode;
    options.decode_backend = request.decode_backend;
    options.decode_device = request.decode_device;
    options.decode_transfer = request.decode_transfer;

    options.decoder_threads_per_source = request.decoder_threads_per_source;
    options.decoded_frame_queue_capacity = request.decoded_frame_queue_capacity;
    auto opened_stream = EvaluationStream::open(std::move(options));
    const auto initialization_end = SteadyClock::now();
    if (!opened_stream) {
        const std::string message = opened_stream.error().message;
        const bool cancelled = opened_stream.error().code == ErrorCode::cancelled;
        if (callbacks.failed) callbacks.failed(message, cancelled);
        return;
    }

    std::unique_ptr<EvaluationStream> stream = std::move(opened_stream).value();
    const FrameDescriptor descriptor = stream->descriptor();
    VideoMetadata reference_metadata = stream->metadata().reference;
    reference_metadata.descriptor = descriptor;
    const auto resolved = resolve_evaluation_config(&reference_metadata);
    request.model = stream->model();
    request.scene = model_scene(request.model);
    request.model_decision_reason = resolved.decision_reason;
    request.scene_policy_version = resolved.policy_version;
    const DeviceInfo device = stream->device_info();
    const EvaluationStreamStatistics opening_statistics = stream->statistics();
    if (callbacks.opened) callbacks.opened(descriptor, device, opening_statistics);

    const auto processing_start = SteadyClock::now();
    auto last_progress_publish = processing_start - progress_publish_interval;
    std::vector<FrameResult> frames;
    std::vector<double> scores;
    const auto publish_progress = [&](bool force) {
        const auto now = SteadyClock::now();
        if (!force && now - last_progress_publish < progress_publish_interval) return;
        last_progress_publish = now;
        LiveProgress progress;
        progress.scores = scores;
        progress.frame_count_hint = stream->frame_count_hint();
        progress.decoded_frame_count = stream->decoded_frame_count();
        progress.elapsed_seconds = elapsedMilliseconds(processing_start, now) / 1000.0;
        progress.frames_per_second = progress.elapsed_seconds > 0.0
            ? static_cast<double>(scores.size()) / progress.elapsed_seconds
            : 0.0;
        if (callbacks.progress) callbacks.progress(std::move(progress));
    };

    std::optional<std::string> failure_message;
    for (;;) {
        if (cancellation_requested_.load(std::memory_order_acquire)) break;
        auto available = stream->next();
        if (!available) {
            failure_message = available.error().message;
            break;
        }
        if (available.value().empty()) break;
        for (FrameResult& frame : available.value()) {
            scores.push_back(frame.vmaf);
            frames.push_back(std::move(frame));
        }
        publish_progress(false);
    }

    const bool cancelled_before_finish =
        cancellation_requested_.load(std::memory_order_acquire);
    if (!failure_message && !cancelled_before_finish) {
        auto tail = stream->finish();
        if (!tail) {
            failure_message = tail.error().message;
        } else {
            for (FrameResult& frame : tail.value()) {
                scores.push_back(frame.vmaf);
                frames.push_back(std::move(frame));
            }
            publish_progress(true);
        }
    }

    if (cancelled_before_finish || failure_message) {
        stream.reset();
        if (callbacks.failed) {
            callbacks.failed(failure_message.value_or(i18n::tr("Stopped by the user")),
                             !failure_message && cancelled_before_finish);
        }
        return;
    }

    if (frames.empty()) {
        if (callbacks.failed) callbacks.failed(i18n::tr("The inputs produced no frames to evaluate"), false);
        return;
    }

    auto pooled_score = pool_scores(scores, PoolMethod::mean);
    if (!pooled_score) {
        const std::string message = pooled_score.error().message;
        if (callbacks.failed) callbacks.failed(message, false);
        return;
    }

    const auto processing_end = SteadyClock::now();
    auto report = std::make_shared<EvaluationReport>();
    report->request = std::move(request);
    report->model = report->request.model;
    report->media = stream->metadata();
    report->device = device;
    report->descriptor = descriptor;
    report->stream_statistics = stream->statistics();
    report->frames = std::move(frames);
    report->pooled_score = pooled_score.value();
    report->initialization_milliseconds = elapsedMilliseconds(
        initialization_start, initialization_end);
    report->processing_milliseconds = elapsedMilliseconds(
        processing_start, processing_end);
    report->end_to_end_milliseconds = elapsedMilliseconds(
        total_start, processing_end);
    report->frames_per_second = report->processing_milliseconds > 0.0
        ? static_cast<double>(report->frames.size()) * 1000.0 /
            report->processing_milliseconds
        : 0.0;
    report->completed_at = std::chrono::system_clock::now();

    if (callbacks.succeeded) callbacks.succeeded(std::move(report));
}

} // namespace crossvmaf::studio
