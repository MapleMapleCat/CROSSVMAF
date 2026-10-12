#include "evaluation/task_queue.hpp"
#include "evaluation/scene_policy.hpp"

#include "i18n/translator.hpp"

#include <crossvmaf/pooling.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <exception>
#include <utility>

namespace crossvmaf::studio {
namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr std::chrono::milliseconds progress_publish_interval{150};
constexpr std::size_t history_limit = 32U;

[[nodiscard]] InputRole task_input_role(ErrorInput input) noexcept {
    switch (input) {
    case ErrorInput::reference: return InputRole::reference;
    case ErrorInput::distorted: return InputRole::distorted;
    case ErrorInput::both: return InputRole::both;
    case ErrorInput::none: return InputRole::none;
    }
    return InputRole::none;
}

[[nodiscard]] double elapsed_seconds_since(SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

[[nodiscard]] std::string make_task_id(std::uint64_t serial) {
    const auto now = std::chrono::system_clock::now();
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_time{};
#ifdef _WIN32
    gmtime_s(&utc_time, &raw_time);
#else
    gmtime_r(&raw_time, &utc_time);
#endif
    char timestamp[24]{};
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &utc_time);
    return std::string("t-") + timestamp + "-" + std::to_string(serial);
}

}  // namespace

struct TaskQueue::TaskControl final {
    TaskSnapshot snapshot;
    std::stop_source stop_source;
    TaskError exception_error{ErrorCode::backend_error, "Unexpected task exception"};
};

TaskQueue::TaskQueue(Callbacks callbacks)
    : callbacks_(std::move(callbacks)) {
    history_.reserve(history_limit + 1U);
    worker_ = std::thread([this] { worker_main(); });
}

TaskQueue::~TaskQueue() {
    {
        std::lock_guard lock(mutex_);
        shutdown_ = true;
        if (active_) {
            active_->stop_source.request_stop();
            if (active_stream_ != nullptr) active_stream_->request_cancel();
        }
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
}

std::string TaskQueue::enqueue(EvaluationRequest request) {
    auto control = std::make_shared<TaskControl>();
    control->snapshot.task_id = make_task_id(next_task_serial_++);
    control->snapshot.request = std::move(request);
    control->snapshot.state = TaskState::queued;
    control->snapshot.enqueued_at = std::chrono::system_clock::now();
    const std::string task_id = control->snapshot.task_id;
    {
        std::lock_guard lock(mutex_);
        queued_.push_back(std::move(control));
    }
    condition_.notify_all();
    return task_id;
}

void TaskQueue::start_queue() {
    {
        std::lock_guard lock(mutex_);
        started_ = true;
        paused_ = false;
    }
    condition_.notify_all();
}

void TaskQueue::pause_queue() {
    std::lock_guard lock(mutex_);
    paused_ = true;
}

bool TaskQueue::cancel(const std::string& task_id) {
    std::shared_ptr<TaskControl> emit_copy;
    {
        std::lock_guard lock(mutex_);
        const auto found = std::find_if(
            queued_.begin(), queued_.end(),
            [&task_id](const std::shared_ptr<TaskControl>& control) {
                return control->snapshot.task_id == task_id;
            });
        if (found != queued_.end()) {
            TaskControl& control = **found;
            control.snapshot.state = TaskState::cancelled;
            control.snapshot.finished_at = std::chrono::system_clock::now();
            control.snapshot.error = TaskError{
                ErrorCode::cancelled, {}, TaskStage::opening, InputRole::none,
                std::nullopt, control.snapshot.task_id, true};
            auto terminal = std::move(*found);
            queued_.erase(found);
            history_.insert(history_.begin(), terminal);
            if (history_.size() > history_limit) history_.pop_back();
            emit_copy = terminal;
        } else if (active_ && active_->snapshot.task_id == task_id &&
                   active_->snapshot.state != TaskState::completed &&
                   active_->snapshot.state != TaskState::failed &&
                   active_->snapshot.state != TaskState::cancelled) {
            active_->stop_source.request_stop();
            if (active_stream_ != nullptr) active_stream_->request_cancel();
            if (active_->snapshot.state != TaskState::cancelling) {
                active_->snapshot.state = TaskState::cancelling;
                emit_copy = active_;
            }
        } else {
            return false;
        }
    }
    if (emit_copy) emit_state_changed(*emit_copy);
    return true;
}

bool TaskQueue::remove_queued(const std::string& task_id) {
    std::lock_guard lock(mutex_);
    const auto found = std::find_if(
        queued_.begin(), queued_.end(),
        [&task_id](const std::shared_ptr<TaskControl>& control) {
            return control->snapshot.task_id == task_id;
        });
    if (found == queued_.end()) return false;
    queued_.erase(found);
    return true;
}

bool TaskQueue::move_queued(
    const std::string& task_id,
    std::size_t position) {
    std::lock_guard lock(mutex_);
    const auto found = std::find_if(
        queued_.begin(), queued_.end(),
        [&task_id](const std::shared_ptr<TaskControl>& control) {
            return control->snapshot.task_id == task_id;
        });
    if (found == queued_.end()) return false;
    auto control = std::move(*found);
    queued_.erase(found);
    const std::size_t target = std::min(position, queued_.size());
    queued_.insert(queued_.begin() + static_cast<std::ptrdiff_t>(target),
                   std::move(control));
    return true;
}

bool TaskQueue::retry(const std::string& task_id) {
    std::shared_ptr<TaskControl> retried;
    {
        std::lock_guard lock(mutex_);
        const auto found = std::find_if(
            history_.begin(), history_.end(),
            [&task_id](const std::shared_ptr<TaskControl>& control) {
                return control->snapshot.task_id == task_id;
            });
        if (found == history_.end()) return false;
        const TaskState state = (*found)->snapshot.state;
        if (state != TaskState::failed && state != TaskState::cancelled) {
            return false;
        }
        retried = std::move(*found);
        history_.erase(found);
        // Fresh attempt: new stop source, cleared outcome, new events only.
        retried->stop_source = std::stop_source();
        retried->exception_error = TaskError{ErrorCode::backend_error, "Unexpected task exception"};
        retried->snapshot.state = TaskState::queued;
        retried->snapshot.attempt += 1U;
        retried->snapshot.error.reset();
        retried->snapshot.completed_frames = 0U;
        retried->snapshot.frame_count_hint.reset();
        retried->snapshot.report_id.clear();
        retried->snapshot.final_statistics.reset();
        retried->snapshot.enqueued_at = std::chrono::system_clock::now();
        queued_.push_back(retried);
    }
    condition_.notify_all();
    emit_state_changed(*retried);
    return true;
}

void TaskQueue::set_failure_policy(QueueFailurePolicy policy) {
    std::lock_guard lock(mutex_);
    failure_policy_ = policy;
}

std::vector<TaskSnapshot> TaskQueue::list_tasks() const {
    std::vector<TaskSnapshot> snapshots;
    std::lock_guard lock(mutex_);
    if (active_) snapshots.push_back(active_->snapshot);
    for (const auto& control : queued_) {
        snapshots.push_back(control->snapshot);
    }
    for (const auto& control : history_) {
        snapshots.push_back(control->snapshot);
    }
    return snapshots;
}

std::optional<TaskSnapshot> TaskQueue::task_snapshot(
    const std::string& task_id) const {
    std::lock_guard lock(mutex_);
    const auto matches = [&task_id](
                             const std::shared_ptr<TaskControl>& control) {
        return control->snapshot.task_id == task_id;
    };
    if (active_ && matches(active_)) return active_->snapshot;
    const auto queued_found =
        std::find_if(queued_.begin(), queued_.end(), matches);
    if (queued_found != queued_.end()) return (*queued_found)->snapshot;
    const auto history_found =
        std::find_if(history_.begin(), history_.end(), matches);
    if (history_found != history_.end()) return (*history_found)->snapshot;
    return std::nullopt;
}

bool TaskQueue::queue_active() const {
    std::lock_guard lock(mutex_);
    return started_ && !paused_;
}

bool TaskQueue::has_active_task() const {
    std::lock_guard lock(mutex_);
    return active_ != nullptr;
}

void TaskQueue::emit_state_changed(TaskControl& task) {
    if (callbacks_.state_changed) {
        TaskSnapshot snapshot;
        {
            std::lock_guard lock(mutex_);
            snapshot = task.snapshot;
        }
        callbacks_.state_changed(snapshot);
    }
}

void TaskQueue::fail_task(
    TaskControl& task,
    TaskError error,
    bool cancelled) {
    {
        std::lock_guard lock(mutex_);
        task.snapshot.error = std::move(error);
        task.snapshot.state =
            cancelled ? TaskState::cancelled : TaskState::failed;
        task.snapshot.finished_at = std::chrono::system_clock::now();
        if (!cancelled &&
            failure_policy_ == QueueFailurePolicy::pause_on_failure) {
            // Pause so the user can inspect, retry, or remove the failed
            // task before anything else runs.
            paused_ = true;
        }
    }
    emit_state_changed(task);
}

void TaskQueue::worker_main() noexcept {
    for (;;) {
        std::shared_ptr<TaskControl> task;
        {
            std::unique_lock lock(mutex_);
            condition_.wait(lock, [this] {
                return shutdown_ ||
                    (started_ && !paused_ && !queued_.empty());
            });
            if (shutdown_) return;
            task = queued_.front();
            queued_.pop_front();
            active_ = task;
            task->snapshot.state = TaskState::opening;
            task->snapshot.started_at = std::chrono::system_clock::now();
        }
        try {
            emit_state_changed(*task);
            run_task(*task);
        } catch (...) {
            const auto exception = std::current_exception();
            // Preallocated at enqueue time: even an allocation failure must
            // leave a queryable failed task. Never overwrite a committed error.
            TaskError error = std::move(task->exception_error);
            {
                std::lock_guard lock(mutex_);
                error.stage = task->snapshot.state == TaskState::finishing
                    ? TaskStage::finishing : task->snapshot.state == TaskState::opening
                    ? TaskStage::opening : TaskStage::running;
            }
            try {
                error.task_id = task->snapshot.task_id;
                std::rethrow_exception(exception);
            } catch (const std::exception& cause) {
                try { error.message = cause.what(); } catch (...) {}
            } catch (...) {}
            bool has_error;
            {
                std::lock_guard lock(mutex_);
                has_error = task->snapshot.error.has_value();
            }
            if (!has_error) {
                // State is committed before notifying. A broken notification
                // callback must not terminate the worker or lose that state.
                try { fail_task(*task, std::move(error), false); } catch (...) {}
            }
        }
        {
            std::lock_guard lock(mutex_);
            // run_task always ends in a terminal state; keep the outcome in
            // the bounded history.
            history_.insert(history_.begin(), task);
            if (history_.size() > history_limit) history_.pop_back();
            active_.reset();
            active_stream_ = nullptr;
        }
    }
}

void TaskQueue::run_task(TaskControl& task) {
    const std::string task_id = task.snapshot.task_id;
    EvaluationRequest request = task.snapshot.request;
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

    auto opened_stream = EvaluationStream::open(
        std::move(options), task.stop_source.get_token());
    const double initialization_milliseconds =
        elapsed_seconds_since(initialization_start) * 1000.0;
    if (!opened_stream) {
        const Error error = opened_stream.error();
        const bool cancelled = error.code == ErrorCode::cancelled;
        fail_task(
            task,
            TaskError{error.code,
                      error.message, TaskStage::opening, task_input_role(error.input),
                      error.frame_index, task_id, cancelled},
            cancelled);
        return;
    }

    std::unique_ptr<EvaluationStream> stream =
        std::move(opened_stream).value();
    struct ActiveStreamGuard {
        std::mutex& mutex;
        EvaluationStream*& active;
        ~ActiveStreamGuard() {
            std::lock_guard lock(mutex);
            active = nullptr;
        }
    } stream_guard{mutex_, active_stream_};
    VideoMetadata reference_metadata = stream->metadata().reference;
    reference_metadata.descriptor = stream->descriptor();
    const auto resolved = resolve_evaluation_config(&reference_metadata);
    request.model = stream->model();
    request.scene = model_scene(request.model);
    request.model_decision_reason = resolved.decision_reason;
    request.scene_policy_version = resolved.policy_version;
    {
        std::lock_guard lock(mutex_);
        task.snapshot.request = request;
        active_stream_ = stream.get();
        if (task.stop_source.stop_requested()) stream->request_cancel();
        task.snapshot.state = TaskState::running;
        task.snapshot.device = stream->device_info();
    }
    emit_state_changed(task);
    if (callbacks_.opened) {
        OpenedInfo opened{
            stream->descriptor(), stream->device_info(), stream->metadata(),
            stream->statistics()};
        callbacks_.opened(task_id, opened);
    }

    const StreamMediaMetadata media = stream->metadata();
    std::optional<double> source_frames_per_second;
    if (media.reference.frame_rate.average.valid()) {
        source_frames_per_second = media.reference.frame_rate.average.value();
    }

    const auto processing_start = SteadyClock::now();
    auto last_progress_publish =
        processing_start - progress_publish_interval;
    std::vector<FrameResult> frames;
    std::vector<double> scores;

    const auto publish_progress = [&](bool force) {
        const auto now = SteadyClock::now();
        if (!force && now - last_progress_publish < progress_publish_interval) {
            return;
        }
        last_progress_publish = now;
        const double elapsed = elapsed_seconds_since(processing_start);
        ProgressSnapshot snapshot;
        snapshot.completed_frames = scores.size();
        snapshot.frame_count_hint = stream->frame_count_hint();
        snapshot.decoded_frame_count = stream->decoded_frame_count();
        snapshot.elapsed_seconds = elapsed;
        snapshot.frames_per_second = elapsed > 0.0
            ? static_cast<double>(scores.size()) / elapsed
            : 0.0;
        snapshot.source_frames_per_second = source_frames_per_second;
        {
            std::lock_guard lock(mutex_);
            task.snapshot.completed_frames = snapshot.completed_frames;
            task.snapshot.frame_count_hint = snapshot.frame_count_hint;
        }
        if (callbacks_.progress) callbacks_.progress(task_id, snapshot);
    };

    std::optional<Error> failure;
    TaskStage failure_stage = TaskStage::running;
    for (;;) {
        auto available = stream->next();
        if (!available) {
            failure = available.error();
            break;
        }
        if (available.value().empty()) break;
        std::vector<FrameResult> batch = std::move(available).value();
        scores.reserve(scores.size() + batch.size());
        for (const FrameResult& frame : batch) scores.push_back(frame.vmaf);
        if (callbacks_.frames_appended) {
            callbacks_.frames_appended(task_id, batch);
        }
        for (FrameResult& frame : batch) {
            frames.push_back(std::move(frame));
        }
        publish_progress(false);
    }

    bool cancelled = failure.has_value() &&
        (failure->code == ErrorCode::cancelled);
    if (!failure && !cancelled) {
        {
            std::lock_guard lock(mutex_);
            task.snapshot.state = TaskState::finishing;
        }
        emit_state_changed(task);
        failure_stage = TaskStage::finishing;
        auto tail = stream->finish();
        if (!tail) {
            failure = tail.error();
            cancelled = failure->code == ErrorCode::cancelled;
        } else if (!tail.value().empty()) {
            std::vector<FrameResult> batch = std::move(tail).value();
            for (const FrameResult& frame : batch) scores.push_back(frame.vmaf);
            if (callbacks_.frames_appended) {
                callbacks_.frames_appended(task_id, batch);
            }
            for (FrameResult& frame : batch) {
                frames.push_back(std::move(frame));
            }
        }
    }
    const double processing_milliseconds =
        elapsed_seconds_since(processing_start) * 1000.0;

    if (failure.has_value()) {
        // Collect optional diagnostics before the terminal notification, but
        // never let a secondary statistics exception replace the core failure.
        try {
            auto statistics = stream->statistics();
            std::lock_guard lock(mutex_);
            task.snapshot.final_statistics = std::move(statistics);
        } catch (...) {}
        fail_task(
            task,
            TaskError{failure->code, failure->message, failure_stage,
                      task_input_role(failure->input), failure->frame_index,
                      task_id, cancelled},
            cancelled);
        return;
    }

    const EvaluationStreamStatistics final_statistics = stream->statistics();
    {
        std::lock_guard lock(mutex_);
        task.snapshot.final_statistics = final_statistics;
    }

    if (scores.empty()) {
        fail_task(
            task,
            TaskError{ErrorCode::unsupported_format,
                      i18n::tr("The inputs produced no frames to evaluate"), TaskStage::running,
                      InputRole::both, std::nullopt, task_id, false},
            false);
        return;
    }

    // A final snapshot is required even when finish() returns no extra frames.
    publish_progress(true);
    auto pooled_score = pool_scores(scores, PoolMethod::mean);
    if (!pooled_score) {
        fail_task(
            task,
            TaskError{pooled_score.error().code,
                      pooled_score.error().message, TaskStage::finishing,
                      InputRole::none, std::nullopt, task_id, false},
            false);
        return;
    }

    auto report = std::make_shared<EvaluationReport>();
    report->report_id = task_id;
    report->request = request;
    report->device = stream->device_info();
    report->descriptor = stream->descriptor();
    report->model = request.model;
    report->media = media;
    report->stream_statistics = final_statistics;
    report->frames = std::move(frames);
    report->pooled_score = pooled_score.value();
    report->initialization_milliseconds = initialization_milliseconds;
    report->processing_milliseconds = processing_milliseconds;
    report->end_to_end_milliseconds =
        elapsed_seconds_since(total_start) * 1000.0;
    report->frames_per_second = report->processing_milliseconds > 0.0
        ? static_cast<double>(report->frames.size()) * 1000.0 /
            report->processing_milliseconds
        : 0.0;
    report->completed_at = std::chrono::system_clock::now();

    {
        std::lock_guard lock(mutex_);
        task.snapshot.state = TaskState::completed;
        task.snapshot.report_id = report->report_id;
        task.snapshot.finished_at = report->completed_at;
    }
    if (callbacks_.completed) {
        callbacks_.completed(std::shared_ptr<const EvaluationReport>(report));
    }
    emit_state_changed(task);
}

}  // namespace crossvmaf::studio
