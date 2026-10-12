#pragma once

// Studio task queue service (P1). Owns the evaluation worker thread and the
// EvaluationStream lifecycle; tasks carry a fixed configuration snapshot and
// move through queued → opening → running → finishing → completed (or
// cancelling → cancelled / failed). Progress travels as overwrite-able
// snapshots plus incremental frame batches, never as full-history copies.
//
// All callbacks fire on the queue's worker thread (state_changed may also
// fire from the calling thread of cancel/retry/remove). Receivers must be
// cheap and thread-safe; the controller marshals them to the UI thread.

#include "types.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stop_token>
#include <string>
#include <thread>
#include <vector>

namespace crossvmaf::studio {

struct OpenedInfo final {
    FrameDescriptor descriptor;
    DeviceInfo device;
    StreamMediaMetadata media;
    EvaluationStreamStatistics statistics;
};

class TaskQueue final {
public:
    struct Callbacks final {
        std::function<void(const TaskSnapshot&)> state_changed;
        std::function<void(const std::string& task_id, const OpenedInfo&)>
            opened;
        std::function<void(
            const std::string& task_id, const ProgressSnapshot&)>
            progress;
        // Incremental frame batches (appends only, never full history). The
        // span borrows the worker's batch; copy what you keep.
        std::function<void(
            const std::string& task_id, std::span<const FrameResult>)>
            frames_appended;
        std::function<void(std::shared_ptr<const EvaluationReport>)>
            completed;
    };

    explicit TaskQueue(Callbacks callbacks);
    ~TaskQueue();

    TaskQueue(const TaskQueue&) = delete;
    TaskQueue& operator=(const TaskQueue&) = delete;

    // The request is stored as the immutable configuration snapshot of the
    // task; returns the stable task id.
    std::string enqueue(EvaluationRequest request);
    // Starts (or resumes) picking queued tasks. No-op when already active.
    void start_queue();
    // Stops picking the next task; the running task continues.
    void pause_queue();
    // Queued task: marks it cancelled without running. Active task: requests
    // cooperative cancellation (cancelling → cancelled). Idempotent.
    bool cancel(const std::string& task_id);
    bool remove_queued(const std::string& task_id);
    bool move_queued(const std::string& task_id, std::size_t position);
    // Requeues a failed/cancelled task with a fresh attempt; never replays
    // the old terminal events. The queue must be started to run it again.
    bool retry(const std::string& task_id);
    void set_failure_policy(QueueFailurePolicy policy);

    [[nodiscard]] std::vector<TaskSnapshot> list_tasks() const;
    [[nodiscard]] std::optional<TaskSnapshot> task_snapshot(
        const std::string& task_id) const;
    [[nodiscard]] bool queue_active() const;
    [[nodiscard]] bool has_active_task() const;

private:
    struct TaskControl;

    void worker_main() noexcept;
    void run_task(TaskControl& task);
    void emit_state_changed(TaskControl& task);
    void fail_task(
        TaskControl& task,
        TaskError error,
        bool cancelled);

    Callbacks callbacks_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    // Pending tasks in execution order.
    std::deque<std::shared_ptr<TaskControl>> queued_;
    std::shared_ptr<TaskControl> active_;
    // Terminal tasks, newest first, capped for memory.
    std::vector<std::shared_ptr<TaskControl>> history_;
    EvaluationStream* active_stream_{};
    QueueFailurePolicy failure_policy_{QueueFailurePolicy::pause_on_failure};
    std::uint64_t next_task_serial_{1U};
    bool started_{};
    bool paused_{};
    bool shutdown_{};
    std::thread worker_;
};

}  // namespace crossvmaf::studio
