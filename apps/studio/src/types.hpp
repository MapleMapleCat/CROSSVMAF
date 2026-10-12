#pragma once

// UI-independent Studio domain types. These DTOs deliberately avoid the
// Slint-generated app.h so analysis, export, persistence, and the task queue
// can be tested without a window.

#include "reports/evaluation_report.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace crossvmaf::studio {

// Overwrite-able progress snapshot; frame data itself travels in append
// batches so progress messages never copy the full history.
struct ProgressSnapshot final {
    std::uint64_t completed_frames{};
    std::optional<std::uint64_t> frame_count_hint;
    std::uint64_t decoded_frame_count{};
    double elapsed_seconds{};
    double frames_per_second{};
    // Nominal source frame rate from the opened media metadata, when known.
    // Playback speed = frames_per_second / source_frames_per_second (CFR).
    std::optional<double> source_frames_per_second;
};

enum class TaskStage : std::uint8_t {
    opening,
    running,
    finishing,
};

enum class InputRole : std::uint8_t {
    none,
    reference,
    distorted,
    both,
};

// Structured task failure: keeps the core error code and raw message and
// adds the context only the application knows. UI text is derived from the
// code, never by parsing the English message.
struct TaskError final {
    ErrorCode code{ErrorCode::backend_error};
    std::string message;
    TaskStage stage{TaskStage::opening};
    InputRole input_role{InputRole::none};
    std::optional<std::uint64_t> frame_index;
    std::string task_id;
    bool cancelled{};
};

// UI summary with an actionable suggestion, keyed on the error code, in the
// active UI language.
[[nodiscard]] std::string describe_task_error(const TaskError& error);

enum class TaskState : std::uint8_t {
    queued,
    opening,
    running,
    finishing,
    completed,
    failed,
    cancelling,
    cancelled,
};

// Localized label for the UI.
[[nodiscard]] std::string task_state_name(TaskState state);
// Stable English identifier for diagnostics and other machine-read output.
[[nodiscard]] std::string_view task_state_id(TaskState state) noexcept;

enum class QueueFailurePolicy : std::uint8_t {
    // Default: a failed task pauses the queue so the user can inspect,
    // retry, or remove it before anything else runs.
    pause_on_failure,
    continue_on_failure,
};

struct TaskSnapshot final {
    std::string task_id;
    EvaluationRequest request;
    TaskState state{TaskState::queued};
    std::uint32_t attempt{1U};
    std::optional<TaskError> error;
    std::uint64_t completed_frames{};
    std::optional<std::uint64_t> frame_count_hint;
    std::string report_id;
    // Diagnostics captured at the terminal point when a stream existed.
    std::optional<EvaluationStreamStatistics> final_statistics;
    std::optional<DeviceInfo> device;
    std::chrono::system_clock::time_point enqueued_at;
    std::chrono::system_clock::time_point started_at;
    std::chrono::system_clock::time_point finished_at;
};

}  // namespace crossvmaf::studio
