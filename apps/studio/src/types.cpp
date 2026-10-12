#include "types.hpp"

#include "i18n/translator.hpp"

namespace crossvmaf::studio {

std::string scene_display_name(SceneId scene) {
    switch (scene) {
    case SceneId::tv_standard:
        return i18n::tr("TV / monitor · standard distance");
    case SceneId::tv_4k_close:
        return i18n::tr("4K TV · close distance");
    case SceneId::tv_4k_standard:
        return i18n::tr("4K TV · standard distance");
    case SceneId::cinema_far:
        return i18n::tr("Large screen / projector · far distance");
    }
    return i18n::tr("Unknown scene");
}

std::string task_state_name(TaskState state) {
    switch (state) {
    case TaskState::queued: return i18n::tr("Queued");
    case TaskState::opening: return i18n::tr("Opening");
    case TaskState::running: return i18n::tr("Evaluating");
    case TaskState::finishing: return i18n::tr("Finishing");
    case TaskState::completed: return i18n::tr("Completed");
    case TaskState::failed: return i18n::tr("Failed");
    case TaskState::cancelling: return i18n::tr("Cancelling");
    case TaskState::cancelled: return i18n::tr("Cancelled");
    }
    return i18n::tr("Unknown");
}

std::string_view task_state_id(TaskState state) noexcept {
    switch (state) {
    case TaskState::queued: return "queued";
    case TaskState::opening: return "opening";
    case TaskState::running: return "running";
    case TaskState::finishing: return "finishing";
    case TaskState::completed: return "completed";
    case TaskState::failed: return "failed";
    case TaskState::cancelling: return "cancelling";
    case TaskState::cancelled: return "cancelled";
    }
    return "unknown";
}

std::string describe_task_error(const TaskError& error) {
    if (error.cancelled || error.code == ErrorCode::cancelled) {
        return i18n::tr("The task was stopped; partial results were not saved");
    }
    std::string description;
    switch (error.code) {
    case ErrorCode::hardware_decode_unavailable:
        description = i18n::tr(
            "Hardware-only decoding is unavailable, so the task stopped. Use automatic "
            "or software decoding, or choose a GPU that supports this video");
        break;
    case ErrorCode::unsupported_format:
        description = i18n::tr(
            "The video format or codec is not supported, or the timelines cannot be aligned");
        break;
    case ErrorCode::invalid_argument:
        description = i18n::tr(
            "The inputs do not meet the evaluation requirements (reference and distorted "
            "specs must match)");
        break;
    case ErrorCode::io_error:
        description = i18n::tr(
            "Cannot read an input file (it may have been moved, deleted or locked)");
        break;
    case ErrorCode::no_compatible_device:
        description = i18n::tr("No compatible compute device is available");
        break;
    case ErrorCode::device_lost:
        description = i18n::tr(
            "The compute device was lost during the evaluation (the driver may have reset)");
        break;
    case ErrorCode::device_self_test_failed:
        description = i18n::tr(
            "The device failed its self-test and cannot run the evaluation");
        break;
    case ErrorCode::unsupported_model:
        description = i18n::tr("The selected quality model is not supported");
        break;
    case ErrorCode::model_error:
        description = i18n::tr("The quality model data is invalid");
        break;
    case ErrorCode::invalid_state:
        description = i18n::tr("The evaluation pipeline entered an invalid state");
        break;
    case ErrorCode::backend_error:
    case ErrorCode::cancelled:
    default:
        description = i18n::tr("The core backend reported an error");
        break;
    }
    switch (error.stage) {
    case TaskStage::opening:
        description = i18n::tr("{} (while opening inputs)", description);
        break;
    case TaskStage::running:
        description = i18n::tr("{} (during evaluation)", description);
        break;
    case TaskStage::finishing:
        description = i18n::tr("{} (while finishing)", description);
        break;
    }
    if (error.frame_index.has_value()) {
        description = i18n::tr("{}, frame #{}", description,
                               std::to_string(*error.frame_index));
    }
    if (!error.message.empty()) {
        // The raw failure message, usually from the core library, is not translated.
        description = i18n::trc("error detail", "{}: {}", description, error.message);
    }
    return description;
}

}  // namespace crossvmaf::studio
