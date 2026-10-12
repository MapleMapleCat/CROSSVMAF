#include "platform/diagnostics.hpp"

#include "i18n/translator.hpp"
#include "reports/report_dto.hpp"
#include "crossvmaf/version.hpp"

#include <chrono>
#include <fstream>

namespace crossvmaf::studio {
namespace {

[[nodiscard]] std::string path_to_utf8(const std::filesystem::path& path) {
    const std::u8string utf8 = path.u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

[[nodiscard]] std::string_view error_code_name(ErrorCode code) {
    switch (code) {
    case ErrorCode::invalid_argument: return "invalid_argument";
    case ErrorCode::invalid_state: return "invalid_state";
    case ErrorCode::unsupported_format: return "unsupported_format";
    case ErrorCode::unsupported_model: return "unsupported_model";
    case ErrorCode::no_compatible_device: return "no_compatible_device";
    case ErrorCode::device_self_test_failed:
        return "device_self_test_failed";
    case ErrorCode::device_lost: return "device_lost";
    case ErrorCode::io_error: return "io_error";
    case ErrorCode::model_error: return "model_error";
    case ErrorCode::backend_error: return "backend_error";
    case ErrorCode::cancelled: return "cancelled";
    case ErrorCode::hardware_decode_unavailable: return "hardware_decode_unavailable";
    }
    return "unknown";
}

[[nodiscard]] std::string_view task_stage_name(TaskStage stage) {
    switch (stage) {
    case TaskStage::opening: return "opening";
    case TaskStage::running: return "running";
    case TaskStage::finishing: return "finishing";
    }
    return "unknown";
}

[[nodiscard]] nlohmann::json input_statistics_json(
    const StreamInputStatistics& statistics) {
    return {
        {"packets_read", statistics.packets_read},
        {"decode_backend", to_string(statistics.decode_backend)},
            {"decode_device", statistics.decode.device_id},
            {"decode_storage", to_string(statistics.decode.native_storage)},
            {"decode_transfer", to_string(statistics.decode.transfer_path)},
            {"decode_attempts", statistics.decode.attempts},
            {"host_transfer_bytes", statistics.decode.host_transfer_bytes},
            {"gpu_copy_bytes", statistics.decode.gpu_copy_bytes},
            {"transfer_nanoseconds", statistics.decode.transfer_nanoseconds},
        {"compressed_bytes", statistics.compressed_bytes},
        {"decoded_frames", statistics.decoded_frames},
        {"decoder_threads", statistics.decoder_threads},
        {"decode_active_ms",
         static_cast<double>(statistics.decode_active_nanoseconds) / 1.0e6},
        {"output_queue_blocked_ms",
         static_cast<double>(statistics.output_queue_blocked_nanoseconds) /
             1.0e6},
        {"end_of_stream", statistics.end_of_stream},
        {"cancellation_requested", statistics.cancellation_requested},
    };
}

[[nodiscard]] nlohmann::json stream_statistics_json(
    const EvaluationStreamStatistics& statistics) {
    return {
        {"frame_count_hint",
         statistics.frame_count_hint.has_value()
             ? nlohmann::json(*statistics.frame_count_hint)
             : nlohmann::json(nullptr)},
        {"decoded_frame_count", statistics.decoded_frame_count},
        {"scheduled_frames", statistics.scheduled_frames},
        {"acquired_frames", statistics.acquired_frames},
        {"record_capacity", statistics.record_capacity},
        {"owned_pairs", statistics.owned_pairs},
        {"ready_lead_frames", statistics.ready_lead_frames},
        {"frame_pair_copy_calls", statistics.frame_pair_copy_calls},
        {"frame_pair_copy_bytes", statistics.frame_pair_copy_bytes},
        {"frame_pair_copy_ms",
         static_cast<double>(statistics.frame_pair_copy_nanoseconds) / 1.0e6},
        {"staging_input_wait_ms",
         static_cast<double>(statistics.staging_input_wait_nanoseconds) /
             1.0e6},
        {"staging_grant_wait_ms",
         static_cast<double>(statistics.staging_grant_wait_nanoseconds) /
             1.0e6},
        {"coordinator_wait_ms",
         static_cast<double>(statistics.coordinator_wait_nanoseconds) / 1.0e6},
        {"input_lease_capacity", statistics.input_lease_capacity},
        {"scheduling_complete", statistics.scheduling_complete},
        {"cancellation_requested", statistics.cancellation_requested},
        {"reference", input_statistics_json(statistics.reference)},
        {"distorted", input_statistics_json(statistics.distorted)},
    };
}

[[nodiscard]] std::int64_t to_unix_seconds(
    std::chrono::system_clock::time_point time_point) {
    return std::chrono::duration_cast<std::chrono::seconds>(
        time_point.time_since_epoch()).count();
}

}  // namespace

nlohmann::json build_task_diagnostics(const TaskSnapshot& task) {
    nlohmann::json document;
    document["schema_version"] = std::string(report_schema_version);
    document["crossvmaf_version"] = std::string(software_version);
    document["kind"] = "task_diagnostics";
    document["task"] = {
        {"task_id", task.task_id},
        {"state", std::string(task_state_id(task.state))},
        {"attempt", task.attempt},
        {"enqueued_at_utc_seconds", to_unix_seconds(task.enqueued_at)},
        {"started_at_utc_seconds", to_unix_seconds(task.started_at)},
        {"finished_at_utc_seconds", to_unix_seconds(task.finished_at)},
        {"completed_frames", task.completed_frames},
        {"frame_count_hint",
         task.frame_count_hint.has_value()
             ? nlohmann::json(*task.frame_count_hint)
             : nlohmann::json(nullptr)},
        {"report_id", task.report_id},
    };
    document["request"] = {
        {"reference", path_to_utf8(task.request.reference_path)},
        {"distorted", path_to_utf8(task.request.distorted_path)},
        {"scene_id", static_cast<int>(task.request.scene)},
        {"model", to_string(task.request.model)},
        {"model_decision_reason", task.request.model_decision_reason},
        {"scene_policy_version", task.request.scene_policy_version},
        {"device_uuid", task.request.device.uuid},
        {"decode_mode", to_string(task.request.decode_mode)},
        {"decode_backend", to_string(task.request.decode_backend)},
        {"decode_transfer", to_string(task.request.decode_transfer)},
        {"decode_device", task.request.decode_device},
        {"decoder_threads_per_source", task.request.decoder_threads_per_source},
        {"decoded_frame_queue_capacity", task.request.decoded_frame_queue_capacity},
    };
    if (task.device.has_value()) {
        document["device"] = {
            {"name", task.device->name},
            {"uuid", task.device->uuid},
            {"vendor_id", task.device->vendor_id},
            {"device_id", task.device->device_id},
            {"api_version_raw", task.device->api_version},
            {"driver_version_raw", task.device->driver_version},
            {"discrete", task.device->discrete},
            {"device_local_heap_bytes",
             task.device->device_local_heap_bytes},
        };
    }
    if (task.error.has_value()) {
        document["error"] = {
            {"code", error_code_name(task.error->code)},
            {"stage", task_stage_name(task.error->stage)},
            {"input_role", task.error->input_role == InputRole::reference ? "reference" :
                           task.error->input_role == InputRole::distorted ? "distorted" :
                           task.error->input_role == InputRole::both ? "both" : "none"},
            {"cancelled", task.error->cancelled},
            {"frame_index",
             task.error->frame_index.has_value()
                 ? nlohmann::json(*task.error->frame_index)
                 : nlohmann::json(nullptr)},
            {"message", task.error->message},
        };
    }
    if (task.final_statistics.has_value()) {
        document["stream_statistics"] =
            stream_statistics_json(*task.final_statistics);
    }
    return document;
}

Result<void> export_support_bundle(
    const nlohmann::json& diagnostics,
    const std::filesystem::path& destination) try {
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) {
        return Result<void>::failure(
            {ErrorCode::io_error,
             i18n::tr("Cannot create the support file: {}", path_to_utf8(destination))});
    }
    output << diagnostics.dump(2) << '\n';
    output.close();
    if (!output) {
        return Result<void>::failure(
            {ErrorCode::io_error,
             i18n::tr("Failed to write the support file: {}", path_to_utf8(destination))});
    }
    return Result<void>::success();
} catch (const std::exception& exception) {
    return Result<void>::failure({ErrorCode::io_error,
        std::string("writing output: ") + exception.what()});
}

}  // namespace crossvmaf::studio
