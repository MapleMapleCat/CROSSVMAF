#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "evaluation/channel.hpp"
#include "evaluation/frame_pair_record.hpp"
#include "evaluation/frame_records.hpp"
#include "evaluation/initialization_observer.hpp"
#include "evaluation/input_lease_pool.hpp"
#include "evaluation/input_staging_ring.hpp"
#include "evaluation/session_detail.hpp"
#include "evaluation/session_telemetry.hpp"
#include "evaluation/stream_opener.hpp"
#include "gpu/error.hpp"
#include "gpu/memory.hpp"
#include "video/backends/vulkan.hpp"
#include "video/decode_backend.hpp"
#include "video/backend_selection.hpp"
#include "scoring/model.hpp"
#include "shaders/embedded_spirv_manifest.hpp"
#include "crossvmaf/video.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace crossvmaf::evaluation::session_detail {

FrameSlot::FrameSlot() = default;
FrameSlot::~FrameSlot() = default;
FrameSlot::FrameSlot(FrameSlot &&) noexcept = default;
FrameSlot &FrameSlot::operator=(FrameSlot &&) noexcept = default;

void FrameSlot::begin_frame(std::uint64_t index) noexcept {
    frame_index = index;
    pending_resource_transaction_count = 0U;
    in_use = true;
}

} // namespace crossvmaf::evaluation::session_detail

namespace crossvmaf::evaluation {

Result<std::vector<DeviceInfo>> enumerate_devices() {
    const auto runtime_instance = gpu::Runtime::acquire();
    if (!runtime_instance) {
        return Result<std::vector<DeviceInfo>>::failure(runtime_instance.error());
    }
    auto candidates = runtime_instance.value()->enumerate_candidates();
    if (!candidates) {
        return Result<std::vector<DeviceInfo>>::failure(candidates.error());
    }
    std::vector<DeviceInfo> devices;
    devices.reserve(candidates.value().size());
    for (auto &candidate : candidates.value()) {
        devices.push_back(std::move(candidate.info));
    }
    return Result<std::vector<DeviceInfo>>::success(std::move(devices));
}

} // namespace crossvmaf::evaluation

namespace crossvmaf::evaluation {
struct DecoderOpenOptions final {
    std::filesystem::path reference_path;
    std::filesystem::path distorted_path;
    std::size_t decoded_frame_queue_capacity{4U};
    std::size_t decoder_threads_per_source{};
    std::size_t record_capacity{6U};
    std::shared_ptr<video::DecodeDevice> decode_device;
    std::shared_ptr<video::VulkanDecodeDevice> transfer_device;
    gpu::Device* compute_device{};
    StreamOptions selection;
    std::vector<std::string> attempts;
    bool respect_automatic_vulkan_policy{};
    // Cooperative cancellation for the blocking decoder-open phase.
    std::stop_token stop_token{};
    bool require_hardware_decode{};
};

struct ReadyInput final {
    static constexpr std::size_t invalid_slot_index = std::numeric_limits<std::size_t>::max();

    FramePairRecord *record{};
    std::uint64_t generation{};
    std::size_t slot_index{invalid_slot_index};

    [[nodiscard]] explicit operator bool() const noexcept {
        return record != nullptr && generation != 0U && slot_index != invalid_slot_index;
    }
};

enum class FramePairAcquireStatus : std::uint8_t {
    ready,
    unavailable,
    input_starved,
    end_of_stream,
};

enum class FramePairInputState : std::uint8_t {
    waiting_for_decoder,
    waiting_for_storage,
    copying,
    terminal,
};

struct FramePairAcquireResult final {
    FramePairAcquireStatus status{FramePairAcquireStatus::unavailable};
    ReadyInput pair;
    std::size_t scheduled_pairs{};
    std::uint64_t input_epoch{};
    std::uint64_t progress_epoch{};
    FramePairInputState input_state{FramePairInputState::waiting_for_decoder};
};

struct InputStatistics final {
    std::optional<std::uint64_t> frame_count_hint;
    std::uint64_t decoded_frame_count{};
    std::uint64_t next_frame_to_schedule{};
    std::uint64_t next_frame_to_acquire{};
    std::size_t record_capacity{};
    std::size_t owned_pair_count{};
    std::size_t ordered_pair_count{};
    std::size_t ready_lead_frames{};
    std::uint64_t frame_pair_copy_calls{};
    std::uint64_t frame_pair_copy_bytes{};
    std::uint64_t frame_pair_copy_nanoseconds{};
    std::uint64_t staging_input_wait_nanoseconds{};
    std::uint64_t staging_input_wait_count{};
    std::uint64_t staging_grant_wait_nanoseconds{};
    std::uint64_t staging_grant_wait_count{};
    std::uint64_t coordinator_wait_nanoseconds{};
    std::uint64_t coordinator_wait_count{};
    bool scheduling_complete{};
    bool cancellation_requested{};
    bool drained{};
    video::VideoDecoderStatistics reference_decoder;
    video::VideoDecoderStatistics distorted_decoder;
};

struct PreparedVideoDecoderPair final {
    std::unique_ptr<video::Decoder> reference;
    std::unique_ptr<video::Decoder> distorted;
    FrameDescriptor descriptor;
    std::optional<std::uint64_t> frame_count_hint;
    bool vulkan_image_input{};
    video::HardwareSurfaceFormat hardware_surface_format{video::HardwareSurfaceFormat::none};
};

} // namespace crossvmaf::evaluation
namespace crossvmaf::evaluation {
namespace {

namespace adm3 = metrics::adm3::vulkan;
namespace cambi = metrics::cambi::vulkan;
namespace motion3 = metrics::motion3::vulkan;
namespace speed_chroma = metrics::speed_chroma::vulkan;
using session_detail::feature_lane_count;
using session_detail::feature_lane_of;
using FeatureTransactionKind = session_detail::FeatureTransactionKind;
using FrameSlot = session_detail::FrameSlot;
using MetricFrameRecord = FrameRecords::FrameRecord;
using PendingTransaction = session_detail::PendingTransaction;
using session_detail::invalid_resource_index;

[[nodiscard]] std::vector<evaluation::MotionFeature>
convert_motion_scores(std::vector<motion3::MotionScore> scores) {
    std::vector<evaluation::MotionFeature> converted_scores;
    converted_scores.reserve(scores.size());
    for (const motion3::MotionScore &score : scores) {
        converted_scores.push_back({score.index, score.value});
    }
    return converted_scores;
}

void append_completions(evaluation::CompletionBatch &destination,
                        evaluation::CompletionBatch source) {
    destination.spatial.insert(destination.spatial.end(),
                               std::make_move_iterator(source.spatial.begin()),
                               std::make_move_iterator(source.spatial.end()));
    destination.motion.insert(destination.motion.end(),
                              std::make_move_iterator(source.motion.begin()),
                              std::make_move_iterator(source.motion.end()));
}

[[nodiscard]] bool should_submit_ready_wave(std::uint32_t ready_count,
                                            std::uint32_t maximum_ready_backlog,
                                            std::uint64_t prefixes_since_wave,
                                            std::uint32_t capacity) noexcept {
    if (ready_count == 0U) return false;
    const std::uint32_t half_capacity_trigger = std::max(1U, maximum_ready_backlog / 2U);
    if (ready_count >= half_capacity_trigger) return true;
    return prefixes_since_wave >= static_cast<std::uint64_t>(capacity);
}

[[nodiscard]] std::optional<std::size_t>
select_lease_keyed_frame_slot(const std::vector<FrameSlot> &frame_slots,
                              const evaluation::FrameStorageHandle &spatial_lease,
                              std::size_t active_slot_count) noexcept {
    if (frame_slots.empty() || !spatial_lease || active_slot_count == 0U) {
        return std::nullopt;
    }
    active_slot_count = std::min(active_slot_count, frame_slots.size());

    // Start every search from the physical input-lease identity. Linear
    // probing keeps all free command/resource slots usable when a lease is
    // recycled before the slot selected for its previous generation retires.
    const std::uint64_t lease_identity =
        (static_cast<std::uint64_t>(spatial_lease.segment_index) << 32U) |
        static_cast<std::uint64_t>(spatial_lease.slot_index);
    const std::size_t preferred_slot = static_cast<std::size_t>(lease_identity % active_slot_count);
    for (std::size_t slot_offset = 0U; slot_offset < active_slot_count; ++slot_offset) {
        const std::size_t frame_slot_index = (preferred_slot + slot_offset) % active_slot_count;
        if (!frame_slots[frame_slot_index].in_use) {
            return frame_slot_index;
        }
    }
    return std::nullopt;
}

void trace_memory_pool(const gpu::DeviceMemoryPoolStats &statistics,
                       const FrameDescriptor &descriptor, std::size_t frame_slot_count) noexcept {
#if defined(_WIN32)
    char *trace_value = nullptr;
    std::size_t trace_value_length = 0U;
    const int trace_status = _dupenv_s(&trace_value, &trace_value_length, "CROSSVMAF_VULKAN_TRACE");
    const bool trace_enabled =
        trace_status == 0 && trace_value != nullptr && std::string_view(trace_value) != "0";
    std::free(trace_value);
#else
    const char *const trace_value = std::getenv("CROSSVMAF_VULKAN_TRACE");
    const bool trace_enabled = trace_value != nullptr && std::string_view(trace_value) != "0";
#endif
    if (!trace_enabled) return;
    std::fprintf(stderr,
                 "[crossvmaf-vulkan] memory_pool width=%u height=%u slots=%zu "
                 "blocks=%llu reserved_bytes=%llu used_bytes=%llu peak_used_bytes=%llu "
                 "fallback=%d budget=%d heap_bytes=%llu heap_budget=%llu heap_usage=%llu\n",
                 descriptor.width, descriptor.height, frame_slot_count,
                 static_cast<unsigned long long>(statistics.block_count),
                 static_cast<unsigned long long>(statistics.reserved_bytes),
                 static_cast<unsigned long long>(statistics.used_bytes),
                 static_cast<unsigned long long>(statistics.peak_used_bytes),
                 statistics.preferred_memory_fallback ? 1 : 0,
                 statistics.memory_budget_available ? 1 : 0,
                 static_cast<unsigned long long>(statistics.device_local_heap_bytes),
                 static_cast<unsigned long long>(statistics.device_local_heap_budget),
                 static_cast<unsigned long long>(statistics.device_local_heap_usage));
}

// CROSSVMAF_VULKAN_QUEUES clamps how many device queues the feature lanes
// spread over: 1, 2, or 4. Any other value (including unset) spreads lanes
// over every queue the device exposes.
[[nodiscard]] std::size_t lane_queue_modulus_from_environment() noexcept {
#if defined(_WIN32)
    char *value = nullptr;
    std::size_t value_length = 0U;
    const int status = _dupenv_s(&value, &value_length, "CROSSVMAF_VULKAN_QUEUES");
    if (status != 0 || value == nullptr) return 0U;
    const unsigned long parsed = std::strtoul(value, nullptr, 10);
    std::free(value);
#else
    const char *const value = std::getenv("CROSSVMAF_VULKAN_QUEUES");
    if (value == nullptr) return 0U;
    const unsigned long parsed = std::strtoul(value, nullptr, 10);
#endif
    return parsed == 1UL || parsed == 2UL || parsed == 4UL ? static_cast<std::size_t>(parsed) : 0U;
}

[[nodiscard]] bool input_hash_trace_enabled() noexcept {
    static const bool enabled = [] {
#if defined(_WIN32)
        char *value = nullptr;
        std::size_t value_length = 0U;
        const int status = _dupenv_s(&value, &value_length, "CROSSVMAF_INPUT_HASH_TRACE");
        const bool result = status == 0 && value != nullptr && std::string_view(value) != "0";
        std::free(value);
        return result;
#else
        const char *const value = std::getenv("CROSSVMAF_INPUT_HASH_TRACE");
        return value != nullptr && std::string_view(value) != "0";
#endif
    }();
    return enabled;
}

[[nodiscard]] Result<std::array<std::uint64_t, 3U>>
hash_gpu_frame_planes(const video::GpuFrameView &frame) {
    std::array<std::uint64_t, 3U> hashes{};
    for (std::size_t plane_index = 0U; plane_index < frame.planes.size(); ++plane_index) {
        const video::GpuPlaneView &plane = frame.planes[plane_index];
        const auto plane_hash =
            plane.buffer->sampled_hash(plane.size_bytes, plane.offset_bytes, 4'096U);
        if (!plane_hash) {
            return Result<std::array<std::uint64_t, 3U>>::failure(plane_hash.error());
        }
        hashes[plane_index] = plane_hash.value();
    }
    return Result<std::array<std::uint64_t, 3U>>::success(hashes);
}

} // namespace

struct EvaluationGpuConfig {
    FrameDescriptor frame;
    DeviceSelector device;
    bool vulkan_image_input{};
    std::shared_ptr<telemetry::SnapshotSink> telemetry_snapshot_sink;
    InitializationObserver *initialization_observer{};
};

struct SegmentedPushOutcome final {
    evaluation::CompletionBatch completions;
    bool slot_busy{};
};

struct SegmentedPollOutcome final {
    evaluation::CompletionBatch completions;
    bool made_progress{};
};

using SegmentedInputToken = SegmentedInputLeaseToken;

} // namespace crossvmaf::evaluation
namespace crossvmaf::evaluation {
namespace {

[[nodiscard]] bool input_descriptors_match(const FrameDescriptor &left,
                                           const FrameDescriptor &right) noexcept {
    return left.width == right.width && left.height == right.height &&
           left.bit_depth == right.bit_depth && left.pixel_format == right.pixel_format;
}

[[nodiscard]] Result<void> validate_decoder_open_options(const DecoderOpenOptions &options) {
    if (options.reference_path.empty() || options.distorted_path.empty()) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "both compressed video input paths must be provided"});
    }
    if (options.decoded_frame_queue_capacity == 0U || options.decoded_frame_queue_capacity > 64U ||
        options.decoder_threads_per_source > 64U) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "decoder queue capacity must be 1..64 and threads 0..64"});
    }
    if (options.record_capacity < 2U || options.record_capacity > 128U) {
        return Result<void>::failure({ErrorCode::invalid_argument,
                                      "decoded frame-pair record capacity must be within 2..128"});
    }
    return Result<void>::success();
}

[[nodiscard]] Error cancelled_error() {
    return {ErrorCode::cancelled, "decoded frame-pair prefetcher was cancelled"};
}

[[nodiscard]] std::uint64_t frame_pair_active_bytes(const FrameDescriptor &descriptor) noexcept {
    const std::uint64_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    std::uint64_t frame_bytes = 0U;
    for (std::size_t plane_index = 0U; plane_index < 3U; ++plane_index) {
        frame_bytes += static_cast<std::uint64_t>(plane_width(descriptor, plane_index)) *
                       static_cast<std::uint64_t>(plane_height(descriptor, plane_index)) *
                       bytes_per_sample;
    }
    return frame_bytes * 2U;
}

void record_elapsed_nanoseconds(std::atomic<std::uint64_t> &destination,
                                std::chrono::steady_clock::time_point started_at) noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_at);
    if (elapsed.count() <= 0) return;
    destination.fetch_add(static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
}

} // namespace

Result<PreparedVideoDecoderPair> prepare_decoder_candidate(const DecoderOpenOptions &options) {
    const auto options_validation = validate_decoder_open_options(options);
    if (!options_validation) {
        return Result<PreparedVideoDecoderPair>::failure(options_validation.error());
    }

    video::VideoDecoderOptions decoder_options;
    decoder_options.output_queue_capacity = options.decoded_frame_queue_capacity;
    decoder_options.codec_thread_count = options.decoder_threads_per_source;
    decoder_options.decode_device = options.decode_device;
    decoder_options.transfer_device = options.transfer_device;
    decoder_options.transfer_policy = options.selection.decode_transfer;
    decoder_options.open_attempts = options.attempts;
    decoder_options.respect_automatic_vulkan_policy = options.respect_automatic_vulkan_policy;
    decoder_options.concurrent_decoder_count = 2U;
    decoder_options.stop_token = options.stop_token;
    decoder_options.require_hardware_decode = options.require_hardware_decode;
    auto decoders =
        video::Decoder::open_pair(options.reference_path, options.distorted_path, decoder_options);
    if (!decoders) {
        return Result<PreparedVideoDecoderPair>::failure(decoders.error());
    }
    auto &reference_decoder = decoders.value().reference;
    auto &distorted_decoder = decoders.value().distorted;

    const bool reference_uses_hardware =
        reference_decoder->decode_backend() != video::VideoDecodeBackend::software;
    const bool distorted_uses_hardware =
        distorted_decoder->decode_backend() != video::VideoDecodeBackend::software;
    if (options.require_hardware_decode && (!reference_uses_hardware || !distorted_uses_hardware)) {
        return Result<PreparedVideoDecoderPair>::failure(
            {ErrorCode::hardware_decode_unavailable,
             "Hardware-only decoding requires both inputs to use hardware"});
    }
    if (reference_uses_hardware != distorted_uses_hardware) {
        reference_decoder->request_cancel();
        distorted_decoder->request_cancel();
        reference_decoder->drain();
        distorted_decoder->drain();
        reference_decoder.reset();
        distorted_decoder.reset();

        decoder_options.decode_device.reset();
        auto software_decoders = video::Decoder::open_pair(options.reference_path,
                                                           options.distorted_path, decoder_options);
        if (!software_decoders) {
            return Result<PreparedVideoDecoderPair>::failure(software_decoders.error());
        }
        reference_decoder = std::move(software_decoders.value().reference);
        distorted_decoder = std::move(software_decoders.value().distorted);
    }
    if (!input_descriptors_match(reference_decoder->descriptor(),
                                 distorted_decoder->descriptor())) {
        return Result<PreparedVideoDecoderPair>::failure(
            {ErrorCode::invalid_argument,
             "reference and distorted decoded descriptors do not match"});
    }

    std::optional<std::uint64_t> frame_count_hint;
    if (reference_decoder->frame_count_hint().has_value() &&
        reference_decoder->frame_count_hint() == distorted_decoder->frame_count_hint()) {
        frame_count_hint = reference_decoder->frame_count_hint();
    }
    const FrameDescriptor descriptor = reference_decoder->descriptor();
    const bool vulkan_image_input = reference_decoder->has_vulkan_output();
    if (vulkan_image_input != distorted_decoder->has_vulkan_output())
        return Result<PreparedVideoDecoderPair>::failure({ErrorCode::hardware_decode_unavailable,
            "the input pair requires different storage; retry both inputs with host transfer"});
    const video::HardwareSurfaceFormat hardware_surface_format =
        reference_decoder->hardware_surface_format();
    if (vulkan_image_input &&
        hardware_surface_format != distorted_decoder->hardware_surface_format()) {
        return Result<PreparedVideoDecoderPair>::failure(
            {ErrorCode::invalid_argument,
             "reference and distorted hardware surface formats do not match"});
    }
    return Result<PreparedVideoDecoderPair>::success(
        {std::move(reference_decoder), std::move(distorted_decoder), descriptor, frame_count_hint,
         vulkan_image_input, hardware_surface_format});
}
Result<PreparedVideoDecoderPair> prepare_decoders(const DecoderOpenOptions& options) {
    if (!options.compute_device) return prepare_decoder_candidate(options);
    auto candidate_options = options;
    Error last_error{ErrorCode::hardware_decode_unavailable, "no eligible decode backend"};
    const auto plan = video::plan_decode_candidates(options.selection,
                                                     options.compute_device->device_info());
    candidate_options.attempts.insert(candidate_options.attempts.end(), plan.skipped.begin(),
                                      plan.skipped.end());
    for (const auto& candidate : plan.candidates) {
        if (options.stop_token.stop_requested())
            return Result<PreparedVideoDecoderPair>::failure({ErrorCode::cancelled, "decoder opening cancelled"});
        const auto backend = candidate.backend;
        candidate_options.decode_device.reset();
        candidate_options.transfer_device.reset();
        candidate_options.selection.decode_transfer = candidate.transfer;
        candidate_options.require_hardware_decode = backend != StreamDecodeBackend::software;
        candidate_options.respect_automatic_vulkan_policy =
            backend == StreamDecodeBackend::vulkan_video && options.selection.decode_mode == DecodeMode::automatic &&
            options.selection.decode_backend == DecodeBackendPreference::automatic;
        if (backend != StreamDecodeBackend::software) {
            auto device = video::open_decode_device(backend, options.selection.decode_device, *options.compute_device);
            if (!device) {
                last_error = device.error();
                if (last_error.code != ErrorCode::hardware_decode_unavailable && last_error.code != ErrorCode::unsupported_format)
                    return Result<PreparedVideoDecoderPair>::failure(last_error);
                candidate_options.attempts.push_back(std::string(to_string(backend)) + "/" +
                    std::string(to_string(candidate.transfer)) + ": " + last_error.message);
                continue;
            }
            candidate_options.decode_device = std::move(device).value();
            const auto& physical_id = backend == StreamDecodeBackend::nvdec ? options.compute_device->device_info().uuid :
                backend == StreamDecodeBackend::vulkan_video ? options.compute_device->device_info().uuid : options.compute_device->device_info().physical_id;
            if (physical_id.empty() || candidate_options.decode_device->info.physical_id != physical_id) {
                if (candidate.transfer == DecodeTransferPolicy::gpu_only) {
                    last_error = {ErrorCode::hardware_decode_unavailable, "GPU-only transfer requires a verified matching physical device"};
                    candidate_options.attempts.push_back(std::string(to_string(backend)) + "/gpu: " + last_error.message);
                    continue;
                }
            } else candidate_options.transfer_device = options.transfer_device;
        }
        auto prepared = prepare_decoder_candidate(candidate_options);
        if (!prepared && backend != StreamDecodeBackend::software &&
            options.selection.decode_backend != DecodeBackendPreference::automatic &&
            candidate.transfer == DecodeTransferPolicy::automatic &&
            prepared.error().code == ErrorCode::hardware_decode_unavailable) {
            candidate_options.attempts.push_back(std::string(to_string(backend)) + ": " + prepared.error().message);
            candidate_options.selection.decode_transfer = DecodeTransferPolicy::host;
            prepared = prepare_decoder_candidate(candidate_options);
        }
        if (prepared) return prepared;
        last_error = prepared.error();
        if (last_error.code != ErrorCode::hardware_decode_unavailable && last_error.code != ErrorCode::unsupported_format)
            return prepared;
        candidate_options.attempts.push_back(std::string(to_string(backend)) + "/" +
            std::string(to_string(candidate_options.selection.decode_transfer)) + ": " + last_error.message);
    }
    for (const auto& attempt : candidate_options.attempts) last_error.message += "\n  " + attempt;
    return Result<PreparedVideoDecoderPair>::failure(std::move(last_error));
}
} // namespace crossvmaf::evaluation
namespace crossvmaf {
namespace {

[[nodiscard]] Result<void> validate_stream_options(const StreamOptions &options) {
    if (options.reference_path.empty() || options.distorted_path.empty()) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "EvaluationStream requires reference and distorted input paths"});
    }
    if (options.decoded_frame_queue_capacity == 0U || options.decoded_frame_queue_capacity > 64U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "decoded frame queue capacity must be between 1 and 64"});
    }
    if (options.decoder_threads_per_source > 64U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "decoder threads per source must be between 0 and 64"});
    }
    if (to_string(options.decode_mode) == "unknown") {
        return Result<void>::failure({ErrorCode::invalid_argument, "unsupported decode mode"});
    }
    const auto decode_validation = video::validate_decode_selection(options);
    if (!decode_validation) return decode_validation;
    if (options.model && to_string(*options.model) == "unknown") {
        return Result<void>::failure(
            {ErrorCode::unsupported_model, "unsupported CrossVMAF model variant"});
    }
    return Result<void>::success();
}

[[nodiscard]] evaluation::SegmentedInputToken
make_segmented_token(const evaluation::FramePairRecord &record) noexcept {
    return {record.frame_index(), record.frame_pair_storage(), record.reference_luma_storage()};
}

void append_frame_results(std::vector<FrameResult> &destination, std::vector<FrameResult> source) {
    destination.insert(destination.end(), std::make_move_iterator(source.begin()),
                       std::make_move_iterator(source.end()));
}

[[nodiscard]] StreamVideoCodec make_public_video_codec(video::VideoCodec codec) noexcept {
    switch (codec) {
    case video::VideoCodec::h264:
        return StreamVideoCodec::h264;
    case video::VideoCodec::hevc:
        return StreamVideoCodec::hevc;
    case video::VideoCodec::av1:
        return StreamVideoCodec::av1;
    case video::VideoCodec::vp8:
        return StreamVideoCodec::vp8;
    case video::VideoCodec::vp9:
        return StreamVideoCodec::vp9;
    case video::VideoCodec::mpeg1:
        return StreamVideoCodec::mpeg1;
    case video::VideoCodec::mpeg2:
        return StreamVideoCodec::mpeg2;
    case video::VideoCodec::mpeg4:
        return StreamVideoCodec::mpeg4;
    case video::VideoCodec::vc1:
        return StreamVideoCodec::vc1;
    case video::VideoCodec::flv1:
        return StreamVideoCodec::flv1;
    case video::VideoCodec::mjpeg:
        return StreamVideoCodec::mjpeg;
    }
    return StreamVideoCodec::h264;
}

[[nodiscard]] StreamInputStatistics
make_public_input_statistics(const video::VideoDecoderStatistics &statistics) {
    const StreamDecodeBackend decode_backend = statistics.decode_backend;
    StreamHardwarePixelFormat hardware_pixel_format = StreamHardwarePixelFormat::none;
    if (statistics.hardware_surface_format == video::HardwareSurfaceFormat::nv12) {
        hardware_pixel_format = StreamHardwarePixelFormat::nv12;
    } else if (statistics.hardware_surface_format == video::HardwareSurfaceFormat::p010) {
        hardware_pixel_format = StreamHardwarePixelFormat::p010;
    }
    return {
        make_public_video_codec(statistics.codec),
        decode_backend,
        hardware_pixel_format,
        statistics.packets_read,
        statistics.compressed_bytes,
        statistics.decoded_frames,
        statistics.decode_active_nanoseconds,
        statistics.decode_operation_count,
        statistics.output_queue_blocked_nanoseconds,
        statistics.output_queue_blocked_wait_count,
        statistics.queue_capacity,
        statistics.queued_frames,
        statistics.codec_thread_count,
        statistics.end_of_stream,
        statistics.cancellation_requested,
        statistics.decode,
    };
}

[[nodiscard]] std::size_t
prepare_result_buffer_capacity(std::size_t capacity,
                               evaluation::InitializationObserver *initialization_observer) {
    const auto admission = evaluation::observe_initialization(
        initialization_observer, evaluation::InitializationStage::result_buffer);
    if (!admission) throw std::runtime_error(admission.error().message);
    return capacity;
}

[[nodiscard]] Error unexpected_stream_exception(std::string operation,
                                                const std::exception *exception) {
    std::string message =
        "EvaluationStream " + std::move(operation) + " threw an unexpected C++ exception";
    if (exception != nullptr) {
        message += ": ";
        message += exception->what();
    }
    return {ErrorCode::backend_error, std::move(message)};
}

} // namespace

using namespace evaluation;

class EvaluationStream::Impl final {

  public:
    using CreationResult = Result<std::unique_ptr<Impl>>;

    static CreationResult create_prepared(const EvaluationGpuConfig &config,
                                          const scoring::ModelSnapshot &model_snapshot,
                                          std::unique_ptr<gpu::Device> context) {
        if (context == nullptr) {
            return CreationResult::failure(
                {ErrorCode::invalid_argument, "Vulkan session requires a prepared device"});
        }
        const auto input_validated = metrics::cambi::validate_input(config.frame);
        if (!input_validated) return CreationResult::failure(input_validated.error());
        const gpu::DeviceView device_view = context->device_view();
        const DeviceInfo device_info = context->device_info();
        auto hotspot_profiler = std::make_unique<telemetry::SessionTelemetry>(
            device_info, config.frame, config.telemetry_snapshot_sink);

        auto cambi = metrics::cambi::Vulkan::create(device_view, config.frame, model_snapshot.cambi,
                                                    &hotspot_profiler->cambi());
        if (!cambi) return CreationResult::failure(cambi.error());
        auto speed_chroma = metrics::speed_chroma::Vulkan::create(
            device_view, config.frame, model_snapshot.speed_chroma,
            &hotspot_profiler->speed_chroma());
        if (!speed_chroma) return CreationResult::failure(speed_chroma.error());
        auto adm3 = metrics::adm3::Vulkan::create(device_view, config.frame, model_snapshot.adm3,
                                                  &hotspot_profiler->adm3());
        if (!adm3) return CreationResult::failure(adm3.error());
        auto motion3 = metrics::motion3::Vulkan::create(
            device_view, config.frame, model_snapshot.motion3, &hotspot_profiler->motion3());
        if (!motion3) return CreationResult::failure(motion3.error());
        // Track resources added after Motion3's sequence-wide state so the
        // profiler can report storage that scales with admission depth.
        const VkDeviceSize initial_logical_buffer_bytes =
            device_view.memory_pool->statistics().used_bytes;

        constexpr std::size_t physical_frame_slot_count =
            session_detail::FrameSlotAdmissionController::physical_slot_count;
        std::vector<FrameSlot> frame_slots;
        frame_slots.reserve(physical_frame_slot_count);
        const std::size_t motion_history_distance =
            model_snapshot.motion3.five_frame_window ? 2U : 1U;

        for (std::size_t frame_slot_index = 0U; frame_slot_index < physical_frame_slot_count;
             ++frame_slot_index) {
            const auto admission = observe_initialization(
                config.initialization_observer, InitializationStage::frame_slot, frame_slot_index);
            if (!admission) return CreationResult::failure(admission.error());
            const auto cambi_slot = cambi.value()->append_slot();
            if (!cambi_slot) return CreationResult::failure(cambi_slot.error());
            const auto speed_chroma_slot = speed_chroma.value()->append_slot();
            if (!speed_chroma_slot) return CreationResult::failure(speed_chroma_slot.error());
            const auto adm3_slot = adm3.value()->append_slot();
            if (!adm3_slot) return CreationResult::failure(adm3_slot.error());
            const auto motion3_slot = motion3.value()->append_slot();
            if (!motion3_slot) return CreationResult::failure(motion3_slot.error());
            frame_slots.emplace_back();
            frame_slots.back().slot_index = frame_slot_index;
        }

        const std::size_t input_lease_depth = frame_slots.size();
        const auto input_pool_admission =
            observe_initialization(config.initialization_observer, InitializationStage::input_pool);
        if (!input_pool_admission) {
            return CreationResult::failure(input_pool_admission.error());
        }
        auto created_input_pool = evaluation::SegmentedInputLeasePool::create(
            device_view, config.frame, input_lease_depth, motion_history_distance,
            config.vulkan_image_input, &hotspot_profiler->runtime());
        if (!created_input_pool) {
            return CreationResult::failure(created_input_pool.error());
        }
        std::unique_ptr<evaluation::SegmentedInputLeasePool> input_pool =
            std::move(created_input_pool).value();
        std::unique_ptr<evaluation::InputStagingRing> input_staging_ring;
        if (!input_pool->input_memory_is_device_local()) {
            auto created_staging_ring = evaluation::InputStagingRing::create(
                device_view, config.frame, frame_slots.size(), motion_history_distance,
                config.vulkan_image_input, &hotspot_profiler->runtime(),
                hotspot_profiler->input_staging_operation_handle());
            if (!created_staging_ring) {
                return CreationResult::failure(created_staging_ring.error());
            }
            input_staging_ring = std::move(created_staging_ring).value();
        }

        const auto speed_chroma_batch = speed_chroma.value()->initialize_batch();
        if (!speed_chroma_batch) return CreationResult::failure(speed_chroma_batch.error());
        auto *speed_chroma_task_ring = speed_chroma.value()->batch();
        const auto adm3_batch = adm3.value()->initialize_batch();
        if (!adm3_batch) return CreationResult::failure(adm3_batch.error());
        auto *adm3_task_ring = adm3.value()->batch();
        const VkDeviceSize final_logical_buffer_bytes =
            device_view.memory_pool->statistics().used_bytes;
        if (final_logical_buffer_bytes < initial_logical_buffer_bytes) {
            return CreationResult::failure(
                {ErrorCode::backend_error, "Vulkan logical buffer accounting moved backwards"});
        }
        const VkDeviceSize depth_scaled_buffer_bytes =
            final_logical_buffer_bytes - initial_logical_buffer_bytes;

        // Each streaming feature lane can add one suffix wave beyond its
        // per-frame prefix transactions. SubmissionOwner sizes every logical
        // lane equally, so one additional entry covers both independent lanes.
        const std::size_t streaming_wave_submission_count =
            speed_chroma_task_ring == nullptr && adm3_task_ring == nullptr ? 0U : 1U;
        const std::size_t per_lane_pending_submission_count =
            frame_slots.size() * 2U + streaming_wave_submission_count;
        const std::size_t maximum_pending_submission_count =
            per_lane_pending_submission_count * feature_lane_count;
        auto submission_owner = gpu::SubmissionOwner::create(
            device_view, per_lane_pending_submission_count, feature_lane_count,
            lane_queue_modulus_from_environment(), hotspot_profiler->enabled());
        if (!submission_owner) {
            return CreationResult::failure(submission_owner.error());
        }

        hotspot_profiler->runtime().configure_vulkan_runtime(
            physical_frame_slot_count, frame_slots.size(), input_pool->spatial_depth(),
            input_pool->reference_luma_depth(), maximum_pending_submission_count,
            depth_scaled_buffer_bytes);
        hotspot_profiler->runtime().record_pending_submissions(0U);

        trace_memory_pool(device_view.memory_pool->statistics(), config.frame, frame_slots.size());
        return CreationResult::success(std::unique_ptr<Impl>(new Impl(
            device_info, model_snapshot.motion3.five_frame_window, std::move(context),
            std::move(submission_owner).value(), std::move(hotspot_profiler), std::move(input_pool),
            std::move(input_staging_ring), std::move(frame_slots), std::move(cambi).value(),
            std::move(speed_chroma).value(), std::move(adm3).value(), std::move(motion3).value())));
    }

    [[nodiscard]] Result<void> validate_push_state() const {
        if (failure_.has_value()) {
            return Result<void>::failure(*failure_);
        }
        if (cancelled_ || flushed_) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "Vulkan provider session cannot accept another frame"});
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> collect_before_push(evaluation::CompletionBatch &completions) {
        auto polled_completions = collect_signaled_completions();
        if (!polled_completions) {
            return Result<void>::failure(polled_completions.error());
        }
        append_completions(completions, std::move(polled_completions.value().completions));
        return Result<void>::success();
    }

    // Returns a lease-keyed free command/resource slot. Full active depth is
    // reported to the controller before one final non-blocking poll; empty
    // means the caller must retry without committing the published lease.
    [[nodiscard]] Result<std::optional<std::size_t>>
    admit_frame(const evaluation::SegmentedInputLeaseToken &token,
                evaluation::CompletionBatch &completions) {
        std::optional<std::size_t> frame_slot_index = select_lease_keyed_frame_slot(
            frame_slots_, token.spatial, frame_slot_admission_controller_.active_slot_count());
        if (!frame_slot_index.has_value()) {
            frame_slot_admission_controller_.observe_slot_busy();
            frame_slot_index = select_lease_keyed_frame_slot(
                frame_slots_, token.spatial, frame_slot_admission_controller_.active_slot_count());
        }
        if (!frame_slot_index.has_value()) {
            auto polled_completions = collect_signaled_completions();
            if (!polled_completions) {
                return Result<std::optional<std::size_t>>::failure(polled_completions.error());
            }
            append_completions(completions, std::move(polled_completions.value().completions));
            frame_slot_index = select_lease_keyed_frame_slot(
                frame_slots_, token.spatial, frame_slot_admission_controller_.active_slot_count());
        }
        if (!frame_slot_index.has_value()) {
            return Result<std::optional<std::size_t>>::success(std::nullopt);
        }
        if (speed_chroma_task_ring_ != nullptr) {
            // Ring pressure can outlive a single wave: an input-starvation
            // wave may still be in flight while later prefixes refill the
            // remaining slots. Drain until a slot is actually free.
            for (std::uint32_t drain_attempt = 0U; speed_chroma_task_ring_->free_slot_count() == 0U;
                 ++drain_attempt) {
                if (drain_attempt > speed_chroma_task_ring_->capacity() + 2U) {
                    return Result<std::optional<std::size_t>>::failure(
                        {ErrorCode::invalid_state,
                         "Speed Chroma suffix wave did not release ring pressure"});
                }
                const auto &speed_transactions =
                    pending_transactions_[session_detail::speed_chroma_lane];
                if (!speed_chroma_task_ring_->wave_in_flight()) {
                    if (speed_chroma_task_ring_->ready_task_count() == 0U) {
                        // Every slot is still awaiting its prefix. Retiring the
                        // newest prefix publishes those tasks so the next pass
                        // can build a wave from them.
                        if (speed_transactions.empty()) {
                            return Result<std::optional<std::size_t>>::failure(
                                {ErrorCode::invalid_state,
                                 "Speed Chroma ring pressure has no pending "
                                 "prefix or suffix work to drain"});
                        }
                        const auto drained = drain_speed_chroma_lane_through(
                            speed_transactions.back().ticket, completions);
                        if (!drained) {
                            return Result<std::optional<std::size_t>>::failure(drained.error());
                        }
                        continue;
                    }
                    const auto wave_submitted = submit_ready_speed_chroma_wave();
                    if (!wave_submitted) {
                        return Result<std::optional<std::size_t>>::failure(wave_submitted.error());
                    }
                }
                const auto wave_ticket = newest_speed_chroma_wave_ticket();
                if (!wave_ticket.has_value()) {
                    return Result<std::optional<std::size_t>>::failure(
                        {ErrorCode::invalid_state,
                         "Speed Chroma ring pressure could not start a suffix "
                         "wave"});
                }
                const auto drained = drain_speed_chroma_lane_through(*wave_ticket, completions);
                if (!drained) {
                    return Result<std::optional<std::size_t>>::failure(drained.error());
                }
            }
        }
        const auto adm3_drained = drain_adm3_ring_pressure(completions);
        if (!adm3_drained) {
            return Result<std::optional<std::size_t>>::failure(adm3_drained.error());
        }
        return Result<std::optional<std::size_t>>::success(frame_slot_index);
    }

    [[nodiscard]] std::optional<gpu::SubmissionTicket>
    newest_speed_chroma_wave_ticket() const noexcept {
        const auto &speed_transactions = pending_transactions_[session_detail::speed_chroma_lane];
        for (auto transaction = speed_transactions.rbegin();
             transaction != speed_transactions.rend(); ++transaction) {
            if (transaction->kind == FeatureTransactionKind::speed_chroma_wave) {
                return transaction->ticket;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] Result<void> drain_lane_through(std::uint32_t lane, gpu::SubmissionTicket ticket,
                                                  evaluation::CompletionBatch &completions) {
        auto retirements = submission_owner_->wait_for_prefix(lane, ticket);
        if (!retirements) {
            return Result<void>::failure(retirements.error());
        }
        auto retired_completions = process_retirements(retirements.value());
        if (!retired_completions) {
            return Result<void>::failure(retired_completions.error());
        }
        append_completions(completions, std::move(retired_completions).value());
        return Result<void>::success();
    }

    [[nodiscard]] Result<void>
    drain_speed_chroma_lane_through(gpu::SubmissionTicket ticket,
                                    evaluation::CompletionBatch &completions) {
        return drain_lane_through(session_detail::speed_chroma_lane, ticket, completions);
    }

    [[nodiscard]] std::optional<gpu::SubmissionTicket> newest_adm3_wave_ticket() const noexcept {
        const auto &adm3_transactions = pending_transactions_[session_detail::adm3_lane];
        for (auto transaction = adm3_transactions.rbegin(); transaction != adm3_transactions.rend();
             ++transaction) {
            if (transaction->kind == FeatureTransactionKind::adm3_wave) {
                return transaction->ticket;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] Result<void> drain_adm3_lane_through(gpu::SubmissionTicket ticket,
                                                       evaluation::CompletionBatch &completions) {
        return drain_lane_through(session_detail::adm3_lane, ticket, completions);
    }

    [[nodiscard]] Result<void> drain_adm3_ring_pressure(evaluation::CompletionBatch &completions) {
        if (adm3_task_ring_ == nullptr) return Result<void>::success();

        for (std::uint32_t drain_attempt = 0U; adm3_task_ring_->free_slot_count() == 0U;
             ++drain_attempt) {
            if (drain_attempt > adm3_task_ring_->capacity() + 2U) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state, "ADM3 suffix wave did not release ring pressure"});
            }
            const auto &adm3_transactions = pending_transactions_[session_detail::adm3_lane];
            if (!adm3_task_ring_->wave_in_flight()) {
                if (adm3_task_ring_->ready_task_count() == 0U) {
                    if (adm3_transactions.empty()) {
                        return Result<void>::failure({ErrorCode::invalid_state,
                                                      "ADM3 ring pressure has no pending prefix or "
                                                      "suffix work to drain"});
                    }
                    const auto drained =
                        drain_adm3_lane_through(adm3_transactions.back().ticket, completions);
                    if (!drained) return drained;
                    continue;
                }
                const auto wave_submitted = submit_ready_adm3_wave();
                if (!wave_submitted) return wave_submitted;
            }
            const auto wave_ticket = newest_adm3_wave_ticket();
            if (!wave_ticket.has_value()) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state, "ADM3 ring pressure could not start a suffix wave"});
            }
            const auto drained = drain_adm3_lane_through(*wave_ticket, completions);
            if (!drained) return drained;
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<evaluation::CompletionBatch> submit_committed_input(
        std::size_t frame_slot_index, const evaluation::SegmentedInputLeaseToken &token,
        const video::UploadedFramePair &uploaded, evaluation::CompletionBatch completions) {
        const std::uint64_t index = token.frame_index;
        if (frame_slot_index >= frame_slots_.size() || frame_slots_[frame_slot_index].in_use) {
            return fail<evaluation::CompletionBatch>(
                {ErrorCode::invalid_state, "Vulkan admission lost its lease-keyed frame slot"});
        }
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        frame_slot.begin_frame(index);
        if (frames_ == nullptr) {
            return fail<evaluation::CompletionBatch>(
                {ErrorCode::invalid_state, "evaluation frame records are not bound"});
        }
        auto frame_record = frames_->find_frame(index);
        if (!frame_record) return fail<evaluation::CompletionBatch>(frame_record.error());
        frame_record.value()->spatial_input = token.spatial;
        frame_record.value()->reference_luma_input = token.reference_luma;
        if (input_hash_trace_enabled() && !uploaded.has_hardware_frames()) {
            const auto reference_hashes = hash_gpu_frame_planes(uploaded.reference);
            const auto distorted_hashes = hash_gpu_frame_planes(uploaded.distorted);
            if (!reference_hashes || !distorted_hashes) {
                return fail<evaluation::CompletionBatch>(
                    !reference_hashes ? reference_hashes.error() : distorted_hashes.error());
            }
            std::fprintf(stderr,
                         "[crossvmaf-input-hash] frame=%llu "
                         "luma_segment=%u luma_slot=%u luma_generation=%llu "
                         "reference_y=%016llx reference_u=%016llx "
                         "reference_v=%016llx distorted_y=%016llx "
                         "distorted_u=%016llx distorted_v=%016llx\n",
                         static_cast<unsigned long long>(index), token.reference_luma.segment_index,
                         token.reference_luma.slot_index,
                         static_cast<unsigned long long>(token.reference_luma.generation),
                         static_cast<unsigned long long>(reference_hashes.value()[0U]),
                         static_cast<unsigned long long>(reference_hashes.value()[1U]),
                         static_cast<unsigned long long>(reference_hashes.value()[2U]),
                         static_cast<unsigned long long>(distorted_hashes.value()[0U]),
                         static_cast<unsigned long long>(distorted_hashes.value()[1U]),
                         static_cast<unsigned long long>(distorted_hashes.value()[2U]));
        }

        video::UploadedFramePair feature_input = uploaded;
        if (input_staging_ring_ != nullptr) {
            auto staged = stage_input(frame_slot_index, token, uploaded);
            if (!staged) {
                return fail<evaluation::CompletionBatch>(staged.error().with_context(
                    "GPU input staging", ErrorInput::both, index));
            }
            feature_input = staged.value();
        }
        const video::GpuFrameView &gpu_reference = feature_input.reference;
        const video::GpuFrameView &gpu_distorted = feature_input.distorted;

        auto submitted = [&]() -> Result<void> {
            const auto context = [index](Result<void> result, const char *operation) {
                if (!result) return Result<void>::failure(result.error().with_context(
                    operation, ErrorInput::none, index));
                return result;
            };
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::feature_record_submit);
            auto feature_submitted = context(
                submit_cambi_prefix(frame_slot_index, gpu_distorted, token), "CAMBI prefix submit");
            if (!feature_submitted) return feature_submitted;
            feature_submitted = context(
                submit_speed_chroma(frame_slot_index, gpu_reference, gpu_distorted, token),
                "Speed Chroma submit");
            if (!feature_submitted) return feature_submitted;
            feature_submitted = context(
                submit_adm3(frame_slot_index, gpu_reference, gpu_distorted, token), "ADM3 submit");
            if (!feature_submitted) return feature_submitted;
            feature_submitted = context(
                submit_motion3(frame_slot_index, gpu_reference, token), "Motion3 submit");
            if (!feature_submitted) return feature_submitted;
            feature_submitted = context(submit_cambi_suffix(frame_slot_index), "CAMBI suffix submit");
            if (!feature_submitted) return feature_submitted;
            return submit_saturated_suffix_waves();
        }();
        if (!submitted) {
            return fail<evaluation::CompletionBatch>(submitted.error());
        }
        if (input_staging_ring_ == nullptr) {
            const auto input_submitted = input_pool_->finish_submission(token);
            if (!input_submitted) {
                return fail<evaluation::CompletionBatch>(input_submitted.error());
            }
        }
        ++next_submission_index_;

        auto polled_completions = collect_signaled_completions();
        if (!polled_completions) {
            return fail<evaluation::CompletionBatch>(polled_completions.error());
        }
        append_completions(completions, std::move(polled_completions).value().completions);
        return Result<evaluation::CompletionBatch>::success(std::move(completions));
    }

    Result<std::optional<evaluation::SegmentedWritableInputLease>>
    try_reserve_segmented_writable_input(std::uint64_t index) {
        const auto state_validation = validate_push_state();
        if (!state_validation) {
            return Result<std::optional<evaluation::SegmentedWritableInputLease>>::failure(
                state_validation.error());
        }
        // Lease recycling backpressure is independent of whether the decoder
        // has another pair. In particular, it must not drain a suffix wave.
        return input_pool_->try_reserve(index);
    }

    Result<void> publish_segmented_input(const evaluation::SegmentedInputLeaseToken &token) {
        return input_pool_->publish(token);
    }

    void abandon_segmented_input(const evaluation::SegmentedInputLeaseToken &token) noexcept {
        input_pool_->abandon(token);
    }

    Result<SegmentedPushOutcome>
    push_ready_segmented_input(const evaluation::SegmentedInputLeaseToken &token) {
        const auto state_validation = validate_push_state();
        if (!state_validation) {
            return Result<SegmentedPushOutcome>::failure(state_validation.error());
        }
        if (token.frame_index != next_submission_index_) {
            return Result<SegmentedPushOutcome>::failure(
                {ErrorCode::invalid_argument,
                 "submitted input lease indices must be contiguous and start at zero"});
        }
        [[maybe_unused]] auto provider_measurement =
            hotspot_profiler_->measure(telemetry::SessionStage::provider_push);
        evaluation::CompletionBatch completions;
        const auto collected = collect_before_push(completions);
        if (!collected) {
            return fail<SegmentedPushOutcome>(collected.error());
        }

        const auto admitted = admit_frame(token, completions);
        if (!admitted) {
            return fail<SegmentedPushOutcome>(admitted.error());
        }
        if (!admitted.value().has_value()) {
            return Result<SegmentedPushOutcome>::success({std::move(completions), true});
        }

        auto committed = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::input_commit);
            return input_pool_->commit_published(token);
        }();
        if (!committed) {
            return fail<SegmentedPushOutcome>(committed.error());
        }
        auto submitted = submit_committed_input(*admitted.value(), token, committed.value(),
                                                std::move(completions));
        if (!submitted) {
            return fail<SegmentedPushOutcome>(submitted.error());
        }
        frame_slot_admission_controller_.observe_frame_submitted();
        return Result<SegmentedPushOutcome>::success({std::move(submitted).value(), false});
    }

    Result<SegmentedPollOutcome> poll_segmented_completions() {
        const auto state_validation = validate_push_state();
        if (!state_validation) {
            return Result<SegmentedPollOutcome>::failure(state_validation.error());
        }
        auto completions = collect_signaled_completions();
        if (!completions) {
            return fail<SegmentedPollOutcome>(completions.error());
        }
        return completions;
    }

    Result<SegmentedPollOutcome> wait_segmented_progress() {
        const auto state_validation = validate_push_state();
        if (!state_validation) {
            return Result<SegmentedPollOutcome>::failure(state_validation.error());
        }
        if (pending_transaction_count() == 0U) {
            return Result<SegmentedPollOutcome>::success({});
        }

        auto retirements = submission_owner_->wait_for_any_lane_prefix();
        if (!retirements) {
            return fail<SegmentedPollOutcome>(retirements.error().with_context("GPU completion wait"));
        }
        auto completions = process_retirements(retirements.value());
        if (!completions) {
            return fail<SegmentedPollOutcome>(completions.error());
        }
        return Result<SegmentedPollOutcome>::success(
            {std::move(completions).value(), !retirements.value().empty()});
    }

    [[nodiscard]] bool has_pending_segmented_work() const noexcept {
        return pending_transaction_count() != 0U;
    }

    Result<SegmentedPollOutcome> poll_segmented_input_starvation() {
        auto completions = poll_segmented_completions();
        if (!completions) return completions;

        const std::size_t pending_before = pending_transaction_count();
        // Only the decoded-input layer calls this path after observing that no
        // completed pair is currently deliverable. Frame-slot and lease waits
        // use ordinary polling and cannot reach this trigger.
        Result<void> first_wave_submitted =
            prefer_adm3_wave_next_ ? submit_ready_adm3_wave() : submit_ready_speed_chroma_wave();
        if (!first_wave_submitted) {
            return fail<SegmentedPollOutcome>(first_wave_submitted.error());
        }
        Result<void> second_wave_submitted =
            prefer_adm3_wave_next_ ? submit_ready_speed_chroma_wave() : submit_ready_adm3_wave();
        if (!second_wave_submitted) {
            return fail<SegmentedPollOutcome>(second_wave_submitted.error());
        }
        prefer_adm3_wave_next_ = !prefer_adm3_wave_next_;
        completions.value().made_progress =
            completions.value().made_progress || pending_transaction_count() > pending_before;
        return completions;
    }

    [[nodiscard]] Result<void> ensure_segmented_input_capacity(std::size_t pair_target) {
        const auto state_validation = validate_push_state();
        if (!state_validation) return state_validation;
        return input_pool_->ensure_pair_capacity(pair_target);
    }

    [[nodiscard]] std::size_t segmented_input_capacity() const noexcept {
        return std::min(input_pool_->spatial_depth(), input_pool_->reference_luma_depth());
    }

    [[nodiscard]] std::size_t frame_slot_count() const noexcept { return frame_slots_.size(); }

    [[nodiscard]] std::size_t speed_chroma_backlog_capacity() const noexcept {
        if (speed_chroma_task_ring_ != nullptr) {
            return speed_chroma_task_ring_->capacity();
        }
        return 1U;
    }

    [[nodiscard]] std::size_t adm3_backlog_capacity() const noexcept {
        if (adm3_task_ring_ != nullptr) {
            return adm3_task_ring_->capacity();
        }
        return 1U;
    }

    [[nodiscard]] std::size_t maximum_streaming_backlog_capacity() const noexcept {
        return std::max(speed_chroma_backlog_capacity(), adm3_backlog_capacity());
    }

    [[nodiscard]] Result<std::size_t>
    prepare_motion_flush_slot(evaluation::CompletionBatch &completions) {
        std::vector<bool> motion_slot_pending(frame_slots_.size(), false);
        std::optional<std::size_t> earliest_motion_slot;
        gpu::SubmissionTicket earliest_motion_ticket;
        for (const PendingTransaction &transaction :
             pending_transactions_[session_detail::motion3_lane]) {
            if (transaction.kind != FeatureTransactionKind::motion3) {
                continue;
            }
            if (transaction.frame_slot_index >= frame_slots_.size()) {
                return Result<std::size_t>::failure(
                    {ErrorCode::invalid_state,
                     "Vulkan Motion3 transaction references an invalid frame slot"});
            }
            motion_slot_pending[transaction.frame_slot_index] = true;
            if (!earliest_motion_slot.has_value()) {
                earliest_motion_slot = transaction.frame_slot_index;
                earliest_motion_ticket = transaction.ticket;
            }
        }

        const auto available_slot =
            std::find(motion_slot_pending.begin(), motion_slot_pending.end(), false);
        if (available_slot != motion_slot_pending.end()) {
            return Result<std::size_t>::success(
                static_cast<std::size_t>(available_slot - motion_slot_pending.begin()));
        }
        if (!earliest_motion_slot.has_value()) {
            return Result<std::size_t>::failure(
                {ErrorCode::invalid_state,
                 "Vulkan flush could not find a reusable Motion3 command slot"});
        }

        // Only the command context selected for the tail must be reusable.
        // Other feature lanes keep running while this narrow prerequisite is
        // retired, and the tail is submitted before the unified drain below.
        auto retired = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::flush_drain);
            return submission_owner_->wait_for_prefix(session_detail::motion3_lane,
                                                      earliest_motion_ticket);
        }();
        if (!retired) {
            return Result<std::size_t>::failure(retired.error());
        }
        auto retired_completions = process_retirements(retired.value());
        if (!retired_completions) {
            return Result<std::size_t>::failure(retired_completions.error());
        }
        append_completions(completions, std::move(retired_completions).value());
        return Result<std::size_t>::success(*earliest_motion_slot);
    }

    [[nodiscard]] Result<void> submit_motion_flush(std::size_t frame_slot_index,
                                                   evaluation::CompletionBatch &completions) {
        FrameSlot &flush_slot = frame_slots_[frame_slot_index];
        [[maybe_unused]] auto measurement =
            hotspot_profiler_->measure(telemetry::SessionStage::feature_record_submit);
        auto recorded_motion_command = [&] {
            [[maybe_unused]] auto record_measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::motion3_record);
            return motion3_->slot(flush_slot.slot_index).record_flush();
        }();
        if (!recorded_motion_command) {
            return Result<void>::failure(recorded_motion_command.error());
        }
        if (!recorded_motion_command.value().has_value()) {
            auto motion_scores =
                motion3_->slot(flush_slot.slot_index).complete_without_submission();
            if (!motion_scores) {
                return Result<void>::failure(motion_scores.error());
            }
            std::vector<evaluation::MotionFeature> tail_completions =
                convert_motion_scores(std::move(motion_scores).value());
            completions.motion.insert(completions.motion.end(),
                                      std::make_move_iterator(tail_completions.begin()),
                                      std::make_move_iterator(tail_completions.end()));
            return Result<void>::success();
        }

        const std::array command_buffers{*recorded_motion_command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto submit_measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::motion3_submit);
            return submission_owner_->submit(session_detail::motion3_lane, command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                motion3_->slot(flush_slot.slot_index).cancel();
            }
            return Result<void>::failure(submitted.error());
        }
        auto committed = motion3_->slot(flush_slot.slot_index).commit_submission(submitted.value());
        if (!committed) return committed;

        PendingTransaction transaction{};
        transaction.ticket = submitted.value();
        transaction.frame_slot_index = frame_slot_index;
        transaction.kind = FeatureTransactionKind::motion3_flush;
        pending_transactions_[session_detail::motion3_lane].push_back(transaction);
        if (hotspot_profiler_->enabled()) {
            hotspot_profiler_->runtime().record_pending_submissions(pending_transaction_count());
        }
        return Result<void>::success();
    }

    Result<evaluation::CompletionBatch> flush_gpu() {
        if (failure_.has_value()) {
            return Result<evaluation::CompletionBatch>::failure(*failure_);
        }
        if (cancelled_ || flushed_) {
            return Result<evaluation::CompletionBatch>::failure(
                {ErrorCode::invalid_state, "Vulkan session flush may only be called once"});
        }
        flushed_ = true;
        input_pool_->seal_end_of_stream();
        evaluation::CompletionBatch completions;
        std::optional<telemetry::SessionMeasurement> session_measurement{
            hotspot_profiler_->measure(telemetry::SessionStage::provider_flush)};

        auto streaming_ready_on_lane = [this](std::size_t lane) {
            const bool speed_work_remains = lane == session_detail::speed_chroma_lane &&
                                            speed_chroma_task_ring_ != nullptr &&
                                            speed_chroma_task_ring_->ready_task_count() > 0U;
            const bool adm3_work_remains = lane == session_detail::adm3_lane &&
                                           adm3_task_ring_ != nullptr &&
                                           adm3_task_ring_->ready_task_count() > 0U;
            return speed_work_remains || adm3_work_remains;
        };

        // Submit every currently recordable suffix before the general drain.
        // Motion3 may need one narrow lane retirement to reuse a command
        // context, but its tail still enters the queue while other feature
        // lanes remain in flight.
        const auto initial_speed_wave = submit_ready_speed_chroma_wave();
        if (!initial_speed_wave) {
            return fail<evaluation::CompletionBatch>(initial_speed_wave.error());
        }
        const auto initial_adm3_wave = submit_ready_adm3_wave();
        if (!initial_adm3_wave) {
            return fail<evaluation::CompletionBatch>(initial_adm3_wave.error());
        }
        const auto motion_flush_slot = prepare_motion_flush_slot(completions);
        if (!motion_flush_slot) {
            return fail<evaluation::CompletionBatch>(motion_flush_slot.error());
        }
        const auto motion_flush_submitted =
            submit_motion_flush(motion_flush_slot.value(), completions);
        if (!motion_flush_submitted) {
            return fail<evaluation::CompletionBatch>(
                motion_flush_submitted.error().with_context("Motion3 flush"));
        }

        for (;;) {
            const auto speed_wave = submit_ready_speed_chroma_wave();
            if (!speed_wave) {
                return fail<evaluation::CompletionBatch>(speed_wave.error());
            }
            const auto adm3_wave = submit_ready_adm3_wave();
            if (!adm3_wave) {
                return fail<evaluation::CompletionBatch>(adm3_wave.error());
            }

            bool drained_any_lane = false;
            for (std::size_t lane = 0U; lane < feature_lane_count; ++lane) {
                if (pending_transactions_[lane].empty()) continue;
                drained_any_lane = true;
                const gpu::SubmissionTicket final_ticket =
                    pending_transactions_[lane].back().ticket;
                auto retired = [&] {
                    [[maybe_unused]] auto measurement =
                        hotspot_profiler_->measure(telemetry::SessionStage::flush_drain);
                    return submission_owner_->wait_for_prefix(lane, final_ticket);
                }();
                if (!retired) {
                    return fail<evaluation::CompletionBatch>(retired.error());
                }
                auto retired_completions = process_retirements(retired.value());
                if (!retired_completions) {
                    return fail<evaluation::CompletionBatch>(retired_completions.error());
                }
                append_completions(completions, std::move(retired_completions).value());
            }

            bool ready_remains = false;
            for (std::size_t lane = 0U; lane < feature_lane_count; ++lane) {
                if (streaming_ready_on_lane(lane)) {
                    ready_remains = true;
                    break;
                }
            }
            if (!drained_any_lane && !ready_remains) {
                break;
            }
            if (!drained_any_lane && ready_remains) {
                return fail<evaluation::CompletionBatch>(
                    {ErrorCode::invalid_state,
                     "Vulkan flush could not submit remaining streaming waves"});
            }
        }
        if (frames_ != nullptr && !frames_->all_spatial_ready()) {
            return fail<evaluation::CompletionBatch>(
                {ErrorCode::invalid_state, "Vulkan flush left incomplete spatial frames"});
        }

        // Finalize only after the session flush measurement has recorded its
        // full parent interval, including tail submission and retirement.
        session_measurement.reset();
        configure_input_lease_profile();
        hotspot_profiler_->finalize(true);
        report_finalized_ = true;
        return Result<evaluation::CompletionBatch>::success(std::move(completions));
    }

    void cancel_gpu() noexcept {
        if (cancelled_) return;
        cancelled_ = true;
        input_pool_->cancel_producers();
        if (submission_owner_ != nullptr) {
            const auto drained = submission_owner_->drain();
            if (drained) {
                if (speed_chroma_task_ring_ != nullptr) {
                    speed_chroma_task_ring_->cancel_after_device_drain();
                }
                if (adm3_task_ring_ != nullptr) {
                    adm3_task_ring_->cancel_after_device_drain();
                }
            }
        }
        if (!report_finalized_) {
            configure_input_lease_profile();
            hotspot_profiler_->finalize(false);
            report_finalized_ = true;
        }
    }

    ~Impl() {
        if (frames_) frames_->cancel();
        input_request_cancel();
        static_cast<void>(input_drain());
        cancel_gpu();
        if (!report_finalized_) {
            configure_input_lease_profile();
            hotspot_profiler_->finalize(false);
            report_finalized_ = true;
        }
        const gpu::DeviceResourceCleanupOutcome cleanup_outcome =
            submission_owner_ != nullptr ? submission_owner_->prepare_for_destruction()
                                         : gpu::DeviceResourceCleanupOutcome::safe_to_destroy;
        if (cleanup_outcome == gpu::DeviceResourceCleanupOutcome::safe_to_destroy) {
            if (speed_chroma_task_ring_ != nullptr) {
                speed_chroma_task_ring_->cancel_after_device_drain();
            }
            if (adm3_task_ring_ != nullptr) {
                adm3_task_ring_->cancel_after_device_drain();
            }
        }
        gpu::apply_device_resource_cleanup(submission_owner_, cleanup_outcome);
        gpu::apply_device_resource_cleanup(input_staging_ring_, cleanup_outcome);
        gpu::apply_device_resource_cleanup(cambi_, cleanup_outcome);
        gpu::apply_device_resource_cleanup(speed_chroma_, cleanup_outcome);
        gpu::apply_device_resource_cleanup(adm3_, cleanup_outcome);
        gpu::apply_device_resource_cleanup(motion3_, cleanup_outcome);
        frame_slots_.clear();
        gpu::apply_device_resource_cleanup(input_pool_, cleanup_outcome);
        // Native decode images must be retained with abandoned GPU consumers.
        gpu::apply_device_resource_cleanup(frames_, cleanup_outcome);
        gpu::apply_device_resource_cleanup(context_, cleanup_outcome);
    }

    const DeviceInfo &device_info() const noexcept { return device_; }

  private:
    void configure_input_lease_profile() noexcept {
        if (input_pool_ == nullptr || hotspot_profiler_ == nullptr) return;
        const evaluation::SegmentedInputLeasePoolStatistics statistics = input_pool_->statistics();
        hotspot_profiler_->runtime().configure_input_lease_runtime({
            true,
            static_cast<std::uint64_t>(input_pool_->logical_buffer_bytes()),
            statistics.reservation_count,
            statistics.spatial_release_count,
            statistics.reference_luma_release_count,
            static_cast<std::uint64_t>(statistics.maximum_active_spatial_lease_count),
            static_cast<std::uint64_t>(statistics.maximum_active_reference_luma_lease_count),
            statistics.spatial_hold_total,
            statistics.reference_luma_hold_total,
            statistics.spatial_hold_maximum,
            statistics.reference_luma_hold_maximum,
        });
    }

    Impl(DeviceInfo device, bool motion3_five_frame_window, std::unique_ptr<gpu::Device> context,
         std::unique_ptr<gpu::SubmissionOwner> submission_owner,
         std::unique_ptr<telemetry::SessionTelemetry> hotspot_profiler,
         std::unique_ptr<evaluation::SegmentedInputLeasePool> input_pool,
         std::unique_ptr<evaluation::InputStagingRing> input_staging_ring,
         std::vector<FrameSlot> frame_slots, std::unique_ptr<metrics::cambi::Vulkan> cambi,
         std::unique_ptr<metrics::speed_chroma::Vulkan> speed_chroma,
         std::unique_ptr<metrics::adm3::Vulkan> adm3,
         std::unique_ptr<metrics::motion3::Vulkan> motion3) noexcept
        : device_(std::move(device)), motion3_five_frame_window_(motion3_five_frame_window),
          context_(std::move(context)), submission_owner_(std::move(submission_owner)),
          hotspot_profiler_(std::move(hotspot_profiler)), input_pool_(std::move(input_pool)),
          input_staging_ring_(std::move(input_staging_ring)), frame_slots_(std::move(frame_slots)),
          cambi_(std::move(cambi)), speed_chroma_(std::move(speed_chroma)), adm3_(std::move(adm3)),
          motion3_(std::move(motion3)) {
        speed_chroma_task_ring_ = speed_chroma_->batch();
        adm3_task_ring_ = adm3_->batch();
    }

    template <typename Value> [[nodiscard]] Result<Value> fail(Error error) {
        if (!failure_.has_value()) failure_.emplace(std::move(error));
        return Result<Value>::failure(*failure_);
    }

    [[nodiscard]] Result<void> complete_resource_transaction(std::size_t frame_slot_index,
                                                             std::uint64_t frame_index) {
        if (frame_slot_index >= frame_slots_.size()) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "Vulkan transaction references an invalid frame slot"});
        }
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        if (!frame_slot.in_use || frame_slot.frame_index != frame_index ||
            frame_slot.pending_resource_transaction_count == 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "Vulkan frame slot retired an invalid resource transaction"});
        }
        --frame_slot.pending_resource_transaction_count;
        if (frame_slot.pending_resource_transaction_count == 0U) {
            frame_slot.in_use = false;
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> register_frame_transaction(
        std::size_t frame_slot_index, FeatureTransactionKind kind, gpu::SubmissionTicket ticket,
        std::uint64_t frame_index,
        std::span<const evaluation::SegmentedInputLeaseConsumer> input_consumers,
        std::uint32_t task_slot = 0U, std::uint32_t task_generation = 0U) {
        if (input_consumers.size() > 2U) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "Vulkan feature transaction exceeds its fixed input consumer capacity"});
        }
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        PendingTransaction transaction{};
        transaction.ticket = ticket;
        transaction.frame_slot_index = frame_slot_index;
        transaction.frame_index = frame_index;
        transaction.kind = kind;
        transaction.task_slot = task_slot;
        transaction.task_generation = task_generation;
        transaction.input_consumer_count = static_cast<std::uint8_t>(input_consumers.size());
        for (std::size_t consumer_index = 0U; consumer_index < input_consumers.size();
             ++consumer_index) {
            transaction.input_consumers[consumer_index] = input_consumers[consumer_index];
        }
        pending_transactions_[feature_lane_of(kind)].push_back(transaction);
        ++frame_slot.pending_resource_transaction_count;
        const auto frame_record = frames_->find_frame(frame_index);
        if (!frame_record) return Result<void>::failure(frame_record.error());
        frame_record.value()->submissions[feature_lane_of(kind)] = ticket;
        if (hotspot_profiler_->enabled()) {
            hotspot_profiler_->runtime().record_pending_submissions(pending_transaction_count());
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> register_streaming_wave_transaction(gpu::SubmissionTicket ticket,
                                                                   FeatureTransactionKind kind) {
        if (kind != FeatureTransactionKind::speed_chroma_wave &&
            kind != FeatureTransactionKind::adm3_wave) {
            return Result<void>::failure({ErrorCode::invalid_argument,
                                          "Vulkan streaming wave transaction has an invalid kind"});
        }
        PendingTransaction transaction{};
        transaction.ticket = ticket;
        transaction.kind = kind;
        pending_transactions_[feature_lane_of(kind)].push_back(transaction);
        if (hotspot_profiler_->enabled()) {
            hotspot_profiler_->runtime().record_pending_submissions(pending_transaction_count());
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<gpu::SubmissionTicket>
    submit_frame_commands(std::size_t frame_slot_index, std::size_t feature_lane,
                          std::span<const VkCommandBuffer> command_buffers) {
        if (input_staging_ring_ == nullptr || feature_lane == session_detail::input_staging_lane) {
            return submission_owner_->submit(feature_lane, command_buffers);
        }
        const std::size_t wait_semaphore_index =
            feature_lane < session_detail::input_staging_lane ? feature_lane : feature_lane - 1U;
        const VkSemaphore wait_semaphore =
            input_staging_ring_->wait_semaphore(frame_slot_index, wait_semaphore_index);
        if (wait_semaphore == VK_NULL_HANDLE) {
            return Result<gpu::SubmissionTicket>::failure(
                {ErrorCode::invalid_state,
                 "staged input feature submission lost its wait semaphore"});
        }
        const std::array wait_semaphores{wait_semaphore};
        const std::array<VkPipelineStageFlags, 1U> wait_stages{
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};
        return submission_owner_->submit(feature_lane, command_buffers,
                                         {wait_semaphores, wait_stages, {}});
    }

    [[nodiscard]] Result<video::UploadedFramePair>
    stage_input(std::size_t frame_slot_index,
                const evaluation::SegmentedInputLeaseToken &input_token,
                const video::UploadedFramePair &source) {
        evaluation::RecordedInputStagingCopy recorded_input{};
        gpu::SubmissionTicket submission_ticket{};
        const auto feature_ready_semaphores =
            input_staging_ring_->signal_semaphores(frame_slot_index);
        if (source.has_hardware_frames()) {
            auto reference_lock = source.hardware_reference->lock_vulkan_frame();
            if (!reference_lock) {
                return Result<video::UploadedFramePair>::failure(reference_lock.error());
            }
            auto distorted_lock = source.hardware_distorted->lock_vulkan_frame();
            if (!distorted_lock) {
                return Result<video::UploadedFramePair>::failure(distorted_lock.error());
            }

            auto synchronization = video::HardwareFrameSynchronization::create(
                reference_lock.value().access(), distorted_lock.value().access(),
                context_->device_view().unified_video_image_layouts, feature_ready_semaphores);
            if (!synchronization) {
                return Result<video::UploadedFramePair>::failure(synchronization.error());
            }

            auto recorded = input_staging_ring_->record_conversion(
                frame_slot_index, input_token.frame_index, reference_lock.value().access(),
                distorted_lock.value().access());
            if (!recorded) {
                return Result<video::UploadedFramePair>::failure(recorded.error());
            }
            const std::array command_buffers{recorded.value().command_buffer};
            auto submitted =
                submission_owner_->submit(session_detail::input_staging_lane, command_buffers,
                                          synchronization.value().submission());
            if (!submitted) {
                if (!submission_owner_->last_submit_reached_queue()) {
                    input_staging_ring_->cancel(frame_slot_index);
                }
                return Result<video::UploadedFramePair>::failure(submitted.error());
            }
            synchronization.value().mark_submitted(reference_lock.value().access(),
                                                   context_->device_view().queue_family);
            synchronization.value().mark_submitted(distorted_lock.value().access(),
                                                   context_->device_view().queue_family);
            recorded_input = recorded.value();
            submission_ticket = submitted.value();
        } else {
            auto recorded = input_staging_ring_->record_copy(frame_slot_index, source);
            if (!recorded) {
                return Result<video::UploadedFramePair>::failure(recorded.error());
            }
            const std::array command_buffers{recorded.value().command_buffer};
            auto submitted =
                submission_owner_->submit(session_detail::input_staging_lane, command_buffers,
                                          {{}, {}, feature_ready_semaphores});
            if (!submitted) {
                if (!submission_owner_->last_submit_reached_queue()) {
                    input_staging_ring_->cancel(frame_slot_index);
                }
                return Result<video::UploadedFramePair>::failure(submitted.error());
            }
            recorded_input = recorded.value();
            submission_ticket = submitted.value();
        }
        auto marked = input_staging_ring_->mark_submitted(frame_slot_index, submission_ticket);
        if (!marked) {
            return Result<video::UploadedFramePair>::failure(marked.error());
        }
        auto input_consumers =
            input_pool_->register_consumers(input_token, true, true, std::nullopt);
        if (!input_consumers) {
            return Result<video::UploadedFramePair>::failure(input_consumers.error());
        }
        auto registered = register_frame_transaction(
            frame_slot_index, FeatureTransactionKind::input_staging, submission_ticket,
            input_token.frame_index, input_consumers.value());
        if (!registered) {
            return Result<video::UploadedFramePair>::failure(registered.error());
        }
        auto source_submitted = input_pool_->finish_staging_submission(input_token);
        if (!source_submitted) {
            return Result<video::UploadedFramePair>::failure(source_submitted.error());
        }
        return Result<video::UploadedFramePair>::success(recorded_input.uploaded);
    }

    [[nodiscard]] Result<void>
    submit_cambi_prefix(std::size_t frame_slot_index, const video::GpuFrameView &distorted,
                        const evaluation::SegmentedInputLeaseToken &input_token) {
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        auto command = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::cambi_record);
            return cambi_->slot(frame_slot.slot_index).record_prefix(distorted);
        }();
        if (!command) return Result<void>::failure(command.error());
        const std::array command_buffers{command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::cambi_submit);
            return submit_frame_commands(frame_slot_index, session_detail::cambi_lane,
                                         command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                cambi_->slot(frame_slot.slot_index).cancel();
            }
            return Result<void>::failure(submitted.error());
        }
        auto marked = cambi_->slot(frame_slot.slot_index).mark_prefix_submitted(submitted.value());
        if (!marked) return marked;
        if (input_staging_ring_ != nullptr) {
            return register_frame_transaction(frame_slot_index,
                                              FeatureTransactionKind::cambi_prefix,
                                              submitted.value(), frame_slot.frame_index, {});
        }
        auto input_consumer = input_pool_->register_spatial_consumer(input_token);
        if (!input_consumer) {
            return Result<void>::failure(input_consumer.error());
        }
        const std::array input_consumers{input_consumer.value()};
        return register_frame_transaction(frame_slot_index, FeatureTransactionKind::cambi_prefix,
                                          submitted.value(), frame_slot.frame_index,
                                          input_consumers);
    }

    [[nodiscard]] Result<void> submit_cambi_suffix(std::size_t frame_slot_index) {
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        auto command = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::cambi_record);
            return cambi_->slot(frame_slot.slot_index).record_suffix();
        }();
        if (!command) return Result<void>::failure(command.error());
        const std::array command_buffers{command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::cambi_submit);
            return submission_owner_->submit(session_detail::cambi_lane, command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                cambi_->slot(frame_slot.slot_index).cancel();
            }
            return Result<void>::failure(submitted.error());
        }
        auto marked = cambi_->slot(frame_slot.slot_index).mark_submitted(submitted.value());
        if (!marked) return marked;
        return register_frame_transaction(frame_slot_index, FeatureTransactionKind::cambi,
                                          submitted.value(), frame_slot.frame_index, {});
    }

    [[nodiscard]] Result<void>
    submit_speed_chroma(std::size_t frame_slot_index, const video::GpuFrameView &reference,
                        const video::GpuFrameView &distorted,
                        const evaluation::SegmentedInputLeaseToken &input_token) {
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        std::optional<speed_chroma::TaskHandle> streaming_task;
        std::optional<speed_chroma::TaskRingView> streaming_task_view;
        if (speed_chroma_task_ring_ != nullptr) {
            const auto reserved = speed_chroma_task_ring_->reserve(frame_slot.frame_index);
            if (!reserved) return Result<void>::failure(reserved.error());
            if (!reserved.value().has_value()) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "Speed Chroma task ring pressure was not drained before "
                     "prefix submission"});
            }
            streaming_task = *reserved.value();
            const auto task_view = speed_chroma_task_ring_->view(*streaming_task);
            if (!task_view) {
                static_cast<void>(speed_chroma_task_ring_->cancel_prefix(*streaming_task));
                return Result<void>::failure(task_view.error());
            }
            streaming_task_view = task_view.value();
        }
        auto command = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::speed_chroma_record);
            return speed_chroma_task_ring_ != nullptr
                       ? speed_chroma_->slot(frame_slot.slot_index)
                             .record_stream_prefix(reference, distorted, *streaming_task,
                                                   *streaming_task_view)
                       : speed_chroma_->slot(frame_slot.slot_index).record(reference, distorted);
        }();
        if (!command) {
            if (streaming_task.has_value()) {
                static_cast<void>(speed_chroma_task_ring_->cancel_prefix(*streaming_task));
            }
            return Result<void>::failure(command.error());
        }
        const std::array command_buffers{command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::speed_chroma_submit);
            return submit_frame_commands(frame_slot_index, session_detail::speed_chroma_lane,
                                         command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                speed_chroma_->slot(frame_slot.slot_index).cancel();
                if (streaming_task.has_value()) {
                    static_cast<void>(speed_chroma_task_ring_->cancel_prefix(*streaming_task));
                }
            }
            return Result<void>::failure(submitted.error());
        }
        auto marked =
            speed_chroma_task_ring_ != nullptr
                ? speed_chroma_->slot(frame_slot.slot_index)
                      .mark_stream_prefix_submitted(submitted.value())
                : speed_chroma_->slot(frame_slot.slot_index).mark_submitted(submitted.value());
        if (!marked) return marked;
        std::array<evaluation::SegmentedInputLeaseConsumer, 1U> staged_input_consumers{};
        std::span<const evaluation::SegmentedInputLeaseConsumer> input_consumers;
        if (input_staging_ring_ == nullptr) {
            auto input_consumer = input_pool_->register_spatial_consumer(input_token);
            if (!input_consumer) {
                return Result<void>::failure(input_consumer.error());
            }
            staged_input_consumers[0U] = input_consumer.value();
            input_consumers = staged_input_consumers;
        }
        if (speed_chroma_task_ring_ == nullptr) {
            return register_frame_transaction(
                frame_slot_index, FeatureTransactionKind::speed_chroma, submitted.value(),
                frame_slot.frame_index, input_consumers);
        }

        const auto registered = register_frame_transaction(
            frame_slot_index, FeatureTransactionKind::speed_chroma_prefix, submitted.value(),
            frame_slot.frame_index, input_consumers, streaming_task->slot,
            streaming_task->generation);
        if (registered) {
            ++speed_chroma_prefixes_since_wave_;
        }
        return registered;
    }

    struct StreamingWaveLane {
        telemetry::SessionStage record_stage{};
        telemetry::SessionStage submit_stage{};
        FeatureTransactionKind kind{};
        std::size_t lane{};
        const char *prefix_overflow_message{};
    };

    template <typename Ring>
    [[nodiscard]] Result<void> submit_ready_wave(Ring *ring, std::uint64_t &prefixes_since_wave,
                                                 const StreamingWaveLane &lane) {
        if (ring == nullptr || ring->ready_task_count() == 0U || ring->wave_in_flight()) {
            return Result<void>::success();
        }

        auto wave_command = [&] {
            [[maybe_unused]] auto measurement = hotspot_profiler_->measure(lane.record_stage);
            return ring->record_wave();
        }();
        if (!wave_command) {
            return Result<void>::failure(wave_command.error().with_context(
                session_detail::feature_transaction_name(lane.kind)));
        }
        const std::array command_buffers{wave_command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto measurement = hotspot_profiler_->measure(lane.submit_stage);
            return submission_owner_->submit(lane.lane, command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                ring->cancel();
            }
            return Result<void>::failure(submitted.error().with_context(
                session_detail::feature_transaction_name(lane.kind)));
        }
        const auto marked = ring->mark_wave_submitted(submitted.value());
        if (!marked) return marked;
        prefixes_since_wave = 0U;
        return register_streaming_wave_transaction(submitted.value(), lane.kind);
    }

    template <typename Ring>
    [[nodiscard]] Result<void> submit_wave_for_saturated_backlog(Ring *ring,
                                                                 std::uint64_t &prefixes_since_wave,
                                                                 const StreamingWaveLane &lane) {
        if (ring == nullptr || ring->wave_in_flight()) {
            return Result<void>::success();
        }

        const std::uint32_t capacity = ring->capacity();
        const std::uint32_t prefix_pending_count = ring->prefix_pending_task_count();
        const std::uint32_t ready_count = ring->ready_task_count();
        if (prefix_pending_count > capacity) {
            return Result<void>::failure({ErrorCode::invalid_state, lane.prefix_overflow_message});
        }
        const std::uint32_t maximum_ready_backlog = capacity - prefix_pending_count;
        if (!should_submit_ready_wave(ready_count, maximum_ready_backlog, prefixes_since_wave,
                                      capacity)) {
            return Result<void>::success();
        }
        return submit_ready_wave(ring, prefixes_since_wave, lane);
    }

    [[nodiscard]] static constexpr StreamingWaveLane speed_chroma_wave_lane() noexcept {
        return {
            telemetry::SessionStage::speed_chroma_record,
            telemetry::SessionStage::speed_chroma_submit,
            FeatureTransactionKind::speed_chroma_wave,
            session_detail::speed_chroma_lane,
            "Speed Chroma prefix backlog exceeds task ring capacity",
        };
    }

    [[nodiscard]] static constexpr StreamingWaveLane adm3_wave_lane() noexcept {
        return {
            telemetry::SessionStage::adm3_record,
            telemetry::SessionStage::adm3_submit,
            FeatureTransactionKind::adm3_wave,
            session_detail::adm3_lane,
            "ADM3 prefix backlog exceeds task ring capacity",
        };
    }

    [[nodiscard]] Result<void> submit_ready_speed_chroma_wave() {
        return submit_ready_wave(speed_chroma_task_ring_, speed_chroma_prefixes_since_wave_,
                                 speed_chroma_wave_lane());
    }

    [[nodiscard]] Result<void> submit_speed_chroma_wave_for_saturated_backlog() {
        return submit_wave_for_saturated_backlog(
            speed_chroma_task_ring_, speed_chroma_prefixes_since_wave_, speed_chroma_wave_lane());
    }

    [[nodiscard]] Result<void>
    submit_adm3(std::size_t frame_slot_index, const video::GpuFrameView &reference,
                const video::GpuFrameView &distorted,
                const evaluation::SegmentedInputLeaseToken &input_token) {
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        std::optional<adm3::TaskHandle> streaming_task;
        std::optional<adm3::TaskRingView> streaming_task_view;
        if (adm3_task_ring_ != nullptr) {
            const auto reserved = adm3_task_ring_->reserve(frame_slot.frame_index);
            if (!reserved) return Result<void>::failure(reserved.error());
            if (!reserved.value().has_value()) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "ADM3 task ring pressure was not drained before prefix "
                     "submission"});
            }
            streaming_task = *reserved.value();
            const auto task_view = adm3_task_ring_->view(*streaming_task);
            if (!task_view) {
                static_cast<void>(adm3_task_ring_->cancel_prefix(*streaming_task));
                return Result<void>::failure(task_view.error());
            }
            streaming_task_view = task_view.value();
        }
        auto command = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::adm3_record);
            return adm3_task_ring_ != nullptr
                       ? adm3_->slot(frame_slot.slot_index)
                             .record_stream_prefix(reference, distorted, *streaming_task,
                                                   *streaming_task_view)
                       : adm3_->slot(frame_slot.slot_index).record(reference, distorted);
        }();
        if (!command) {
            if (streaming_task.has_value()) {
                static_cast<void>(adm3_task_ring_->cancel_prefix(*streaming_task));
            }
            return Result<void>::failure(command.error());
        }
        const std::array command_buffers{command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::adm3_submit);
            return submit_frame_commands(frame_slot_index, session_detail::adm3_lane,
                                         command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                adm3_->slot(frame_slot.slot_index).cancel();
                if (streaming_task.has_value()) {
                    static_cast<void>(adm3_task_ring_->cancel_prefix(*streaming_task));
                }
            }
            return Result<void>::failure(submitted.error());
        }
        auto marked =
            adm3_task_ring_ != nullptr
                ? adm3_->slot(frame_slot.slot_index).mark_stream_prefix_submitted(submitted.value())
                : adm3_->slot(frame_slot.slot_index).mark_submitted(submitted.value());
        if (!marked) return marked;
        std::vector<evaluation::SegmentedInputLeaseConsumer> input_consumers;
        if (input_staging_ring_ == nullptr) {
            auto registered_consumers =
                input_pool_->register_consumers(input_token, true, true, std::nullopt);
            if (!registered_consumers) {
                return Result<void>::failure(registered_consumers.error());
            }
            input_consumers = std::move(registered_consumers).value();
        }
        if (adm3_task_ring_ == nullptr) {
            return register_frame_transaction(frame_slot_index, FeatureTransactionKind::adm3,
                                              submitted.value(), frame_slot.frame_index,
                                              input_consumers);
        }
        const auto registered =
            register_frame_transaction(frame_slot_index, FeatureTransactionKind::adm3_prefix,
                                       submitted.value(), frame_slot.frame_index, input_consumers,
                                       streaming_task->slot, streaming_task->generation);
        if (registered) {
            ++adm3_prefixes_since_wave_;
        }
        return registered;
    }

    [[nodiscard]] Result<void> submit_ready_adm3_wave() {
        return submit_ready_wave(adm3_task_ring_, adm3_prefixes_since_wave_, adm3_wave_lane());
    }

    [[nodiscard]] Result<void> submit_adm3_wave_for_saturated_backlog() {
        return submit_wave_for_saturated_backlog(adm3_task_ring_, adm3_prefixes_since_wave_,
                                                 adm3_wave_lane());
    }

    [[nodiscard]] Result<void> submit_saturated_suffix_waves() {
        const auto submit_speed = [this] {
            return submit_speed_chroma_wave_for_saturated_backlog();
        };
        const auto submit_adm3 = [this] { return submit_adm3_wave_for_saturated_backlog(); };

        Result<void> first = prefer_adm3_wave_next_ ? submit_adm3() : submit_speed();
        if (!first) return first;
        Result<void> second = prefer_adm3_wave_next_ ? submit_speed() : submit_adm3();
        if (!second) return second;
        prefer_adm3_wave_next_ = !prefer_adm3_wave_next_;
        return Result<void>::success();
    }

    [[nodiscard]] Result<void>
    submit_motion3(std::size_t frame_slot_index, const video::GpuFrameView &reference,
                   const evaluation::SegmentedInputLeaseToken &input_token) {
        FrameSlot &frame_slot = frame_slots_[frame_slot_index];
        video::GpuPlaneView previous_luma{};
        const video::GpuPlaneView *previous_luma_pointer = nullptr;
        const std::uint64_t previous_luma_distance = motion3_five_frame_window_ ? 2U : 1U;
        if (reference.index >= previous_luma_distance) {
            auto previous =
                input_staging_ring_ != nullptr
                    ? input_staging_ring_->reference_luma(reference.index - previous_luma_distance)
                    : input_pool_->reference_luma(reference.index - previous_luma_distance);
            if (!previous) return Result<void>::failure(previous.error());
            previous_luma = previous.value();
            previous_luma_pointer = &previous_luma;
        }

        auto command = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::motion3_record);
            return motion3_->slot(frame_slot.slot_index)
                .record_push(reference, previous_luma_pointer);
        }();
        if (!command) return Result<void>::failure(command.error());
        if (!command.value().has_value()) {
            auto motion_scores =
                motion3_->slot(frame_slot.slot_index).complete_without_submission();
            if (!motion_scores) {
                return Result<void>::failure(motion_scores.error());
            }
            if (!motion_scores.value().empty()) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "Motion3 startup completion unexpectedly produced output"});
            }
            return Result<void>::success();
        }
        const std::array command_buffers{*command.value()};
        auto submitted = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::motion3_submit);
            return submit_frame_commands(frame_slot_index, session_detail::motion3_lane,
                                         command_buffers);
        }();
        if (!submitted) {
            if (!submission_owner_->last_submit_reached_queue()) {
                motion3_->slot(frame_slot.slot_index).cancel();
            }
            return Result<void>::failure(submitted.error());
        }
        auto committed = motion3_->slot(frame_slot.slot_index).commit_submission(submitted.value());
        if (!committed) return committed;
        const std::optional<std::uint64_t> previous_luma_frame =
            reference.index >= previous_luma_distance
                ? std::optional<std::uint64_t>{reference.index - previous_luma_distance}
                : std::nullopt;
        std::vector<evaluation::SegmentedInputLeaseConsumer> input_consumers;
        if (input_staging_ring_ == nullptr) {
            auto registered_consumers =
                input_pool_->register_consumers(input_token, false, true, previous_luma_frame);
            if (!registered_consumers) {
                return Result<void>::failure(registered_consumers.error());
            }
            input_consumers = std::move(registered_consumers).value();
        }
        return register_frame_transaction(frame_slot_index, FeatureTransactionKind::motion3,
                                          submitted.value(), frame_slot.frame_index,
                                          input_consumers);
    }

    [[nodiscard]] std::size_t pending_transaction_count() const noexcept {
        std::size_t count = 0U;
        for (const auto &lane_transactions : pending_transactions_) {
            count += lane_transactions.size();
        }
        return count;
    }

    [[nodiscard]] Result<SegmentedPollOutcome> collect_signaled_completions() {
        auto retirements = [&] {
            [[maybe_unused]] auto measurement =
                hotspot_profiler_->measure(telemetry::SessionStage::nonblocking_retire);
            return submission_owner_->poll_all_lanes();
        }();
        if (!retirements) {
            return Result<SegmentedPollOutcome>::failure(retirements.error().with_context("GPU completion polling"));
        }
        const bool made_progress = !retirements.value().empty();
        auto completions = process_retirements(retirements.value());
        if (!completions) {
            return Result<SegmentedPollOutcome>::failure(completions.error());
        }
        return Result<SegmentedPollOutcome>::success(
            {std::move(completions).value(), made_progress});
    }

    [[nodiscard]] Result<evaluation::CompletionBatch>
    process_retirements(std::span<const gpu::SubmissionRetirement> retirements) {
        [[maybe_unused]] auto measurement =
            hotspot_profiler_->measure(telemetry::SessionStage::readback_retirement);
        evaluation::CompletionBatch completions;
        for (const gpu::SubmissionRetirement &retirement : retirements) {
            auto &lane_transactions = pending_transactions_[retirement.lane];
            if (lane_transactions.empty() ||
                lane_transactions.front().ticket.serial != retirement.ticket.serial) {
                return Result<evaluation::CompletionBatch>::failure(
                    {ErrorCode::invalid_state, "Vulkan submission retirement violated lane FIFO"});
            }
            const PendingTransaction transaction = lane_transactions.front();
            const auto retirement_failed = [&transaction](Error error) {
                const auto frame = transaction.frame_slot_index == invalid_resource_index
                    ? std::nullopt : std::optional<std::uint64_t>{transaction.frame_index};
                return Result<evaluation::CompletionBatch>::failure(error.with_context(
                    std::string(session_detail::feature_transaction_name(transaction.kind)) + " retirement",
                    ErrorInput::none, frame));
            };
            lane_transactions.pop_front();
            if (hotspot_profiler_->enabled()) {
                hotspot_profiler_->runtime().record_pending_submissions(
                    pending_transaction_count());
            }
            FrameSlot *frame_slot = transaction.frame_slot_index == invalid_resource_index
                                        ? nullptr
                                        : &frame_slots_[transaction.frame_slot_index];
            MetricFrameRecord *pending_frame = nullptr;
            if (transaction.kind != FeatureTransactionKind::input_staging &&
                transaction.kind != FeatureTransactionKind::speed_chroma_wave &&
                transaction.kind != FeatureTransactionKind::adm3_wave &&
                transaction.kind != FeatureTransactionKind::motion3 &&
                transaction.kind != FeatureTransactionKind::motion3_flush) {
                const auto pending_frame_result = frames_->find_frame(transaction.frame_index);
                if (!pending_frame_result) {
                    return retirement_failed(
                        {ErrorCode::invalid_state,
                         "Vulkan transaction lost its pending spatial frame"});
                }
                pending_frame = pending_frame_result.value();
            }

            switch (transaction.kind) {
            case FeatureTransactionKind::input_staging: {
                if (input_staging_ring_ == nullptr) {
                    return retirement_failed(
                        {ErrorCode::invalid_state,
                         "input staging retirement lost its staging ring"});
                }
                const auto staging_retired = input_staging_ring_->mark_retired(
                    transaction.frame_slot_index, retirement.ticket, retirement.host_elapsed);
                if (!staging_retired) {
                    return retirement_failed(staging_retired.error());
                }
                break;
            }
            case FeatureTransactionKind::cambi_prefix: {
                const auto prefix_retired =
                    cambi_->slot(frame_slot->slot_index)
                        .retire_prefix(retirement.ticket, retirement.host_elapsed);
                if (!prefix_retired) {
                    return retirement_failed(prefix_retired.error());
                }
                break;
            }
            case FeatureTransactionKind::cambi: {
                auto result = [&] {
                    [[maybe_unused]] auto measurement =
                        hotspot_profiler_->measure(telemetry::SessionStage::cambi_retire);
                    return cambi_->slot(frame_slot->slot_index)
                        .retire(retirement.ticket, retirement.host_elapsed);
                }();
                if (!result) {
                    return retirement_failed(result.error());
                }
                pending_frame->cambi_value = result.value().value;
                break;
            }
            case FeatureTransactionKind::speed_chroma_prefix: {
                const auto prefix_retired =
                    speed_chroma_->slot(frame_slot->slot_index)
                        .retire_stream_prefix(retirement.ticket, retirement.host_elapsed);
                if (!prefix_retired) {
                    return retirement_failed(prefix_retired.error());
                }
                if (speed_chroma_task_ring_ == nullptr) {
                    return retirement_failed(
                        {ErrorCode::invalid_state,
                         "Speed Chroma streaming prefix lost its task ring"});
                }
                const auto published = speed_chroma_task_ring_->publish_ready(
                    {transaction.task_slot, transaction.task_generation});
                if (!published) {
                    return retirement_failed(published.error());
                }
                break;
            }
            case FeatureTransactionKind::speed_chroma_wave: {
                if (speed_chroma_task_ring_ == nullptr) {
                    return retirement_failed(
                        {ErrorCode::invalid_state,
                         "Speed Chroma wave retirement lost its task ring"});
                }
                auto wave_results = speed_chroma_task_ring_->retire_wave(retirement.ticket,
                                                                         retirement.host_elapsed);
                if (!wave_results) {
                    return retirement_failed(wave_results.error());
                }
                for (const speed_chroma::StreamingTaskFrameResult &frame_result :
                     wave_results.value()) {
                    const auto pending_frame_result = frames_->find_frame(frame_result.frame_index);
                    if (!pending_frame_result) {
                        return retirement_failed(
                            {ErrorCode::invalid_state,
                             "Speed Chroma wave result lost its pending frame"});
                    }
                    MetricFrameRecord &wave_pending_frame = *pending_frame_result.value();
                    wave_pending_frame.speed_chroma_value = frame_result.result.uv;
                    wave_pending_frame.speed_chroma_diagnostics = {
                        frame_result.result.u,          frame_result.result.v,
                        frame_result.result.uv,         frame_result.result.u_singular,
                        frame_result.result.v_singular,
                    };
                }
                break;
            }
            case FeatureTransactionKind::speed_chroma: {
                hotspot_profiler_->speed_chroma().record_submission_latency(
                    retirement.host_elapsed);
                auto result = [&] {
                    [[maybe_unused]] auto measurement =
                        hotspot_profiler_->measure(telemetry::SessionStage::speed_chroma_retire);
                    return speed_chroma_->slot(frame_slot->slot_index)
                        .retire(retirement.ticket, retirement.host_elapsed);
                }();
                if (!result) {
                    return retirement_failed(result.error());
                }
                pending_frame->speed_chroma_value = result.value().uv;
                pending_frame->speed_chroma_diagnostics = {
                    result.value().u,          result.value().v,          result.value().uv,
                    result.value().u_singular, result.value().v_singular,
                };
                break;
            }
            case FeatureTransactionKind::adm3_prefix: {
                const auto prefix_retired =
                    adm3_->slot(frame_slot->slot_index)
                        .retire_stream_prefix(retirement.ticket, retirement.host_elapsed);
                if (!prefix_retired) {
                    return retirement_failed(prefix_retired.error());
                }
                if (adm3_task_ring_ == nullptr) {
                    return retirement_failed(
                        {ErrorCode::invalid_state, "ADM3 streaming prefix lost its task ring"});
                }
                const auto published = adm3_task_ring_->publish_ready(
                    {transaction.task_slot, transaction.task_generation});
                if (!published) {
                    return retirement_failed(published.error());
                }
                break;
            }
            case FeatureTransactionKind::adm3_wave: {
                if (adm3_task_ring_ == nullptr) {
                    return retirement_failed(
                        {ErrorCode::invalid_state, "ADM3 wave retirement lost its task ring"});
                }
                auto wave_results =
                    adm3_task_ring_->retire_wave(retirement.ticket, retirement.host_elapsed);
                if (!wave_results) {
                    return retirement_failed(wave_results.error());
                }
                for (const adm3::StreamingTaskFrameResult &frame_result : wave_results.value()) {
                    const auto pending_frame_result = frames_->find_frame(frame_result.frame_index);
                    if (!pending_frame_result) {
                        return retirement_failed(
                            {ErrorCode::invalid_state, "ADM3 wave result lost its pending frame"});
                    }
                    pending_frame_result.value()->adm3_value = frame_result.result.value;
                }
                break;
            }
            case FeatureTransactionKind::adm3: {
                hotspot_profiler_->adm3().record_submission_latency(retirement.host_elapsed);
                auto result = [&] {
                    [[maybe_unused]] auto measurement =
                        hotspot_profiler_->measure(telemetry::SessionStage::adm3_retire);
                    return adm3_->slot(frame_slot->slot_index)
                        .retire(retirement.ticket, retirement.host_elapsed);
                }();
                if (!result) {
                    return retirement_failed(result.error());
                }
                pending_frame->adm3_value = result.value().value;
                break;
            }
            case FeatureTransactionKind::motion3:
            case FeatureTransactionKind::motion3_flush: {
                hotspot_profiler_->motion3().record_submission_latency(retirement.host_elapsed);
                auto result = [&] {
                    [[maybe_unused]] auto measurement =
                        hotspot_profiler_->measure(telemetry::SessionStage::motion3_retire);
                    return motion3_->slot(frame_slot->slot_index)
                        .retire(retirement.ticket, retirement.host_elapsed);
                }();
                if (!result) {
                    return retirement_failed(result.error());
                }
                std::vector<evaluation::MotionFeature> motion_completions =
                    convert_motion_scores(std::move(result).value());
                completions.motion.insert(completions.motion.end(),
                                          std::make_move_iterator(motion_completions.begin()),
                                          std::make_move_iterator(motion_completions.end()));
                break;
            }
            }

            for (std::size_t consumer_index = 0U; consumer_index < transaction.input_consumer_count;
                 ++consumer_index) {
                const auto input_retired =
                    input_pool_->retire_consumer(transaction.input_consumers[consumer_index]);
                if (!input_retired) {
                    return retirement_failed(input_retired.error());
                }
            }

            const bool owns_frame_slot_resource =
                frame_slot != nullptr && transaction.kind != FeatureTransactionKind::motion3_flush;
            if (owns_frame_slot_resource) {
                const auto resource_completed = complete_resource_transaction(
                    transaction.frame_slot_index, transaction.frame_index);
                if (!resource_completed) {
                    return retirement_failed(resource_completed.error());
                }
            }
        }
        const auto saturated_wave_submitted = submit_saturated_suffix_waves();
        if (!saturated_wave_submitted) {
            return Result<evaluation::CompletionBatch>::failure(saturated_wave_submitted.error());
        }
        return Result<evaluation::CompletionBatch>::success(std::move(completions));
    }

    DeviceInfo device_;
    bool motion3_five_frame_window_{};
    std::unique_ptr<gpu::Device> context_;
    std::unique_ptr<gpu::SubmissionOwner> submission_owner_;
    std::unique_ptr<telemetry::SessionTelemetry> hotspot_profiler_;
    std::unique_ptr<evaluation::SegmentedInputLeasePool> input_pool_;
    std::unique_ptr<evaluation::InputStagingRing> input_staging_ring_;
    std::vector<FrameSlot> frame_slots_;
    session_detail::FrameSlotAdmissionController frame_slot_admission_controller_;
    std::unique_ptr<metrics::cambi::Vulkan> cambi_;
    std::unique_ptr<metrics::speed_chroma::Vulkan> speed_chroma_;
    std::unique_ptr<metrics::adm3::Vulkan> adm3_;
    std::unique_ptr<metrics::motion3::Vulkan> motion3_;
    speed_chroma::StreamingTaskRingResources *speed_chroma_task_ring_{};
    adm3::StreamingTaskRingResources *adm3_task_ring_{};
    std::unique_ptr<FrameRecords> frames_;
    std::array<std::deque<PendingTransaction>, feature_lane_count> pending_transactions_;
    std::optional<Error> failure_;
    std::uint64_t next_submission_index_{};
    bool prefer_adm3_wave_next_{};
    std::uint64_t speed_chroma_prefixes_since_wave_{};
    std::uint64_t adm3_prefixes_since_wave_{};
    bool flushed_{};
    bool cancelled_{};
    bool report_finalized_{};

  public:
  public:
    [[nodiscard]] const FrameDescriptor &descriptor() const noexcept { return descriptor_; }

    [[nodiscard]] const StreamMediaMetadata &metadata() const noexcept { return metadata_; }
    [[nodiscard]] ModelVariant model() const noexcept { return model_; }

    [[nodiscard]] std::optional<std::uint64_t> frame_count_hint() const noexcept {
        return input_frame_count_hint();
    }

    [[nodiscard]] std::uint64_t decoded_frame_count() const noexcept {
        return input_decoded_frame_count();
    }

    [[nodiscard]] EvaluationStreamStatistics statistics() const {
        const evaluation::InputStatistics prefetch_statistics = input_statistics();
        return {
            prefetch_statistics.frame_count_hint,
            prefetch_statistics.decoded_frame_count,
            prefetch_statistics.next_frame_to_schedule,
            prefetch_statistics.next_frame_to_acquire,
            prefetch_statistics.record_capacity,
            prefetch_statistics.owned_pair_count,
            prefetch_statistics.ready_lead_frames,
            prefetch_statistics.frame_pair_copy_calls,
            prefetch_statistics.frame_pair_copy_bytes,
            prefetch_statistics.frame_pair_copy_nanoseconds,
            prefetch_statistics.staging_input_wait_nanoseconds,
            prefetch_statistics.staging_input_wait_count,
            prefetch_statistics.staging_grant_wait_nanoseconds,
            prefetch_statistics.staging_grant_wait_count,
            prefetch_statistics.coordinator_wait_nanoseconds,
            prefetch_statistics.coordinator_wait_count,
            segmented_input_capacity(),
            prefetch_statistics.scheduling_complete,
            cancel_requested_.load(std::memory_order_acquire) ||
                prefetch_statistics.cancellation_requested,
            make_public_input_statistics(prefetch_statistics.reference_decoder),
            make_public_input_statistics(prefetch_statistics.distorted_decoder),
        };
    }

    // Cross-thread cancellation entry. Only touches the atomic flag and the
    // prefetcher's own thread-safe cancel path; GPU-side wind-down stays on
    // the worker thread at the next commit point.
    void request_cancel() noexcept {
        cancel_requested_.store(true, std::memory_order_release);
        input_request_cancel();
    }

    [[nodiscard]] bool cancellation_requested() const noexcept {
        return cancel_requested_.load(std::memory_order_acquire);
    }

    [[nodiscard]] Result<std::vector<FrameResult>> next() {
        try {
            return next_impl();
        } catch (const std::exception &exception) {
            return fail(unexpected_stream_exception("next()", &exception));
        } catch (...) {
            return fail(unexpected_stream_exception("next()", nullptr));
        }
    }

    [[nodiscard]] Result<std::vector<FrameResult>> finish() {
        try {
            return finish_impl();
        } catch (const std::exception &exception) {
            return fail(unexpected_stream_exception("finish()", &exception));
        } catch (...) {
            return fail(unexpected_stream_exception("finish()", nullptr));
        }
    }

  private:
    // Records the decoded reference timestamp for a frame at registration
    // time so result batches can be aligned through frame_index later.
    [[nodiscard]] FrameTiming make_frame_timing(std::uint64_t frame_index) {
        FrameTiming timing;
        const auto timestamp = input_take_frame_timestamp(frame_index);
        if (timestamp.has_value() && timestamp->valid && timestamp->time_base_numerator > 0 &&
            timestamp->time_base_denominator > 0) {
            timing.status = FrameTimingStatus::valid;
            timing.presentation_timestamp = timestamp->value;
            timing.time_base = {timestamp->time_base_numerator, timestamp->time_base_denominator};
            if (!timing_origin_pts_.has_value()) {
                timing_origin_pts_ = timestamp->value;
            }
            timing.media_time_seconds =
                static_cast<double>((static_cast<long double>(timestamp->value) -
                                     static_cast<long double>(*timing_origin_pts_)) *
                                    static_cast<long double>(timestamp->time_base_numerator) /
                                    static_cast<long double>(timestamp->time_base_denominator));
            const FrameRateInfo &frame_rate = metadata_.reference.frame_rate;
            if (frame_rate.mode == FrameRateMode::constant && frame_rate.average.valid()) {
                timing.duration_seconds = static_cast<double>(frame_rate.average.denominator) /
                                          static_cast<double>(frame_rate.average.numerator);
            }
        }
        return timing;
    }

    // Flag non-monotonic presentation times without rewriting decoded timestamps.
    void attach_timings(std::vector<FrameResult> &results) {
        for (FrameResult &result : results) {
            if (result.timing.status != FrameTimingStatus::valid) continue;
            if (last_media_time_seconds_.has_value() &&
                result.timing.media_time_seconds < *last_media_time_seconds_ - 1.0e-9) {
                result.timing.status = FrameTimingStatus::non_monotonic;
            } else {
                last_media_time_seconds_ = result.timing.media_time_seconds;
            }
        }
    }

    [[nodiscard]] Result<std::vector<FrameResult>>
    apply_and_drain(const evaluation::CompletionBatch &completions) {
        const auto applied = frames_->apply_completions(completions);
        if (!applied) {
            return Result<std::vector<FrameResult>>::failure(applied.error());
        }
        auto drained = frames_->drain();
        if (drained) attach_timings(drained.value());
        return drained;
    }

    [[nodiscard]] Result<std::vector<FrameResult>>
    collect_push_results(std::uint64_t frame_index,
                         const evaluation::SegmentedPushOutcome &pushed) {
        if (!pushed.slot_busy) {
            const auto frame_record = frames_->find_frame(frame_index);
            if (!frame_record) {
                return Result<std::vector<FrameResult>>::failure(frame_record.error());
            }
            frame_record.value()->timing = make_frame_timing(frame_index);
        }
        return apply_and_drain(pushed.completions);
    }

    [[nodiscard]] Result<std::vector<FrameResult>>
    collect_poll_results(const evaluation::SegmentedPollOutcome &polled) {
        return apply_and_drain(polled.completions);
    }

    [[nodiscard]] Result<std::vector<FrameResult>> wait_for_vulkan_progress() {
        if (!has_pending_segmented_work()) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state, "Vulkan progress wait requires pending work"});
        }
        auto waited = wait_segmented_progress();
        if (!waited) {
            return Result<std::vector<FrameResult>>::failure(waited.error());
        }
        return collect_poll_results(waited.value());
    }

    [[nodiscard]] Result<std::vector<FrameResult>> next_impl() {
        if (finished_) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state,
                 "EvaluationStream::next() cannot be called after finish()"});
        }
        if (failed_) {
            return Result<std::vector<FrameResult>>::failure(terminal_error_.value_or(
                {ErrorCode::invalid_state,
                 "EvaluationStream::next() cannot continue after failure"}));
        }
        if (cancel_requested_.load(std::memory_order_acquire)) {
            return commit_cancelled();
        }
        if (input_exhausted_) {
            return Result<std::vector<FrameResult>>::success({});
        }
        for (;;) {
            if (cancel_requested_.load(std::memory_order_acquire)) {
                return commit_cancelled();
            }
            if (!pending_push_.has_value()) {
                auto acquired = input_try_acquire_next();
                if (!acquired) return fail(acquired.error());

                switch (acquired.value().status) {
                case evaluation::FramePairAcquireStatus::end_of_stream: {
                    input_exhausted_ = true;
                    auto polled = poll_segmented_input_starvation();
                    if (!polled) return fail(polled.error());
                    auto results = collect_poll_results(polled.value());
                    if (!results) return fail(results.error());
                    return results;
                }

                case evaluation::FramePairAcquireStatus::unavailable: {
                    // Lease and queue recycling are normal downstream
                    // backpressure. Ordinary polling must not turn them into a
                    // Speed Chroma suffix-wave trigger.
                    auto polled = poll_segmented_completions();
                    if (!polled) return fail(polled.error());
                    auto results = collect_poll_results(polled.value());
                    if (!results) return fail(results.error());
                    if (!results.value().empty()) {
                        return results;
                    }
                    const bool made_progress =
                        polled.value().made_progress || acquired.value().scheduled_pairs > 0U;
                    if (!made_progress) {
                        const bool staging_copy_can_complete_next =
                            acquired.value().input_state ==
                            evaluation::FramePairInputState::copying;
                        if (!staging_copy_can_complete_next && has_pending_segmented_work()) {
                            auto waited = wait_for_vulkan_progress();
                            if (!waited) return fail(waited.error());
                            if (!waited.value().empty()) return waited;
                        } else if (!staging_copy_can_complete_next && !frames_->has_registration_capacity()) {
                            auto predicted = frames_->wait_for_ready();
                            if (!predicted) return fail(predicted.error());
                            attach_timings(predicted.value());
                            if (!predicted.value().empty()) return predicted;
                        } else {
                            const auto input_progress =
                                input_wait_for_progress(acquired.value().progress_epoch);
                            if (!input_progress) {
                                return fail(input_progress.error());
                            }
                        }
                    }
                    continue;
                }

                case evaluation::FramePairAcquireStatus::input_starved: {
                    const bool new_starvation_epoch =
                        !last_input_starvation_epoch_.has_value() ||
                        *last_input_starvation_epoch_ != acquired.value().input_epoch;
                    auto polled = new_starvation_epoch ? poll_segmented_input_starvation()
                                                       : poll_segmented_completions();
                    if (!polled) return fail(polled.error());
                    if (new_starvation_epoch) {
                        last_input_starvation_epoch_ = acquired.value().input_epoch;
                    }
                    auto results = collect_poll_results(polled.value());
                    if (!results) return fail(results.error());
                    if (!results.value().empty()) {
                        return results;
                    }
                    const bool made_progress =
                        polled.value().made_progress || acquired.value().scheduled_pairs > 0U;
                    if (!made_progress) {
                        const bool staging_copy_can_complete_next =
                            acquired.value().input_state ==
                            evaluation::FramePairInputState::copying;
                        if (!staging_copy_can_complete_next && has_pending_segmented_work()) {
                            auto waited = wait_for_vulkan_progress();
                            if (!waited) return fail(waited.error());
                            if (!waited.value().empty()) return waited;
                        } else if (!staging_copy_can_complete_next && !frames_->has_registration_capacity()) {
                            auto predicted = frames_->wait_for_ready();
                            if (!predicted) return fail(predicted.error());
                            attach_timings(predicted.value());
                            if (!predicted.value().empty()) return predicted;
                        } else {
                            const auto input_progress =
                                input_wait_for_progress(acquired.value().progress_epoch);
                            if (!input_progress) {
                                return fail(input_progress.error());
                            }
                        }
                    }
                    continue;
                }

                case evaluation::FramePairAcquireStatus::ready:
                    break;
                }

                const evaluation::ReadyInput ready_pair = acquired.value().pair;
                const evaluation::SegmentedInputToken input_token =
                    make_segmented_token(*ready_pair.record);

                const auto published = publish_segmented_input(input_token);
                if (!published) return fail(published.error());
                pending_push_ = PendingPush{ready_pair, input_token};
            }

            const evaluation::ReadyInput ready_pair = pending_push_->pair;
            const evaluation::SegmentedInputToken input_token = pending_push_->token;

            auto pushed = push_ready_segmented_input(input_token);
            if (!pushed) {
                // Vulkan owns the token after the submit attempt, including
                // partial-submit failures. Keep pipeline cancellation from
                // abandoning memory that may still have GPU consumers.
                pending_push_.reset();
                const auto ownership_transferred = input_mark_submitted(ready_pair);
                if (ownership_transferred) {
                    static_cast<void>(input_retire(ready_pair));
                }
                return fail(pushed.error());
            }

            auto push_results = collect_push_results(input_token.frame_index, pushed.value());
            if (!push_results) {
                if (!pushed.value().slot_busy) {
                    pending_push_.reset();
                    const auto ownership_transferred = input_mark_submitted(ready_pair);
                    if (ownership_transferred) {
                        static_cast<void>(input_retire(ready_pair));
                    }
                }
                return fail(push_results.error());
            }

            if (pushed.value().slot_busy) {
                if (!push_results.value().empty()) {
                    return push_results;
                }
                auto polled = poll_segmented_completions();
                if (!polled) return fail(polled.error());
                auto poll_results = collect_poll_results(polled.value());
                if (!poll_results) return fail(poll_results.error());
                if (!poll_results.value().empty()) {
                    return poll_results;
                }
                if (!polled.value().made_progress) {
                    auto waited = wait_for_vulkan_progress();
                    if (!waited) return fail(waited.error());
                    if (!waited.value().empty()) return waited;
                }
                continue;
            }

            pending_push_.reset();

            const auto ownership_transferred = input_mark_submitted(ready_pair);
            if (!ownership_transferred) {
                return fail(ownership_transferred.error());
            }

            // Vulkan transactions now own lease retirement, so the
            // record can be recycled immediately after push acceptance.
            const auto record_retired = input_retire(ready_pair);
            if (!record_retired) return fail(record_retired.error());

            if (!push_results.value().empty()) {
                return push_results;
            }
        }
    }

    [[nodiscard]] Result<std::vector<FrameResult>> finish_impl() {
        if (finished_) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state, "EvaluationStream::finish() may only be called once"});
        }
        if (failed_) {
            return Result<std::vector<FrameResult>>::failure(terminal_error_.value_or(
                {ErrorCode::invalid_state,
                 "EvaluationStream::finish() cannot complete after failure"}));
        }
        if (cancel_requested_.load(std::memory_order_acquire)) {
            return commit_cancelled();
        }

        if (!input_exhausted_) {
            return Result<std::vector<FrameResult>>::failure(
                {ErrorCode::invalid_state,
                 "EvaluationStream::finish() requires all input to be consumed by next()"});
        }
        std::vector<FrameResult> accumulated_results;

        if (pending_push_.has_value()) {
            for (;;) {
                if (cancel_requested_.load(std::memory_order_acquire)) {
                    return commit_cancelled();
                }

                auto pushed = push_ready_segmented_input(pending_push_->token);
                if (!pushed) return fail(pushed.error());
                const bool slot_busy = pushed.value().slot_busy;
                auto push_results =
                    collect_push_results(pending_push_->token.frame_index, pushed.value());
                if (!push_results) return fail(push_results.error());
                append_frame_results(accumulated_results, std::move(push_results).value());
                if (!slot_busy) {
                    const evaluation::ReadyInput ready_pair = pending_push_->pair;
                    pending_push_.reset();
                    const auto ownership_transferred = input_mark_submitted(ready_pair);
                    if (!ownership_transferred) {
                        return fail(ownership_transferred.error());
                    }
                    const auto record_retired = input_retire(ready_pair);
                    if (!record_retired) {
                        return fail(record_retired.error());
                    }
                    break;
                }
                auto polled = poll_segmented_completions();
                if (!polled) return fail(polled.error());
                auto poll_results = collect_poll_results(polled.value());
                if (!poll_results) return fail(poll_results.error());
                append_frame_results(accumulated_results, std::move(poll_results).value());
                if (!polled.value().made_progress) {
                    auto waited = wait_for_vulkan_progress();
                    if (!waited) return fail(waited.error());
                    append_frame_results(accumulated_results, std::move(waited).value());
                }
            }
        }

        const auto input_drained = input_drain();
        if (!input_drained) return fail(input_drained.error());
        auto tail = flush_gpu();
        if (!tail) return fail(tail.error().with_context("GPU finish"));
        auto tail_results = apply_and_drain(tail.value());
        if (!tail_results) return fail(tail_results.error());
        append_frame_results(accumulated_results, std::move(tail_results).value());
        auto final_predictions = frames_->finish();
        if (!final_predictions) return fail(final_predictions.error().with_context("finishing predictions"));
        attach_timings(final_predictions.value());
        append_frame_results(accumulated_results, std::move(final_predictions).value());

        finished_ = true;
        return Result<std::vector<FrameResult>>::success(std::move(accumulated_results));
    }

    // Terminal commit for cooperative cancellation, always on the worker
    // thread: stop new input, retire the prediction worker, and let the
    // Vulkan session drain in-flight GPU work. In-flight work is never
    // force-destroyed.
    [[nodiscard]] Result<std::vector<FrameResult>> commit_cancelled() {
        return fail({ErrorCode::cancelled, "evaluation was cancelled"});
    }

    [[nodiscard]] Result<std::vector<FrameResult>> fail(Error error) {
        failed_ = true;
        if (!terminal_error_.has_value()) terminal_error_.emplace(std::move(error));
        frames_->cancel();
        input_request_cancel();
        // Cleanup failures must not replace the error that caused shutdown.
        try { static_cast<void>(input_drain()); } catch (...) {}
        cancel_gpu();
        return Result<std::vector<FrameResult>>::failure(*terminal_error_);
    }

    FrameDescriptor descriptor_;
    StreamMediaMetadata metadata_;
    ModelVariant model_{default_model};
    struct PendingPush final {
        evaluation::ReadyInput pair;
        evaluation::SegmentedInputToken token;
    };
    std::optional<PendingPush> pending_push_;
    std::optional<std::uint64_t> last_input_starvation_epoch_;
    std::optional<std::int64_t> timing_origin_pts_;
    std::optional<double> last_media_time_seconds_;
    std::optional<Error> terminal_error_;
    std::atomic_bool cancel_requested_{};
    bool input_exhausted_{};
    bool finished_{};
    bool failed_{};

  private:
  public:
    using RecordSlot = FrameRecords::FrameRecord;

    enum class WorkerEventKind : std::uint8_t {
        waiting_for_input,
        storage_requested,
        copy_completed,
        end_of_stream,
        failed,
    };

    struct WorkerEvent final {
        WorkerEventKind kind{WorkerEventKind::waiting_for_input};
        std::uint64_t frame_index{};
        std::uint64_t input_epoch{};
        std::size_t slot_index{};
        std::uint64_t generation{};
        std::optional<Error> error;
    };

    struct CopyGrant final {
        RecordSlot *slot{};
        std::size_t slot_index{};
        std::uint64_t frame_index{};
        std::uint64_t generation{};
    };

    struct ScheduleProgress final {
        std::size_t scheduled_pairs{};
        bool storage_unavailable{};
        bool decoder_input_pending{};
    };

    [[nodiscard]] const FrameDescriptor &input_descriptor() const noexcept { return descriptor_; }

    [[nodiscard]] std::optional<std::uint64_t> input_frame_count_hint() const noexcept {
        return frame_count_hint_;
    }

    [[nodiscard]] std::uint64_t input_decoded_frame_count() const noexcept {
        return next_frame_to_schedule_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const VideoMetadata &input_reference_metadata() const noexcept {
        return reference_metadata_;
    }

    [[nodiscard]] const VideoMetadata &input_distorted_metadata() const noexcept {
        return distorted_metadata_;
    }

    [[nodiscard]] std::optional<video::VideoTimestamp>
    input_take_frame_timestamp(std::uint64_t frame_index) noexcept {
        return slot_for_frame(frame_index).decoded_timestamp;
    }

    [[nodiscard]] Result<FramePairAcquireResult> input_try_acquire_next() {
        try {
            const auto scheduling_result = schedule_available();
            if (!scheduling_result) {
                return Result<FramePairAcquireResult>::failure(scheduling_result.error());
            }
            if (cancellation_requested_.load(std::memory_order_acquire)) {
                return Result<FramePairAcquireResult>::failure(cancelled_error());
            }

            const std::uint64_t next_frame = next_frame_to_acquire_.load(std::memory_order_acquire);
            if (scheduling_complete_.load(std::memory_order_acquire) &&
                next_frame >= input_decoded_frame_count()) {
                return Result<FramePairAcquireResult>::success(
                    {FramePairAcquireStatus::end_of_stream,
                     {},
                     scheduling_result.value().scheduled_pairs,
                     input_epoch_.load(std::memory_order_acquire),
                     progress_epoch_.load(std::memory_order_acquire),
                     worker_input_state_.load(std::memory_order_acquire)});
            }

            RecordSlot &slot = slot_for_frame(next_frame);
            if (next_frame < input_decoded_frame_count() &&
                slot.record.frame_index() == next_frame && slot.record.ready()) {
                if (slot.acquired_by_caller) {
                    return Result<FramePairAcquireResult>::failure(
                        {ErrorCode::invalid_state, "decoded frame-pair record was acquired twice"});
                }
                slot.acquired_by_caller = true;
                next_frame_to_acquire_.store(next_frame + 1U, std::memory_order_release);
                return Result<FramePairAcquireResult>::success(
                    {FramePairAcquireStatus::ready,
                     {&slot.record, slot.record.generation(), slot_index_for_frame(next_frame)},
                     scheduling_result.value().scheduled_pairs,
                     input_epoch_.load(std::memory_order_acquire),
                     progress_epoch_.load(std::memory_order_acquire),
                     worker_input_state_.load(std::memory_order_acquire)});
            }

            const bool downstream_backpressure =
                scheduling_result.value().storage_unavailable ||
                owned_pair_count_.load(std::memory_order_acquire) >= record_capacity_;
            return Result<FramePairAcquireResult>::success(
                {downstream_backpressure ? FramePairAcquireStatus::unavailable
                                         : FramePairAcquireStatus::input_starved,
                 {},
                 scheduling_result.value().scheduled_pairs,
                 input_epoch_.load(std::memory_order_acquire),
                 progress_epoch_.load(std::memory_order_acquire),
                 worker_input_state_.load(std::memory_order_acquire)});
        } catch (const std::bad_alloc &) {
            return Result<FramePairAcquireResult>::failure(
                {ErrorCode::backend_error, "failed to allocate while pairing decoded frames"});
        } catch (const std::exception &exception) {
            return Result<FramePairAcquireResult>::failure(
                {ErrorCode::backend_error,
                 "decoded frame pairing failed: " + std::string(exception.what())});
        } catch (...) {
            return Result<FramePairAcquireResult>::failure(
                {ErrorCode::backend_error, "decoded frame pairing failed with an unknown error"});
        }
    }

    [[nodiscard]] Result<void> input_wait_for_progress(std::uint64_t observed_progress_epoch) {
        const auto wait_started_at = std::chrono::steady_clock::now();
        coordinator_wait_count_.fetch_add(1U, std::memory_order_relaxed);
        while (progress_epoch_.load(std::memory_order_acquire) == observed_progress_epoch &&
               !staging_worker_done_.load(std::memory_order_acquire) &&
               !cancellation_requested_.load(std::memory_order_acquire)) {
            progress_epoch_.wait(observed_progress_epoch, std::memory_order_acquire);
        }
        record_elapsed_nanoseconds(coordinator_wait_nanoseconds_, wait_started_at);
        if (cancellation_requested_.load(std::memory_order_acquire)) {
            return Result<void>::failure(cancelled_error());
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> input_mark_submitted(const ReadyInput &pair) {
        const auto slot_result = validate_acquired_pair(pair);
        if (!slot_result) return Result<void>::failure(slot_result.error());
        RecordSlot &slot = *slot_result.value();
        if (!slot.staging_owns_storage) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "decoded frame-pair storage ownership was already transferred"});
        }
        slot.staging_owns_storage = false;
        return slot.record.mark_submitted(pair.generation);
    }

    [[nodiscard]] Result<void> input_retire(const ReadyInput &pair) {
        const auto slot_result = validate_acquired_pair(pair);
        if (!slot_result) return Result<void>::failure(slot_result.error());
        RecordSlot &slot = *slot_result.value();
        if (slot.staging_owns_storage) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "decoded frame pair must be submitted before retirement"});
        }

        const auto retired = slot.record.mark_retired(pair.generation);
        if (!retired) return retired;
        slot.acquired_by_caller = false;
        decrement_owned_pair_count();
        return Result<void>::success();
    }

    void input_request_cancel() noexcept {
        if (cancellation_requested_.exchange(true, std::memory_order_acq_rel)) {
            return;
        }
        if (reference_decoder_) reference_decoder_->request_cancel();
        if (distorted_decoder_) distorted_decoder_->request_cancel();
        if (copy_grants_) copy_grants_->close();
        worker_events_.close();
        signal_worker_progress();
    }

    [[nodiscard]] Result<void> input_drain() {
        std::lock_guard drain_lock(drain_mutex_);
        if (drained_.load(std::memory_order_acquire)) {
            return Result<void>::success();
        }
        const bool cancellation_requested = cancellation_requested_.load(std::memory_order_acquire);
        if (!cancellation_requested && !scheduling_complete_.load(std::memory_order_acquire)) {
            return Result<void>::failure(
                {ErrorCode::invalid_state, "decoder drain requires end of stream or cancellation"});
        }

        if (staging_worker_.joinable()) staging_worker_.join();
        if (reference_decoder_) reference_decoder_->drain();
        if (distorted_decoder_) distorted_decoder_->drain();
        if (cancellation_requested) {
            const auto cleanup_result = cleanup_cancelled_records();
            if (!cleanup_result) return cleanup_result;
        }
        drained_.store(true, std::memory_order_release);
        return Result<void>::success();
    }

    [[nodiscard]] InputStatistics input_statistics() const {
        const std::uint64_t next_frame_to_schedule =
            next_frame_to_schedule_.load(std::memory_order_acquire);
        const std::uint64_t next_frame_to_acquire =
            next_frame_to_acquire_.load(std::memory_order_acquire);
        const std::uint64_t ordered_pair_count =
            next_frame_to_schedule >= next_frame_to_acquire
                ? next_frame_to_schedule - next_frame_to_acquire
                : 0U;
        return {
            frame_count_hint_,
            next_frame_to_schedule,
            next_frame_to_schedule,
            next_frame_to_acquire,
            record_capacity_,
            owned_pair_count_.load(std::memory_order_acquire),
            static_cast<std::size_t>(std::min<std::uint64_t>(ordered_pair_count, record_capacity_)),
            calculate_ready_lead_frames(),
            frame_pair_copy_calls_.load(std::memory_order_relaxed),
            frame_pair_copy_bytes_.load(std::memory_order_relaxed),
            frame_pair_copy_nanoseconds_.load(std::memory_order_relaxed),
            staging_input_wait_nanoseconds_.load(std::memory_order_relaxed),
            staging_input_wait_count_.load(std::memory_order_relaxed),
            staging_grant_wait_nanoseconds_.load(std::memory_order_relaxed),
            staging_grant_wait_count_.load(std::memory_order_relaxed),
            coordinator_wait_nanoseconds_.load(std::memory_order_relaxed),
            coordinator_wait_count_.load(std::memory_order_relaxed),
            scheduling_complete_.load(std::memory_order_acquire),
            cancellation_requested_.load(std::memory_order_acquire),
            drained_.load(std::memory_order_acquire),
            reference_decoder_->statistics(),
            distorted_decoder_->statistics()};
    }

  private:
    [[nodiscard]] Result<ScheduleProgress> schedule_available() {
        if (cancellation_requested_.load(std::memory_order_acquire)) {
            return Result<ScheduleProgress>::failure(cancelled_error());
        }

        ScheduleProgress progress;
        for (;;) {
            auto worker_event = worker_events_.try_receive();
            if (worker_event.status != ChannelReceiveStatus::value) break;
            const auto processed =
                process_worker_event(std::move(*worker_event.value), &progress.scheduled_pairs);
            if (!processed) {
                return Result<ScheduleProgress>::failure(processed.error());
            }
        }

        const std::uint64_t scheduled_frame_count =
            next_frame_to_schedule_.load(std::memory_order_acquire);
        if (scheduled_frame_count < last_reported_scheduled_frame_count_) {
            return Result<ScheduleProgress>::failure(
                {ErrorCode::invalid_state, "decoded staging progress moved backwards"});
        }
        progress.scheduled_pairs = static_cast<std::size_t>(
            std::min<std::uint64_t>(scheduled_frame_count - last_reported_scheduled_frame_count_,
                                    std::numeric_limits<std::size_t>::max()));
        last_reported_scheduled_frame_count_ = scheduled_frame_count;

        if (worker_error_.has_value()) {
            return Result<ScheduleProgress>::failure(*worker_error_);
        }
        if (staging_worker_done_.load(std::memory_order_acquire)) {
            if (staging_exception_) {
                try {
                    std::rethrow_exception(staging_exception_);
                } catch (const std::exception &exception) {
                    worker_error_ = Error{ErrorCode::backend_error, exception.what()}.with_context(
                        "decoded input staging", ErrorInput::both, worker_frame_index_);
                } catch (...) {
                    worker_error_ = Error{ErrorCode::backend_error, "unknown exception"}.with_context(
                        "decoded input staging", ErrorInput::both, worker_frame_index_);
                }
            } else if (staging_error_) {
                worker_error_ = staging_error_->with_context(
                    "decoded input staging", ErrorInput::both, worker_frame_index_);
            }
            if (worker_error_) return Result<ScheduleProgress>::failure(*worker_error_);
            if (!scheduling_complete_.load(std::memory_order_acquire)) {
                const auto ended = process_worker_event(
                    {WorkerEventKind::end_of_stream, worker_frame_index_,
                     input_epoch_.load(std::memory_order_acquire)}, nullptr);
                if (!ended) return Result<ScheduleProgress>::failure(ended.error());
            }
        }
        if (scheduling_complete_.load(std::memory_order_acquire)) {
            return Result<ScheduleProgress>::success(progress);
        }

        const auto grants_filled = fill_copy_grants();
        if (!grants_filled) {
            return Result<ScheduleProgress>::failure(grants_filled.error());
        }
        progress.storage_unavailable = !grants_filled.value();

        progress.decoder_input_pending =
            !progress.storage_unavailable && !scheduling_complete_.load(std::memory_order_acquire);
        return Result<ScheduleProgress>::success(progress);
    }

    [[nodiscard]] Result<void> process_worker_event(WorkerEvent event,
                                                    std::size_t *scheduled_pair_count) {
        input_epoch_.store(event.input_epoch, std::memory_order_release);
        switch (event.kind) {
        case WorkerEventKind::waiting_for_input:
            worker_input_state_.store(FramePairInputState::waiting_for_decoder,
                                      std::memory_order_release);
            return Result<void>::success();

        case WorkerEventKind::storage_requested:
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "decoded staging worker used the retired request/grant protocol"});

        case WorkerEventKind::copy_completed: {
            if (event.slot_index >= frames_->capacity()) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "decoded staging completion references an invalid slot"});
            }
            RecordSlot &slot = frames_->slot(event.slot_index);
            if (slot.record.frame_index() != event.frame_index ||
                slot.record.generation() != event.generation) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state, "decoded staging completion is stale"});
            }
            if (event.error.has_value()) {
                worker_error_ = std::move(event.error);
                return Result<void>::failure(*worker_error_);
            }
            if (!slot.record.ready()) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "decoded staging completion did not publish a ready record: frame=" +
                         std::to_string(event.frame_index) +
                         ", slot=" + std::to_string(event.slot_index) +
                         ", state=" + std::to_string(static_cast<unsigned>(slot.record.state()))});
            }
            const std::uint64_t expected_frame =
                next_frame_to_schedule_.load(std::memory_order_acquire);
            if (event.frame_index != expected_frame) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state, "decoded staging completions were not contiguous"});
            }
            next_frame_to_schedule_.store(event.frame_index + 1U, std::memory_order_release);
            worker_input_state_.store(FramePairInputState::waiting_for_decoder,
                                      std::memory_order_release);
            if (scheduled_pair_count != nullptr) ++*scheduled_pair_count;
            return Result<void>::success();
        }

        case WorkerEventKind::end_of_stream: {
            if (event.frame_index != next_frame_to_schedule_.load(std::memory_order_acquire)) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "decoded staging worker ended before all completions were observed"});
            }
            const auto unused_grants_abandoned = abandon_unused_grants(event.frame_index);
            if (!unused_grants_abandoned) {
                return unused_grants_abandoned;
            }
            scheduling_complete_.store(true, std::memory_order_release);
            worker_input_state_.store(FramePairInputState::terminal, std::memory_order_release);
            return Result<void>::success();
        }

        case WorkerEventKind::failed:
            worker_input_state_.store(FramePairInputState::terminal, std::memory_order_release);
            worker_error_ =
                event.error.has_value()
                    ? std::move(event.error)
                    : std::optional<Error>{Error{ErrorCode::backend_error,
                                                 "decoded staging worker failed without an error"}};
            return Result<void>::failure(*worker_error_);
        }
        return Result<void>::failure(
            {ErrorCode::invalid_state, "decoded staging worker published an unknown event"});
    }

    [[nodiscard]] Result<bool> fill_copy_grants() {
        bool grant_window_has_capacity = true;
        while (owned_pair_count_.load(std::memory_order_acquire) < record_capacity_) {
            if (!frames_->has_registration_capacity()) {
                grant_window_has_capacity = false;
                break;
            }
            const std::uint64_t frame_index = next_frame_to_reserve_;
            RecordSlot &slot = slot_for_frame(frame_index);
            if (slot.record.state() != FramePairRecordState::free) {
                grant_window_has_capacity = false;
                break;
            }

            auto reservation_result = reserve_input(descriptor_, frame_index);
            if (!reservation_result) {
                return Result<bool>::failure(reservation_result.error());
            }
            if (!reservation_result.value().has_value()) {
                grant_window_has_capacity = false;
                break;
            }
            const FramePairStorageReservation reservation = *reservation_result.value();
            const auto reservation_validation = validate_reservation(reservation, frame_index);
            if (!reservation_validation) {
                static_cast<void>(abandon_input(reservation));
                return Result<bool>::failure(reservation_validation.error());
            }
            if (next_generation_ == 0U) {
                static_cast<void>(abandon_input(reservation));
                return Result<bool>::failure({ErrorCode::invalid_state,
                                              "decoded frame-pair generation space was exhausted"});
            }

            const auto registered = frames_->register_frame(frame_index);
            if (!registered) {
                static_cast<void>(abandon_input(reservation));
                return Result<bool>::failure(registered.error());
            }
            const std::uint64_t generation = next_generation_;
            next_generation_ =
                generation == std::numeric_limits<std::uint64_t>::max() ? 0U : generation + 1U;
            const auto scheduled = slot.record.schedule(
                {frame_index, generation, reservation.frame_pair_storage,
                 reservation.reference_luma_storage, reservation.reference_destination,
                 reservation.distorted_destination, reservation.mode});
            if (!scheduled) {
                static_cast<void>(abandon_input(reservation));
                return Result<bool>::failure(scheduled.error());
            }
            slot.staging_owns_storage = true;
            slot.acquired_by_caller = false;
            owned_pair_count_.fetch_add(1U, std::memory_order_acq_rel);
            ++next_frame_to_reserve_;

            const bool grant_sent = copy_grants_->send(
                {&slot, slot_index_for_frame(frame_index), frame_index, generation});
            if (!grant_sent) {
                const auto abandoned = abandon_record(slot, generation);
                if (!abandoned) {
                    return Result<bool>::failure(abandoned.error());
                }
                return Result<bool>::failure(cancelled_error());
            }
        }
        if (owned_pair_count_.load(std::memory_order_acquire) >= record_capacity_) {
            grant_window_has_capacity = false;
        }
        return Result<bool>::success(grant_window_has_capacity);
    }

    void run_staging_worker() noexcept {
        try {
            auto worker_result = run_staging_worker_impl();
            if (!worker_result) {
                staging_error_.emplace(std::move(worker_result).error());
            }
        } catch (...) {
            staging_exception_ = std::current_exception();
        }
        // Terminal publication cannot allocate or block behind a full event
        // queue. The consumer acquires this flag before reading the outcome.
        staging_worker_done_.store(true, std::memory_order_release);
        worker_events_.close();
        signal_worker_progress();
    }

    void signal_worker_progress() noexcept {
        progress_epoch_.fetch_add(1U, std::memory_order_release);
        progress_epoch_.notify_all();
    }

    [[nodiscard]] Result<void> run_staging_worker_impl() {
        for (std::uint64_t frame_index = 0U;; ++frame_index) {
            worker_frame_index_ = frame_index;
            const std::uint64_t input_epoch = frame_index + 1U;
            input_epoch_.store(input_epoch, std::memory_order_release);
            worker_input_state_.store(FramePairInputState::waiting_for_decoder,
                                      std::memory_order_release);

            const auto input_wait_started_at = std::chrono::steady_clock::now();
            staging_input_wait_count_.fetch_add(1U, std::memory_order_relaxed);
            auto reference_pop = reference_decoder_->wait_pop();
            if (!reference_pop) {
                if (cancellation_requested_.load(std::memory_order_acquire)) {
                    return Result<void>::success();
                }
                return Result<void>::failure(reference_pop.error().with_context(
                    "reference input", ErrorInput::reference, frame_index));
            }
            auto distorted_pop = distorted_decoder_->wait_pop();
            if (!distorted_pop) {
                if (cancellation_requested_.load(std::memory_order_acquire)) {
                    return Result<void>::success();
                }
                return Result<void>::failure(distorted_pop.error().with_context(
                    "distorted input", ErrorInput::distorted, frame_index));
            }
            record_elapsed_nanoseconds(staging_input_wait_nanoseconds_, input_wait_started_at);

            video::VideoFramePopResult reference_result = std::move(reference_pop).value();
            video::VideoFramePopResult distorted_result = std::move(distorted_pop).value();
            const bool reference_end_of_stream =
                reference_result.status == video::VideoFramePopStatus::end_of_stream;
            const bool distorted_end_of_stream =
                distorted_result.status == video::VideoFramePopStatus::end_of_stream;
            if (reference_end_of_stream || distorted_end_of_stream) {
                if (reference_end_of_stream && distorted_end_of_stream) {
                    return Result<void>::success();
                }
                return Result<void>::failure(
                    {ErrorCode::invalid_argument,
                     "reference and distorted videos decoded different frame counts"});
            }
            if (reference_result.status != video::VideoFramePopStatus::frame ||
                distorted_result.status != video::VideoFramePopStatus::frame) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state, "blocking video decoder returned no frame"});
            }

            video::Frame reference_frame = std::move(reference_result.frame);
            video::Frame distorted_frame = std::move(distorted_result.frame);
            const auto timestamps_valid =
                validate_frame_timestamps(reference_frame, distorted_frame, frame_index);
            if (!timestamps_valid) return timestamps_valid;

            worker_input_state_.store(FramePairInputState::waiting_for_storage,
                                      std::memory_order_release);
            const auto grant_wait_started_at = std::chrono::steady_clock::now();
            staging_grant_wait_count_.fetch_add(1U, std::memory_order_relaxed);
            auto grant_result = copy_grants_->receive();
            record_elapsed_nanoseconds(staging_grant_wait_nanoseconds_, grant_wait_started_at);
            if (grant_result.status != ChannelReceiveStatus::value) {
                if (cancellation_requested_.load(std::memory_order_acquire)) {
                    return Result<void>::success();
                }
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "decoded staging storage grant channel closed unexpectedly"});
            }
            const CopyGrant grant = std::move(*grant_result.value);
            if (grant.slot == nullptr || grant.frame_index != frame_index ||
                grant.slot_index != slot_index_for_frame(frame_index) || grant.generation == 0U) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "decoded staging worker received an invalid storage grant"});
            }

            worker_input_state_.store(FramePairInputState::copying, std::memory_order_release);
            grant.slot->decoded_timestamp = reference_frame.timestamp();
            const auto copied =
                copy_pair_to_record(*grant.slot, grant.generation, std::move(reference_frame),
                                    std::move(distorted_frame));
            if (!copied) return copied;
            next_frame_to_schedule_.store(frame_index + 1U, std::memory_order_release);
            signal_worker_progress();
        }
    }

    [[nodiscard]] Result<void> validate_frame_timestamps(const video::Frame &reference,
                                                         const video::Frame &distorted,
                                                         std::uint64_t frame_index) {
        const video::VideoTimestamp reference_timestamp = reference.timestamp();
        const video::VideoTimestamp distorted_timestamp = distorted.timestamp();
        if (reference_timestamp.valid != distorted_timestamp.valid) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "reference and distorted video timestamps are not comparable"});
        }
        if (!reference_timestamp.valid) {
            return Result<void>::success();
        }
        if (reference_timestamp.time_base_denominator <= 0 ||
            distorted_timestamp.time_base_denominator <= 0) {
            return Result<void>::failure({ErrorCode::unsupported_format,
                                          "decoded video uses an invalid timestamp time base"});
        }

        if (!reference_timestamp_origin_.has_value()) {
            reference_timestamp_origin_ = reference_timestamp.value;
            distorted_timestamp_origin_ = distorted_timestamp.value;
        }
        const long double reference_seconds =
            (static_cast<long double>(reference_timestamp.value) -
             static_cast<long double>(*reference_timestamp_origin_)) *
            static_cast<long double>(reference_timestamp.time_base_numerator) /
            static_cast<long double>(reference_timestamp.time_base_denominator);
        const long double distorted_seconds =
            (static_cast<long double>(distorted_timestamp.value) -
             static_cast<long double>(*distorted_timestamp_origin_)) *
            static_cast<long double>(distorted_timestamp.time_base_numerator) /
            static_cast<long double>(distorted_timestamp.time_base_denominator);
        const long double reference_tick_seconds =
            std::abs(static_cast<long double>(reference_timestamp.time_base_numerator) /
                     static_cast<long double>(reference_timestamp.time_base_denominator));
        const long double distorted_tick_seconds =
            std::abs(static_cast<long double>(distorted_timestamp.time_base_numerator) /
                     static_cast<long double>(distorted_timestamp.time_base_denominator));
        const long double timestamp_tolerance_seconds =
            std::clamp((reference_tick_seconds + distorted_tick_seconds) / 2.0L, 1.0e-6L, 5.0e-3L);
        if (std::abs(reference_seconds - distorted_seconds) > timestamp_tolerance_seconds) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "reference and distorted video presentation timestamps diverge at frame " +
                     std::to_string(frame_index)});
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> copy_pair_to_record(RecordSlot &slot, std::uint64_t generation,
                                                   video::Frame reference_frame,
                                                   video::Frame distorted_frame) {
        FramePairRecord &record = slot.record;
        const auto reference_claim = record.claim_decode(FramePairSide::reference, generation);
        if (!reference_claim) return reference_claim;
        const auto distorted_claim = record.claim_decode(FramePairSide::distorted, generation);
        if (!distorted_claim) {
            static_cast<void>(
                record.complete_decode(FramePairSide::reference, generation,
                                       Result<void>::failure(distorted_claim.error())));
            return distorted_claim;
        }

        const auto reference_decode_started_at = reference_frame.decode_started_at();
        const auto distorted_decode_started_at = distorted_frame.decode_started_at();
        const auto reference_decode_completed_at = reference_frame.decode_completed_at();
        const auto distorted_decode_completed_at = distorted_frame.decode_completed_at();
        const auto reference_started = record.mark_decode_started(
            FramePairSide::reference, generation, reference_decode_started_at);
        if (!reference_started) {
            static_cast<void>(
                record.complete_decode(FramePairSide::reference, generation,
                                       Result<void>::failure(reference_started.error())));
            static_cast<void>(
                record.complete_decode(FramePairSide::distorted, generation,
                                       Result<void>::failure(reference_started.error())));
            return reference_started;
        }
        const auto distorted_started = record.mark_decode_started(
            FramePairSide::distorted, generation, distorted_decode_started_at);
        if (!distorted_started) {
            static_cast<void>(
                record.complete_decode(FramePairSide::reference, generation,
                                       Result<void>::failure(distorted_started.error())));
            static_cast<void>(
                record.complete_decode(FramePairSide::distorted, generation,
                                       Result<void>::failure(distorted_started.error())));
            return distorted_started;
        }

        const FramePairStorageReservation reservation{
            record.frame_pair_storage(), record.reference_luma_storage(),
            record.destination(FramePairSide::reference),
            record.destination(FramePairSide::distorted), record.storage_mode()};
        const bool cpu_copy = reservation.mode == FramePairStorageReservation::Mode::cpu_writable;
        const auto reference_write_started_at = std::chrono::steady_clock::now();
        const auto reference_write =
            write_decoded_input(reservation, FramePairSide::reference, std::move(reference_frame));
        const auto reference_write_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - reference_write_started_at);
        if (cpu_copy) {
            frame_pair_copy_nanoseconds_.fetch_add(
                static_cast<std::uint64_t>(
                    std::max<std::int64_t>(reference_write_elapsed.count(), 0)),
                std::memory_order_relaxed);
        }
        const auto reference_completed = record.complete_decode(
            FramePairSide::reference, generation, reference_write, reference_decode_completed_at);
        if (!reference_completed) return reference_completed;

        const auto distorted_write_started_at = std::chrono::steady_clock::now();
        const auto distorted_write =
            write_decoded_input(reservation, FramePairSide::distorted, std::move(distorted_frame));
        const auto distorted_write_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - distorted_write_started_at);
        if (cpu_copy) {
            frame_pair_copy_nanoseconds_.fetch_add(
                static_cast<std::uint64_t>(
                    std::max<std::int64_t>(distorted_write_elapsed.count(), 0)),
                std::memory_order_relaxed);
        }
        const auto distorted_completed = record.complete_decode(
            FramePairSide::distorted, generation, distorted_write, distorted_decode_completed_at);
        if (!distorted_completed) return distorted_completed;
        if (!reference_write) return Result<void>::failure(reference_write.error().with_context(
            "reference input copy", ErrorInput::reference, record.frame_index()));
        if (!distorted_write) return Result<void>::failure(distorted_write.error().with_context(
            "distorted input copy", ErrorInput::distorted, record.frame_index()));

        if (cpu_copy) {
            frame_pair_copy_calls_.fetch_add(1U, std::memory_order_relaxed);
            frame_pair_copy_bytes_.fetch_add(frame_pair_active_bytes(descriptor_),
                                             std::memory_order_relaxed);
        }

        return record.wait_for_decodes();
    }

    [[nodiscard]] Result<void> validate_reservation(const FramePairStorageReservation &reservation,
                                                    std::uint64_t frame_index) const {
        if (!reservation.frame_pair_storage || !reservation.reference_luma_storage ||
            reservation.reference_destination.index != frame_index ||
            reservation.distorted_destination.index != frame_index) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "storage provider returned an invalid decoded frame reservation"});
        }
        if (reservation.mode == FramePairStorageReservation::Mode::cpu_writable) {
            for (const WritablePlaneView &plane : reservation.reference_destination.planes) {
                if (plane.data == nullptr || plane.size_bytes == 0U || plane.stride_bytes == 0U) {
                    return Result<void>::failure(
                        {ErrorCode::invalid_argument,
                         "storage provider returned an invalid reference plane"});
                }
            }
            for (const WritablePlaneView &plane : reservation.distorted_destination.planes) {
                if (plane.data == nullptr || plane.size_bytes == 0U || plane.stride_bytes == 0U) {
                    return Result<void>::failure(
                        {ErrorCode::invalid_argument,
                         "storage provider returned an invalid distorted plane"});
                }
            }
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> abandon_record(RecordSlot &slot, std::uint64_t generation) {
        FramePairRecord &record = slot.record;
        // Either these grants were never consumed (dual EOF), or input_drain
        // has joined the staging worker. An exception after claim_decode must
        // not leave wait_for_decodes waiting for a producer that has exited.
        const auto sealed = record.seal_cancelled_decodes(generation);
        if (!sealed) return sealed;
        static_cast<void>(record.wait_for_decodes());
        const FramePairStorageReservation reservation{
            record.frame_pair_storage(), record.reference_luma_storage(),
            record.destination(FramePairSide::reference),
            record.destination(FramePairSide::distorted), record.storage_mode()};
        const auto abandoned = record.abandon(generation);
        if (!abandoned) return abandoned;
        const auto storage_abandoned = abandon_input(reservation);
        if (!storage_abandoned) return storage_abandoned;
        const auto recycled = record.recycle(generation);
        if (!recycled) return recycled;
        slot.staging_owns_storage = false;
        slot.acquired_by_caller = false;
        decrement_owned_pair_count();
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> cleanup_cancelled_records() {
        if (!frames_) return Result<void>::success();
        for (std::size_t slot_index = 0U; slot_index < frames_->capacity(); ++slot_index) {
            RecordSlot &slot = frames_->slot(slot_index);
            if (!slot.staging_owns_storage) continue;
            const auto abandoned = abandon_record(slot, slot.record.generation());
            if (!abandoned) return abandoned;
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> abandon_unused_grants(std::uint64_t decoded_frame_count) {
        while (next_frame_to_reserve_ > decoded_frame_count) {
            const std::uint64_t frame_index = next_frame_to_reserve_ - 1U;
            RecordSlot &slot = slot_for_frame(frame_index);
            if (!slot.staging_owns_storage || slot.record.frame_index() != frame_index) {
                return Result<void>::failure(
                    {ErrorCode::invalid_state,
                     "decoded staging grant window lost an unused reservation"});
            }
            const auto abandoned = abandon_record(slot, slot.record.generation());
            if (!abandoned) return abandoned;
            const auto abandoned_frame = frames_->abandon_latest(frame_index);
            if (!abandoned_frame) return abandoned_frame;
            --next_frame_to_reserve_;
        }
        return Result<void>::success();
    }

    [[nodiscard]] Result<RecordSlot *> validate_acquired_pair(const ReadyInput &pair) {
        if (!pair || pair.slot_index >= frames_->capacity()) {
            return Result<RecordSlot *>::failure(
                {ErrorCode::invalid_argument, "decoded frame-pair handle is invalid"});
        }
        RecordSlot &slot = frames_->slot(pair.slot_index);
        if (&slot.record != pair.record || !slot.acquired_by_caller ||
            slot.record.generation() != pair.generation) {
            return Result<RecordSlot *>::failure(
                {ErrorCode::invalid_state, "decoded frame-pair handle no longer owns its record"});
        }
        return Result<RecordSlot *>::success(&slot);
    }

    [[nodiscard]] std::size_t slot_index_for_frame(std::uint64_t frame_index) const noexcept {
        return static_cast<std::size_t>(frame_index %
                                        static_cast<std::uint64_t>(frames_->capacity()));
    }

    [[nodiscard]] RecordSlot &slot_for_frame(std::uint64_t frame_index) noexcept {
        return frames_->slot(slot_index_for_frame(frame_index));
    }

    [[nodiscard]] const RecordSlot &slot_for_frame(std::uint64_t frame_index) const noexcept {
        return frames_->slot(slot_index_for_frame(frame_index));
    }

    [[nodiscard]] std::size_t calculate_ready_lead_frames() const noexcept {
        const auto acquired = next_frame_to_acquire_.load(std::memory_order_acquire);
        const auto decoded = next_frame_to_schedule_.load(std::memory_order_acquire);
        return static_cast<std::size_t>(
            decoded > acquired ? std::min<std::uint64_t>(decoded - acquired, record_capacity_)
                               : 0U);
    }

    void decrement_owned_pair_count() noexcept {
        const std::size_t previous_count =
            owned_pair_count_.fetch_sub(1U, std::memory_order_acq_rel);
        if (previous_count == 0U) {
            owned_pair_count_.store(0U, std::memory_order_release);
        }
    }

    std::optional<std::uint64_t> frame_count_hint_;
    VideoMetadata reference_metadata_;
    VideoMetadata distorted_metadata_;
    std::size_t record_capacity_{};
    std::unique_ptr<video::Decoder> reference_decoder_;
    std::unique_ptr<video::Decoder> distorted_decoder_;
    BlockingBoundedChannel<WorkerEvent> worker_events_{8U};
    std::unique_ptr<BlockingBoundedChannel<CopyGrant>> copy_grants_;
    std::thread staging_worker_;
    std::atomic<bool> staging_worker_done_{};
    std::exception_ptr staging_exception_;
    std::optional<Error> staging_error_;
    std::optional<Error> worker_error_;
    std::atomic<FramePairInputState> worker_input_state_{FramePairInputState::waiting_for_decoder};
    std::optional<std::int64_t> reference_timestamp_origin_;
    std::optional<std::int64_t> distorted_timestamp_origin_;
    std::uint64_t worker_frame_index_{};
    std::uint64_t next_frame_to_reserve_{};
    std::uint64_t last_reported_scheduled_frame_count_{};
    std::uint64_t next_generation_{1U};
    std::atomic<std::uint64_t> input_epoch_{};
    std::atomic<std::uint64_t> progress_epoch_{};
    std::atomic<std::uint64_t> next_frame_to_schedule_{};
    std::atomic<std::uint64_t> next_frame_to_acquire_{};
    std::atomic<std::size_t> owned_pair_count_{};
    std::atomic<std::uint64_t> frame_pair_copy_calls_{};
    std::atomic<std::uint64_t> frame_pair_copy_bytes_{};
    std::atomic<std::uint64_t> frame_pair_copy_nanoseconds_{};
    std::atomic<std::uint64_t> staging_input_wait_nanoseconds_{};
    std::atomic<std::uint64_t> staging_input_wait_count_{};
    std::atomic<std::uint64_t> staging_grant_wait_nanoseconds_{};
    std::atomic<std::uint64_t> staging_grant_wait_count_{};
    std::atomic<std::uint64_t> coordinator_wait_nanoseconds_{};
    std::atomic<std::uint64_t> coordinator_wait_count_{};
    std::atomic<bool> scheduling_complete_{};
    std::atomic<bool> cancellation_requested_{};
    std::atomic<bool> drained_{};
    std::mutex drain_mutex_;

  public:
    [[nodiscard]] Result<void> initialize_input(const DecoderOpenOptions &options,
                                                PreparedVideoDecoderPair prepared,
                                                ModelVariant model,
                                                InitializationObserver *observer) {
        model_ = model;
        descriptor_ = prepared.descriptor;
        frame_count_hint_ = prepared.frame_count_hint;
        vulkan_image_input_ = prepared.vulkan_image_input;
        record_capacity_ = options.record_capacity;
        reference_decoder_ = std::move(prepared.reference);
        distorted_decoder_ = std::move(prepared.distorted);
        reference_metadata_ = reference_decoder_->metadata();
        distorted_metadata_ = distorted_decoder_->metadata();
        metadata_ = {reference_metadata_, distorted_metadata_};
        const auto storage_admission =
            observe_initialization(observer, InitializationStage::storage_provider);
        if (!storage_admission) return storage_admission;
        const auto capacity_admission =
            observe_initialization(observer, InitializationStage::input_capacity);
        if (!capacity_admission) return capacity_admission;
        const auto capacity = ensure_segmented_input_capacity(record_capacity_);
        if (!capacity) return capacity;
        copy_grants_ = std::make_unique<BlockingBoundedChannel<CopyGrant>>(record_capacity_);
        frames_ = std::make_unique<FrameRecords>(
            model, prepare_result_buffer_capacity(maximum_streaming_backlog_capacity() +
                                                      frame_slot_count() + record_capacity_ + 2U,
                                                  observer));
        staging_worker_ = std::thread([this] { run_staging_worker(); });
        return Result<void>::success();
    }

  private:
    [[nodiscard]] Result<std::optional<FramePairStorageReservation>>
    reserve_input(const FrameDescriptor &, std::uint64_t frame_index) {
        auto reserved = input_pool_->try_reserve(frame_index);
        if (!reserved)
            return Result<std::optional<FramePairStorageReservation>>::failure(reserved.error());
        if (!reserved.value())
            return Result<std::optional<FramePairStorageReservation>>::success(std::nullopt);
        const auto &writable = *reserved.value();
        return Result<std::optional<FramePairStorageReservation>>::success(
            FramePairStorageReservation{writable.token.spatial, writable.token.reference_luma,
                                        writable.reference, writable.distorted,
                                        vulkan_image_input_
                                            ? FramePairStorageReservation::Mode::vulkan_decoded
                                            : FramePairStorageReservation::Mode::cpu_writable});
    }

    [[nodiscard]] Result<void> abandon_input(const FramePairStorageReservation &reservation) {
        input_pool_->abandon({reservation.reference_destination.index,
                              reservation.frame_pair_storage, reservation.reference_luma_storage});
        return Result<void>::success();
    }

    [[nodiscard]] Result<void> write_decoded_input(const FramePairStorageReservation &reservation,
                                                   FramePairSide side, video::Frame frame) {
        if (!vulkan_image_input_) {
            return frame.copy_to(side == FramePairSide::reference
                                     ? reservation.reference_destination
                                     : reservation.distorted_destination);
        }
        auto &logical_frame = slot_for_frame(reservation.reference_destination.index);
        auto &owned_frame = side == FramePairSide::reference ? logical_frame.hardware_reference
                                                             : logical_frame.hardware_distorted;
        owned_frame = std::move(frame);
        return input_pool_->store_hardware_decoded_frame({reservation.reference_destination.index,
                                                          reservation.frame_pair_storage,
                                                          reservation.reference_luma_storage},
                                                         side, owned_frame);
    }

    bool vulkan_image_input_{};
};

Result<std::unique_ptr<EvaluationStream>> EvaluationStream::open(StreamOptions options) {
    return detail::EvaluationStreamTelemetryOpener::open(std::move(options));
}

Result<std::unique_ptr<EvaluationStream>> EvaluationStream::open(StreamOptions options,
                                                                 std::stop_token stop_token) {
    return detail::EvaluationStreamTelemetryOpener::open(std::move(options), nullptr, nullptr,
                                                         stop_token);
}

namespace {

[[nodiscard]] Error open_cancelled_error() {
    return {ErrorCode::cancelled, "evaluation stream open was cancelled"};
}

} // namespace

Result<std::unique_ptr<EvaluationStream>> detail::EvaluationStreamTelemetryOpener::open(
    StreamOptions options,
    std::shared_ptr<evaluation::telemetry::SnapshotSink> telemetry_snapshot_sink,
    evaluation::InitializationObserver *initialization_observer, std::stop_token stop_token) {
    try {
        const auto options_validation = validate_stream_options(options);
        if (!options_validation) {
            return Result<std::unique_ptr<EvaluationStream>>::failure(options_validation.error());
        }
        if (stop_token.stop_requested()) {
            return Result<std::unique_ptr<EvaluationStream>>::failure(open_cancelled_error());
        }

        evaluation::DecoderOpenOptions decoder_open_options;
        decoder_open_options.reference_path = options.reference_path;
        decoder_open_options.distorted_path = options.distorted_path;
        decoder_open_options.decoded_frame_queue_capacity = options.decoded_frame_queue_capacity;
        decoder_open_options.decoder_threads_per_source = options.decoder_threads_per_source;
        decoder_open_options.stop_token = stop_token;
        decoder_open_options.require_hardware_decode = options.decode_mode == DecodeMode::hardware;

        auto prepared_vulkan_device =
            gpu::Device::create(options.device, gpu::generated::embedded_spirv_manifest_sha256,
                gpu::VideoImageLayoutPolicy::automatic,
                {options.decode_mode != DecodeMode::software &&
                     (options.decode_backend == DecodeBackendPreference::automatic || options.decode_backend == DecodeBackendPreference::vulkan_video),
                 options.decode_mode != DecodeMode::software && options.decode_transfer != DecodeTransferPolicy::host &&
                     (options.decode_backend == DecodeBackendPreference::qsv || options.decode_backend == DecodeBackendPreference::nvdec ||
                      options.decode_backend == DecodeBackendPreference::vaapi || options.decode_mode == DecodeMode::hardware)});
        if (!prepared_vulkan_device) {
            return Result<std::unique_ptr<EvaluationStream>>::failure(
                prepared_vulkan_device.error());
        }
        if (initialization_observer != nullptr) {
            initialization_observer->device_prepared(*prepared_vulkan_device.value());
        }
        decoder_open_options.selection = options;
        decoder_open_options.compute_device = prepared_vulkan_device.value().get();
        decoder_open_options.transfer_device = video::prepare_decode_device(*prepared_vulkan_device.value());
        auto prepared_decoders = evaluation::prepare_decoders(decoder_open_options);
        if (!prepared_decoders) {
            // The decoder-open phase is interruptible through the stop
            // token; the decoder classifies an actual FFmpeg abort as cancelled.
            return Result<std::unique_ptr<EvaluationStream>>::failure(
                prepared_decoders.error());
        }
        if (initialization_observer != nullptr) {
            initialization_observer->decoders_prepared(
                prepared_decoders.value().reference->decode_backend() != StreamDecodeBackend::software);
        }

        if (options.decode_mode == DecodeMode::automatic && options.decode_backend == DecodeBackendPreference::automatic &&
            options.decode_transfer != DecodeTransferPolicy::gpu_only && decoder_open_options.transfer_device != nullptr &&
            !prepared_decoders.value().vulkan_image_input) {
            // Both decoders are now software-only, so no worker can submit
            // Vulkan work. A failed hardware attempt may have lost the shared
            // device even though reopening the inputs in software succeeded.
            const auto device_view = prepared_vulkan_device.value()->device_view();
            const VkResult idle_result = device_view.table->vkDeviceWaitIdle(device_view.device);
            if (idle_result == VK_ERROR_DEVICE_LOST) {
                // Release our decoders and device references before replacing
                // the context; FFmpeg-owned native references retain their own
                // device lifetime. Reopen in software without the new device.
                prepared_decoders.value() = {};
                decoder_open_options.decode_device.reset();
                decoder_open_options.transfer_device.reset();
                prepared_vulkan_device.value().reset();
                prepared_vulkan_device = gpu::Device::create(
                    options.device, gpu::generated::embedded_spirv_manifest_sha256);
                if (!prepared_vulkan_device) {
                    return Result<std::unique_ptr<EvaluationStream>>::failure(
                        prepared_vulkan_device.error());
                }
                if (initialization_observer != nullptr) {
                    initialization_observer->device_prepared(*prepared_vulkan_device.value());
                }
                decoder_open_options.compute_device = prepared_vulkan_device.value().get();
                decoder_open_options.selection.decode_mode = DecodeMode::software;
                decoder_open_options.selection.decode_backend = DecodeBackendPreference::automatic;
                prepared_decoders = evaluation::prepare_decoders(decoder_open_options);
                if (!prepared_decoders) {
                    return Result<std::unique_ptr<EvaluationStream>>::failure(
                        prepared_decoders.error());
                }
                if (initialization_observer != nullptr) {
                    initialization_observer->decoders_prepared(false);
                }
            } else if (idle_result != VK_SUCCESS) {
                return Result<std::unique_ptr<EvaluationStream>>::failure(gpu::make_error(
                    "vkDeviceWaitIdle after software decode fallback", idle_result));
            }
        }
        const FrameDescriptor descriptor = prepared_decoders.value().descriptor;
        if (stop_token.stop_requested()) {
            return Result<std::unique_ptr<EvaluationStream>>::failure(open_cancelled_error());
        }

        const ModelVariant model = options.model.value_or(select_model(
            descriptor, prepared_decoders.value().reference->metadata().frame_rate));
        auto model_snapshot = scoring::make_model_snapshot(model);
        if (!model_snapshot) {
            return Result<std::unique_ptr<EvaluationStream>>::failure(model_snapshot.error());
        }

        evaluation::EvaluationGpuConfig session_config{};
        session_config.initialization_observer = initialization_observer;
        session_config.frame = descriptor;
        session_config.device = options.device;
        session_config.vulkan_image_input = prepared_decoders.value().vulkan_image_input;
        session_config.telemetry_snapshot_sink = std::move(telemetry_snapshot_sink);
        auto evaluation = EvaluationStream::Impl::create_prepared(
            session_config, model_snapshot.value(), std::move(prepared_vulkan_device).value());
        if (!evaluation) {
            return Result<std::unique_ptr<EvaluationStream>>::failure(evaluation.error());
        }
        decoder_open_options.record_capacity =
            std::max<std::size_t>(2U, evaluation.value()->frame_slot_count() + 1U);
        const auto initialized = evaluation.value()->initialize_input(
            decoder_open_options, std::move(prepared_decoders).value(), model,
            initialization_observer);
        if (!initialized)
            return Result<std::unique_ptr<EvaluationStream>>::failure(initialized.error());
        return Result<std::unique_ptr<EvaluationStream>>::success(
            std::unique_ptr<EvaluationStream>(new EvaluationStream(std::move(evaluation).value())));
    } catch (const std::exception &exception) {
        return Result<std::unique_ptr<EvaluationStream>>::failure(
            unexpected_stream_exception("decoded open()", &exception));
    } catch (...) {
        return Result<std::unique_ptr<EvaluationStream>>::failure(
            unexpected_stream_exception("decoded open()", nullptr));
    }
}

Result<std::vector<DeviceInfo>> EvaluationStream::enumerate_devices() {
    return evaluation::enumerate_devices();
}

EvaluationStream::EvaluationStream(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
}

EvaluationStream::~EvaluationStream() = default;
EvaluationStream::EvaluationStream(EvaluationStream &&) noexcept = default;
EvaluationStream &EvaluationStream::operator=(EvaluationStream &&) noexcept = default;

const DeviceInfo &EvaluationStream::device_info() const noexcept {
    return impl_->device_info();
}

const FrameDescriptor &EvaluationStream::descriptor() const noexcept {
    return impl_->descriptor();
}

ModelVariant EvaluationStream::model() const noexcept {
    return impl_->model();
}

const StreamMediaMetadata &EvaluationStream::metadata() const noexcept {
    return impl_->metadata();
}

std::optional<std::uint64_t> EvaluationStream::frame_count_hint() const noexcept {
    return impl_->frame_count_hint();
}

std::uint64_t EvaluationStream::decoded_frame_count() const noexcept {
    return impl_->decoded_frame_count();
}

EvaluationStreamStatistics EvaluationStream::statistics() const {
    return impl_->statistics();
}

void EvaluationStream::request_cancel() noexcept {
    impl_->request_cancel();
}

bool EvaluationStream::cancellation_requested() const noexcept {
    return impl_->cancellation_requested();
}

Result<std::vector<FrameResult>> EvaluationStream::next() {
    return impl_->next();
}

Result<std::vector<FrameResult>> EvaluationStream::finish() {
    return impl_->finish();
}

} // namespace crossvmaf

namespace crossvmaf {

DeviceSelfTestResult
detail::EvaluationStreamTelemetryOpener::run_self_test(const DeviceSelector &device) {
    // Same small input the CLI self-test uses: large enough to exercise every
    // extractor path, small enough to finish quickly.
    constexpr FrameDescriptor descriptor{320U, 216U, 8U, PixelFormat::yuv420p};

    DeviceSelfTestResult result;
    const auto start = std::chrono::steady_clock::now();
    const auto finish = [&result, start] {
        result.elapsed_milliseconds =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        return result;
    };

    auto model_snapshot = scoring::make_model_snapshot(default_model);
    if (!model_snapshot) {
        result.checks.push_back({"model_snapshot", false, model_snapshot.error().message});
        result.error = model_snapshot.error();
        return finish();
    }
    result.checks.push_back({"model_snapshot", true, {}});

    evaluation::EvaluationGpuConfig session_config{};
    session_config.frame = descriptor;
    session_config.device = device;
    auto prepared_device =
        gpu::Device::create(device, gpu::generated::embedded_spirv_manifest_sha256);
    auto session =
        prepared_device
            ? EvaluationStream::Impl::create_prepared(session_config, model_snapshot.value(),
                                                      std::move(prepared_device).value())
            : Result<std::unique_ptr<EvaluationStream::Impl>>::failure(prepared_device.error());
    if (!session) {
        result.checks.push_back({"device_and_integer_pipeline", false, session.error().message});
        result.error = session.error();
        return finish();
    }
    result.checks.push_back({"device_and_integer_pipeline", true, {}});
    result.device = session.value()->device_info();
    result.passed = true;
    return finish();
}

} // namespace crossvmaf

namespace crossvmaf {
DeviceSelfTestResult run_device_self_test(const DeviceSelector &device) {
    return detail::EvaluationStreamTelemetryOpener::run_self_test(device);
}
} // namespace crossvmaf
