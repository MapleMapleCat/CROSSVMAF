#include "video/decoder.hpp"
#include "video/ffmpeg.hpp"
#include "video/decode_backend.hpp"
#include "video/frame_transfer.hpp"
#include "video/backends/vulkan.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/cpu.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/pixfmt.h>
}

namespace crossvmaf::video {
namespace {

[[nodiscard]] bool descriptors_match(const FrameDescriptor &left,
                                     const FrameDescriptor &right) noexcept {
    return left.width == right.width && left.height == right.height &&
           left.bit_depth == right.bit_depth && left.pixel_format == right.pixel_format;
}

void add_elapsed_nanoseconds(std::atomic<std::uint64_t> &destination,
                             std::chrono::steady_clock::time_point started_at) noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started_at);
    if (elapsed.count() <= 0) return;
    destination.fetch_add(static_cast<std::uint64_t>(elapsed.count()), std::memory_order_relaxed);
}

} // namespace

bool automatic_decode_should_attempt_hardware(VideoCodec codec) noexcept {
    switch (codec) {
    case VideoCodec::h264:
    case VideoCodec::vp8:
    case VideoCodec::mpeg1:
    case VideoCodec::mpeg2:
    case VideoCodec::mpeg4:
    case VideoCodec::flv1:
    case VideoCodec::mjpeg:
        return false;
    default:
        return true;
    }
}

std::size_t plan_software_decoder_thread_count(VideoCodec codec, std::size_t requested_thread_count,
                                               std::size_t concurrent_decoder_count,
                                               std::size_t logical_processor_count) noexcept {
    if (requested_thread_count != 0U || codec != VideoCodec::av1 ||
        concurrent_decoder_count <= 1U || logical_processor_count == 0U) {
        return requested_thread_count;
    }

    // dav1d owns its worker pool. Reserve roughly one sixth of the host for
    // decoder drivers, staging, Vulkan submission, and model prediction.
    // Keep the existing per-source automatic cap to bound frame buffering;
    // callers can still explicitly request more workers for large streams.
    constexpr std::size_t maximum_automatic_av1_thread_count = 16U;
    const std::size_t coordination_thread_reserve = logical_processor_count / 6U;
    const std::size_t shared_codec_thread_budget =
        std::max(concurrent_decoder_count, logical_processor_count - coordination_thread_reserve);
    const std::size_t planned_thread_count =
        std::max<std::size_t>(1U, shared_codec_thread_budget / concurrent_decoder_count);
    return std::min(planned_thread_count, maximum_automatic_av1_thread_count);
}

HardwareDecoderThreadPlan
plan_hardware_decoder_threads(std::size_t requested_thread_count,
                              bool consumers_extend_decoder_timeline,
                              std::size_t logical_processor_count,
                              bool dedicated_decode_queue) noexcept {
    if (dedicated_decode_queue) {
        // With one queue per H.264 stream, two frame workers provide useful
        // run-ahead without the surface retention and startup cost of eight.
        return {true, requested_thread_count == 0U ? 2U : requested_thread_count};
    }
    // FFmpeg's Vulkan hwaccels are HWACCEL_CAP_ASYNC_SAFE but not
    // HWACCEL_CAP_THREAD_SAFE, so frame threads take turns under the frame
    // thread hwaccel mutex: they add no decode parallelism, but each retains
    // one decode surface and delays output by thread_count - 1 frames.
    // ff_vk_decode_init sizes its in-flight GPU decode pool from
    // thread_count, so that count remains the decode depth in either mode.
    if (!consumers_extend_decoder_timeline) {
        // Consumers only wait on the decoder timeline, so decoding never
        // depends on them and no output delay is needed. Depth two exposes
        // host fence wake-up and queue submission latency on the shared
        // decode engine; four keeps it busy with two decode surfaces less
        // than the frame-threaded arrangement.
        return {false, requested_thread_count == 0U ? default_hardware_decode_depth
                                                    : requested_thread_count};
    }
    // Every consumer conversion advances the image timeline, so a decode
    // referencing that image and submitted after the conversion waits for
    // it. Frame threading submits the following decodes before the frame is
    // output, which keeps typical reference distances ahead of consumers.
    // FFmpeg's automatic count would retain up to sixteen surfaces per
    // source; eight already covers ordinary reference structures.
    if (requested_thread_count != 0U) {
        return {true, requested_thread_count};
    }
    if (logical_processor_count == 0U) {
        return {true, 0U};
    }
    return {true, std::min(logical_processor_count, maximum_automatic_hardware_frame_threads)};
}

DecoderPairThreadPlan plan_decoder_pair_threads(const SoftwareDecodeWorkload &reference,
                                                const SoftwareDecodeWorkload &distorted,
                                                std::size_t requested_thread_count,
                                                std::size_t logical_processor_count) noexcept {
    const DecoderPairThreadPlan unchanged{requested_thread_count, requested_thread_count};
    constexpr std::size_t native_auto_thread_limit = 16U;
    if (requested_thread_count != 0U || logical_processor_count <= native_auto_thread_limit ||
        reference.codec != VideoCodec::h264 || distorted.codec != VideoCodec::h264 ||
        reference.descriptor.width == 0U || reference.descriptor.height == 0U ||
        !descriptors_match(reference.descriptor, distorted.descriptor) ||
        !std::isfinite(reference.encoded_bits_per_frame) ||
        !std::isfinite(distorted.encoded_bits_per_frame) ||
        reference.encoded_bits_per_frame <= 0.0 || distorted.encoded_bits_per_frame <= 0.0) {
        return unchanged;
    }

    const bool reference_is_heavier =
        reference.encoded_bits_per_frame > distorted.encoded_bits_per_frame;
    const double heavier_workload =
        std::max(reference.encoded_bits_per_frame, distorted.encoded_bits_per_frame);
    const double lighter_workload =
        std::min(reference.encoded_bits_per_frame, distorted.encoded_bits_per_frame);
    // Use the measured metadata ratio continuously instead of classifying
    // inputs as heavy/light. Square-root weighting leaves reconstruction
    // headroom on the lower-bitrate source. Normalize before taking the root
    // so even extreme finite estimates cannot overflow the calculation.
    constexpr std::size_t pair_thread_budget = native_auto_thread_limit * 2U;
    constexpr std::size_t minimum_source_threads = 2U;
    const double relative_lighter_weight = std::sqrt(lighter_workload / heavier_workload);
    const auto weighted_threads = static_cast<std::size_t>(
        std::round(static_cast<double>(pair_thread_budget) / (1.0 + relative_lighter_weight)));
    // Frame workers can wait on reference frames; they are not reserved CPU
    // cores. Bound the combined pools, not each source by the CPU count.
    const std::size_t heavier_threads =
        std::min(weighted_threads, pair_thread_budget - minimum_source_threads);
    const std::size_t lighter_threads = pair_thread_budget - heavier_threads;
    return reference_is_heavier ? DecoderPairThreadPlan{heavier_threads, lighter_threads}
                                : DecoderPairThreadPlan{lighter_threads, heavier_threads};
}

class Decoder::State final {
  public:
    static Result<std::unique_ptr<State>> prepare(const std::filesystem::path &path,
                                                  VideoDecoderOptions options) {
        if (path.empty()) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::invalid_argument, "video decoder requires an input path"});
        }
        if (options.output_queue_capacity == 0U || options.output_queue_capacity > 64U ||
            options.codec_thread_count > 64U || options.concurrent_decoder_count == 0U ||
            options.concurrent_decoder_count > 64U) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::invalid_argument,
                 "decoder queue capacity must be 1..64, threads 0..64, "
                 "and concurrent decoders 1..64"});
        }

        try {
            const auto compatibility = require_ffmpeg_abi();
            if (!compatibility)
                return Result<std::unique_ptr<State>>::failure(compatibility.error());
            auto state = std::unique_ptr<State>(new State(path, options));
            const auto initialized = state->initialize_input();
            if (!initialized) {
                return Result<std::unique_ptr<State>>::failure(initialized.error());
            }
            return Result<std::unique_ptr<State>>::success(std::move(state));
        } catch (const std::exception &exception) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::backend_error,
                 "failed to prepare FFmpeg input: " + std::string(exception.what())});
        } catch (...) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::backend_error, "failed to prepare FFmpeg input"});
        }
    }

    static Result<std::unique_ptr<State>> start(std::unique_ptr<State> state) {
        try {
            auto initialized = state->initialize_decoder();
            if (!initialized && state->options_.decode_device != nullptr &&
                !state->options_.require_hardware_decode &&
                (initialized.error().code == ErrorCode::unsupported_format || initialized.error().code == ErrorCode::hardware_decode_unavailable)) {
                // Hardware capability is stream/profile dependent. Reopen the
                // local input from byte zero so fallback never reuses packets
                // consumed by a failed Vulkan decoder attempt.
                VideoDecoderOptions software_options = state->options_;
                software_options.decode_device.reset();
                const std::filesystem::path path = state->path_;
                state.reset();
                auto fallback = prepare(path, std::move(software_options));
                if (!fallback) return fallback;
                state = std::move(fallback).value();
                initialized = state->initialize_decoder();
            }
            if (!initialized) {
                if (state->options_.require_hardware_decode &&
                    (initialized.error().code == ErrorCode::unsupported_format ||
                     initialized.error().code == ErrorCode::hardware_decode_unavailable)) {
                    return Result<std::unique_ptr<State>>::failure(
                        {ErrorCode::hardware_decode_unavailable,
                         "Hardware-only decoding failed: " + initialized.error().message});
                }
                return Result<std::unique_ptr<State>>::failure(initialized.error());
            }
            state->worker_ =
                std::thread([state_pointer = state.get()] { state_pointer->run_worker(); });
            return Result<std::unique_ptr<State>>::success(std::move(state));
        } catch (const std::bad_alloc &) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::backend_error, "failed to allocate FFmpeg decoder state"});
        } catch (const std::exception &exception) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::backend_error,
                 "failed to start FFmpeg decoder worker: " + std::string(exception.what())});
        } catch (...) {
            return Result<std::unique_ptr<State>>::failure(
                {ErrorCode::backend_error, "failed to start FFmpeg decoder worker"});
        }
    }

    State(const State &) = delete;
    State &operator=(const State &) = delete;

    ~State() {
        request_cancel();
        drain();
        av_frame_free(&decode_frame_);
        av_packet_free(&packet_);
        avcodec_free_context(&codec_context_);
        if (format_context_ != nullptr) {
            avformat_close_input(&format_context_);
        }
    }

    [[nodiscard]] Result<VideoFramePopResult> try_pop() {
        std::lock_guard queue_lock(queue_mutex_);
        return pop_locked();
    }

    [[nodiscard]] Result<VideoFramePopResult> wait_pop() {
        std::unique_lock queue_lock(queue_mutex_);
        queue_not_empty_.wait(queue_lock, [this] {
            return !output_queue_.empty() || first_error_.has_value() || worker_exception_ || end_of_stream_ ||
                   cancellation_requested_.load(std::memory_order_acquire);
        });
        if (output_queue_.empty() && !first_error_.has_value() && !worker_exception_ && !end_of_stream_ &&
            cancellation_requested_.load(std::memory_order_acquire)) {
            return Result<VideoFramePopResult>::failure(
                {ErrorCode::cancelled, "video decoding was cancelled"});
        }
        return pop_locked();
    }

    [[nodiscard]] Result<VideoFramePopResult> pop_locked() {
        if (!output_queue_.empty()) {
            Frame frame = std::move(output_queue_.front());
            output_queue_.pop_front();
            queue_not_full_.notify_one();
            return Result<VideoFramePopResult>::success(
                {VideoFramePopStatus::frame, std::move(frame)});
        }
        if (first_error_.has_value()) {
            return Result<VideoFramePopResult>::failure(first_error_->with_context(
                "decoding " + path_as_utf8(path_), ErrorInput::none,
                decoded_frames_.load(std::memory_order_relaxed)));
        }
        if (worker_exception_) {
            try {
                std::rethrow_exception(worker_exception_);
            } catch (const std::exception &exception) {
                return Result<VideoFramePopResult>::failure(
                    Error{ErrorCode::backend_error, exception.what()}.with_context(
                        "decoder worker for " + path_as_utf8(path_), ErrorInput::none,
                        decoded_frames_.load(std::memory_order_relaxed)));
            } catch (...) {
                return Result<VideoFramePopResult>::failure(
                    Error{ErrorCode::backend_error, "unknown exception"}.with_context(
                        "decoder worker for " + path_as_utf8(path_), ErrorInput::none,
                        decoded_frames_.load(std::memory_order_relaxed)));
            }
        }
        if (end_of_stream_) {
            return Result<VideoFramePopResult>::success({VideoFramePopStatus::end_of_stream, {}});
        }
        return Result<VideoFramePopResult>::success({VideoFramePopStatus::unavailable, {}});
    }

    [[nodiscard]] VideoDecoderStatistics statistics() const {
        std::lock_guard queue_lock(queue_mutex_);
        return {codec_,
                decode_backend_,
                hardware_surface_format_,
                packets_read_.load(std::memory_order_relaxed),
                compressed_bytes_.load(std::memory_order_relaxed),
                decoded_frames_.load(std::memory_order_relaxed),
                decode_active_nanoseconds_.load(std::memory_order_relaxed),
                decode_operation_count_.load(std::memory_order_relaxed),
                output_queue_blocked_nanoseconds_.load(std::memory_order_relaxed),
                output_queue_blocked_wait_count_.load(std::memory_order_relaxed),
                options_.output_queue_capacity,
                output_queue_.size(),
                codec_thread_count_,
                end_of_stream_,
                cancellation_requested_.load(std::memory_order_acquire),
                {backend_storage(decode_backend_), transfer_path_, options_.decode_device ? options_.decode_device->info.id : "",
                 options_.open_attempts, host_transfer_bytes_.load(std::memory_order_relaxed), gpu_copy_bytes_.load(std::memory_order_relaxed),
                 transfer_nanoseconds_.load(std::memory_order_relaxed)}};
    }

    void request_cancel() noexcept {
        cancellation_requested_.store(true, std::memory_order_release);
        queue_not_full_.notify_all();
        queue_not_empty_.notify_all();
    }

    void drain() noexcept {
        std::lock_guard drain_lock(drain_mutex_);
        if (worker_.joinable()) worker_.join();
    }

    [[nodiscard]] SoftwareDecodeWorkload software_workload() const {
        const AVStream *const stream = format_context_->streams[video_stream_index_];
        SoftwareDecodeWorkload workload;
        workload.codec = codec_;
        // Short clips cannot amortize a deeper frame-thread pipeline. Missing
        // metadata keeps the native policy without an extra decode/seek pass.
        if (stream->nb_frames > 0 && stream->nb_frames < 64) return workload;
        if (stream->codecpar->bit_rate <= 0 || stream->avg_frame_rate.num <= 0 ||
            stream->avg_frame_rate.den <= 0) {
            return workload;
        }
        AVFrame metadata_frame{};
        metadata_frame.width = stream->codecpar->width;
        metadata_frame.height = stream->codecpar->height;
        metadata_frame.format = stream->codecpar->format;
        const auto descriptor = descriptor_from_frame(metadata_frame);
        if (!descriptor) return workload;
        workload.descriptor = descriptor.value();
        workload.encoded_bits_per_frame = static_cast<double>(stream->codecpar->bit_rate) *
                                          stream->avg_frame_rate.den / stream->avg_frame_rate.num;
        return workload;
    }

    std::filesystem::path path_;
    VideoDecoderOptions options_;
    FrameDescriptor descriptor_{};
    VideoCodec codec_{VideoCodec::h264};
    VideoDecodeBackend decode_backend_{VideoDecodeBackend::software};
    bool has_vulkan_output_{};
    DecodeTransferPath transfer_path_{DecodeTransferPath::host};
    HardwareSurfaceFormat hardware_surface_format_{HardwareSurfaceFormat::none};
    std::optional<std::uint64_t> frame_count_hint_;
    VideoMetadata metadata_;

  private:
    State(std::filesystem::path path, VideoDecoderOptions options) noexcept
        : path_(std::move(path)), options_(options) {}

    static int interrupt_callback(void *opaque) noexcept {
        const auto *const state = static_cast<const State *>(opaque);
        return state->cancellation_requested_.load(std::memory_order_acquire) ? 1 : 0;
    }

    [[nodiscard]] Result<void> initialize_input() {
        if (options_.stop_token.stop_possible()) {
            // Arms the cooperative-cancellation bridge before the first
            // potentially blocking FFmpeg call; the interrupt callback then
            // aborts demux/decode waits when stop is requested.
            stop_callback_.emplace(options_.stop_token, [this] { request_cancel(); });
        }
        format_context_ = avformat_alloc_context();
        if (format_context_ == nullptr) {
            return Result<void>::failure(
                {ErrorCode::backend_error, "failed to allocate FFmpeg format context"});
        }
        format_context_->interrupt_callback = {interrupt_callback, this};

        AVDictionary *open_options = nullptr;
        av_dict_set(&open_options, "protocol_whitelist", "file", 0);
        av_dict_set(&open_options, "format_whitelist",
                    "h264,hevc,av1,obu,ivf,m4v,mpegvideo,mpeg,mpegts,vc1,"
                    "mjpeg,mpjpeg,mov,matroska,webm,avi,asf,flv",
                    0);
        const std::string utf8_path = path_as_utf8(path_);
        const int open_result =
            avformat_open_input(&format_context_, utf8_path.c_str(), nullptr, &open_options);
        av_dict_free(&open_options);
        if (open_result < 0) {
            return Result<void>::failure(
                {open_result == AVERROR_EXIT && cancellation_requested_.load(std::memory_order_acquire)
                     ? ErrorCode::cancelled : ErrorCode::io_error,
                 ffmpeg_error_message(open_result, "opening video input")});
        }

        const int stream_info_result = avformat_find_stream_info(format_context_, nullptr);
        if (stream_info_result < 0) {
            return Result<void>::failure(
                {stream_info_result == AVERROR_EXIT && cancellation_requested_.load(std::memory_order_acquire)
                     ? ErrorCode::cancelled : ErrorCode::unsupported_format,
                 ffmpeg_error_message(stream_info_result, "reading video stream metadata")});
        }

        const int selected_stream_index =
            av_find_best_stream(format_context_, AVMEDIA_TYPE_VIDEO, -1, -1, &selected_decoder_, 0);
        if (selected_stream_index < 0 || selected_decoder_ == nullptr) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format, "input does not contain a decodable video stream"});
        }
        video_stream_index_ = selected_stream_index;
        AVStream *const video_stream = format_context_->streams[video_stream_index_];
        stream_time_base_ = video_stream->time_base;

        const auto codec_result = video_codec_from_id(video_stream->codecpar->codec_id);
        if (!codec_result) {
            return Result<void>::failure(codec_result.error());
        }
        codec_ = codec_result.value();
        if (video_stream->nb_frames > 0) {
            frame_count_hint_ = static_cast<std::uint64_t>(video_stream->nb_frames);
        }
        metadata_ = extract_video_metadata(format_context_, video_stream_index_, path_);

        return Result<void>::success();
    }

    [[nodiscard]] Result<void> initialize_decoder() {
        if (options_.decode_device && options_.decode_device->backend() == StreamDecodeBackend::vulkan_video &&
            options_.respect_automatic_vulkan_policy && !automatic_decode_should_attempt_hardware(codec_))
            return Result<void>::failure({ErrorCode::hardware_decode_unavailable, "Vulkan automatic policy uses software for this codec"});
        const bool should_attempt_hardware_decode = options_.decode_device &&
            (options_.decode_device->backend() != StreamDecodeBackend::vulkan_video ||
             options_.require_hardware_decode || automatic_decode_should_attempt_hardware(codec_));
        decode_backend_ = should_attempt_hardware_decode ? options_.decode_device->backend() : StreamDecodeBackend::software;
        selected_decoder_ = find_backend_codec(decode_backend_, format_context_->streams[video_stream_index_]->codecpar->codec_id);
        if (!selected_decoder_ || !codec_supports_backend(selected_decoder_, decode_backend_) ||
            (options_.require_hardware_decode && decode_backend_ == StreamDecodeBackend::software)) {
            return Result<void>::failure({ErrorCode::hardware_decode_unavailable,
                "selected backend has no compatible decoder for this codec"});
        }
        const bool use_vulkan_hardware_decoder = decode_backend_ == StreamDecodeBackend::vulkan_video;

        AVStream *const video_stream = format_context_->streams[video_stream_index_];

        codec_context_ = avcodec_alloc_context3(selected_decoder_);
        if (codec_context_ == nullptr) {
            return Result<void>::failure(
                {ErrorCode::backend_error, "failed to allocate FFmpeg codec context"});
        }
        const int parameter_result =
            avcodec_parameters_to_context(codec_context_, video_stream->codecpar);
        if (parameter_result < 0) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format,
                 ffmpeg_error_message(parameter_result, "copying video codec parameters")});
        }
        codec_context_->pkt_timebase = stream_time_base_;
        // Quality measurements must not silently score pixels reconstructed
        // after a detected bitstream error (e.g. MJPEG logs overread but can
        // otherwise still return a successful frame without a corrupt flag).
        codec_context_->err_recognition |= AV_EF_EXPLODE;
        const int detected_logical_processor_count = av_cpu_count();
        const std::size_t logical_processor_count =
            detected_logical_processor_count > 0
                ? static_cast<std::size_t>(detected_logical_processor_count)
                : 0U;
        const std::size_t configured_codec_thread_count =
            should_attempt_hardware_decode
                ? options_.codec_thread_count
                : plan_software_decoder_thread_count(codec_, options_.codec_thread_count,
                                                     options_.concurrent_decoder_count,
                                                     logical_processor_count);
        codec_context_->thread_count = static_cast<int>(configured_codec_thread_count);
        codec_context_->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        if (use_vulkan_hardware_decoder) {
            auto native = options_.decode_device->retain_decoder_context(
                codec_ == VideoCodec::h264 && options_.concurrent_decoder_count > 1U);
            if (!native) return Result<void>::failure(native.error());
            codec_context_->hw_device_ctx = static_cast<AVBufferRef*>(native.value().native_context);
            dedicated_decode_queue_ = native.value().dedicated_queue;
            const HardwareDecoderThreadPlan hardware_plan = plan_hardware_decoder_threads(
                options_.codec_thread_count,
                !options_.decode_device->unified_video_image_layouts(),
                logical_processor_count, dedicated_decode_queue_);
            codec_context_->thread_type = hardware_plan.frame_threading
                                              ? (FF_THREAD_FRAME | FF_THREAD_SLICE)
                                              : FF_THREAD_SLICE;
            codec_context_->thread_count = static_cast<int>(hardware_plan.thread_count);
            if (options_.decode_device->serializes_decode_commands() && !dedicated_decode_queue_) {
                // The asynchronous Vulkan pool supplies run-ahead on this driver.
                // Frame workers retain extra surfaces without useful run-ahead here.
                codec_context_->thread_type = FF_THREAD_SLICE;
            }
            select_backend_format(*codec_context_, decode_backend_);
            const std::size_t requested_extra_frames = options_.output_queue_capacity + 4U;
            codec_context_->extra_hw_frames = static_cast<int>(std::min(
                requested_extra_frames, static_cast<std::size_t>(std::numeric_limits<int>::max())));
        }
        if (should_attempt_hardware_decode && !use_vulkan_hardware_decoder) {
            auto native = options_.decode_device->retain_decoder_context(false);
            if (!native) return Result<void>::failure(native.error());
            codec_context_->hw_device_ctx = static_cast<AVBufferRef*>(native.value().native_context);
            codec_context_->thread_count = static_cast<int>(options_.codec_thread_count == 0 ? 1 : options_.codec_thread_count);
            codec_context_->thread_type = FF_THREAD_SLICE;
            codec_context_->extra_hw_frames = static_cast<int>(options_.output_queue_capacity + 8U);
            select_backend_format(*codec_context_, decode_backend_);
        }
        const int codec_open_result = avcodec_open2(codec_context_, selected_decoder_, nullptr);
        if (codec_open_result < 0) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format,
                 ffmpeg_error_message(codec_open_result,
                                      use_vulkan_hardware_decoder
                                          ? "opening Vulkan hardware video decoder"
                                          : "opening software video decoder")});
        }
        codec_thread_count_ = static_cast<std::size_t>(codec_context_->thread_count);

        packet_ = av_packet_alloc();
        decode_frame_ = av_frame_alloc();
        if (packet_ == nullptr || decode_frame_ == nullptr) {
            return Result<void>::failure(
                {ErrorCode::backend_error, "failed to allocate FFmpeg decode buffers"});
        }

        auto first_frame_result = decode_next_frame_measured();
        if (!first_frame_result) {
            return Result<void>::failure(first_frame_result.error());
        }
        if (!first_frame_result.value().has_value()) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format, "video input contains no decoded frames"});
        }
        descriptor_ = first_frame_result.value()->descriptor();
        has_vulkan_output_ = first_frame_result.value()->has_vulkan_storage();
        transfer_path_ = transfer_.path();
        if (!transfer_.fallback_reason().empty()) options_.open_attempts.push_back(transfer_.fallback_reason());
        hardware_surface_format_ = first_frame_result.value()->hardware_surface_format();
        metadata_.descriptor = descriptor_;
        metadata_.descriptor_confirmed = true;
        if (use_vulkan_hardware_decoder &&
            (hardware_surface_format_ == HardwareSurfaceFormat::none)) {
            return Result<void>::failure(
                {ErrorCode::unsupported_format,
                 "Vulkan decoder did not produce an NV12 or P010 hardware frame"});
        }
        output_queue_.push_back(std::move(*first_frame_result.value()));
        return Result<void>::success();
    }

    [[nodiscard]] ErrorCode decode_failure_code(int error) const noexcept {
        if (error == AVERROR_EXIT && cancellation_requested_.load(std::memory_order_acquire))
            return ErrorCode::cancelled;
        if (error == AVERROR_INVALIDDATA) return ErrorCode::unsupported_format;
        return error == AVERROR(ENOMEM) || decode_backend_ == StreamDecodeBackend::software
            ? ErrorCode::backend_error : ErrorCode::hardware_decode_unavailable;
    }

    [[nodiscard]] Result<std::optional<Frame>> decode_next_frame() {
        const auto decode_started_at = std::chrono::steady_clock::now();
        for (;;) {
            const int receive_result = receive_frame();
            if (receive_result >= 0) {
                if ((decode_frame_->flags & AV_FRAME_FLAG_CORRUPT) != 0 ||
                    decode_frame_->decode_error_flags != 0) {
                    av_frame_unref(decode_frame_);
                    return Result<std::optional<Frame>>::failure(
                        {ErrorCode::unsupported_format, "decoder returned a corrupt or concealed video frame"});
                }
                // QSV may synthesize timestamps for elementary streams whose
                // packets carry none. Do not turn those into source timing.
                const bool timestamp_valid = input_has_timestamps_ && decode_frame_->best_effort_timestamp != AV_NOPTS_VALUE;
                const VideoTimestamp timestamp{
                    timestamp_valid ? decode_frame_->best_effort_timestamp : AV_NOPTS_VALUE,
                    stream_time_base_.num, stream_time_base_.den, timestamp_valid};
                if (decode_backend_ != StreamDecodeBackend::software &&
                    decode_frame_->format != backend_pixel_format(decode_backend_)) {
                    av_frame_unref(decode_frame_);
                    return Result<std::optional<Frame>>::failure({ErrorCode::unsupported_format,
                        "hardware decoder changed its native output format"});
                }
                const auto transfer_started = std::chrono::steady_clock::now();
                auto transferred = transfer_.convert(*decode_frame_, options_.transfer_policy, options_.transfer_device);
                const auto native_surface_format = hardware_surface_format_from_frame(*decode_frame_);
                av_frame_unref(decode_frame_);
                if (!transferred) return Result<std::optional<Frame>>::failure(transferred.error());
                AVFrame* owned_frame = transferred.value();
                const auto descriptor_result = descriptor_from_frame(*owned_frame);
                if (!descriptor_result) {
                    av_frame_free(&owned_frame);
                    return Result<std::optional<Frame>>::failure(descriptor_result.error());
                }
                if (decode_backend_ != StreamDecodeBackend::software && transfer_.path() != DecodeTransferPath::native) {
                    add_elapsed_nanoseconds(transfer_nanoseconds_, transfer_started);
                    const auto& d = descriptor_result.value();
                    const std::uint64_t bytes = (static_cast<std::uint64_t>(d.width) * d.height +
                        2ULL * ((d.width + 1U) / 2U) * ((d.height + 1U) / 2U)) * (d.bit_depth > 8 ? 2U : 1U);
                    if (transfer_.path() == DecodeTransferPath::host) host_transfer_bytes_.fetch_add(bytes, std::memory_order_relaxed);
                    if (transfer_.path() == DecodeTransferPath::gpu_copy) gpu_copy_bytes_.fetch_add(transfer_.last_gpu_copy_bytes(), std::memory_order_relaxed);
                }
                const bool vulkan_storage = owned_frame->format == AV_PIX_FMT_VULKAN;
                if (decoded_frames_.load(std::memory_order_relaxed) != 0U &&
                    (!descriptors_match(descriptor_, descriptor_result.value()) ||
                     has_vulkan_output_ != vulkan_storage || hardware_surface_format_ != native_surface_format)) {
                    av_frame_free(&owned_frame);
                    return Result<std::optional<Frame>>::failure({ErrorCode::unsupported_format,
                        "decoded video format or input storage changed during the stream"});
                }
                const auto decode_completed_at = std::chrono::steady_clock::now();
                const auto release_frame = [](AVFrame* value) { av_frame_free(&value); };
                std::unique_ptr<AVFrame, decltype(release_frame)> frame_guard(owned_frame, release_frame);
                auto frame_impl =
                    std::make_unique<Frame::Impl>(owned_frame, descriptor_result.value(), timestamp,
                                                  decode_started_at, decode_completed_at, decode_backend_);
                static_cast<void>(frame_guard.release());
                frame_impl->hardware_surface_format_ = native_surface_format;
                decoded_frames_.fetch_add(1U, std::memory_order_relaxed);
                return Result<std::optional<Frame>>::success(
                    std::optional<Frame>{Frame(std::move(frame_impl))});
            }
            if (receive_result == AVERROR_EOF) {
                return Result<std::optional<Frame>>::success(std::nullopt);
            }
            if (receive_result != AVERROR(EAGAIN)) {
                return Result<std::optional<Frame>>::failure(
                    {decode_failure_code(receive_result),
                     ffmpeg_error_message(receive_result, "receiving decoded video frame")});
            }

            if (demux_end_of_stream_) {
                if (flush_packet_sent_) {
                    return Result<std::optional<Frame>>::failure(
                        {ErrorCode::backend_error,
                         "FFmpeg decoder requested input after its flush packet"});
                }
                const int flush_result = send_packet(nullptr);
                if (flush_result < 0 && flush_result != AVERROR_EOF) {
                    return Result<std::optional<Frame>>::failure(
                        {ErrorCode::backend_error,
                         ffmpeg_error_message(flush_result, "flushing video decoder")});
                }
                flush_packet_sent_ = true;
                continue;
            }

            int read_result = 0;
            for (;;) {
                av_packet_unref(packet_);
                read_result = av_read_frame(format_context_, packet_);
                if (read_result < 0 || packet_->stream_index == video_stream_index_) {
                    break;
                }
            }

            if (read_result == AVERROR_EOF) {
                demux_end_of_stream_ = true;
                continue;
            }
            if (read_result < 0) {
                if (cancellation_requested_.load(std::memory_order_acquire)) {
                    return Result<std::optional<Frame>>::failure(
                        {ErrorCode::cancelled, "video decoding was cancelled"});
                }
                return Result<std::optional<Frame>>::failure(
                    {ErrorCode::io_error,
                     ffmpeg_error_message(read_result, "reading compressed video packet")});
            }
            if ((packet_->flags & AV_PKT_FLAG_CORRUPT) != 0) {
                av_packet_unref(packet_);
                return Result<std::optional<Frame>>::failure(
                    {ErrorCode::unsupported_format, "demuxer returned a corrupt video packet"});
            }

            packets_read_.fetch_add(1U, std::memory_order_relaxed);
            if (packet_->size > 0) {
                compressed_bytes_.fetch_add(static_cast<std::uint64_t>(packet_->size),
                                            std::memory_order_relaxed);
            }
            input_has_timestamps_ = input_has_timestamps_ || packet_->pts != AV_NOPTS_VALUE || packet_->dts != AV_NOPTS_VALUE;
            const int send_result = send_packet(packet_);
            av_packet_unref(packet_);
            if (send_result < 0) {
                return Result<std::optional<Frame>>::failure(
                    {decode_failure_code(send_result),
                     ffmpeg_error_message(send_result, "submitting compressed video packet")});
            }
        }
    }

    [[nodiscard]] std::unique_lock<std::mutex> lock_codec_calls() const {
        // A dedicated queue lets H.264 streams progress independently while
        // the shared dispatch still serializes decode-command recording.
        return codec_ == VideoCodec::h264 && !dedicated_decode_queue_ && codec_context_->hw_device_ctx &&
                       options_.decode_device
                   ? options_.decode_device->lock_codec_calls()
                   : std::unique_lock<std::mutex>{};
    }

    [[nodiscard]] int receive_frame() {
        const auto guard = lock_codec_calls();
        return avcodec_receive_frame(codec_context_, decode_frame_);
    }

    [[nodiscard]] int send_packet(const AVPacket* packet) {
        const auto guard = lock_codec_calls();
        return avcodec_send_packet(codec_context_, packet);
    }

    [[nodiscard]] Result<std::optional<Frame>> decode_next_frame_measured() {
        const auto started_at = std::chrono::steady_clock::now();
        const auto previous_transfer = transfer_nanoseconds_.load(std::memory_order_relaxed);
        auto decoded_frame = decode_next_frame();
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - started_at).count();
        const auto transfer = transfer_nanoseconds_.load(std::memory_order_relaxed) - previous_transfer;
        const auto total = elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0U;
        decode_active_nanoseconds_.fetch_add(total > transfer ? total - transfer : 0U, std::memory_order_relaxed);
        decode_operation_count_.fetch_add(1U, std::memory_order_relaxed);
        return decoded_frame;
    }

    void run_worker() noexcept {
        try {
            for (;;) {
                {
                    std::unique_lock queue_lock(queue_mutex_);
                    const bool output_queue_is_full =
                        output_queue_.size() >= options_.output_queue_capacity;
                    const auto wait_started_at = std::chrono::steady_clock::now();
                    queue_not_full_.wait(queue_lock, [this] {
                        return output_queue_.size() < options_.output_queue_capacity ||
                               cancellation_requested_.load(std::memory_order_acquire);
                    });
                    if (output_queue_is_full) {
                        add_elapsed_nanoseconds(output_queue_blocked_nanoseconds_, wait_started_at);
                        output_queue_blocked_wait_count_.fetch_add(1U, std::memory_order_relaxed);
                    }
                    if (cancellation_requested_.load(std::memory_order_acquire)) {
                        return;
                    }
                }

                auto decoded_frame = decode_next_frame_measured();
                if (!decoded_frame) {
                    if (!cancellation_requested_.load(std::memory_order_acquire)) {
                        remember_error(std::move(decoded_frame).error());
                    }
                    return;
                }
                if (!decoded_frame.value().has_value()) {
                    {
                        std::lock_guard queue_lock(queue_mutex_);
                        end_of_stream_ = true;
                    }
                    queue_not_empty_.notify_all();
                    return;
                }

                {
                    std::lock_guard queue_lock(queue_mutex_);
                    output_queue_.push_back(std::move(*decoded_frame.value()));
                }
                queue_not_empty_.notify_one();
            }
        } catch (...) {
            // Publishing an exception must not allocate another error string or
            // queue entry on the worker that just ran out of memory.
            {
                std::lock_guard queue_lock(queue_mutex_);
                worker_exception_ = std::current_exception();
            }
            queue_not_empty_.notify_all();
        }
    }

    void remember_error(Error error) {
        {
            std::lock_guard queue_lock(queue_mutex_);
            if (!first_error_.has_value()) {
                first_error_.emplace(std::move(error));
            }
        }
        queue_not_empty_.notify_all();
    }

    FrameTransfer transfer_;
    std::atomic<std::uint64_t> host_transfer_bytes_{};
    std::atomic<std::uint64_t> gpu_copy_bytes_{};
    std::atomic<std::uint64_t> transfer_nanoseconds_{};
    AVFormatContext *format_context_{};
    const AVCodec *selected_decoder_{};
    AVCodecContext *codec_context_{};
    AVPacket *packet_{};
    AVFrame *decode_frame_{};
    int video_stream_index_{-1};
    AVRational stream_time_base_{};
    std::size_t codec_thread_count_{};
    bool dedicated_decode_queue_{};
    bool input_has_timestamps_{};
    bool demux_end_of_stream_{};
    bool flush_packet_sent_{};
    std::atomic<bool> cancellation_requested_{};
    std::atomic<std::uint64_t> packets_read_{};
    std::atomic<std::uint64_t> compressed_bytes_{};
    std::atomic<std::uint64_t> decoded_frames_{};
    std::atomic<std::uint64_t> decode_active_nanoseconds_{};
    std::atomic<std::uint64_t> decode_operation_count_{};
    std::atomic<std::uint64_t> output_queue_blocked_nanoseconds_{};
    std::atomic<std::uint64_t> output_queue_blocked_wait_count_{};
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_not_full_;
    std::condition_variable queue_not_empty_;
    std::deque<Frame> output_queue_;
    std::optional<Error> first_error_;
    std::exception_ptr worker_exception_;
    bool end_of_stream_{};
    std::thread worker_;
    std::mutex drain_mutex_;
    std::optional<std::stop_callback<std::function<void()>>> stop_callback_;
};

Result<std::unique_ptr<Decoder>> Decoder::open(const std::filesystem::path &path,
                                               VideoDecoderOptions options) {
    auto state = State::prepare(path, options);
    if (state) state = State::start(std::move(state).value());
    if (!state) {
        return Result<std::unique_ptr<Decoder>>::failure(
            state.error().with_context("opening " + path_as_utf8(path)));
    }
    return Result<std::unique_ptr<Decoder>>::success(
        std::unique_ptr<Decoder>(new Decoder(std::move(state).value())));
}

Result<VideoDecoderPair> Decoder::open_pair(const std::filesystem::path &reference_path,
                                            const std::filesystem::path &distorted_path,
                                            VideoDecoderOptions options) {
    options.concurrent_decoder_count = 2U;
    const auto input_error = [](Error error) {
        // Demux/stream discovery does not depend on the selected hardware
        // backend. A malformed input must not trigger another candidate.
        if (error.code == ErrorCode::unsupported_format)
            error.code = ErrorCode::invalid_argument;
        return Result<VideoDecoderPair>::failure(std::move(error));
    };
    auto reference_state = State::prepare(reference_path, options);
    if (!reference_state) {
        return input_error(reference_state.error().with_context(
            "opening reference " + path_as_utf8(reference_path), ErrorInput::reference));
    }
    auto distorted_state = State::prepare(distorted_path, options);
    if (!distorted_state) {
        return input_error(distorted_state.error().with_context(
            "opening distorted " + path_as_utf8(distorted_path), ErrorInput::distorted));
    }
    const int detected_processors = av_cpu_count();
    const auto plan = plan_decoder_pair_threads(
        reference_state.value()->software_workload(), distorted_state.value()->software_workload(),
        options.codec_thread_count,
        detected_processors > 0 ? static_cast<std::size_t>(detected_processors) : 0U);
    reference_state.value()->options_.codec_thread_count = plan.reference_thread_count;
    distorted_state.value()->options_.codec_thread_count = plan.distorted_thread_count;
    reference_state = State::start(std::move(reference_state).value());
    if (!reference_state) {
        return Result<VideoDecoderPair>::failure(reference_state.error().with_context(
            "opening reference " + path_as_utf8(reference_path), ErrorInput::reference));
    }
    distorted_state = State::start(std::move(distorted_state).value());
    if (!distorted_state) {
        return Result<VideoDecoderPair>::failure(distorted_state.error().with_context(
            "opening distorted " + path_as_utf8(distorted_path), ErrorInput::distorted));
    }
    return Result<VideoDecoderPair>::success(
        {std::unique_ptr<Decoder>(new Decoder(std::move(reference_state).value())),
         std::unique_ptr<Decoder>(new Decoder(std::move(distorted_state).value()))});
}

Decoder::Decoder(std::unique_ptr<State> state) noexcept : state_(std::move(state)) {
}

Decoder::~Decoder() = default;
Decoder::Decoder(Decoder &&) noexcept = default;
Decoder &Decoder::operator=(Decoder &&) noexcept = default;

const std::filesystem::path &Decoder::path() const noexcept {
    return state_->path_;
}

const FrameDescriptor &Decoder::descriptor() const noexcept {
    return state_->descriptor_;
}

const VideoMetadata &Decoder::metadata() const noexcept {
    return state_->metadata_;
}

VideoCodec Decoder::codec() const noexcept {
    return state_->codec_;
}

VideoDecodeBackend Decoder::decode_backend() const noexcept {
    return state_->decode_backend_;
}

bool Decoder::has_vulkan_output() const noexcept { return state_->has_vulkan_output_; }

HardwareSurfaceFormat Decoder::hardware_surface_format() const noexcept {
    return state_->hardware_surface_format_;
}

std::optional<std::uint64_t> Decoder::frame_count_hint() const noexcept {
    return state_->frame_count_hint_;
}

Result<VideoFramePopResult> Decoder::try_pop() {
    return state_->try_pop();
}

Result<VideoFramePopResult> Decoder::wait_pop() {
    return state_->wait_pop();
}

VideoDecoderStatistics Decoder::statistics() const {
    return state_->statistics();
}

void Decoder::request_cancel() noexcept {
    state_->request_cancel();
}

void Decoder::drain() noexcept {
    state_->drain();
}

} // namespace crossvmaf::video
