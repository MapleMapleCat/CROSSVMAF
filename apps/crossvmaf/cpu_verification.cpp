#include "cpu_verification.hpp"

#include "hotspot_profiler.hpp"
#include "options.hpp"
#include "reference_evaluator.hpp"
#include "video/decoder.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace crossvmaf::cli {
namespace {

constexpr auto decoder_poll_interval = std::chrono::microseconds(50);

[[nodiscard]] bool descriptors_match(
    const crossvmaf::FrameDescriptor& left,
    const crossvmaf::FrameDescriptor& right) noexcept {
    return left.width == right.width && left.height == right.height &&
        left.bit_depth == right.bit_depth &&
        left.pixel_format == right.pixel_format;
}

class OwnedFrameStorage final {
public:
    explicit OwnedFrameStorage(
        const crossvmaf::FrameDescriptor& descriptor) {
        const std::size_t bytes_per_sample =
            descriptor.bit_depth == 8U ? 1U : 2U;
        for (std::size_t plane_index = 0U; plane_index < 3U;
             ++plane_index) {
            strides_[plane_index] =
                static_cast<std::size_t>(crossvmaf::plane_width(
                    descriptor, plane_index)) * bytes_per_sample;
            planes_[plane_index].resize(
                strides_[plane_index] *
                crossvmaf::plane_height(descriptor, plane_index));
        }
    }

    [[nodiscard]] crossvmaf::WritableFrameView writable_view(
        std::uint64_t frame_index) noexcept {
        crossvmaf::WritableFrameView frame{};
        frame.index = frame_index;
        for (std::size_t plane_index = 0U; plane_index < 3U;
             ++plane_index) {
            frame.planes[plane_index] = {
                planes_[plane_index].data(),
                planes_[plane_index].size(),
                strides_[plane_index]};
        }
        return frame;
    }

    [[nodiscard]] crossvmaf::FrameView view(
        std::uint64_t frame_index) const noexcept {
        crossvmaf::FrameView frame{};
        frame.index = frame_index;
        for (std::size_t plane_index = 0U; plane_index < 3U;
             ++plane_index) {
            frame.planes[plane_index] = {
                planes_[plane_index].data(), strides_[plane_index]};
        }
        return frame;
    }

private:
    std::array<std::vector<std::byte>, 3U> planes_;
    std::array<std::size_t, 3U> strides_{};
};

[[nodiscard]] std::size_t calculate_reference_batch_size(
    const crossvmaf::FrameDescriptor& descriptor) noexcept {
    constexpr std::size_t maximum_batch_size = 4U;
    constexpr std::uint64_t maximum_batch_input_bytes =
        std::uint64_t{128} * 1024U * 1024U;
    constexpr std::uint64_t maximum_u64 =
        (std::numeric_limits<std::uint64_t>::max)();

    const std::uint64_t bytes_per_sample =
        descriptor.bit_depth == 8U ? 1U : 2U;
    std::uint64_t frame_bytes = 0U;
    for (std::size_t plane_index = 0U; plane_index < 3U; ++plane_index) {
        const std::uint64_t width = crossvmaf::plane_width(
            descriptor, plane_index);
        const std::uint64_t height = crossvmaf::plane_height(
            descriptor, plane_index);
        if (height != 0U && width > maximum_u64 / height) return 1U;
        const std::uint64_t plane_samples = width * height;
        if (plane_samples > maximum_u64 / bytes_per_sample) return 1U;
        const std::uint64_t plane_bytes =
            plane_samples * bytes_per_sample;
        if (frame_bytes > maximum_u64 - plane_bytes) return 1U;
        frame_bytes += plane_bytes;
    }
    if (frame_bytes == 0U || frame_bytes > maximum_u64 / 2U) return 1U;

    const std::uint64_t frame_pair_bytes = frame_bytes * 2U;
    const std::uint64_t memory_limited_batch_size =
        maximum_batch_input_bytes / frame_pair_bytes;
    return static_cast<std::size_t>(std::clamp<std::uint64_t>(
        memory_limited_batch_size, 1U, maximum_batch_size));
}

[[nodiscard]] crossvmaf::Result<
    std::vector<crossvmaf::FrameResult>>
evaluate_reference_sequence(
    crossvmaf::verification::ReferenceEvaluator& evaluator,
    const crossvmaf::cli::Options& options,
    const crossvmaf::FrameDescriptor& expected_descriptor,
    std::size_t batch_size,
    crossvmaf::cli::CliHotspotProfiler& hotspot_profiler) {
    crossvmaf::video::VideoDecoderOptions decoder_options{
        options.decoded_frame_queue_capacity,
        options.decoder_threads_per_source};
    decoder_options.concurrent_decoder_count = 2U;
    auto reference_decoder = crossvmaf::video::Decoder::open(
        options.reference_path, decoder_options);
    if (!reference_decoder) {
        return crossvmaf::Result<
            std::vector<crossvmaf::FrameResult>>::failure(
                reference_decoder.error());
    }
    auto distorted_decoder = crossvmaf::video::Decoder::open(
        options.distorted_path, decoder_options);
    if (!distorted_decoder) {
        return crossvmaf::Result<
            std::vector<crossvmaf::FrameResult>>::failure(
                distorted_decoder.error());
    }
    if (!descriptors_match(
            expected_descriptor, reference_decoder.value()->descriptor()) ||
        !descriptors_match(
            expected_descriptor, distorted_decoder.value()->descriptor())) {
        return crossvmaf::Result<
            std::vector<crossvmaf::FrameResult>>::failure(
                {crossvmaf::ErrorCode::invalid_argument,
                 "CPU verification decoders produced a different descriptor"});
    }

    std::vector<OwnedFrameStorage> reference_storage;
    std::vector<OwnedFrameStorage> distorted_storage;
    reference_storage.reserve(batch_size);
    distorted_storage.reserve(batch_size);
    for (std::size_t frame_offset = 0U;
         frame_offset < batch_size; ++frame_offset) {
        reference_storage.emplace_back(expected_descriptor);
        distorted_storage.emplace_back(expected_descriptor);
    }
    std::vector<crossvmaf::FrameView> reference_batch;
    std::vector<crossvmaf::FrameView> distorted_batch;
    reference_batch.reserve(batch_size);
    distorted_batch.reserve(batch_size);
    std::optional<crossvmaf::video::Frame> reference_frame;
    std::optional<crossvmaf::video::Frame> distorted_frame;
    bool reference_end_of_stream = false;
    bool distorted_end_of_stream = false;
    std::uint64_t frame_index = 0U;
    std::vector<crossvmaf::FrameResult> results;

    const auto evaluate_batch = [&]() -> crossvmaf::Result<void> {
        if (reference_batch.empty()) {
            return crossvmaf::Result<void>::success();
        }
        auto evaluated = hotspot_profiler.measure_operation(
            crossvmaf::cli::CliHotspotStage::reference_push,
            [&] {
                return evaluator.push_batch(
                    std::span<const crossvmaf::FrameView>(reference_batch),
                    std::span<const crossvmaf::FrameView>(distorted_batch));
            });
        if (!evaluated) {
            return crossvmaf::Result<void>::failure(evaluated.error());
        }
        results.insert(
            results.end(),
            std::make_move_iterator(evaluated.value().begin()),
            std::make_move_iterator(evaluated.value().end()));
        reference_batch.clear();
        distorted_batch.clear();
        return crossvmaf::Result<void>::success();
    };

    for (;;) {
        if (!reference_frame.has_value() && !reference_end_of_stream) {
            auto popped = reference_decoder.value()->try_pop();
            if (!popped) {
                return crossvmaf::Result<
                    std::vector<crossvmaf::FrameResult>>::failure(
                        popped.error());
            }
            if (popped.value().status ==
                crossvmaf::video::VideoFramePopStatus::frame) {
                reference_frame = std::move(popped).value().frame;
            } else if (popped.value().status ==
                       crossvmaf::video::VideoFramePopStatus::end_of_stream) {
                reference_end_of_stream = true;
            }
        }
        if (!distorted_frame.has_value() && !distorted_end_of_stream) {
            auto popped = distorted_decoder.value()->try_pop();
            if (!popped) {
                return crossvmaf::Result<
                    std::vector<crossvmaf::FrameResult>>::failure(
                        popped.error());
            }
            if (popped.value().status ==
                crossvmaf::video::VideoFramePopStatus::frame) {
                distorted_frame = std::move(popped).value().frame;
            } else if (popped.value().status ==
                       crossvmaf::video::VideoFramePopStatus::end_of_stream) {
                distorted_end_of_stream = true;
            }
        }

        if (reference_frame.has_value() && distorted_frame.has_value()) {
            const std::size_t frame_offset = reference_batch.size();
            auto copied = hotspot_profiler.measure_operation(
                crossvmaf::cli::CliHotspotStage::frame_read,
                [&] {
                    const auto reference_copy = reference_frame->copy_to(
                        reference_storage[frame_offset].writable_view(
                            frame_index));
                    if (!reference_copy) return reference_copy;
                    return distorted_frame->copy_to(
                        distorted_storage[frame_offset].writable_view(
                            frame_index));
                });
            if (!copied) {
                return crossvmaf::Result<
                    std::vector<crossvmaf::FrameResult>>::failure(
                        copied.error());
            }

            reference_batch.push_back(
                reference_storage[frame_offset].view(frame_index));
            distorted_batch.push_back(
                distorted_storage[frame_offset].view(frame_index));
            reference_frame.reset();
            distorted_frame.reset();
            ++frame_index;
            if (reference_batch.size() == batch_size) {
                const auto evaluated = evaluate_batch();
                if (!evaluated) {
                    return crossvmaf::Result<
                        std::vector<crossvmaf::FrameResult>>::failure(
                            evaluated.error());
                }
            }
            continue;
        }

        if (reference_end_of_stream || distorted_end_of_stream) {
            if (reference_end_of_stream && distorted_end_of_stream &&
                !reference_frame.has_value() &&
                !distorted_frame.has_value()) {
                break;
            }
            return crossvmaf::Result<
                std::vector<crossvmaf::FrameResult>>::failure(
                    {crossvmaf::ErrorCode::invalid_argument,
                     "CPU verification videos decoded different frame counts"});
        }
        std::this_thread::sleep_for(decoder_poll_interval);
    }

    reference_decoder.value()->drain();
    distorted_decoder.value()->drain();
    const auto evaluated = evaluate_batch();
    if (!evaluated) {
        return crossvmaf::Result<
            std::vector<crossvmaf::FrameResult>>::failure(
                evaluated.error());
    }
    auto tail = hotspot_profiler.measure_operation(
        crossvmaf::cli::CliHotspotStage::reference_finish,
        [&evaluator] { return evaluator.flush(); });
    if (!tail) {
        return crossvmaf::Result<
            std::vector<crossvmaf::FrameResult>>::failure(tail.error());
    }
    results.insert(results.end(), tail.value().begin(), tail.value().end());
    return crossvmaf::Result<
        std::vector<crossvmaf::FrameResult>>::success(std::move(results));
}

} // namespace

CpuVerification::CpuVerification(
    std::unique_ptr<verification::ReferenceEvaluator> evaluator,
    FrameDescriptor descriptor, std::size_t batch_size)
    : evaluator_(std::move(evaluator)), descriptor_(descriptor), batch_size_(batch_size) {}

CpuVerification::~CpuVerification() = default;

Result<std::unique_ptr<CpuVerification>> CpuVerification::create(
    const FrameDescriptor& descriptor, ModelVariant model,
    CliHotspotProfiler& hotspot_profiler) {
    const std::size_t batch_size = calculate_reference_batch_size(descriptor);
    verification::ReferenceEvaluatorConfig config{};
    config.frame = descriptor;
    config.model = model;
    config.maximum_batch_size = batch_size;
    auto created = hotspot_profiler.measure_operation(
        CliHotspotStage::reference_open,
        [&config] { return verification::ReferenceEvaluator::create(config); });
    if (!created) {
        return Result<std::unique_ptr<CpuVerification>>::failure(created.error());
    }
    return Result<std::unique_ptr<CpuVerification>>::success(
        std::unique_ptr<CpuVerification>(new CpuVerification(
            std::move(created).value(), descriptor, batch_size)));
}

Result<std::vector<FrameResult>> CpuVerification::evaluate(
    const Options& options, CliHotspotProfiler& hotspot_profiler) {
    return evaluate_reference_sequence(
        *evaluator_, options, descriptor_, batch_size_, hotspot_profiler);
}

[[nodiscard]] crossvmaf::Result<void> verify_cpu_results(
    std::span<const FrameResult> gpu_results,
    std::span<const FrameResult> reference_results) {
    constexpr std::uint64_t adm3_maximum_raw_error = 429'496U;
    constexpr double vmaf_maximum_error = 1.0e-3;
    if (gpu_results.size() != reference_results.size()) {
        return crossvmaf::Result<void>::failure(
            {crossvmaf::ErrorCode::backend_error,
             "GPU/reference verification emitted a different result count"});
    }

    const auto absolute_raw_difference = [](
        crossvmaf::Q32Value left,
        crossvmaf::Q32Value right) noexcept -> std::uint64_t {
        return left.raw_q32_32 >= right.raw_q32_32
            ? static_cast<std::uint64_t>(left.raw_q32_32) -
                  static_cast<std::uint64_t>(right.raw_q32_32)
            : static_cast<std::uint64_t>(right.raw_q32_32) -
                  static_cast<std::uint64_t>(left.raw_q32_32);
    };
    const auto values_equal = [](
        crossvmaf::Q32Value left,
        crossvmaf::Q32Value right) noexcept {
        return left.raw_q32_32 == right.raw_q32_32;
    };

    for (std::size_t result_index = 0U;
         result_index < gpu_results.size(); ++result_index) {
        const crossvmaf::FrameResult& gpu_result = gpu_results[result_index];
        const crossvmaf::FrameResult& reference_result =
            reference_results[result_index];
        if (gpu_result.index != reference_result.index ||
            !values_equal(
                gpu_result.features.cambi,
                reference_result.features.cambi) ||
            !values_equal(
                gpu_result.features.speed_chroma_uv,
                reference_result.features.speed_chroma_uv) ||
            !values_equal(
                gpu_result.speed_chroma.u,
                reference_result.speed_chroma.u) ||
            !values_equal(
                gpu_result.speed_chroma.v,
                reference_result.speed_chroma.v) ||
            !values_equal(
                gpu_result.speed_chroma.uv,
                reference_result.speed_chroma.uv) ||
            gpu_result.speed_chroma.u_singular !=
                reference_result.speed_chroma.u_singular ||
            gpu_result.speed_chroma.v_singular !=
                reference_result.speed_chroma.v_singular ||
            absolute_raw_difference(
                gpu_result.features.adm3,
                reference_result.features.adm3) >
                adm3_maximum_raw_error ||
            !values_equal(
                gpu_result.features.motion3,
                reference_result.features.motion3) ||
            std::abs(gpu_result.vmaf - reference_result.vmaf) >
                vmaf_maximum_error) {
            return crossvmaf::Result<void>::failure(
                {crossvmaf::ErrorCode::backend_error,
                 "GPU/reference verification mismatch at frame " +
                     std::to_string(gpu_result.index)});
        }
    }
    return crossvmaf::Result<void>::success();
}

} // namespace crossvmaf::cli
