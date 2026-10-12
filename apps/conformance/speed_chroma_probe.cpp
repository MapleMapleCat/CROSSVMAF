#include "speed_chroma_probe.hpp"

#include "gpu/device.hpp"
#include "frame_uploader.hpp"
#include "metrics/speed_chroma/reference.hpp"
#include "metrics/speed_chroma/vulkan.hpp"
#include "scoring/model.hpp"
#include "sha256.hpp"
#include "synthetic_frame.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace crossvmaf::conformance {
namespace {

constexpr crossvmaf::numeric::Q32_32 canonical_sigma_nn =
    crossvmaf::numeric::Q32_32::from_raw(816'043'776LL);
constexpr crossvmaf::numeric::Q32_32 canonical_nn_floor =
    crossvmaf::numeric::Q32_32::from_raw(429'496'736LL);

[[nodiscard]] metrics::speed_chroma::Config make_config(
    const scoring::PackedModel& packed) noexcept {
    return {
        crossvmaf::numeric::Q32_32::from_raw(packed.speed_chroma.kernel_scale_raw),
        crossvmaf::numeric::Q32_32::from_raw(packed.speed_chroma.prescale_raw),
        packed.speed_chroma.prescale_method == scoring::ScalingMethod::bilinear
            ? metrics::speed_chroma::PrescaleMethod::bilinear
            : metrics::speed_chroma::PrescaleMethod::nearest,
        crossvmaf::numeric::Q32_32::from_raw(packed.speed_chroma.sigma_nn_raw),
        crossvmaf::numeric::Q32_32::from_raw(packed.speed_chroma.nn_floor_raw),
        packed.speed_chroma.weight_variance_mode,
        crossvmaf::numeric::Q32_32::from_raw(packed.speed_chroma.maximum_raw),
    };
}

void append_i64_le(std::vector<std::byte>& output, std::int64_t value) {
    const auto bits = static_cast<std::uint64_t>(value);
    for (unsigned shift = 0U; shift < 64U; shift += 8U) {
        output.push_back(static_cast<std::byte>((bits >> shift) & 0xffU));
    }
}

[[nodiscard]] crossvmaf::numeric::Q32_32 absolute_q32(crossvmaf::numeric::Q32_32 value) noexcept {
    if (value.raw() >= 0) return value;
    if (value.raw() == std::numeric_limits<std::int64_t>::min()) {
        return crossvmaf::numeric::Q32_32::from_raw(std::numeric_limits<std::int64_t>::max());
    }
    return crossvmaf::numeric::Q32_32::from_raw(-value.raw());
}

[[nodiscard]] bool print_channel_diagnostics(
    std::string_view label, const PlaneView& reference,
    const PlaneView& distorted,
    const metrics::speed_chroma::Dimensions& dimensions,
    const metrics::speed_chroma::Config& config,
    std::uint8_t bit_depth, const std::filesystem::path& output_directory,
    metrics::speed_chroma::reference::ChannelStatistics& reference_output,
    metrics::speed_chroma::reference::ChannelStatistics& distorted_output,
    metrics::speed_chroma::ParameterDiagnostics& reference_parameters_output,
    metrics::speed_chroma::ParameterDiagnostics& distorted_parameters_output) {
    const auto reference_samples =
        metrics::speed_chroma::reference::detail::preprocess_channel(
            reference, dimensions, config, bit_depth);
    const auto distorted_samples =
        metrics::speed_chroma::reference::detail::preprocess_channel(
            distorted, dimensions, config, bit_depth);
    if (!reference_samples || !distorted_samples) {
        std::cerr << (!reference_samples ? reference_samples.error().message
                                         : distorted_samples.error().message)
                  << '\n';
        return false;
    }
    const auto write_samples = [&](std::string_view side,
                                   std::span<const std::int32_t> samples) {
        const std::filesystem::path path =
            output_directory /
            ("speed-" + std::string(label) + "-" + std::string(side) +
             "-q20.bin");
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) return false;
        stream.write(reinterpret_cast<const char*>(samples.data()),
                     static_cast<std::streamsize>(samples.size_bytes()));
        return stream.good();
    };
    if (!write_samples("reference", reference_samples.value()) ||
        !write_samples("distorted", distorted_samples.value())) {
        std::cerr << "failed to write Speed Chroma sample diagnostics\n";
        return false;
    }
    metrics::speed_chroma::ParameterDiagnostics reference_parameters;
    metrics::speed_chroma::ParameterDiagnostics distorted_parameters;
    const auto reference_statistics =
        metrics::speed_chroma::reference::estimate_parameters(
            dimensions, reference_samples.value(), canonical_sigma_nn,
            &reference_parameters);
    const auto distorted_statistics =
        metrics::speed_chroma::reference::estimate_parameters(
            dimensions, distorted_samples.value(), canonical_sigma_nn,
            &distorted_parameters);
    if (!reference_statistics || !distorted_statistics) {
        std::cerr << (!reference_statistics
                          ? reference_statistics.error().message
                          : distorted_statistics.error().message)
                  << '\n';
        return false;
    }
    reference_output = reference_statistics.value();
    distorted_output = distorted_statistics.value();
    reference_parameters_output = reference_parameters;
    distorted_parameters_output = distorted_parameters;
    std::cout << "diagnostic_" << label
              << "_preprocessed_sample_count="
              << reference_samples.value().size() << '\n';
    constexpr std::int64_t sample_scale = std::int64_t{1} <<
        (crossvmaf::numeric::Q32_32::fractional_bits -
         metrics::speed_chroma::reference::sample_fractional_bits);
    for (std::size_t index = 0; index < reference_samples.value().size();
         ++index) {
        const crossvmaf::numeric::Q32_32 reference_sample = crossvmaf::numeric::Q32_32::from_raw(
            static_cast<std::int64_t>(reference_samples.value()[index]) *
            sample_scale);
        const crossvmaf::numeric::Q32_32 distorted_sample = crossvmaf::numeric::Q32_32::from_raw(
            static_cast<std::int64_t>(distorted_samples.value()[index]) *
            sample_scale);
        std::cout << "diagnostic_" << label << "_reference_sample_" << index
                  << '=' << reference_sample.to_string(9) << '\n'
                  << "diagnostic_" << label << "_distorted_sample_" << index
                  << '=' << distorted_sample.to_string(9) << '\n';
    }
    for (std::size_t index = 0; index < reference_parameters.means.size();
         ++index) {
        std::cout << "diagnostic_" << label << "_reference_mean_" << index
                  << '=' << reference_parameters.means[index].to_string(9)
                  << '\n'
                  << "diagnostic_" << label << "_distorted_mean_" << index
                  << '=' << distorted_parameters.means[index].to_string(9)
                  << '\n';
    }
    for (std::size_t index = 0;
         index < reference_parameters.covariance.size(); ++index) {
        std::cout << "diagnostic_" << label << "_reference_covariance_"
                  << index << '='
                  << reference_parameters.covariance[index].to_string(9)
                  << '\n'
                  << "diagnostic_" << label << "_distorted_covariance_"
                  << index << '='
                  << distorted_parameters.covariance[index].to_string(9)
                  << '\n';
    }
    for (std::size_t index = 0;
         index < reference_parameters.tridiagonal_diagonal.size(); ++index) {
        std::cout << "diagnostic_" << label
                  << "_reference_tridiagonal_diagonal_" << index << '='
                  << reference_parameters.tridiagonal_diagonal[index].to_string(9)
                  << '\n'
                  << "diagnostic_" << label
                  << "_distorted_tridiagonal_diagonal_" << index << '='
                  << distorted_parameters.tridiagonal_diagonal[index].to_string(9)
                  << '\n';
    }
    for (std::size_t index = 0;
         index < reference_parameters.tridiagonal_subdiagonal.size(); ++index) {
        std::cout << "diagnostic_" << label
                  << "_reference_tridiagonal_subdiagonal_" << index << '='
                  << reference_parameters.tridiagonal_subdiagonal[index].to_string(9)
                  << '\n'
                  << "diagnostic_" << label
                  << "_distorted_tridiagonal_subdiagonal_" << index << '='
                  << distorted_parameters.tridiagonal_subdiagonal[index].to_string(9)
                  << '\n';
    }
    metrics::speed_chroma::reference::ChannelScoreDiagnostics score_diagnostics;
    const auto score = metrics::speed_chroma::reference::channel_score(
        dimensions, reference_statistics.value(), distorted_statistics.value(),
        canonical_sigma_nn, canonical_nn_floor, config.weight_variance_mode,
        &score_diagnostics);
    if (!score) {
        std::cerr << score.error().message << '\n';
        return false;
    }
    for (std::size_t block = 0;
         block < reference_statistics.value().entropies.size(); ++block) {
        std::cout
            << "diagnostic_" << label << "_block_" << block
            << "_reference_entropy="
            << reference_statistics.value().entropies[block].to_string(9)
            << '\n'
            << "diagnostic_" << label << "_block_" << block
            << "_reference_variance="
            << reference_statistics.value().variances[block].to_string(9)
            << '\n'
            << "diagnostic_" << label << "_block_" << block
            << "_distorted_entropy="
            << distorted_statistics.value().entropies[block].to_string(9)
            << '\n'
            << "diagnostic_" << label << "_block_" << block
            << "_distorted_variance="
            << distorted_statistics.value().variances[block].to_string(9)
            << '\n'
            << "diagnostic_" << label << "_block_" << block
            << "_spatial_reference="
            << score_diagnostics.blocks[block].spatial_reference.to_string(9)
            << '\n'
            << "diagnostic_" << label << "_block_" << block
            << "_spatial_distorted="
            << score_diagnostics.blocks[block].spatial_distorted.to_string(9)
            << '\n'
            << "diagnostic_" << label << "_block_" << block
            << "_absolute_difference="
            << score_diagnostics.blocks[block].absolute_difference.to_string(9)
            << '\n';
    }
    return true;
}

[[nodiscard]] bool compare_statistic(
    std::string_view channel, std::string_view statistic,
    std::span<const crossvmaf::numeric::Q32_32> cpu, std::span<const crossvmaf::numeric::Q32_32> gpu) {
    std::size_t mismatch_count = 0U;
    std::size_t first_mismatch = 0U;
    const std::size_t common = std::min(cpu.size(), gpu.size());
    for (std::size_t index = 0U; index < common; ++index) {
        if (cpu[index].raw() != gpu[index].raw()) {
            if (mismatch_count == 0U) first_mismatch = index;
            ++mismatch_count;
        }
    }
    mismatch_count += cpu.size() > gpu.size() ? cpu.size() - gpu.size()
                                               : gpu.size() - cpu.size();
    const bool equal = mismatch_count == 0U;
    std::cout << "gpu_statistic_" << channel << '_' << statistic
              << "_equal=" << (equal ? "true" : "false") << '\n'
              << "gpu_statistic_" << channel << '_' << statistic
              << "_mismatch_count=" << mismatch_count << '\n';
    if (!equal && first_mismatch < common) {
        std::cout << "gpu_statistic_" << channel << '_' << statistic
                  << "_first_index=" << first_mismatch << '\n'
                  << "gpu_statistic_" << channel << '_' << statistic
                  << "_cpu_raw=" << cpu[first_mismatch].raw() << '\n'
                  << "gpu_statistic_" << channel << '_' << statistic
                  << "_gpu_raw=" << gpu[first_mismatch].raw() << '\n';
    }
    return equal;
}

}  // namespace

int run_speed_chroma_probe(const std::filesystem::path& output_directory) {
    std::error_code error;
    std::filesystem::create_directories(output_directory, error);
    if (error) {
        std::cerr << "failed to create probe output directory: "
                  << error.message() << '\n';
        return 3;
    }
    const FrameDescriptor descriptor{
        216U, 216U, 10U, PixelFormat::yuv444p};
    const OwnedFrame reference_frame =
        make_speed_chroma_frame(descriptor, false);
    const OwnedFrame distorted_frame =
        make_speed_chroma_frame(descriptor, true);
    const auto reference_path = output_directory / "speed-reference.yuv";
    const auto distorted_path = output_directory / "speed-distorted.yuv";
    const auto wrote_reference =
        reference_frame.write_raw(reference_path, descriptor);
    const auto wrote_distorted =
        distorted_frame.write_raw(distorted_path, descriptor);
    if (!wrote_reference || !wrote_distorted) {
        std::cerr << (!wrote_reference ? wrote_reference.error().message
                                      : wrote_distorted.error().message)
                  << '\n';
        return 4;
    }
    const scoring::PackedModel* packed =
        scoring::find_packed_model(ModelVariant::v1_0_16_3d0h);
    if (packed == nullptr) {
        std::cerr << "pinned model is unavailable\n";
        return 5;
    }
    const auto config = make_config(*packed);
    metrics::speed_chroma::reference::Reference reference(
        descriptor, config);
    const auto cpu = reference.compute_detailed(
        reference_frame.view(), distorted_frame.view());
    if (!cpu) {
        std::cerr << cpu.error().message << '\n';
        return 6;
    }
    std::array<metrics::speed_chroma::reference::ChannelStatistics, 2U>
        cpu_reference_statistics;
    std::array<metrics::speed_chroma::reference::ChannelStatistics, 2U>
        cpu_distorted_statistics;
    std::array<metrics::speed_chroma::ParameterDiagnostics, 2U>
        cpu_reference_parameters;
    std::array<metrics::speed_chroma::ParameterDiagnostics, 2U>
        cpu_distorted_parameters;
    if (!print_channel_diagnostics(
            "u", reference_frame.view().planes[1],
            distorted_frame.view().planes[1], cpu.value().dimensions, config,
            descriptor.bit_depth, output_directory,
            cpu_reference_statistics[0], cpu_distorted_statistics[0],
            cpu_reference_parameters[0], cpu_distorted_parameters[0]) ||
        !print_channel_diagnostics(
            "v", reference_frame.view().planes[2],
            distorted_frame.view().planes[2], cpu.value().dimensions, config,
            descriptor.bit_depth, output_directory,
            cpu_reference_statistics[1], cpu_distorted_statistics[1],
            cpu_reference_parameters[1], cpu_distorted_parameters[1])) {
        return 6;
    }

    auto context = gpu::Device::create({});
    if (!context) {
        std::cerr << context.error().message << '\n';
        return 7;
    }
    auto uploader = crossvmaf::conformance::FrameUploader::create(
        context.value()->device_view(), descriptor);
    auto executor = metrics::speed_chroma::vulkan::Executor::create(
        context.value()->device_view(), descriptor, config);
    if (!uploader || !executor) {
        std::cerr << (!uploader ? uploader.error().message
                               : executor.error().message)
                  << '\n';
        return 7;
    }
    const auto uploaded = uploader.value()->upload(
        reference_frame.view(), distorted_frame.view());
    if (!uploaded) {
        std::cerr << uploaded.error().message << '\n';
        return 7;
    }
    bool preprocess_equal = true;
    for (std::size_t plane = 1U; plane < 3U; ++plane) {
        const std::string_view label = plane == 1U ? "u" : "v";
        const auto cpu_reference =
            metrics::speed_chroma::reference::detail::preprocess_channel(
                reference_frame.view().planes[plane], cpu.value().dimensions,
                config, descriptor.bit_depth);
        const auto cpu_distorted =
            metrics::speed_chroma::reference::detail::preprocess_channel(
                distorted_frame.view().planes[plane], cpu.value().dimensions,
                config, descriptor.bit_depth);
        const auto gpu_reference = executor.value()->preprocess_for_conformance(
            uploaded.value().reference.planes[plane]);
        const auto gpu_distorted = executor.value()->preprocess_for_conformance(
            uploaded.value().distorted.planes[plane]);
        if (!cpu_reference || !cpu_distorted || !gpu_reference ||
            !gpu_distorted) {
            const Error probe_error = !cpu_reference ? cpu_reference.error()
                : !cpu_distorted ? cpu_distorted.error()
                : !gpu_reference ? gpu_reference.error()
                                 : gpu_distorted.error();
            std::cerr << probe_error.message << '\n';
            return 7;
        }
        const auto compare_samples = [&](
            std::string_view side, std::span<const std::int32_t> cpu_samples,
            std::span<const std::int32_t> gpu_samples) {
            std::size_t mismatch_count = 0U;
            std::size_t first_mismatch = 0U;
            for (std::size_t index = 0U;
                 index < std::min(cpu_samples.size(), gpu_samples.size());
                 ++index) {
                if (cpu_samples[index] != gpu_samples[index]) {
                    if (mismatch_count == 0U) first_mismatch = index;
                    ++mismatch_count;
                }
            }
            mismatch_count += cpu_samples.size() > gpu_samples.size()
                ? cpu_samples.size() - gpu_samples.size()
                : gpu_samples.size() - cpu_samples.size();
            const bool equal = mismatch_count == 0U;
            std::cout << "gpu_preprocess_" << label << '_' << side
                      << "_equal=" << (equal ? "true" : "false") << '\n'
                      << "gpu_preprocess_" << label << '_' << side
                      << "_mismatch_count=" << mismatch_count << '\n';
            if (!equal && first_mismatch < cpu_samples.size() &&
                first_mismatch < gpu_samples.size()) {
                std::cout << "gpu_preprocess_" << label << '_' << side
                          << "_first_index=" << first_mismatch << '\n'
                          << "gpu_preprocess_" << label << '_' << side
                          << "_cpu_q20=" << cpu_samples[first_mismatch] << '\n'
                          << "gpu_preprocess_" << label << '_' << side
                          << "_gpu_q20=" << gpu_samples[first_mismatch] << '\n';
            }
            preprocess_equal = preprocess_equal && equal;
        };
        compare_samples("reference", cpu_reference.value(),
                        gpu_reference.value());
        compare_samples("distorted", cpu_distorted.value(),
                        gpu_distorted.value());
    }
    if (!preprocess_equal) return 9;
    metrics::speed_chroma::DetailedDiagnostics gpu_diagnostics;
    const auto gpu = executor.value()->compute_detailed(
        uploaded.value().reference, uploaded.value().distorted,
        &gpu_diagnostics);
    if (!gpu) {
        std::cerr << gpu.error().message << '\n';
        return 7;
    }
    const std::array<const metrics::speed_chroma::ChannelDiagnostics*,
                     2U>
        gpu_channels{&gpu_diagnostics.u, &gpu_diagnostics.v};
    bool statistics_equal = true;
    for (std::size_t channel = 0U; channel < gpu_channels.size(); ++channel) {
        const std::string_view label = channel == 0U ? "u" : "v";
        const auto& gpu_channel = *gpu_channels[channel];
        std::cout << "gpu_sturm_" << label << "_reference_negative_pivots="
                  << gpu_channel.reference_parameters.negative_pivot_count
                  << '\n'
                  << "gpu_sturm_" << label
                  << "_reference_stabilized_pivots="
                  << gpu_channel.reference_parameters.stabilized_pivot_count
                  << '\n'
                  << "gpu_sturm_" << label << "_distorted_negative_pivots="
                  << gpu_channel.distorted_parameters.negative_pivot_count
                  << '\n'
                  << "gpu_sturm_" << label
                  << "_distorted_stabilized_pivots="
                  << gpu_channel.distorted_parameters.stabilized_pivot_count
                  << '\n';
        statistics_equal = compare_statistic(
            label, "reference_mean",
            cpu_reference_parameters[channel].means,
            gpu_channel.reference_parameters.means) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "reference_covariance",
            cpu_reference_parameters[channel].covariance,
            gpu_channel.reference_parameters.covariance) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "reference_tridiagonal_diagonal",
            cpu_reference_parameters[channel].tridiagonal_diagonal,
            gpu_channel.reference_parameters.tridiagonal_diagonal) &&
            statistics_equal;
        statistics_equal = compare_statistic(
            label, "reference_tridiagonal_subdiagonal",
            cpu_reference_parameters[channel].tridiagonal_subdiagonal,
            gpu_channel.reference_parameters.tridiagonal_subdiagonal) &&
            statistics_equal;
        statistics_equal = compare_statistic(
            label, "distorted_mean",
            cpu_distorted_parameters[channel].means,
            gpu_channel.distorted_parameters.means) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "distorted_covariance",
            cpu_distorted_parameters[channel].covariance,
            gpu_channel.distorted_parameters.covariance) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "distorted_tridiagonal_diagonal",
            cpu_distorted_parameters[channel].tridiagonal_diagonal,
            gpu_channel.distorted_parameters.tridiagonal_diagonal) &&
            statistics_equal;
        statistics_equal = compare_statistic(
            label, "distorted_tridiagonal_subdiagonal",
            cpu_distorted_parameters[channel].tridiagonal_subdiagonal,
            gpu_channel.distorted_parameters.tridiagonal_subdiagonal) &&
            statistics_equal;
        statistics_equal = compare_statistic(
            label, "reference_entropy",
            cpu_reference_statistics[channel].entropies,
            gpu_channel.reference_entropies) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "reference_variance",
            cpu_reference_statistics[channel].variances,
            gpu_channel.reference_variances) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "distorted_entropy",
            cpu_distorted_statistics[channel].entropies,
            gpu_channel.distorted_entropies) && statistics_equal;
        statistics_equal = compare_statistic(
            label, "distorted_variance",
            cpu_distorted_statistics[channel].variances,
            gpu_channel.distorted_variances) && statistics_equal;
    }
    const std::array<const metrics::speed_chroma::vulkan::
                         ParameterDiagnostics*, 4U>
        wave_parameter_sources{
            &gpu_diagnostics.u.reference_parameters,
            &gpu_diagnostics.u.distorted_parameters,
            &gpu_diagnostics.v.reference_parameters,
            &gpu_diagnostics.v.distorted_parameters,
        };
    bool wave_conformance_pass = true;
    for (const std::uint32_t wave_task_count :
         {1U, 3U, 4U, 7U, 8U, 16U, 31U, 32U}) {
        std::vector<crossvmaf::numeric::Q32_32> covariance_wave;
        covariance_wave.reserve(
            static_cast<std::size_t>(wave_task_count) *
            wave_parameter_sources.size() *
            metrics::speed_chroma::vulkan::
                score_covariance_elements);
        for (std::uint32_t task_index = 0U;
             task_index < wave_task_count; ++task_index) {
            for (const auto* parameter_source : wave_parameter_sources) {
                covariance_wave.insert(
                    covariance_wave.end(),
                    parameter_source->covariance.begin(),
                    parameter_source->covariance.end());
            }
        }
        const auto wave = executor.value()->
            factorize_covariance_wave_for_conformance(
                covariance_wave, wave_task_count);
        if (!wave) {
            std::cerr << wave.error().message << '\n';
            return 7;
        }
        wave_conformance_pass = wave_conformance_pass &&
            wave.value().stream_diagnostics.size() ==
                static_cast<std::size_t>(wave_task_count) *
                    wave_parameter_sources.size();
    }
    std::cout << "gpu_wave_conformance_pass="
              << (wave_conformance_pass ? "true" : "false") << '\n';
    if (!wave_conformance_pass) return 9;
    const bool raw_equal =
        cpu.value().u.raw_q32_32 == gpu.value().u.raw_q32_32 &&
        cpu.value().v.raw_q32_32 == gpu.value().v.raw_q32_32 &&
        cpu.value().uv.raw_q32_32 == gpu.value().uv.raw_q32_32 &&
        cpu.value().u_singular == gpu.value().u_singular &&
        cpu.value().v_singular == gpu.value().v_singular;
    const auto official_u = crossvmaf::numeric::Q32_32::parse_decimal("7.855857849");
    const auto official_v = crossvmaf::numeric::Q32_32::parse_decimal("15.988956451");
    const auto official_uv = crossvmaf::numeric::Q32_32::parse_decimal("11.922407150");
    const auto official_tolerance = crossvmaf::numeric::Q32_32::parse_decimal("0.000010000");
    if (!official_u || !official_v || !official_uv || !official_tolerance) {
        std::cerr << "failed to parse pinned official Speed Chroma values\n";
        return 6;
    }
    const crossvmaf::numeric::Q32_32 official_u_error = crossvmaf::numeric::Q32_32::subtract_saturating(
        crossvmaf::numeric::Q32_32::from_raw(gpu.value().u.raw_q32_32), official_u.value());
    const crossvmaf::numeric::Q32_32 official_v_error = crossvmaf::numeric::Q32_32::subtract_saturating(
        crossvmaf::numeric::Q32_32::from_raw(gpu.value().v.raw_q32_32), official_v.value());
    const crossvmaf::numeric::Q32_32 official_uv_error = crossvmaf::numeric::Q32_32::subtract_saturating(
        crossvmaf::numeric::Q32_32::from_raw(gpu.value().uv.raw_q32_32), official_uv.value());
    const bool official_compatibility_pass =
        absolute_q32(official_u_error) <= official_tolerance.value() &&
        absolute_q32(official_v_error) <= official_tolerance.value() &&
        absolute_q32(official_uv_error) <= official_tolerance.value();
    std::vector<std::byte> raw_stream;
    raw_stream.reserve(3U * sizeof(std::int64_t));
    append_i64_le(raw_stream, gpu.value().u.raw_q32_32);
    append_i64_le(raw_stream, gpu.value().v.raw_q32_32);
    append_i64_le(raw_stream, gpu.value().uv.raw_q32_32);
    const auto& device = context.value()->device_info();
    std::cout << "model=" << packed->name << '\n'
              << "pixel_format=yuv444p10le\n"
              << "width=" << descriptor.width << '\n'
              << "height=" << descriptor.height << '\n'
              << "device_name=" << device.name << '\n'
              << "device_uuid=" << device.uuid << '\n'
              << "vendor_id=" << device.vendor_id << '\n'
              << "driver_version_raw=" << device.driver_version << '\n'
              << "reference_path=" << reference_path.string() << '\n'
              << "distorted_path=" << distorted_path.string() << '\n'
              << "cpu_speed_chroma_u_raw=" << cpu.value().u.raw_q32_32 << '\n'
              << "gpu_speed_chroma_u_raw=" << gpu.value().u.raw_q32_32 << '\n'
              << "speed_chroma_u=" << gpu.value().u.decimal() << '\n'
              << "cpu_speed_chroma_v_raw=" << cpu.value().v.raw_q32_32 << '\n'
              << "gpu_speed_chroma_v_raw=" << gpu.value().v.raw_q32_32 << '\n'
              << "speed_chroma_v=" << gpu.value().v.decimal() << '\n'
              << "cpu_speed_chroma_uv_raw=" << cpu.value().uv.raw_q32_32 << '\n'
              << "gpu_speed_chroma_uv_raw=" << gpu.value().uv.raw_q32_32 << '\n'
              << "speed_chroma_uv=" << gpu.value().uv.decimal() << '\n'
              << "u_singular=" << (gpu.value().u_singular ? "true" : "false")
              << '\n'
              << "v_singular=" << (gpu.value().v_singular ? "true" : "false")
              << '\n'
              << "cpu_gpu_raw_equal=" << (raw_equal ? "true" : "false")
              << '\n'
              << "cpu_gpu_statistics_equal="
              << (statistics_equal ? "true" : "false") << '\n'
              << "official_speed_chroma_u="
              << official_u.value().to_string(9) << '\n'
              << "official_speed_chroma_u_error="
              << official_u_error.to_string(9) << '\n'
              << "official_speed_chroma_u_abs_error="
              << absolute_q32(official_u_error).to_string(9) << '\n'
              << "official_speed_chroma_v="
              << official_v.value().to_string(9) << '\n'
              << "official_speed_chroma_v_error="
              << official_v_error.to_string(9) << '\n'
              << "official_speed_chroma_v_abs_error="
              << absolute_q32(official_v_error).to_string(9) << '\n'
              << "official_speed_chroma_uv="
              << official_uv.value().to_string(9) << '\n'
              << "official_speed_chroma_uv_error="
              << official_uv_error.to_string(9) << '\n'
              << "official_speed_chroma_uv_abs_error="
              << absolute_q32(official_uv_error).to_string(9) << '\n'
              << "official_compatibility_tolerance="
              << official_tolerance.value().to_string(9) << '\n'
              << "official_compatibility_pass="
              << (official_compatibility_pass ? "true" : "false") << '\n'
              << "raw_stream_sha256=" << sha256_hex(raw_stream) << '\n';
    return raw_equal && official_compatibility_pass ? 0 : 8;
}

}  // namespace crossvmaf::conformance
