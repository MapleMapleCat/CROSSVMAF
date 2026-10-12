#define CROSSVMAF_EVALUATION_IMPLEMENTATION
#include "metrics/cambi/cambi.hpp"
#include "crossvmaf/video.hpp"
#include "video/frame.hpp"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace crossvmaf {
namespace {

// Average frame rates whose relative difference exceeds this fraction are
// different timelines; below it they are the same rate expressed with
// different rationals (e.g. 24000/1001 vs 23.976).
constexpr double frame_rate_mismatch_tolerance = 1.0e-3;
// Durations differing by more than this fraction are worth a warning.
constexpr double duration_mismatch_tolerance = 2.0e-2;

[[nodiscard]] std::string rational_text(const Rational &value) {
    if (!value.valid()) return "unknown";
    return std::to_string(value.numerator) + "/" + std::to_string(value.denominator);
}

[[nodiscard]] std::string dimensions_text(const FrameDescriptor &descriptor) {
    return std::to_string(descriptor.width) + "x" + std::to_string(descriptor.height);
}

[[nodiscard]] std::string bit_depth_text(const FrameDescriptor &descriptor) {
    return std::to_string(descriptor.bit_depth) + "-bit";
}

[[nodiscard]] std::string pixel_format_text(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::yuv420p:
        return "yuv420p";
    case PixelFormat::yuv422p:
        return "yuv422p";
    case PixelFormat::yuv444p:
        return "yuv444p";
    }
    return "unknown";
}

[[nodiscard]] std::string seconds_text(double seconds) {
    char buffer[32]{};
    std::snprintf(buffer, sizeof(buffer), "%.3f s", seconds);
    return buffer;
}

void add_issue(PairValidation &validation, std::string reason_code, std::string field,
               PairIssueSeverity severity, std::string reference_value, std::string distorted_value,
               std::string message, bool requires_runtime_validation) {
    PairIssue issue;
    issue.reason_code = std::move(reason_code);
    issue.field = std::move(field);
    issue.severity = severity;
    issue.reference_value = std::move(reference_value);
    issue.distorted_value = std::move(distorted_value);
    issue.message = std::move(message);
    issue.requires_runtime_validation = requires_runtime_validation;
    validation.issues.push_back(std::move(issue));
}

void compare_descriptors(const VideoMetadata &reference, const VideoMetadata &distorted,
                         PairValidation &validation) {
    if (!reference.descriptor.has_value() || !distorted.descriptor.has_value()) {
        add_issue(
            validation, "spec_unconfirmed", "descriptor", PairIssueSeverity::info,
            reference.descriptor.has_value() ? dimensions_text(*reference.descriptor) : "unknown",
            distorted.descriptor.has_value() ? dimensions_text(*distorted.descriptor) : "unknown",
            "video spec is not available from container metadata and will be "
            "validated at runtime",
            true);
        return;
    }
    const FrameDescriptor &reference_descriptor = *reference.descriptor;
    const FrameDescriptor &distorted_descriptor = *distorted.descriptor;
    if (reference_descriptor.width != distorted_descriptor.width ||
        reference_descriptor.height != distorted_descriptor.height) {
        add_issue(validation, "dimension_mismatch", "dimensions", PairIssueSeverity::error,
                  dimensions_text(reference_descriptor), dimensions_text(distorted_descriptor),
                  "reference and distorted video dimensions differ", false);
    }
    if (reference_descriptor.bit_depth != distorted_descriptor.bit_depth) {
        add_issue(validation, "bit_depth_mismatch", "bit_depth", PairIssueSeverity::error,
                  bit_depth_text(reference_descriptor), bit_depth_text(distorted_descriptor),
                  "reference and distorted video bit depths differ", false);
    }
    if (reference_descriptor.pixel_format != distorted_descriptor.pixel_format) {
        add_issue(validation, "pixel_format_mismatch", "pixel_format", PairIssueSeverity::error,
                  pixel_format_text(reference_descriptor.pixel_format),
                  pixel_format_text(distorted_descriptor.pixel_format),
                  "reference and distorted video chroma formats differ", false);
    }
}

void compare_frame_rates(const VideoMetadata &reference, const VideoMetadata &distorted,
                         PairValidation &validation) {
    const FrameRateInfo &reference_rate = reference.frame_rate;
    const FrameRateInfo &distorted_rate = distorted.frame_rate;
    if (reference_rate.average.valid() && distorted_rate.average.valid()) {
        const double reference_value = reference_rate.average.value();
        const double distorted_value = distorted_rate.average.value();
        const double relative_difference = std::fabs(reference_value - distorted_value) /
                                           std::max(reference_value, distorted_value);
        if (relative_difference > frame_rate_mismatch_tolerance) {
            add_issue(validation, "frame_rate_mismatch", "frame_rate", PairIssueSeverity::error,
                      rational_text(reference_rate.average), rational_text(distorted_rate.average),
                      "reference and distorted average frame rates differ; the "
                      "presentation timelines diverge",
                      false);
        }
    } else {
        add_issue(validation, "frame_rate_unknown", "frame_rate", PairIssueSeverity::info,
                  rational_text(reference_rate.average), rational_text(distorted_rate.average),
                  "average frame rate is unknown for at least one input and will "
                  "be validated per frame at runtime",
                  true);
    }
    if (reference_rate.mode == FrameRateMode::variable ||
        distorted_rate.mode == FrameRateMode::variable) {
        add_issue(validation, "variable_frame_rate", "frame_rate", PairIssueSeverity::warning,
                  reference_rate.mode == FrameRateMode::variable ? "variable" : "constant",
                  distorted_rate.mode == FrameRateMode::variable ? "variable" : "constant",
                  "at least one input appears to use variable frame rate "
                  "timestamps; runtime timeline validation applies",
                  true);
    }
}

void compare_durations(const VideoMetadata &reference, const VideoMetadata &distorted,
                       PairValidation &validation) {
    if (!reference.duration_seconds.has_value() || !distorted.duration_seconds.has_value()) {
        return;
    }
    const double reference_value = *reference.duration_seconds;
    const double distorted_value = *distorted.duration_seconds;
    const double relative_difference =
        std::fabs(reference_value - distorted_value) / std::max(reference_value, distorted_value);
    if (relative_difference > duration_mismatch_tolerance) {
        add_issue(validation, "duration_mismatch", "duration", PairIssueSeverity::warning,
                  seconds_text(reference_value), seconds_text(distorted_value),
                  "container-reported durations differ; evaluation stops at the "
                  "shorter input",
                  false);
    }
}

void compare_frame_counts(const VideoMetadata &reference, const VideoMetadata &distorted,
                          PairValidation &validation) {
    // Estimated counts are too unreliable to compare.
    if (!reference.frame_count.has_value() || !distorted.frame_count.has_value() ||
        reference.frame_count_estimated || distorted.frame_count_estimated) {
        return;
    }
    if (*reference.frame_count != *distorted.frame_count) {
        add_issue(validation, "frame_count_mismatch", "frame_count", PairIssueSeverity::warning,
                  std::to_string(*reference.frame_count), std::to_string(*distorted.frame_count),
                  "container-reported frame counts differ; evaluation stops at "
                  "the shorter input",
                  false);
    }
}

[[nodiscard]] std::string path_text(const std::filesystem::path &path) {
    const std::u8string utf8 = path.u8string();
    return std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size());
}

void compare_file_identities(const VideoMetadata &reference, const VideoMetadata &distorted,
                             PairValidation &validation) {
    if (!reference.identity.captured || !distorted.identity.captured) return;
    if (reference.identity.path.lexically_normal() == distorted.identity.path.lexically_normal() &&
        !reference.identity.path.empty()) {
        add_issue(validation, "same_input_file", "path", PairIssueSeverity::error,
                  path_text(reference.identity.path), path_text(distorted.identity.path),
                  "reference and distorted inputs are the same file", false);
    }
}

} // namespace

Result<PairValidation> validate_video_pair(const VideoMetadata &reference,
                                           const VideoMetadata &distorted) {
    PairValidation validation;
    compare_file_identities(reference, distorted, validation);
    compare_descriptors(reference, distorted, validation);
    compare_frame_rates(reference, distorted, validation);
    compare_durations(reference, distorted, validation);
    compare_frame_counts(reference, distorted, validation);

    bool has_error = false;
    bool requires_runtime = false;
    for (const PairIssue &issue : validation.issues) {
        if (issue.severity == PairIssueSeverity::error) has_error = true;
        if (issue.requires_runtime_validation) requires_runtime = true;
    }
    validation.status = has_error          ? PairValidationStatus::invalid
                        : requires_runtime ? PairValidationStatus::needs_runtime_validation
                                           : PairValidationStatus::ready;
    return Result<PairValidation>::success(std::move(validation));
}

} // namespace crossvmaf

namespace crossvmaf {
Result<void> validate_frame_descriptor(const FrameDescriptor &descriptor) noexcept {
    // Preserve the SDK's validation order while keeping video metric-free.
    if (descriptor.width == 0U || descriptor.height == 0U) {
        return video::validate_frame_layout(descriptor);
    }
    const auto metric_limit = metrics::cambi::validate_input(descriptor);
    if (!metric_limit) return metric_limit;
    return video::validate_frame_layout(descriptor);
}
Result<void> validate_frame_view(const FrameDescriptor &descriptor,
                                 const FrameView &frame) noexcept {
    const auto validated = validate_frame_descriptor(descriptor);
    if (!validated) return validated;
    return video::validate_plane_views(descriptor, frame);
}
} // namespace crossvmaf
