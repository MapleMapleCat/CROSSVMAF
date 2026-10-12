#include "reports/comparison.hpp"

#include "i18n/translator.hpp"
#include "reports/report_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>

namespace crossvmaf::studio {
namespace {

constexpr double frame_rate_match_tolerance = 1.0e-3;

[[nodiscard]] bool descriptors_match(
    const FrameDescriptor& left,
    const FrameDescriptor& right) noexcept {
    return left.width == right.width && left.height == right.height &&
        left.bit_depth == right.bit_depth &&
        left.pixel_format == right.pixel_format;
}

[[nodiscard]] bool same_reference_identity(
    const EvaluationReport& baseline,
    const EvaluationReport& candidate) {
    const std::filesystem::path& baseline_path =
        baseline.request.reference_path;
    const std::filesystem::path& candidate_path =
        candidate.request.reference_path;
    return !baseline_path.empty() &&
        baseline_path.lexically_normal() == candidate_path.lexically_normal();
}

[[nodiscard]] bool frame_rates_compatible(
    const VideoMetadata& baseline,
    const VideoMetadata& candidate) {
    // Unknown frame rates cannot prove a timeline difference; known rates
    // must match.
    if (!baseline.frame_rate.average.valid() ||
        !candidate.frame_rate.average.valid()) {
        return true;
    }
    const double baseline_value = baseline.frame_rate.average.value();
    const double candidate_value = candidate.frame_rate.average.value();
    return std::fabs(baseline_value - candidate_value) /
            std::max(baseline_value, candidate_value) <=
        frame_rate_match_tolerance;
}

void validate_candidate(
    const EvaluationReport& baseline,
    const EvaluationReport& candidate,
    std::size_t index,
    ComparisonValidation& validation) {
    const auto reject = [&validation, &candidate](
                            std::string reason_code, std::string message) {
        validation.issues.push_back(
            {std::move(reason_code), std::move(message),
             candidate.report_id});
    };
    bool eligible = true;
    if (!same_reference_identity(baseline, candidate)) {
        reject(
            "different_reference",
            i18n::tr("Different reference videos cannot be aligned frame by frame"));
        eligible = false;
    }
    if (!descriptors_match(baseline.descriptor, candidate.descriptor)) {
        reject(
            "different_spec",
            i18n::tr("Picture specs differ (resolution, bit depth or chroma format)"));
        eligible = false;
    }
    if (baseline.frames.size() != candidate.frames.size()) {
        reject(
            "different_frame_count",
            i18n::tr("Frame counts differ, so the curves cannot be aligned"));
        eligible = false;
    }
    if (baseline.model != candidate.model) {
        reject(
            "different_model",
            i18n::tr("Different quality models were used, so the scores are not comparable"));
        eligible = false;
    }
    if (!frame_rates_compatible(baseline.media.reference,
                                candidate.media.reference)) {
        reject(
            "different_timeline",
            i18n::tr("Reference timelines (frame rates) differ, so media times are not comparable"));
        eligible = false;
    }
    if (eligible) validation.eligible_indices.push_back(index);
}

[[nodiscard]] std::string display_name_of(const EvaluationReport& report) {
    const std::filesystem::path filename =
        report.request.distorted_path.filename();
    const std::u8string utf8 =
        (filename.empty() ? report.request.distorted_path : filename)
            .u8string();
    return std::string(
        reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

}  // namespace

ComparisonValidation validate_comparison(
    std::span<const std::shared_ptr<const EvaluationReport>> reports) {
    ComparisonValidation validation;
    std::vector<std::shared_ptr<const EvaluationReport>> valid_reports;
    for (const auto& report : reports) {
        if (report && !report->frames.empty()) valid_reports.push_back(report);
    }
    if (valid_reports.size() < 2U) {
        validation.issues.push_back(
            {"insufficient_reports",
             i18n::tr("Needs at least two comparable evaluations of the same reference video"),
             {}});
        return validation;
    }

    const EvaluationReport& baseline = *valid_reports.front();
    std::set<std::string> seen_ids{baseline.report_id};
    validation.eligible_indices.push_back(0U);
    for (std::size_t index = 1U; index < valid_reports.size(); ++index) {
        const EvaluationReport& candidate = *valid_reports[index];
        if (!seen_ids.insert(candidate.report_id).second) {
            validation.issues.push_back(
                {"duplicate_report",
                 i18n::tr("The same report was added to the comparison twice"),
                 candidate.report_id});
            continue;
        }
        validate_candidate(baseline, candidate, index, validation);
    }
    validation.comparable = validation.eligible_indices.size() >= 2U;
    if (!validation.comparable) {
        validation.issues.push_back(
            {"insufficient_reports",
             i18n::tr("No other evaluation is comparable with the baseline report"),
             {}});
    }
    return validation;
}

ComparisonResult build_comparison(
    std::span<const std::shared_ptr<const EvaluationReport>> reports,
    const ComparisonOptions& options) {
    ComparisonResult result;
    std::vector<std::shared_ptr<const EvaluationReport>> valid_reports;
    for (const auto& report : reports) {
        if (report && !report->frames.empty()) valid_reports.push_back(report);
    }
    result.validation = validate_comparison(reports);
    if (!result.validation.comparable) return result;

    std::size_t kept = 0U;
    std::vector<const EvaluationReport*> eligible;
    eligible.reserve(result.validation.eligible_indices.size());
    for (const std::size_t index : result.validation.eligible_indices) {
        if (kept >= options.maximum_reports) {
            result.warnings.push_back(i18n::trn(
                "More reports are comparable than the limit; only the latest {0} is compared",
                "More reports are comparable than the limit; only the latest {0} are compared",
                options.maximum_reports, std::to_string(options.maximum_reports)));
            break;
        }
        eligible.push_back(valid_reports[index].get());
        ++kept;
    }

    std::size_t baseline_position = 0U;
    if (!options.baseline_report_id.empty()) {
        bool found = false;
        for (std::size_t index = 0U; index < eligible.size(); ++index) {
            if (eligible[index]->report_id == options.baseline_report_id) {
                baseline_position = index;
                found = true;
                break;
            }
        }
        if (!found) {
            result.warnings.push_back(
                i18n::tr("The requested baseline report is not comparable; "
                         "the first comparable report is used instead"));
        }
    }
    const EvaluationReport& baseline = *eligible[baseline_position];
    result.baseline_report_id = baseline.report_id;
    const double baseline_mean =
        calculate_statistics(report_scores(baseline)).mean;

    for (std::size_t index = 0U; index < eligible.size(); ++index) {
        const EvaluationReport& report = *eligible[index];
        const SeriesStatistics statistics =
            calculate_statistics(report_scores(report));
        ComparisonRow row;
        row.report_id = report.report_id;
        row.display_name = display_name_of(report);
        row.mean = statistics.mean;
        row.minimum = statistics.minimum;
        row.percentile_5 = statistics.percentile_5;
        row.delta_vs_baseline = statistics.mean - baseline_mean;
        row.is_baseline = index == baseline_position;
        row.average_bit_rate = report.media.distorted.average_bit_rate;
        row.bit_rate_source = report.media.distorted.bit_rate_source;
        if (report.media.distorted.identity.captured) {
            row.file_size_bytes = report.media.distorted.identity.size_bytes;
        }
        result.rows.push_back(std::move(row));
    }

    // Bit-rate order for the RD story; unknown rates keep input order last.
    std::stable_sort(
        result.rows.begin(), result.rows.end(),
        [](const ComparisonRow& left, const ComparisonRow& right) {
            if (left.average_bit_rate.has_value() !=
                right.average_bit_rate.has_value()) {
                return left.average_bit_rate.has_value();
            }
            if (!left.average_bit_rate.has_value()) return false;
            return *left.average_bit_rate < *right.average_bit_rate;
        });

    bool any_estimated_rate = false;
    bool any_missing_rate = false;
    for (const ComparisonRow& row : result.rows) {
        any_estimated_rate = any_estimated_rate ||
            (row.average_bit_rate.has_value() &&
             row.bit_rate_source != BitRateSource::video_stream);
        any_missing_rate =
            any_missing_rate || !row.average_bit_rate.has_value();
    }
    if (any_estimated_rate) {
        result.warnings.push_back(
            i18n::tr("Some bitrates are estimated from the container or file size, "
                     "not the video stream alone"));
    }
    if (any_missing_rate) {
        result.warnings.push_back(
            i18n::tr("Some reports lack a valid video bitrate, so no encoding tier is recommended"));
    }
    return result;
}

ComparisonHighlights summarize_comparison(
    const ComparisonResult& comparison,
    std::span<const std::shared_ptr<const EvaluationReport>> reports,
    const AnalysisPolicy& policy) {
    ComparisonHighlights highlights;
    const std::vector<ComparisonRow>& rows = comparison.rows;
    if (!comparison.validation.comparable || rows.empty()) return highlights;

    highlights.all_rates_known = true;
    double largest_drop = 0.0;
    for (std::size_t index = 0U; index < rows.size(); ++index) {
        const ComparisonRow& row = rows[index];
        if (!highlights.highest_mean ||
            row.mean > rows[*highlights.highest_mean].mean) {
            highlights.highest_mean = index;
        }
        if (row.delta_vs_baseline < largest_drop) {
            largest_drop = row.delta_vs_baseline;
            highlights.largest_drop = index;
        }
        highlights.all_rates_known =
            highlights.all_rates_known && row.average_bit_rate.has_value();
    }
    if (highlights.all_rates_known) {
        for (std::size_t index = 0U; index < rows.size(); ++index) {
            const ComparisonRow& row = rows[index];
            if (row.mean < policy.visually_lossless_threshold) continue;
            if (!highlights.lowest_visually_lossless_rate ||
                *row.average_bit_rate <
                    *rows[*highlights.lowest_visually_lossless_rate]
                         .average_bit_rate) {
                highlights.lowest_visually_lossless_rate = index;
            }
        }
    }

    std::vector<std::vector<double>> scores;
    for (const auto& report : reports) {
        if (report && !report->frames.empty()) {
            scores.push_back(report_scores(*report));
        }
    }
    if (scores.size() < 2U) return highlights;
    std::size_t frame_count = scores.front().size();
    for (const std::vector<double>& series : scores) {
        frame_count = std::min(frame_count, series.size());
    }
    for (std::size_t frame = 0U; frame < frame_count; ++frame) {
        double lowest = scores.front()[frame];
        double highest = lowest;
        for (const std::vector<double>& series : scores) {
            lowest = std::min(lowest, series[frame]);
            highest = std::max(highest, series[frame]);
        }
        if (!highlights.widest_gap_frame ||
            highest - lowest > highlights.widest_gap) {
            highlights.widest_gap = highest - lowest;
            highlights.widest_gap_frame = frame;
        }
    }
    return highlights;
}

}  // namespace crossvmaf::studio
