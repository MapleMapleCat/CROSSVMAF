#include "reports/report_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace crossvmaf::studio {

std::vector<double> report_scores(const EvaluationReport& report) {
    std::vector<double> scores;
    scores.reserve(report.frames.size());
    for (const FrameResult& frame : report.frames) {
        scores.push_back(frame.vmaf);
    }
    return scores;
}

SeriesStatistics calculate_statistics(std::span<const double> values) {
    SeriesStatistics statistics;
    if (values.empty()) return statistics;

    statistics.minimum = std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double reciprocal_sum = 0.0;
    for (std::size_t index = 0U; index < values.size(); ++index) {
        const double value = values[index];
        sum += value;
        reciprocal_sum += 1.0 / std::max(value, 1.0e-6);
        if (value < statistics.minimum) {
            statistics.minimum = value;
            statistics.minimum_index = index;
        }
    }
    statistics.mean = sum / static_cast<double>(values.size());
    statistics.harmonic_mean =
        static_cast<double>(values.size()) / reciprocal_sum;

    std::vector<double> sorted_values(values.begin(), values.end());
    std::sort(sorted_values.begin(), sorted_values.end());
    // Nearest-rank-on-midpoint quantile rule, versioned with AnalysisPolicy.
    const auto percentile = [&sorted_values](double fraction) {
        const double scaled_index = fraction *
            static_cast<double>(sorted_values.size() - 1U);
        const std::size_t index = static_cast<std::size_t>(
            std::floor(scaled_index + 0.5));
        return sorted_values[index];
    };
    statistics.percentile_5 = percentile(0.05);
    statistics.median = percentile(0.5);

    double squared_difference_sum = 0.0;
    for (const double value : values) {
        const double difference = value - statistics.mean;
        squared_difference_sum += difference * difference;
    }
    statistics.standard_deviation = std::sqrt(
        squared_difference_sum / static_cast<double>(values.size()));
    return statistics;
}

namespace {

[[nodiscard]] std::vector<LowScoreSegment> find_low_score_segments(
    std::span<const double> scores,
    const EvaluationReport& report,
    const AnalysisPolicy& policy) {
    std::vector<LowScoreSegment> segments;

    std::size_t index = 0U;
    while (index < scores.size()) {
        if (scores[index] >= policy.low_score_threshold) {
            ++index;
            continue;
        }

        LowScoreSegment segment{index, index, index, {}, {}, {}};
        while (segment.end + 1U < scores.size()) {
            const bool next_is_low =
                scores[segment.end + 1U] < policy.low_score_threshold;
            const bool following_is_low =
                segment.end + 2U < scores.size() &&
                scores[segment.end + 2U] < policy.low_score_threshold;
            if (!next_is_low && !following_is_low) break;
            ++segment.end;
            if (scores[segment.end] < scores[segment.minimum_index]) {
                segment.minimum_index = segment.end;
            }
        }

        if (!segments.empty() &&
            segment.start - segments.back().end <
                policy.merge_distance_frames) {
            LowScoreSegment& previous = segments.back();
            previous.end = segment.end;
            if (scores[segment.minimum_index] <
                scores[previous.minimum_index]) {
                previous.minimum_index = segment.minimum_index;
            }
        } else {
            segments.push_back(segment);
        }
        index = segment.end + 1U;
    }

    for (LowScoreSegment& segment : segments) {
        segment.start_seconds =
            frame_media_time_seconds(report.frames[segment.start]);
        segment.end_seconds =
            frame_media_time_seconds(report.frames[segment.end]);
        segment.minimum_seconds =
            frame_media_time_seconds(report.frames[segment.minimum_index]);
    }
    return segments;
}

}  // namespace

std::size_t count_visually_lossless(
    std::span<const double> values,
    const AnalysisPolicy& policy) {
    return static_cast<std::size_t>(
        std::count_if(values.begin(), values.end(), [&policy](double value) {
            return value >= policy.visually_lossless_threshold;
        }));
}

ReportAnalysis analyze_report(
    const EvaluationReport& report,
    const AnalysisPolicy& policy) {
    ReportAnalysis analysis;
    analysis.policy_version = AnalysisPolicy::version;
    if (report.frames.empty()) return analysis;

    const std::vector<double> scores = report_scores(report);
    analysis.statistics = calculate_statistics(scores);
    analysis.segments = find_low_score_segments(scores, report, policy);
    analysis.top_segments = analysis.segments;
    std::stable_sort(
        analysis.top_segments.begin(),
        analysis.top_segments.end(),
        [&scores](const LowScoreSegment& left, const LowScoreSegment& right) {
            return scores[left.minimum_index] < scores[right.minimum_index];
        });
    if (analysis.top_segments.size() > policy.maximum_listed_segments) {
        analysis.top_segments.resize(policy.maximum_listed_segments);
    }
    analysis.visually_lossless_count = count_visually_lossless(scores, policy);
    analysis.visually_lossless_fraction =
        static_cast<double>(analysis.visually_lossless_count) /
        static_cast<double>(scores.size());
    return analysis;
}

std::vector<std::pair<std::size_t, double>> decimate_series(
    std::span<const double> values,
    std::size_t maximum_points) {
    std::vector<std::pair<std::size_t, double>> output;
    if (values.size() <= maximum_points) {
        output.reserve(values.size());
        for (std::size_t index = 0U; index < values.size(); ++index) {
            output.emplace_back(index, values[index]);
        }
        return output;
    }

    const std::size_t bucket_count =
        std::max<std::size_t>(1U, maximum_points / 2U);
    const double bucket_width =
        static_cast<double>(values.size()) / static_cast<double>(bucket_count);
    output.reserve(bucket_count * 2U);
    for (std::size_t bucket = 0U; bucket < bucket_count; ++bucket) {
        const std::size_t start = static_cast<std::size_t>(
            std::floor(static_cast<double>(bucket) * bucket_width));
        const std::size_t end = std::max(
            start + 1U,
            std::min(
                values.size(),
                static_cast<std::size_t>(std::floor(
                    static_cast<double>(bucket + 1U) * bucket_width))));
        std::size_t minimum_index = start;
        std::size_t maximum_index = start;
        for (std::size_t index = start; index < end; ++index) {
            if (values[index] < values[minimum_index]) minimum_index = index;
            if (values[index] > values[maximum_index]) maximum_index = index;
        }
        const std::size_t first_index = std::min(minimum_index, maximum_index);
        const std::size_t second_index = std::max(minimum_index, maximum_index);
        output.emplace_back(first_index, values[first_index]);
        if (second_index != first_index) {
            output.emplace_back(second_index, values[second_index]);
        }
    }
    return output;
}

const FrameResult* get_frame(
    const EvaluationReport& report,
    std::uint64_t frame_index) noexcept {
    if (frame_index >= report.frames.size()) return nullptr;
    return &report.frames[frame_index];
}

std::optional<double> frame_media_time_seconds(
    const FrameResult& frame) noexcept {
    if (frame.timing.status == FrameTimingStatus::missing) {
        return std::nullopt;
    }
    return frame.timing.media_time_seconds;
}

int qualityBandKind(double value) noexcept {
    if (value >= 93.0) return 0;
    if (value >= 80.0) return 1;
    if (value >= 60.0) return 2;
    return 3;
}

} // namespace crossvmaf::studio
