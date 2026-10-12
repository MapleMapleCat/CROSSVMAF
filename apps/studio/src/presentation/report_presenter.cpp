#include "presentation/report_presenter.hpp"
#include "presentation/chart_data_builder.hpp"
#include "presentation/display_format.hpp"
#include "i18n/translator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <numeric>
#include <string_view>
#include <utility>

namespace crossvmaf::studio {
namespace {

[[nodiscard]] slint::SharedString sharedString(const std::string& value) {
    return slint::SharedString(std::string_view(value));
}

template <typename Item>
[[nodiscard]] std::shared_ptr<slint::VectorModel<Item>> vectorModel(
    std::vector<Item> values) {
    return std::make_shared<slint::VectorModel<Item>>(std::move(values));
}

// Heuristic phenomenon hints: possible phenomena with the analysis rule
// version behind them, never a diagnosis from a single feature threshold.
[[nodiscard]] std::pair<std::string, int> describePhenomenon(
    const FrameResult& frame) {
    if (frame.features.cambi.value() > 1.6) {
        return {i18n::tr("Possible dark banding"), 2};
    }
    if (frame.features.motion3.value() > 5.5) {
        return {i18n::tr("Possible motion-related drop"), 4};
    }
    if (frame.features.adm3.value() < 0.945) {
        return {i18n::tr("Possible detail loss / blur"), 3};
    }
    return {i18n::tr("Brief quality dip"), 2};
}

void applyChartBaseToReport(const VM& view_model, ChartBase base) {
    view_model.set_rep_zones(vectorModel(std::move(base.zones)));
    view_model.set_rep_grid(vectorModel(std::move(base.horizontal_grid)));
    view_model.set_rep_vgrid(vectorModel(std::move(base.vertical_grid)));
    view_model.set_rep_ylabels(vectorModel(std::move(base.y_labels)));
    view_model.set_rep_xlabels(vectorModel(std::move(base.x_labels)));
    view_model.set_rep_guides(vectorModel(std::move(base.guides)));
}

void applyChartBaseToLive(const VM& view_model, ChartBase base) {
    view_model.set_live_zones(vectorModel(std::move(base.zones)));
    view_model.set_live_grid(vectorModel(std::move(base.horizontal_grid)));
    view_model.set_live_vgrid(vectorModel(std::move(base.vertical_grid)));
    view_model.set_live_ylabels(vectorModel(std::move(base.y_labels)));
    view_model.set_live_xlabels(vectorModel(std::move(base.x_labels)));
    view_model.set_live_guides(vectorModel(std::move(base.guides)));
}

void applyChartBaseToComparison(const VM& view_model, ChartBase base) {
    view_model.set_cmp_zones(vectorModel(std::move(base.zones)));
    view_model.set_cmp_grid(vectorModel(std::move(base.horizontal_grid)));
    view_model.set_cmp_vgrid(vectorModel(std::move(base.vertical_grid)));
    view_model.set_cmp_ylabels(vectorModel(std::move(base.y_labels)));
    view_model.set_cmp_xlabels(vectorModel(std::move(base.x_labels)));
    view_model.set_cmp_guides(vectorModel(std::move(base.guides)));
}

[[nodiscard]] std::string segmentTimeLabel(
    const LowScoreSegment& segment) {
    std::string label =
        "#" + formatInteger(static_cast<double>(segment.start)) + " – #" +
        formatInteger(static_cast<double>(segment.end));
    if (segment.start_seconds.has_value() &&
        segment.end_seconds.has_value()) {
        label += " · " + formatMediaTime(*segment.start_seconds) + " – " +
            formatMediaTime(*segment.end_seconds);
    }
    return label;
}

[[nodiscard]] std::string segmentDurationLabel(
    const LowScoreSegment& segment) {
    const std::size_t frame_count = segment.end - segment.start + 1U;
    std::string label = i18n::trn("{0} frame", "{0} frames", frame_count,
        formatInteger(static_cast<double>(frame_count)));
    if (segment.start_seconds.has_value() &&
        segment.end_seconds.has_value() &&
        *segment.end_seconds >= *segment.start_seconds) {
        label += " · " +
            formatDecimal(*segment.end_seconds - *segment.start_seconds, 1) +
            " s";
    }
    return label;
}

[[nodiscard]] std::string bitRateLabel(
    const std::optional<std::uint64_t>& bit_rate, BitRateSource source) {
    if (!bit_rate.has_value()) return i18n::tr("Unknown");
    std::string label = formatBitRate(*bit_rate);
    if (source != BitRateSource::video_stream) label += "*";
    return label;
}

// Differences that round to zero read as 0.0, never as -0.0 or +0.0.
[[nodiscard]] std::string signedDecimal(double value) {
    if (std::abs(value) < 0.05) return formatDecimal(0.0, 1);
    return (value > 0.0 ? "+" : "") + formatDecimal(value, 1);
}

[[nodiscard]] std::string frameLabel(
    const EvaluationReport& report, std::size_t frame_index) {
    std::string label = i18n::tr("Frame #{}",
        formatInteger(static_cast<double>(frame_index)));
    if (frame_index < report.frames.size()) {
        if (const auto media_time =
                frame_media_time_seconds(report.frames[frame_index]);
            media_time.has_value()) {
            label += " · " + formatMediaTime(*media_time);
        }
    }
    return label;
}

// CmpTrack.state of a video that is not compared (yet).
[[nodiscard]] std::int32_t pendingTrackState(const ComparisonSession::PendingTrack& track) {
    switch (track.state) {
    case TaskState::queued: return 1;
    case TaskState::opening:
    case TaskState::running:
    case TaskState::finishing:
    case TaskState::cancelling: return 2;
    case TaskState::failed: return 3;
    case TaskState::cancelled: return 4;
    case TaskState::completed: return 5;
    }
    return 1;
}

[[nodiscard]] std::string pendingTrackStatus(const ComparisonSession::PendingTrack& track) {
    switch (track.state) {
    case TaskState::queued: return i18n::tr("Waiting to be evaluated");
    case TaskState::opening: return i18n::tr("Opening the videos…");
    case TaskState::running:
    case TaskState::finishing:
        return track.progress.has_value()
            ? i18n::tr("Evaluating · {}", formatDecimal(*track.progress * 100.0, 0) + "%")
            : i18n::tr("Evaluating…");
    case TaskState::cancelling: return i18n::tr("Stopping…");
    case TaskState::failed:
        return track.error.has_value() ? describe_task_error(*track.error)
                                       : i18n::tr("Evaluation failed");
    case TaskState::cancelled: return i18n::tr("Stopped before it finished");
    case TaskState::completed:
        return track.issue.has_value()
            ? i18n::tr("Not compared: {}", track.issue->message)
            : std::string{};
    }
    return {};
}

[[nodiscard]] std::string videoSpecLabel(const EvaluationReport& report) {
    std::string label =
        std::to_string(report.descriptor.width) + " × " +
        std::to_string(report.descriptor.height) + " · " +
        pixelFormatName(report.descriptor.pixel_format) + " · " +
        std::to_string(report.descriptor.bit_depth) + " bit";
    if (report.media.reference.frame_rate.average.valid()) {
        label += " · " +
            formatDecimal(report.media.reference.frame_rate.average.value(), 3) +
            " fps";
    }
    return label + " · " + i18n::trn(
        "{0} frame", "{0} frames", report.frames.size(),
        formatInteger(static_cast<double>(report.frames.size())));
}

[[nodiscard]] std::string decodeBackendLabel(StreamDecodeBackend backend) {
    switch (backend) {
    case StreamDecodeBackend::software: return i18n::tr("Software decoding");
    case StreamDecodeBackend::vulkan_video: return "Vulkan Video";
    case StreamDecodeBackend::qsv: return "Intel QSV";
    case StreamDecodeBackend::nvdec: return "NVIDIA NVDEC";
    case StreamDecodeBackend::vaapi: return "VA-API";
    }
    return i18n::tr("Unknown");
}

}  // namespace

ReportPresenter::ReportPresenter() = default;

void ReportPresenter::clearReport(const VM& view_model) {
    report_.reset();
    analysis_ = {};
    report_y_minimum_ = 50.0;
    view_model.set_report_available(false);
    view_model.set_report_description(
        sharedString(i18n::tr("Results appear here after an evaluation completes")));
    view_model.set_rep_score(sharedString("—"));
    view_model.set_rep_grade(sharedString(i18n::tr("No report")));
    std::vector<QuantItem> quantities(3U);
    for (QuantItem& quantity : quantities) {
        quantity.value = sharedString("—");
        quantity.label = sharedString(i18n::tr("No data"));
        quantity.sub = sharedString("");
        quantity.value_kind = -1;
        quantity.locate = -1;
    }
    view_model.set_rep_quants(vectorModel(std::move(quantities)));
    ChartGeometry geometry;
    applyChartBaseToReport(view_model, buildChartBase(geometry));
    view_model.set_rep_path(sharedString(""));
    view_model.set_report_meta_line(sharedString(""));
    view_model.set_rep_markers(vectorModel(std::vector<MarkerDot>{}));
    view_model.set_rep_strip(vectorModel(std::vector<HeatCell>{}));
    view_model.set_rep_segs(vectorModel(std::vector<SegRow>{}));
    view_model.set_hist_bars(vectorModel(std::vector<HistoBar>{}));
    view_model.set_hist_labels(vectorModel(std::vector<AxisLabel>{}));
    view_model.set_hist_legend(vectorModel(std::vector<LegendDot>{}));
    view_model.set_hist_summary(sharedString(""));
}

void ReportPresenter::setReport(std::shared_ptr<const EvaluationReport> report) {
    report_ = std::move(report);
    if (!report_ || report_->frames.empty()) {
        analysis_ = {};
        report_y_minimum_ = 50.0;
        return;
    }
    analysis_ = analyze_report(*report_);
    report_y_minimum_ = std::clamp(
        std::floor(analysis_.statistics.minimum / 5.0) * 5.0 - 5.0,
        0.0,
        95.0);
}

void ReportPresenter::applyReport(const VM& view_model) const {
    if (!report_ || report_->frames.empty()) return;

    const std::vector<double> scores = report_scores(*report_);
    const SeriesStatistics& statistics = analysis_.statistics;

    ChartGeometry geometry;
    geometry.y_minimum = static_cast<float>(report_y_minimum_);
    geometry.frame_count = scores.size();
    applyChartBaseToReport(view_model, buildChartBase(geometry));
    view_model.set_rep_path(sharedString(seriesPath(scores, geometry, 1100U)));

    std::vector<MarkerDot> markers;
    markers.reserve(analysis_.top_segments.size());
    for (const LowScoreSegment& segment : analysis_.top_segments) {
        markers.push_back({
            static_cast<float>(geometry.x(segment.minimum_index)),
            static_cast<float>(geometry.y(scores[segment.minimum_index]))});
    }
    view_model.set_rep_markers(vectorModel(std::move(markers)));
    HeatStripBundle heat_strip = buildHeatStrip(scores, scores.size());
    view_model.set_rep_strip(vectorModel(std::move(heat_strip.cells)));
    view_model.set_rep_strip_frame_w(heat_strip.frame_width);

    const int mean_kind = qualityBandKind(statistics.mean);
    view_model.set_rep_score(sharedString(formatDecimal(statistics.mean, 1)));
    view_model.set_rep_score_kind(mean_kind);
    view_model.set_rep_grade(sharedString(qualityBandName(mean_kind)));
    view_model.set_rep_grade_kind(mean_kind);

    std::vector<QuantItem> quantities(3U);
    quantities[0].value = sharedString(formatDecimal(statistics.minimum, 1));
    quantities[0].label = sharedString(i18n::tr("Lowest frame · #{}",
        formatInteger(static_cast<double>(statistics.minimum_index))));
    quantities[0].value_kind = qualityBandKind(statistics.minimum);
    if (const auto minimum_time =
            frame_media_time_seconds(report_->frames[statistics.minimum_index]);
        minimum_time.has_value()) {
        quantities[0].sub = sharedString(formatMediaTime(*minimum_time) + " · ");
    }
    quantities[0].locate = static_cast<std::int32_t>(statistics.minimum_index);
    quantities[1].value = sharedString(formatDecimal(statistics.percentile_5, 1));
    quantities[1].label = sharedString(i18n::tr("95% of frames ≥ this"));
    quantities[1].sub = sharedString(i18n::tr("5th percentile"));
    quantities[1].value_kind = -1;
    quantities[1].locate = -1;
    const std::size_t below_threshold = static_cast<std::size_t>(std::count_if(
        scores.begin(), scores.end(), [](double score) {
            return score < AnalysisPolicy{}.low_score_threshold;
        }));
    quantities[2].value = sharedString(formatDecimal(
        static_cast<double>(below_threshold) /
            static_cast<double>(scores.size()) * 100.0,
        0) + "%");
    quantities[2].label = sharedString(i18n::tr("Frames below 90"));
    quantities[2].sub = sharedString(i18n::trn("{0} / {1} frame",
        "{0} / {1} frames", scores.size(),
        formatInteger(static_cast<double>(below_threshold)),
        formatInteger(static_cast<double>(scores.size()))));
    quantities[2].value_kind = below_threshold == 0U ? 0 : 2;
    quantities[2].locate = -1;
    view_model.set_rep_quants(vectorModel(std::move(quantities)));

    std::vector<SegRow> segment_rows;
    segment_rows.reserve(analysis_.top_segments.size());
    for (const LowScoreSegment& segment : analysis_.top_segments) {
        SegRow row;
        row.time = sharedString(segmentTimeLabel(segment));
        row.duration = sharedString(segmentDurationLabel(segment));
        row.score = sharedString(formatDecimal(scores[segment.minimum_index], 1));
        row.score_kind = qualityBandKind(scores[segment.minimum_index]);
        const auto [phenomenon, phenomenon_kind] =
            describePhenomenon(report_->frames[segment.minimum_index]);
        row.phenomenon = sharedString(phenomenon);
        row.phen_kind = phenomenon_kind;
        row.frame = static_cast<std::int32_t>(segment.minimum_index);
        segment_rows.push_back(row);
    }
    view_model.set_rep_segs(vectorModel(std::move(segment_rows)));

    HistogramBundle histogram = buildHistogram(scores, statistics.minimum);
    view_model.set_hist_bars(vectorModel(std::move(histogram.bars)));
    view_model.set_hist_labels(vectorModel(std::move(histogram.labels)));
    view_model.set_hist_legend(vectorModel(std::move(histogram.legend)));
    view_model.set_hist_summary(sharedString(i18n::tr("P5 {0} · P50 {1} · mean {2}",
        formatDecimal(statistics.percentile_5, 1),
        formatDecimal(statistics.median, 1),
        formatDecimal(statistics.mean, 1))));

    const std::string frame_count = i18n::trn(
        "{0} frame", "{0} frames", report_->frames.size(),
        formatInteger(static_cast<double>(report_->frames.size())));
    view_model.set_report_available(true);
    view_model.set_report_description(sharedString(i18n::tr(
        "{0} · {1} · took {2} s", pairDisplayName(report_->request),
        formatTimestamp(report_->completed_at),
        formatDecimal(report_->end_to_end_milliseconds / 1000.0, 1))));

    std::string duration_label = frame_count;
    if (report_->media.reference.duration_seconds.has_value()) {
        duration_label += " · " +
            formatMediaTime(*report_->media.reference.duration_seconds);
    }
    view_model.set_report_duration_label(sharedString(duration_label));

    std::string elapsed_label = i18n::tr(
        "{0} s ({1} fps throughput)",
        formatDecimal(report_->end_to_end_milliseconds / 1000.0, 1),
        formatDecimal(report_->frames_per_second, 1));
    if (report_->media.reference.frame_rate.average.valid() &&
        report_->frames_per_second > 0.0) {
        elapsed_label = i18n::tr(
            "{0} · playback speed ×{1}", elapsed_label,
            formatDecimal(
                report_->frames_per_second /
                    report_->media.reference.frame_rate.average.value(),
                2));
    }
    const auto reference_backend = report_->stream_statistics.reference.decode_backend;
    const auto distorted_backend = report_->stream_statistics.distorted.decode_backend;
    const std::string decoder_label = reference_backend == distorted_backend
        ? decodeBackendLabel(reference_backend)
        : i18n::tr("Decoder mismatch in report");
    // One line under the hero carries what the old metadata card showed.
    std::string meta_line = videoSpecLabel(*report_) + " | " + decoder_label;
    if (!report_->device.name.empty()) {
        meta_line += " · " + report_->device.name;
    }
    meta_line += " | " + std::string(scene_display_name(report_->request.scene)) +
        " · " + std::string(to_string(report_->model));
    meta_line += " | " + elapsed_label;
    view_model.set_report_meta_line(sharedString(meta_line));
}

void ReportPresenter::applyLive(
    const VM& view_model,
    std::span<const double> scores,
    const ProgressSnapshot& progress) const {
    if (scores.empty()) {
        ChartGeometry geometry;
        geometry.height = 220.0F;
        applyChartBaseToLive(view_model, buildChartBase(geometry));
        view_model.set_live_path(sharedString(""));
        view_model.set_live_dot_visible(false);
        view_model.set_live_cur(sharedString("—"));
        view_model.set_live_mean(sharedString("—"));
        view_model.set_live_strip(vectorModel(std::vector<HeatCell>{}));
        view_model.set_live_strip_frac(0.0F);
        view_model.set_live_strip_frame_w(0.0F);
        return;
    }

    const std::size_t total_frame_count = static_cast<std::size_t>(std::max(
        static_cast<std::uint64_t>(scores.size()),
        progress.frame_count_hint.value_or(progress.decoded_frame_count)));
    const auto minimum_iterator = std::min_element(
        scores.begin(), scores.end());
    const double y_minimum = std::clamp(
        std::floor(*minimum_iterator / 5.0) * 5.0 - 5.0,
        0.0,
        95.0);
    ChartGeometry geometry;
    geometry.height = 220.0F;
    geometry.y_minimum = static_cast<float>(y_minimum);
    geometry.frame_count = std::max<std::size_t>(2U, total_frame_count);
    applyChartBaseToLive(view_model, buildChartBase(geometry));
    view_model.set_live_path(sharedString(
        seriesPath(scores, geometry, 900U)));
    view_model.set_live_dot_visible(true);
    view_model.set_live_dot_x(static_cast<float>(
        geometry.x(scores.size() - 1U)));
    view_model.set_live_dot_y(static_cast<float>(
        geometry.y(scores.back())));
    HeatStripBundle strip = buildHeatStrip(scores, total_frame_count);
    view_model.set_live_strip(vectorModel(std::move(strip.cells)));
    view_model.set_live_strip_frac(strip.progress_fraction);
    view_model.set_live_strip_frame_w(strip.frame_width);

    const double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    view_model.set_live_cur(sharedString(formatDecimal(scores.back(), 1)));
    view_model.set_live_mean(sharedString(formatDecimal(
        sum / static_cast<double>(scores.size()), 1)));
}

void ReportPresenter::applyHover(
    const VM& view_model,
    double pointer_x,
    double pointer_y,
    double chart_width,
    double chart_height) const {
    static_cast<void>(pointer_y);
    static_cast<void>(chart_height);
    if (!report_ || report_->frames.empty() || chart_width <= 0.0) return;

    ChartGeometry geometry;
    geometry.y_minimum = static_cast<float>(report_y_minimum_);
    geometry.frame_count = report_->frames.size();
    const double viewbox_x = pointer_x / chart_width * geometry.width;
    const double frame_fraction = (viewbox_x - geometry.left) / geometry.plotWidth();
    const std::size_t frame_index = static_cast<std::size_t>(std::clamp(
        std::floor(frame_fraction * static_cast<double>(geometry.frame_count - 1U) + 0.5),
        0.0,
        static_cast<double>(geometry.frame_count - 1U)));
    const FrameResult& frame = report_->frames[frame_index];
    const double value = frame.vmaf;
    const int kind = qualityBandKind(value);
    view_model.set_hov_visible(true);
    view_model.set_hov_x_frac(static_cast<float>(
        geometry.x(frame_index) / geometry.width));
    view_model.set_hov_y_frac(static_cast<float>(
        geometry.y(value) / geometry.height));
    view_model.set_hov_title(sharedString(frameLabel(*report_, frame_index)));
    view_model.set_hov_value(sharedString(formatDecimal(value, 1)));
    view_model.set_hov_band(sharedString(qualityBandName(kind)));
    view_model.set_hov_kind(kind);
}

void ReportPresenter::setComparison(
    const ComparisonSession& session, ComparisonResult comparison) {
    comparison_entries_ = session.selection().entries();
    comparison_pending_ = session.pending();
    comparison_reference_ = session.reference_path();
    comparison_ = std::move(comparison);
    comparison_scores_.clear();
    comparison_statistics_.clear();
    std::set<std::string> hidden;
    for (const ComparisonSelection::Entry& entry : comparison_entries_) {
        comparison_scores_.push_back(report_scores(*entry.report));
        comparison_statistics_.push_back(
            calculate_statistics(comparison_scores_.back()));
        if (hidden_comparison_ids_.contains(entry.report->report_id)) {
            hidden.insert(entry.report->report_id);
        }
    }
    hidden_comparison_ids_ = std::move(hidden);
}

void ReportPresenter::setComparisonPending(
    std::vector<ComparisonSession::PendingTrack> pending) {
    comparison_pending_ = std::move(pending);
}

void ReportPresenter::toggleComparisonSeries(std::size_t track) {
    if (track >= comparison_entries_.size()) return;
    const std::string& report_id = comparison_entries_[track].report->report_id;
    if (hidden_comparison_ids_.erase(report_id) == 0U) {
        hidden_comparison_ids_.insert(report_id);
    }
}

bool ReportPresenter::comparisonTrackVisible(std::size_t track) const {
    return track < comparison_entries_.size() &&
        !hidden_comparison_ids_.contains(
            comparison_entries_[track].report->report_id);
}

std::size_t ReportPresenter::comparisonBaselineTrack() const {
    for (std::size_t track = 0U; track < comparison_entries_.size(); ++track) {
        if (comparison_entries_[track].report->report_id ==
            comparison_.baseline_report_id) {
            return track;
        }
    }
    return 0U;
}

std::vector<std::size_t> ReportPresenter::comparisonDisplayOrder() const {
    std::vector<std::size_t> order;
    for (const ComparisonRow& row : comparison_.rows) {
        for (std::size_t track = 0U; track < comparison_entries_.size(); ++track) {
            if (comparison_entries_[track].report->report_id == row.report_id) {
                order.push_back(track);
            }
        }
    }
    if (order.size() != comparison_entries_.size()) {
        order.resize(comparison_entries_.size());
        std::iota(order.begin(), order.end(), std::size_t{0U});
    }
    return order;
}

ChartGeometry ReportPresenter::comparisonGeometry() const {
    double y_minimum = 95.0;
    for (std::size_t track = 0U; track < comparison_statistics_.size(); ++track) {
        if (comparisonTrackVisible(track)) {
            y_minimum = std::min(y_minimum, comparison_statistics_[track].minimum);
        }
    }
    ChartGeometry geometry;
    geometry.y_minimum = static_cast<float>(std::clamp(
        std::floor(y_minimum / 5.0) * 5.0 - 5.0, 0.0, 95.0));
    geometry.frame_count =
        comparison_scores_.empty() ? 2U : comparison_scores_.front().size();
    return geometry;
}

void ReportPresenter::applyComparison(const VM& view_model) const {
    const bool comparable = comparison_.validation.comparable;
    view_model.set_compare_available(comparable);
    view_model.set_cmp_hov_visible(false);

    const EvaluationReport* anchor = comparison_entries_.empty()
        ? nullptr
        : comparison_entries_.front().report.get();
    const std::string reference_name = fileDisplayName(
        anchor ? anchor->request.reference_path : comparison_reference_);
    view_model.set_cmp_reference(sharedString(reference_name));
    view_model.set_compare_description(sharedString(
        anchor ? i18n::tr("Reference {0} · {1}", reference_name, videoSpecLabel(*anchor))
        : !reference_name.empty() ? i18n::tr("Reference {}", reference_name)
        : i18n::tr("Compare several encodes of the same reference video side by side")));
    applyComparisonTracks(view_model);

    std::vector<slint::SharedString> notes;
    std::vector<slint::SharedString> insights;
    std::vector<SeriesPath> series;
    std::vector<RdSample> rate_samples;
    ChartBase chart_base;
    std::optional<std::size_t> visually_lossless_row;
    if (comparable) {
        for (const std::string& warning : comparison_.warnings) {
            notes.push_back(sharedString(warning));
        }

        const ChartGeometry geometry = comparisonGeometry();
        chart_base = buildChartBase(geometry);
        for (std::size_t track = 0U; track < comparison_entries_.size(); ++track) {
            SeriesPath path;
            path.name = sharedString(fileDisplayName(
                comparison_entries_[track].report->request.distorted_path));
            path.path = sharedString(
                seriesPath(comparison_scores_[track], geometry, 700U));
            path.slot = static_cast<std::int32_t>(
                comparison_entries_[track].slot + 1U);
            path.width = 1.6F;
            path.opacity = 0.95F;
            path.visible = comparisonTrackVisible(track);
            series.push_back(path);
        }

        std::vector<std::shared_ptr<const EvaluationReport>> reports;
        for (const ComparisonSelection::Entry& entry : comparison_entries_) {
            reports.push_back(entry.report);
        }
        const ComparisonHighlights highlights =
            summarize_comparison(comparison_, reports);
        const std::vector<ComparisonRow>& rows = comparison_.rows;
        if (highlights.highest_mean) {
            const ComparisonRow& row = rows[*highlights.highest_mean];
            insights.push_back(sharedString(i18n::tr(
                "{0} has the highest mean VMAF: {1}.",
                row.display_name, formatDecimal(row.mean, 1))));
        }
        if (highlights.all_rates_known) {
            if (highlights.lowest_visually_lossless_rate) {
                visually_lossless_row = highlights.lowest_visually_lossless_rate;
                const ComparisonRow& row = rows[*visually_lossless_row];
                insights.push_back(sharedString(i18n::tr(
                    "{0} is the lowest bitrate with a mean of at least 93: {1} at {2}.",
                    row.display_name, formatDecimal(row.mean, 1),
                    bitRateLabel(row.average_bit_rate, row.bit_rate_source))));
            } else {
                insights.push_back(sharedString(i18n::tr(
                    "No evaluation reaches a mean of 93, the visually lossless level.")));
            }
        }
        if (highlights.largest_drop) {
            const ComparisonRow& row = rows[*highlights.largest_drop];
            insights.push_back(sharedString(i18n::tr(
                "{0} averages {1} below the baseline; its worst frame scores {2}.",
                row.display_name, formatDecimal(-row.delta_vs_baseline, 1),
                formatDecimal(row.minimum, 1))));
        }
        if (highlights.widest_gap_frame && highlights.widest_gap >= 0.05) {
            insights.push_back(sharedString(i18n::tr(
                "Largest gap between the curves: {0} points ({1}).",
                formatDecimal(highlights.widest_gap, 1),
                frameLabel(*anchor, *highlights.widest_gap_frame))));
        }

        for (std::size_t index = 0U; index < rows.size(); ++index) {
            if (!rows[index].average_bit_rate.has_value()) continue;
            for (const ComparisonSelection::Entry& entry : comparison_entries_) {
                if (entry.report->report_id != rows[index].report_id) continue;
                rate_samples.push_back({
                    static_cast<double>(*rows[index].average_bit_rate),
                    rows[index].mean,
                    static_cast<std::int32_t>(entry.slot + 1U),
                    visually_lossless_row == index});
            }
        }
    }
    applyChartBaseToComparison(view_model, std::move(chart_base));
    view_model.set_cmp_series(vectorModel(std::move(series)));
    view_model.set_cmp_notes(vectorModel(std::move(notes)));
    view_model.set_cmp_insights(vectorModel(std::move(insights)));

    const bool rate_chart = rate_samples.size() >= 2U;
    RdChartBundle rate_bundle =
        rate_chart ? buildRdChart(rate_samples) : RdChartBundle{};
    view_model.set_rd_available(rate_chart);
    view_model.set_rd_grid(vectorModel(std::move(rate_bundle.grid)));
    view_model.set_rd_ylabels(vectorModel(std::move(rate_bundle.y_labels)));
    view_model.set_rd_xlabels(vectorModel(std::move(rate_bundle.x_labels)));
    view_model.set_rd_guide_dash(sharedString(rate_bundle.guide_path));
    view_model.set_rd_guide_label(sharedString(rate_bundle.guide_label));
    view_model.set_rd_guide_label_x(rate_bundle.guide_label_x);
    view_model.set_rd_guide_label_y(rate_bundle.guide_label_y);
    view_model.set_rd_line_dash(sharedString(rate_bundle.line_path));
    view_model.set_rd_points(vectorModel(std::move(rate_bundle.points)));
}

void ReportPresenter::applyComparisonTracks(const VM& view_model) const {
    const bool comparable = comparison_.validation.comparable;
    const std::vector<std::size_t> order = comparisonDisplayOrder();
    const std::size_t baseline = comparisonBaselineTrack();
    std::vector<CmpTrack> tracks;
    tracks.reserve(order.size() + comparison_pending_.size());
    for (const std::size_t track : order) {
        const ComparisonSelection::Entry& entry = comparison_entries_[track];
        const EvaluationReport& report = *entry.report;
        const SeriesStatistics& statistics = comparison_statistics_[track];
        const bool is_baseline = comparable && track == baseline;
        const double delta =
            statistics.mean - comparison_statistics_[baseline].mean;
        const int mean_kind = qualityBandKind(statistics.mean);
        CmpTrack row{};
        row.track = static_cast<std::int32_t>(track);
        row.slot = static_cast<std::int32_t>(entry.slot + 1U);
        row.name = sharedString(fileDisplayName(report.request.distorted_path));
        row.state = 0;
        row.visible = comparisonTrackVisible(track);
        row.baseline = is_baseline;
        row.rate = sharedString(bitRateLabel(
            report.media.distorted.average_bit_rate,
            report.media.distorted.bit_rate_source));
        row.mean = sharedString(formatDecimal(statistics.mean, 1));
        row.delta = sharedString(
            comparable && !is_baseline ? signedDecimal(delta) : "—");
        row.delta_kind = !comparable || is_baseline || std::abs(delta) < 0.05
            ? 0 : delta > 0.0 ? 1 : 2;
        row.low5 = sharedString(formatDecimal(statistics.percentile_5, 1));
        row.worst = sharedString(formatDecimal(statistics.minimum, 1));
        row.lossless = sharedString(formatDecimal(
            static_cast<double>(count_visually_lossless(comparison_scores_[track])) /
                static_cast<double>(comparison_scores_[track].size()) * 100.0,
            0) + "%");
        row.grade = sharedString(qualityBandName(mean_kind));
        row.grade_kind = mean_kind;
        tracks.push_back(row);
    }
    for (std::size_t index = 0U; index < comparison_pending_.size(); ++index) {
        const ComparisonSession::PendingTrack& pending = comparison_pending_[index];
        CmpTrack row{};
        row.track = static_cast<std::int32_t>(comparison_entries_.size() + index);
        row.name = sharedString(fileDisplayName(pending.distorted_path));
        row.state = pendingTrackState(pending);
        row.progress = pending.progress.has_value()
            ? static_cast<float>(*pending.progress) : -1.0F;
        row.status = sharedString(pendingTrackStatus(pending));
        row.can_retry = pending.state == TaskState::failed ||
            pending.state == TaskState::cancelled;
        tracks.push_back(row);
    }
    view_model.set_cmp_tracks(vectorModel(std::move(tracks)));

}

void ReportPresenter::applyComparisonHover(
    const VM& view_model, double pointer_x, double chart_width) const {
    if (!comparison_.validation.comparable || comparison_scores_.empty() ||
        comparison_scores_.front().empty() || chart_width <= 0.0) {
        view_model.set_cmp_hov_visible(false);
        return;
    }
    const ChartGeometry geometry = comparisonGeometry();
    const double viewbox_x = pointer_x / chart_width * geometry.width;
    const double frame_fraction =
        (viewbox_x - geometry.left) / geometry.plotWidth();
    const std::size_t last_frame = geometry.frame_count - 1U;
    const std::size_t frame_index = static_cast<std::size_t>(std::clamp(
        std::floor(frame_fraction * static_cast<double>(last_frame) + 0.5),
        0.0,
        static_cast<double>(last_frame)));

    const std::size_t baseline = comparisonBaselineTrack();
    const double baseline_value = comparison_scores_[baseline][frame_index];
    std::vector<CmpHoverRow> rows;
    for (const std::size_t track : comparisonDisplayOrder()) {
        if (!comparisonTrackVisible(track)) continue;
        const double value = comparison_scores_[track][frame_index];
        CmpHoverRow row;
        row.slot = static_cast<std::int32_t>(comparison_entries_[track].slot + 1U);
        row.name = sharedString(fileDisplayName(
            comparison_entries_[track].report->request.distorted_path));
        row.value = sharedString(formatDecimal(value, 1));
        row.delta = sharedString(track == baseline
            ? i18n::tr("baseline")
            : signedDecimal(value - baseline_value));
        row.y_frac = static_cast<float>(geometry.y(value) / geometry.height);
        rows.push_back(row);
    }
    if (rows.empty()) {
        view_model.set_cmp_hov_visible(false);
        return;
    }
    view_model.set_cmp_hov_title(sharedString(
        frameLabel(*comparison_entries_.front().report, frame_index)));
    view_model.set_cmp_hov_rows(vectorModel(std::move(rows)));
    view_model.set_cmp_hov_x_frac(static_cast<float>(
        geometry.x(frame_index) / geometry.width));
    view_model.set_cmp_hov_visible(true);
}

double ReportPresenter::reportX(std::size_t frame_index) const {
    ChartGeometry geometry;
    geometry.frame_count = report_ ? report_->frames.size() : 2U;
    return geometry.x(std::min(
        frame_index,
        geometry.frame_count > 0U ? geometry.frame_count - 1U : 0U));
}

double ReportPresenter::reportY(double value) const {
    ChartGeometry geometry;
    geometry.y_minimum = static_cast<float>(report_y_minimum_);
    return geometry.y(value);
}

}  // namespace crossvmaf::studio
