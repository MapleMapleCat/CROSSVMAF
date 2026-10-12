#include "reports/report_pdf.hpp"

#include "i18n/translator.hpp"
#include "reports/pdf_document.hpp"
#include "reports/report_analysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <locale>
#include <limits>
#include <sstream>

namespace crossvmaf::studio {
namespace {

using pdf::Document;
constexpr double left = 42;
constexpr double right = Document::width - left;
constexpr double content_width = right - left;
constexpr std::array<pdf::Color, 4> bands{{pdf::accent, {0.30, 0.48, 0.64}, {0.75, 0.49, 0.18}, {0.71, 0.29, 0.27}}};

std::string decimal(double value, int precision = 2) {
    if (!std::isfinite(value)) return i18n::tr("Unknown");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}
std::string utf8(const std::filesystem::path& path) {
    const auto value = path.u8string();
    return {reinterpret_cast<const char*>(value.data()), value.size()};
}
std::string known(std::string value) { return value.empty() ? i18n::tr("Unknown") : value; }
std::string timestamp(std::chrono::system_clock::time_point time) {
    if (time == std::chrono::system_clock::time_point{}) return i18n::tr("Unknown");
    const auto seconds = std::chrono::system_clock::to_time_t(time);
    std::tm date{};
#ifdef _WIN32
    if (gmtime_s(&date, &seconds) != 0) return i18n::tr("Unknown");
#else
    if (!gmtime_r(&seconds, &date)) return i18n::tr("Unknown");
#endif
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::put_time(&date, "%Y-%m-%d %H:%M UTC");
    return out.str();
}

void section(Document& pdf, double y, std::string_view number, std::string_view title) {
    pdf.text(left, y, number, 9, pdf::accent, true);
    pdf.text(left + 26, y, title, 13, pdf::ink, true);
}
void chrome(Document& pdf, std::size_t page, std::size_t count) {
    pdf.page();
    pdf.rect(left, 32, 18, 3, pdf::accent);
    pdf.text(left + 26, 37, "CrossVMAF", 11, pdf::ink, true);
    pdf.right(right, 37, "STUDIO / QUALITY REPORT", 8, pdf::muted);
    pdf.line(left, 53, right, 53);
    pdf.line(left, 794, right, 794);
    pdf.text(left, 812, "CrossVMAF  /  " + i18n::tr("Video quality assessment"), 8, pdf::muted);
    pdf.right(right, 812, std::to_string(page) + " / " + std::to_string(count), 8, pdf::muted);
}

void trend(Document& pdf, const EvaluationReport& report, const ReportAnalysis& analysis) {
    section(pdf, 371, "01", i18n::tr("Quality over time"));
    const auto scores = report_scores(report);
    const auto bounds = std::minmax_element(scores.begin(), scores.end());
    const double low = std::min(0.0, std::floor(*bounds.first / 20.0) * 20.0);
    const double high = std::max(100.0, std::ceil(*bounds.second / 20.0) * 20.0);
    constexpr double plot_left = left + 30;
    constexpr double plot_top = 397;
    constexpr double plot_bottom = 541;
    constexpr double plot_width = right - plot_left;
    const auto y = [&](double value) { return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top); };
    const auto x = [&](std::size_t index) {
        return plot_left + (scores.size() == 1U ? 0.5 : static_cast<double>(index) /
            static_cast<double>(scores.size() - 1U)) * plot_width;
    };
    // Scores outside 0..100 remain visible, never silently clipped.
    pdf.rect(plot_left, y(100), plot_width, y(90) - y(100), pdf::paper);
    for (int tick = 0; tick <= 5; ++tick) {
        const double value = low + (high - low) * tick / 5.0;
        pdf.line(plot_left, y(value), right, y(value));
        pdf.right(plot_left - 8, y(value) + 3, decimal(value, 0), 8, pdf::muted);
    }
    std::vector<std::pair<double, double>> points;
    for (const auto& [index, value] : decimate_series(scores, 1000U)) points.emplace_back(x(index), y(value));
    pdf.polyline(points, pdf::accent, 1.3);
    const auto minimum = analysis.statistics.minimum_index;
    pdf.rect(x(minimum) - 2.2, y(scores[minimum]) - 2.2, 4.4, 4.4, bands[3]);
    bool timed = true;
    double previous = -std::numeric_limits<double>::infinity();
    for (const auto& frame : report.frames) {
        if (frame.timing.status != FrameTimingStatus::valid ||
            !std::isfinite(frame.timing.media_time_seconds) || frame.timing.media_time_seconds <= previous) {
            timed = false;
            break;
        }
        previous = frame.timing.media_time_seconds;
    }
    // X is always frame order: timestamp labels would distort VFR spacing.
    std::optional<std::size_t> last_tick;
    for (std::size_t tick = 0; tick <= 4; ++tick) {
        if (scores.size() == 1U && tick != 2U) continue;
        const auto index = (scores.size() - 1U) * tick / 4U;
        if (last_tick == index) continue;
        last_tick = index;
        const auto label = "#" + std::to_string(report.frames[index].index);
        const auto position = x(index);
        if (tick == 4) pdf.right(position, 557, label, 8, pdf::muted);
        else pdf.text(position, 557, label, 8, pdf::muted);
    }
    pdf.text(plot_left, 576, i18n::tr("Frame order; valleys preserved when downsampled."), 8, pdf::muted);
    if (timed) {
        pdf.right(right, 576, decimal(report.frames.front().timing.media_time_seconds, 2) + " - " +
            decimal(report.frames.back().timing.media_time_seconds, 2) + " s", 8, pdf::muted);
    }
}

void overview(Document& pdf, const EvaluationReport& report, const ReportAnalysis& analysis) {
    pdf.text(left, 97, i18n::tr("Video quality report"), 27, pdf::ink, true);
    pdf.text(left, 119, i18n::tr("Reference-based evaluation"), 10, pdf::muted);
    pdf.right(right, 119, timestamp(report.completed_at), 9, pdf::muted);
    pdf.rect(left, 139, content_width, 121, pdf::paper);
    pdf.rect(left, 139, 3, 121, pdf::accent);
    pdf.text(left + 18, 163, i18n::tr("Pooled VMAF"), 10, pdf::muted);
    pdf.text(left + 16, 218, decimal(report.pooled_score), 43, pdf::accent, true);
    pdf.text(left + 18, 243, i18n::tr("Model score"), 8, pdf::muted);
    const auto metric = [&](double x, double y, std::string_view label, std::string_view value) {
        pdf.text(x, y, label, 9, pdf::muted);
        pdf.text(x, y + 23, value, 20, pdf::ink, true);
    };
    metric(left + 227, 164, i18n::tr("Mean"), decimal(analysis.statistics.mean));
    metric(left + 368, 164, i18n::tr("5th percentile"), decimal(analysis.statistics.percentile_5));
    metric(left + 227, 215, i18n::tr("Minimum"), decimal(analysis.statistics.minimum));
    metric(left + 368, 215, i18n::tr("Evaluated frames"), std::to_string(report.frames.size()));
    pdf.text(left, 284, i18n::tr("Reference"), 8, pdf::muted);
    pdf.paragraph(left + 76, 284, content_width - 76, known(utf8(report.request.reference_path.filename())), 10, pdf::ink, 2);
    pdf.text(left, 322, i18n::tr("Distorted"), 8, pdf::muted);
    pdf.paragraph(left + 76, 322, content_width - 76, known(utf8(report.request.distorted_path.filename())), 10, pdf::ink, 2);
    trend(pdf, report, analysis);
    section(pdf, 619, "02", i18n::tr("Score distribution"));
    const std::array<std::string, 4> labels{"93+", "80 - <93", "60 - <80", "<60"};
    std::array<std::size_t, 4> counts{};
    for (const auto& frame : report.frames) ++counts[static_cast<std::size_t>(qualityBandKind(frame.vmaf))];
    for (std::size_t i = 0; i < 4; ++i) {
        const double y = 644 + static_cast<double>(i) * 25;
        const double fraction = static_cast<double>(counts[i]) / static_cast<double>(report.frames.size());
        pdf.text(left, y + 8, labels[i], 9, pdf::muted);
        pdf.rect(left + 74, y, 295, 8, pdf::paper);
        if (counts[i] != 0U) pdf.rect(left + 74, y, 295 * fraction, 8, bands[i]);
        pdf.right(right, y + 8, decimal(fraction * 100, 1) + "%  /  " + std::to_string(counts[i]), 9, pdf::ink);
    }
    pdf.text(left, 762, i18n::tr("Bands summarize model scores; they are not a pass/fail verdict."), 8, pdf::muted);
}

bool has_segment_times(const EvaluationReport& report, const LowScoreSegment& segment) {
    double previous = -std::numeric_limits<double>::infinity();
    for (std::size_t i = segment.start; i <= segment.end; ++i) {
        const auto& timing = report.frames[i].timing;
        if (timing.status != FrameTimingStatus::valid || !std::isfinite(timing.media_time_seconds) ||
            timing.media_time_seconds < previous) return false;
        previous = timing.media_time_seconds;
    }
    return true;
}

double segments(Document& pdf, double y, const EvaluationReport& report,
                std::span<const LowScoreSegment> rows) {
    pdf.rect(left, y, content_width, 26, pdf::paper);
    pdf.text(left + 10, y + 17, i18n::tr("Frames / media time"), 8, pdf::muted);
    pdf.text(left + 285, y + 17, i18n::tr("Frame count"), 8, pdf::muted);
    pdf.right(right - 10, y + 17, i18n::tr("Minimum VMAF"), 8, pdf::muted);
    y += 26;
    if (rows.empty()) {
        pdf.text(left + 10, y + 27, i18n::tr("No segments need review"), 11, pdf::accent);
        y += 53;
    }
    for (const auto& segment : rows) {
        pdf.text(left + 10, y + 20, "#" + std::to_string(report.frames[segment.start].index) + " - #" +
            std::to_string(report.frames[segment.end].index), 10);
        const auto time = has_segment_times(report, segment) ?
            decimal(*segment.start_seconds) + " - " + decimal(*segment.end_seconds) + " s" :
            i18n::tr("Media time unavailable");
        pdf.text(left + 10, y + 36, time, 8, pdf::muted);
        pdf.text(left + 285, y + 25, std::to_string(segment.end - segment.start + 1U), 10);
        pdf.right(right - 10, y + 25, decimal(report.frames[segment.minimum_index].vmaf), 13, pdf::accent, true);
        y += 49;
        pdf.line(left, y, right, y);
    }
    return y;
}

void details(Document& pdf, const EvaluationReport& report, const ReportAnalysis& analysis) {
    pdf.text(left, 97, i18n::tr("Review & methodology"), 25, pdf::ink, true);
    section(pdf, 137, "03", i18n::tr("Segments to review"));
    pdf.text(left, 157, i18n::tr("Lowest-scoring {} of {} segments; review threshold <90.",
        std::to_string(analysis.top_segments.size()), std::to_string(analysis.segments.size())), 9, pdf::muted);
    const double after_table = segments(pdf, 174, report, analysis.top_segments);
    double y = after_table + 31;
    section(pdf, y, "04", i18n::tr("Evaluation details"));
    y += 24;
    const auto row = [&](std::string_view label, const std::string& value) {
        pdf.text(left, y, label, 8, pdf::muted);
        const auto next = pdf.paragraph(left + 118, y, content_width - 118, value, 9, pdf::ink, 2);
        y = std::max(y + 24, next + 7);
    };
    row(i18n::tr("Model / scene"), std::string(to_string(report.model)) + " / " + scene_display_name(report.request.scene));
    const auto& spec = report.descriptor;
    const std::string pixel = spec.pixel_format == PixelFormat::yuv420p ? "yuv420p" :
        spec.pixel_format == PixelFormat::yuv422p ? "yuv422p" : "yuv444p";
    row(i18n::tr("Video format"), std::to_string(spec.width) + " x " + std::to_string(spec.height) +
        " / " + std::to_string(spec.bit_depth) + " bit / " + pixel);
    row(i18n::tr("Compute device"), known(report.device.name));
    row(i18n::tr("Processing"), decimal(report.end_to_end_milliseconds / 1000.0, 2) + " s / " +
        decimal(report.frames_per_second, 1) + " fps");
    row(i18n::tr("Software / policy"), "CrossVMAF " + known(report.producer_version) +
        " / " + std::string(analysis.policy_version));
    row(i18n::tr("Report ID"), known(report.report_id));
    pdf.line(left, y, right, y);
    pdf.paragraph(left, y + 18, content_width,
        i18n::tr("Statistics use every evaluated frame. The 5th percentile follows the shared analysis policy. Segments merge nearby dips and may include intervening frames. Scores depend on the model and viewing scene; review the source video before making delivery decisions."),
        8, pdf::muted, 4);
}

} // namespace

Result<void> writeEvaluationPdf(const std::filesystem::path& destination, const EvaluationReport& report) try {
    // Validate before opening the destination: bad reports must not truncate
    // a previously exported customer report.
    if (report.frames.empty() || !std::isfinite(report.pooled_score) ||
        std::any_of(report.frames.begin(), report.frames.end(), [](const FrameResult& frame) {
            return !std::isfinite(frame.vmaf) || std::abs(frame.vmaf) > 1000000;
        })) {
        return Result<void>::failure({ErrorCode::invalid_argument,
            i18n::tr("Cannot generate a PDF without valid frame scores.")});
    }
    const auto analysis = analyze_report(report);
    // Include a chronological appendix when the six highlights are not the
    // complete set, so long reports do not silently lose review segments.
    constexpr std::size_t per_page = 11;
    const std::size_t appendix_pages = analysis.segments.size() > analysis.top_segments.size() ?
        (analysis.segments.size() + per_page - 1U) / per_page : 0U;
    Document pdf;
    chrome(pdf, 1, 2U + appendix_pages);
    overview(pdf, report, analysis);
    chrome(pdf, 2, 2U + appendix_pages);
    details(pdf, report, analysis);
    for (std::size_t page = 0; page < appendix_pages; ++page) {
        chrome(pdf, 3U + page, 2U + appendix_pages);
        pdf.text(left, 97, i18n::tr("All review segments"), 25, pdf::ink, true);
        pdf.text(left, 121, i18n::tr("Chronological order / frame indices are zero-based."), 9, pdf::muted);
        const auto start = page * per_page;
        const auto count = std::min(per_page, analysis.segments.size() - start);
        segments(pdf, 147, report, std::span(analysis.segments).subspan(start, count));
    }
    const auto bytes = pdf.finish(i18n::tr("Video quality report") + " - " + utf8(report.request.distorted_path.filename()));
    std::ofstream output(destination, std::ios::binary | std::ios::trunc);
    if (!output) return Result<void>::failure({ErrorCode::io_error,
        i18n::tr("Cannot open the export file: {}", utf8(destination))});
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    output.close();
    if (!output) return Result<void>::failure({ErrorCode::io_error,
        i18n::tr("Failed to write the export file: {}", utf8(destination))});
    return Result<void>::success();
} catch (const std::exception& exception) {
    return Result<void>::failure({ErrorCode::io_error,
        i18n::tr("Could not generate the PDF: {}", std::string(exception.what()))});
}

} // namespace crossvmaf::studio
