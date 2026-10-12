#include "presentation/chart_data_builder.hpp"

#include "i18n/translator.hpp"
#include "presentation/display_format.hpp"
#include "reports/report_analysis.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string_view>
#include <utility>

namespace crossvmaf::studio {

namespace {

[[nodiscard]] slint::SharedString sharedString(const std::string& value) {
    return slint::SharedString(std::string_view(value));
}

} // namespace

[[nodiscard]] slint::Color rgb(
    std::uint8_t red,
    std::uint8_t green,
    std::uint8_t blue) {
    return slint::Color::from_rgb_uint8(red, green, blue);
}

namespace {

[[nodiscard]] std::string oneDecimal(double value) {
    char buffer[32]{};
    std::snprintf(buffer, sizeof(buffer), "%.1f", value);
    return buffer;
}

} // namespace

[[nodiscard]] std::string seriesPath(
    std::span<const double> values,
    const ChartGeometry& geometry,
    std::size_t maximum_points) {
    std::string path;
    const auto points = decimate_series(values, maximum_points);
    path.reserve(points.size() * 18U);
    bool is_first_point = true;
    for (const auto& [index, value] : points) {
        path += is_first_point ? "M " : " L ";
        is_first_point = false;
        path += oneDecimal(geometry.x(index));
        path += ' ';
        path += oneDecimal(geometry.y(value));
    }
    return path;
}

namespace {

[[nodiscard]] std::string dashedHorizontalLine(
    double start_x,
    double end_x,
    double y,
    double dash_length,
    double gap_length) {
    std::string path;
    for (double x = start_x; x < end_x; x += dash_length + gap_length) {
        path += "M " + oneDecimal(x) + " " + oneDecimal(y) + " L " +
            oneDecimal(std::min(x + dash_length, end_x)) + " " + oneDecimal(y);
    }
    return path;
}

// Dashes along a polyline; the pattern restarts at every vertex.
[[nodiscard]] std::string dashedPolyline(
    std::span<const std::pair<double, double>> points,
    double dash_length,
    double gap_length) {
    std::string path;
    for (std::size_t index = 1U; index < points.size(); ++index) {
        const auto [start_x, start_y] = points[index - 1U];
        const double delta_x = points[index].first - start_x;
        const double delta_y = points[index].second - start_y;
        const double length = std::hypot(delta_x, delta_y);
        if (length <= 0.0) continue;
        for (double offset = 0.0; offset < length;
             offset += dash_length + gap_length) {
            const double end = std::min(offset + dash_length, length);
            path += "M " + oneDecimal(start_x + delta_x * offset / length) +
                " " + oneDecimal(start_y + delta_y * offset / length) +
                " L " + oneDecimal(start_x + delta_x * end / length) + " " +
                oneDecimal(start_y + delta_y * end / length);
        }
    }
    return path;
}

// A 1, 2, 2.5 or 5 × 10^n step giving at most `maximum_ticks` intervals.
[[nodiscard]] double niceStep(double range, int maximum_ticks) {
    const double raw_step = range / static_cast<double>(maximum_ticks);
    const double magnitude = std::pow(10.0, std::floor(std::log10(raw_step)));
    for (const double factor : {1.0, 2.0, 2.5, 5.0}) {
        if (factor * magnitude >= raw_step) return factor * magnitude;
    }
    return 10.0 * magnitude;
}

} // namespace

[[nodiscard]] ChartBase buildChartBase(const ChartGeometry& geometry) {
    ChartBase base;
    struct Zone final {
        double minimum;
        double maximum;
        int quality_kind;
    };
    const std::array<Zone, 3U> zones{{
        {93.0, 100.0, 0},
        {60.0, 80.0, 2},
        {0.0, 60.0, 3},
    }};
    for (const Zone& zone : zones) {
        const double minimum = std::clamp(
            zone.minimum,
            static_cast<double>(geometry.y_minimum),
            static_cast<double>(geometry.y_maximum));
        const double maximum = std::clamp(
            zone.maximum,
            static_cast<double>(geometry.y_minimum),
            static_cast<double>(geometry.y_maximum));
        if (maximum <= minimum) continue;
        ZoneRect rectangle;
        rectangle.x = geometry.left;
        rectangle.y = static_cast<float>(geometry.y(maximum));
        rectangle.w = static_cast<float>(geometry.plotWidth());
        rectangle.h = static_cast<float>(geometry.y(minimum) - geometry.y(maximum));
        rectangle.quality_kind = zone.quality_kind;
        base.zones.push_back(rectangle);
    }

    const int first_y_tick = static_cast<int>(
        std::ceil(static_cast<double>(geometry.y_minimum) / 10.0)) * 10;
    for (int value = first_y_tick; value <= 100; value += 10) {
        const float y = static_cast<float>(geometry.y(value));
        base.horizontal_grid.push_back(
            {geometry.left, y,
             static_cast<float>(geometry.left + geometry.plotWidth()), y});
        AxisLabel label;
        label.x = geometry.left - 8.0F;
        label.y = y;
        label.text = sharedString(std::to_string(value));
        label.anchor = 2;
        base.y_labels.push_back(label);
    }

    constexpr std::size_t x_tick_count = 6U;
    for (std::size_t tick = 0U; tick <= x_tick_count; ++tick) {
        const std::size_t frame_index = geometry.frame_count <= 1U
            ? 0U
            : tick * (geometry.frame_count - 1U) / x_tick_count;
        const float x = static_cast<float>(geometry.x(frame_index));
        base.vertical_grid.push_back(
            {x, geometry.top, x,
             static_cast<float>(geometry.top + geometry.plotHeight())});
        AxisLabel label;
        label.x = x;
        label.y = geometry.height - 9.5F;
        label.text = sharedString(compactFrameLabel(frame_index));
        label.anchor = 1;
        base.x_labels.push_back(label);
    }

    if (geometry.y_minimum <= 93.0F) {
        GuideLine guide;
        guide.dash_path = sharedString(dashedHorizontalLine(
            geometry.left,
            geometry.left + geometry.plotWidth(),
            geometry.y(93.0),
            4.0,
            5.0));
        guide.quality_kind = 0;
        guide.label = sharedString(i18n::tr("93 visually lossless"));
        guide.label_x = static_cast<float>(
            geometry.left + geometry.plotWidth() - 5.0);
        guide.label_y = static_cast<float>(geometry.y(93.0) - 8.5);
        base.guides.push_back(guide);
    }
    return base;
}

namespace {

[[nodiscard]] slint::Color heatColor(double value) {
    struct Stop final {
        double value;
        double red;
        double green;
        double blue;
    };
    constexpr std::array<Stop, 4U> stops{{
        {60.0, 201.0, 97.0, 95.0},
        {75.0, 217.0, 160.0, 63.0},
        {90.0, 78.0, 169.0, 124.0},
        {98.0, 111.0, 168.0, 220.0},
    }};
    if (value <= stops.front().value) return rgb(201, 97, 95);
    for (std::size_t index = 0U; index + 1U < stops.size(); ++index) {
        if (value > stops[index + 1U].value) continue;
        const Stop& left = stops[index];
        const Stop& right = stops[index + 1U];
        const double fraction = (value - left.value) / (right.value - left.value);
        return rgb(
            static_cast<std::uint8_t>(std::lround(
                left.red + (right.red - left.red) * fraction)),
            static_cast<std::uint8_t>(std::lround(
                left.green + (right.green - left.green) * fraction)),
            static_cast<std::uint8_t>(std::lround(
                left.blue + (right.blue - left.blue) * fraction)));
    }
    return rgb(111, 168, 220);
}

} // namespace

[[nodiscard]] HeatStripBundle buildHeatStrip(
    std::span<const double> scores,
    std::size_t total_frame_count) {
    HeatStripBundle bundle;
    if (scores.empty() || total_frame_count == 0U) return bundle;

    constexpr std::size_t maximum_cell_count = 720U;
    const double plot_width = chart_viewbox_width - chart_left - chart_right;
    const std::size_t cell_count = std::min(maximum_cell_count, scores.size());
    const double frames_per_cell =
        static_cast<double>(scores.size()) / static_cast<double>(cell_count);
    const double evaluated_width = total_frame_count > 1U
        ? plot_width * static_cast<double>(scores.size() - 1U) /
            static_cast<double>(total_frame_count - 1U)
        : plot_width;
    const double cell_width = evaluated_width / static_cast<double>(cell_count);
    bundle.cells.reserve(cell_count);
    for (std::size_t cell = 0U; cell < cell_count; ++cell) {
        const std::size_t start = static_cast<std::size_t>(
            std::floor(static_cast<double>(cell) * frames_per_cell));
        const std::size_t end = std::max(
            start + 1U,
            std::min(
                scores.size(),
                static_cast<std::size_t>(std::floor(
                    static_cast<double>(cell + 1U) * frames_per_cell))));
        const double minimum = *std::min_element(
            scores.begin() + static_cast<std::ptrdiff_t>(start),
            scores.begin() + static_cast<std::ptrdiff_t>(end));
        HeatCell heat_cell;
        heat_cell.x = static_cast<float>(chart_left + cell * cell_width);
        heat_cell.w = static_cast<float>(cell_width + 0.4);
        heat_cell.color = heatColor(minimum);
        bundle.cells.push_back(heat_cell);
    }
    bundle.frame_width = static_cast<float>(
        total_frame_count > scores.size() ? plot_width : evaluated_width);
    bundle.progress_fraction = static_cast<float>(std::min(
        1.0,
        static_cast<double>(scores.size()) /
            static_cast<double>(total_frame_count)));
    return bundle;
}

[[nodiscard]] HistogramBundle buildHistogram(
    std::span<const double> scores, double minimum_score) {
    HistogramBundle bundle;
    if (scores.empty()) return bundle;
    constexpr float histogram_width = 560.0F;
    constexpr float histogram_height = 180.0F;
    constexpr float histogram_left = 10.0F;
    constexpr float histogram_right = 10.0F;
    constexpr float histogram_top = 10.0F;
    constexpr float histogram_bottom = 22.0F;
    constexpr double bin_width = 2.5;
    const double histogram_minimum = std::clamp(
        std::floor(minimum_score / 5.0) * 5.0, 0.0, 97.5);
    const double maximum_score = *std::max_element(scores.begin(), scores.end());
    // Round outward on both sides. Keep a non-zero range for constant scores,
    // including the 100-point endpoint, while retaining the 2.5-point bins.
    const double histogram_maximum = std::min(100.0, std::max(
        histogram_minimum + 5.0, std::ceil(maximum_score / 5.0) * 5.0));
    const double histogram_range = histogram_maximum - histogram_minimum;
    const std::size_t bin_count = static_cast<std::size_t>(
        std::ceil(histogram_range / bin_width));
    std::vector<std::size_t> bins(std::max<std::size_t>(1U, bin_count), 0U);
    for (const double score : scores) {
        const std::ptrdiff_t unclamped_index = static_cast<std::ptrdiff_t>(
            std::floor((score - histogram_minimum) / bin_width));
        const std::size_t bin_index = static_cast<std::size_t>(std::clamp(
            unclamped_index,
            std::ptrdiff_t{0},
            static_cast<std::ptrdiff_t>(bins.size() - 1U)));
        ++bins[bin_index];
    }
    const std::size_t maximum_bin_count = *std::max_element(bins.begin(), bins.end());
    const double histogram_plot_width =
        histogram_width - histogram_left - histogram_right;
    const double histogram_plot_height =
        histogram_height - histogram_top - histogram_bottom;
    std::vector<HistoBar> histogram_bars;
    histogram_bars.reserve(bins.size());
    for (std::size_t index = 0U; index < bins.size(); ++index) {
        const double height = maximum_bin_count == 0U
            ? 0.0
            : static_cast<double>(bins[index]) /
                static_cast<double>(maximum_bin_count) * histogram_plot_height;
        HistoBar bar;
        bar.x = static_cast<float>(
            histogram_left + static_cast<double>(index) /
                static_cast<double>(bins.size()) * histogram_plot_width + 1.0);
        bar.y = static_cast<float>(histogram_top + histogram_plot_height - height);
        bar.w = static_cast<float>(
            histogram_plot_width / static_cast<double>(bins.size()) - 2.0);
        bar.h = static_cast<float>(std::max(height, bins[index] == 0U ? 0.0 : 2.0));
        bar.quality_kind = qualityBandKind(
            histogram_minimum + (static_cast<double>(index) + 0.5) * bin_width);
        histogram_bars.push_back(bar);
    }
    bundle.bars = std::move(histogram_bars);

    std::vector<AxisLabel> histogram_labels;
    const double tick_step = histogram_range <= 10.0 ? 2.5
        : histogram_range <= 30.0 ? 5.0 : 10.0;
    const auto add_histogram_label = [&](double value) {
        AxisLabel label;
        label.x = static_cast<float>(
            histogram_left + (value - histogram_minimum) /
                histogram_range * histogram_plot_width);
        label.y = histogram_height - 8.5F;
        label.text = sharedString(formatDecimal(value, value == std::floor(value) ? 0 : 1));
        label.anchor = 1;
        histogram_labels.push_back(label);
    };
    // Always label the actual bounds, even when they fall between major ticks.
    add_histogram_label(histogram_minimum);
    for (double value = std::ceil(histogram_minimum / tick_step) * tick_step;
         value < histogram_maximum; value += tick_step) {
        if (value > histogram_minimum) add_histogram_label(value);
    }
    add_histogram_label(histogram_maximum);
    bundle.labels = std::move(histogram_labels);

    const auto share = [&scores](auto predicate) {
        const auto count = std::count_if(scores.begin(), scores.end(), predicate);
        return formatDecimal(
            static_cast<double>(count) / static_cast<double>(scores.size()) * 100.0,
            1) + "%";
    };
    std::vector<LegendDot> histogram_legend(4U);
    histogram_legend[0] = {
        sharedString("≥93 · " + share([](double value) { return value >= 93.0; })),
        0};
    histogram_legend[1] = {
        sharedString("80–93 · " + share([](double value) {
            return value >= 80.0 && value < 93.0;
        })),
        1};
    histogram_legend[2] = {
        sharedString("60–80 · " + share([](double value) {
            return value >= 60.0 && value < 80.0;
        })),
        2};
    histogram_legend[3] = {
        sharedString("<60 · " + share([](double value) { return value < 60.0; })),
        3};
    bundle.legend = std::move(histogram_legend);

    return bundle;
}

RdChartBundle buildRdChart(std::span<const RdSample> samples) {
    RdChartBundle bundle;
    if (samples.empty()) return bundle;
    constexpr double width = 560.0;
    constexpr double height = 240.0;
    constexpr double left = 40.0;
    constexpr double right = 18.0;
    constexpr double top = 22.0;
    constexpr double bottom = 28.0;
    constexpr double bits_per_megabit = 1.0e6;
    constexpr double plot_width = width - left - right;
    constexpr double plot_height = height - top - bottom;

    double lowest_rate = samples.front().bits_per_second / bits_per_megabit;
    double highest_rate = lowest_rate;
    double lowest_mean = samples.front().mean;
    for (const RdSample& sample : samples) {
        const double rate = sample.bits_per_second / bits_per_megabit;
        lowest_rate = std::min(lowest_rate, rate);
        highest_rate = std::max(highest_rate, rate);
        lowest_mean = std::min(lowest_mean, sample.mean);
    }

    // Y: 5-point steps from just below the lowest mean up to 100.
    const double y_minimum = std::clamp(
        std::floor((lowest_mean - 2.0) / 5.0) * 5.0, 0.0, 90.0);
    const double y_step = 100.0 - y_minimum <= 20.0 ? 5.0 : 10.0;
    const auto y_of = [&](double value) {
        return top + (100.0 - std::clamp(value, y_minimum, 100.0)) /
            (100.0 - y_minimum) * plot_height;
    };

    // X: nice Mbps ticks from zero, unless the rates sit far from zero.
    const bool from_zero = lowest_rate < highest_rate * 0.4;
    const double spread =
        std::max(highest_rate - lowest_rate, highest_rate * 0.2);
    const double x_step = std::max(
        niceStep(from_zero ? highest_rate * 1.08 : spread * 1.4, 5), 1.0e-3);
    const double x_minimum = from_zero
        ? 0.0
        : std::max(0.0,
                   std::floor((lowest_rate - spread * 0.2) / x_step) * x_step);
    const double x_maximum = std::max(
        x_minimum + x_step,
        std::ceil((highest_rate + spread * 0.08) / x_step) * x_step);
    const auto x_of = [&](double rate) {
        return left + (rate - x_minimum) / (x_maximum - x_minimum) * plot_width;
    };
    const int x_decimals = x_step >= 1.0 ? 0 : x_step >= 0.1 ? 1 : 2;

    const int y_tick_count =
        static_cast<int>(std::lround((100.0 - y_minimum) / y_step));
    for (int tick = 0; tick <= y_tick_count; ++tick) {
        const double value = y_minimum + tick * y_step;
        const float y = static_cast<float>(y_of(value));
        bundle.grid.push_back({static_cast<float>(left), y,
                               static_cast<float>(left + plot_width), y});
        AxisLabel label;
        label.x = static_cast<float>(left - 8.0);
        label.y = y;
        label.text = sharedString(formatDecimal(value, 0));
        label.anchor = 2;
        bundle.y_labels.push_back(label);
    }
    const int x_tick_count =
        static_cast<int>(std::lround((x_maximum - x_minimum) / x_step));
    for (int tick = 0; tick <= x_tick_count; ++tick) {
        const double value = x_minimum + tick * x_step;
        AxisLabel label;
        label.x = static_cast<float>(x_of(value));
        label.y = static_cast<float>(height - 9.5);
        label.text = sharedString(formatDecimal(value, x_decimals));
        label.anchor = 1;
        bundle.x_labels.push_back(label);
    }

    const double visually_lossless = AnalysisPolicy{}.visually_lossless_threshold;
    if (y_minimum <= visually_lossless) {
        bundle.guide_path = dashedHorizontalLine(
            left, left + plot_width, y_of(visually_lossless), 4.0, 5.0);
        bundle.guide_label = formatDecimal(visually_lossless, 0);
        bundle.guide_label_x = static_cast<float>(left + plot_width);
        bundle.guide_label_y = static_cast<float>(y_of(visually_lossless) + 9.0);
    }

    std::vector<RdSample> ordered(samples.begin(), samples.end());
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const RdSample& left_sample, const RdSample& right_sample) {
            return left_sample.bits_per_second < right_sample.bits_per_second;
        });
    std::vector<std::pair<double, double>> line;
    line.reserve(ordered.size());
    for (const RdSample& sample : ordered) {
        const double x = x_of(sample.bits_per_second / bits_per_megabit);
        const double y = y_of(sample.mean);
        line.emplace_back(x, y);
        RdPoint point;
        point.x = static_cast<float>(x);
        point.y = static_cast<float>(y);
        point.slot = sample.slot;
        point.value_label = sharedString(formatDecimal(sample.mean, 1));
        point.sweet = sample.highlighted;
        bundle.points.push_back(point);
    }
    bundle.line_path = dashedPolyline(line, 2.0, 3.0);
    return bundle;
}

} // namespace crossvmaf::studio
