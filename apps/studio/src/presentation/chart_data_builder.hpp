#pragma once

#include "app.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace crossvmaf::studio {

constexpr float chart_viewbox_width = 1160.0F;
constexpr float report_viewbox_height = 280.0F;
constexpr float chart_left = 44.0F;
constexpr float chart_right = 14.0F;
constexpr float chart_top = 12.0F;
constexpr float chart_bottom = 24.0F;

struct ChartGeometry final {
    float width{chart_viewbox_width};
    float height{report_viewbox_height};
    float left{chart_left};
    float right{chart_right};
    float top{chart_top};
    float bottom{chart_bottom};
    float y_minimum{50.0F};
    float y_maximum{100.0F};
    std::size_t frame_count{2U};

    [[nodiscard]] double plotWidth() const noexcept {
        return width - left - right;
    }

    [[nodiscard]] double plotHeight() const noexcept {
        return height - top - bottom;
    }

    [[nodiscard]] double x(std::size_t frame_index) const noexcept {
        if (frame_count <= 1U) return left;
        return left + static_cast<double>(frame_index) /
            static_cast<double>(frame_count - 1U) * plotWidth();
    }

    [[nodiscard]] double y(double value) const noexcept {
        const double clamped_value = std::clamp(
            value,
            static_cast<double>(y_minimum),
            static_cast<double>(y_maximum));
        return top + (y_maximum - clamped_value) /
            (y_maximum - y_minimum) * plotHeight();
    }
};

struct ChartBase final {
    std::vector<ZoneRect> zones;
    std::vector<SegLine> horizontal_grid;
    std::vector<SegLine> vertical_grid;
    std::vector<AxisLabel> y_labels;
    std::vector<AxisLabel> x_labels;
    std::vector<GuideLine> guides;
};

struct HeatStripBundle final {
    std::vector<HeatCell> cells;
    float frame_width{};
    float progress_fraction{};
};

struct HistogramBundle final {
    std::vector<HistoBar> bars;
    std::vector<AxisLabel> labels;
    std::vector<LegendDot> legend;
};

// One evaluation on the rate-distortion chart.
struct RdSample final {
    double bits_per_second{};
    double mean{};
    std::int32_t slot{};  // 1-based SeriesPalette slot
    bool highlighted{};
};

// Rate-distortion chart geometry in its 560 × 240 viewbox.
struct RdChartBundle final {
    std::vector<SegLine> grid;
    std::vector<AxisLabel> y_labels;
    std::vector<AxisLabel> x_labels;
    std::string guide_path;
    std::string guide_label;
    float guide_label_x{};
    float guide_label_y{};
    std::string line_path;
    std::vector<RdPoint> points;
};

[[nodiscard]] slint::Color rgb(std::uint8_t red, std::uint8_t green, std::uint8_t blue);
[[nodiscard]] std::string seriesPath(
    std::span<const double> values,
    const ChartGeometry& geometry,
    std::size_t maximum_points);
[[nodiscard]] ChartBase buildChartBase(const ChartGeometry& geometry);
[[nodiscard]] HeatStripBundle buildHeatStrip(
    std::span<const double> scores, std::size_t total_frame_count);
[[nodiscard]] HistogramBundle buildHistogram(
    std::span<const double> scores, double minimum_score);
// Mean VMAF over bit rate in Mbps; points are joined in bit-rate order.
[[nodiscard]] RdChartBundle buildRdChart(std::span<const RdSample> samples);

} // namespace crossvmaf::studio
