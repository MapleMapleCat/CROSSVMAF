#pragma once

// Small vector-only PDF backend. Font metrics and compressed font programs
// come from trusted build assets, never from imported reports or system fonts.
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace crossvmaf::studio::pdf {

struct Glyph { std::uint32_t codepoint; std::uint16_t index; std::uint16_t width; };
struct FontAsset {
    const char* name;
    const unsigned char* compressed;
    std::size_t compressed_size;
    std::size_t original_size;
    const Glyph* glyphs;
    std::size_t glyph_count;
    std::array<int, 6> metrics; // bbox, ascent, descent in 1/1000 em
};
extern const FontAsset font_assets[3];

struct Color { double r; double g; double b; };
inline constexpr Color ink{0.10, 0.15, 0.18};
inline constexpr Color muted{0.39, 0.45, 0.48};
inline constexpr Color accent{0.02, 0.44, 0.42};
inline constexpr Color rule{0.85, 0.89, 0.89};
inline constexpr Color paper{0.95, 0.97, 0.97};

class Document final {
public:
    static constexpr double width{595.28};
    static constexpr double height{841.89};
    void page();
    void text(double x, double y, std::string_view value, double size = 10,
              Color color = ink, bool bold = false);
    // Wraps on measured font advances, including long filenames and CJK.
    // Returns the first baseline after the paragraph; ellipsizes only at cap.
    double paragraph(double x, double y, double available_width, std::string_view value,
                     double size = 10, Color color = ink, std::size_t maximum_lines = 8);
    void right(double x, double y, std::string_view value, double size = 10,
               Color color = ink, bool bold = false);
    void rect(double x, double y, double w, double h, Color color);
    void line(double x1, double y1, double x2, double y2, Color color = rule, double thickness = 0.6);
    void polyline(std::span<const std::pair<double, double>> points, Color color, double thickness);
    [[nodiscard]] std::string finish(std::string_view title);

private:
    struct UsedGlyph { std::uint32_t codepoint; Glyph glyph; };
    std::array<std::vector<UsedGlyph>, 3> used_;
    std::array<std::map<std::uint32_t, std::uint16_t>, 3> cids_;
    std::vector<std::string> pages_;
    [[nodiscard]] static std::pair<std::size_t, Glyph> glyph(std::uint32_t codepoint, bool bold);
    [[nodiscard]] static double measure(std::span<const std::uint32_t> value, double size, bool bold);
    void text_codes(double x, double y, std::span<const std::uint32_t> value, double size, Color color, bool bold);
};

} // namespace crossvmaf::studio::pdf
