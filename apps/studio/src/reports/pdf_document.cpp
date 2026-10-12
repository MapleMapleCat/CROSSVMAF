#include "reports/pdf_document.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace crossvmaf::studio::pdf {
namespace {

std::string number(double value) {
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite PDF coordinate");
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::fixed << std::setprecision(3) << value;
    return stream.str();
}
std::string rgb(Color color) { return number(color.r) + " " + number(color.g) + " " + number(color.b); }
std::string hex16(std::uint16_t value) {
    constexpr char digits[] = "0123456789ABCDEF";
    std::string result(4, '0');
    for (int index = 3; index >= 0; --index) {
        result[static_cast<std::size_t>(index)] = digits[value & 15U];
        value = static_cast<std::uint16_t>(value >> 4U);
    }
    return result;
}
std::string unicode_hex(std::uint32_t value) {
    if (value <= 0xFFFFU) return hex16(static_cast<std::uint16_t>(value));
    value -= 0x10000U;
    return hex16(static_cast<std::uint16_t>(0xD800U + (value >> 10U))) +
        hex16(static_cast<std::uint16_t>(0xDC00U + (value & 0x3FFU)));
}
std::vector<std::uint32_t> decode(std::string_view input) {
    std::vector<std::uint32_t> result;
    for (std::size_t i = 0; i < input.size();) {
        const auto first = static_cast<unsigned char>(input[i++]);
        if (first < 0x80U) {
            // Control bytes in report metadata cannot inject PDF commands.
            result.push_back(first < 32U ? 32U : first);
            continue;
        }
        const unsigned count = first >= 0xF0U && first <= 0xF4U ? 3U :
            first >= 0xE0U && first <= 0xEFU ? 2U : first >= 0xC2U && first <= 0xDFU ? 1U : 0U;
        std::uint32_t code = first & (count == 3U ? 7U : count == 2U ? 15U : 31U);
        bool valid = count != 0 && i + count <= input.size();
        for (unsigned j = 0; valid && j < count; ++j) {
            const auto next = static_cast<unsigned char>(input[i + j]);
            valid = (next & 0xC0U) == 0x80U;
            code = (code << 6U) | (next & 0x3FU);
        }
        valid = valid && code >= (count == 3U ? 0x10000U : count == 2U ? 0x800U : 0x80U) &&
            code <= 0x10FFFFU && !(code >= 0xD800U && code <= 0xDFFFU);
        if (valid) i += count;
        result.push_back(valid ? code : 0xFFFDU);
    }
    return result;
}
std::string stream_object(std::string_view data, std::string_view extra = {}) {
    return "<< /Length " + std::to_string(data.size()) + " " + std::string(extra) +
        " >>\nstream\n" + std::string(data) + "\nendstream";
}

} // namespace

std::pair<std::size_t, Glyph> Document::glyph(std::uint32_t codepoint, bool bold) {
    for (const std::size_t index : {bold ? 1U : 0U, 2U}) {
        const auto& font = font_assets[index];
        const auto* end = font.glyphs + font.glyph_count;
        const auto* found = std::lower_bound(font.glyphs, end, codepoint,
            [](const Glyph& item, std::uint32_t code) { return item.codepoint < code; });
        if (found != end && found->codepoint == codepoint) return {index, *found};
    }
    if (codepoint != '?') return glyph('?', bold);
    throw std::runtime_error("Missing PDF fallback glyph");
}

double Document::measure(std::span<const std::uint32_t> value, double size, bool bold) {
    double width_value = 0;
    for (auto code : value) width_value += glyph(code, bold).second.width * size / 1000.0;
    return width_value;
}

void Document::page() { pages_.emplace_back(); }

void Document::text_codes(double x, double y, std::span<const std::uint32_t> value,
                         double size, Color color, bool bold) {
    auto& out = pages_.back();
    out += "BT " + rgb(color) + " rg 1 0 0 1 " + number(x) + " " + number(height - y) + " Tm\n";
    std::size_t current = 3;
    for (const auto code : value) {
        const auto [font_index, font_glyph] = glyph(code, bold);
        if (current != font_index) {
            if (current != 3) out += "> Tj\n";
            out += "/F" + std::to_string(font_index) + " " + number(size) + " Tf <";
            current = font_index;
        }
        auto& mapping = cids_[font_index];
        auto found = mapping.find(code);
        if (found == mapping.end()) {
            if (used_[font_index].size() >= 65534U) throw std::runtime_error("Too many PDF glyphs");
            const auto cid = static_cast<std::uint16_t>(used_[font_index].size() + 1U);
            used_[font_index].push_back({code, font_glyph});
            found = mapping.emplace(code, cid).first;
        }
        out += hex16(found->second);
    }
    if (current != 3) out += "> Tj\n";
    out += "ET\n";
}

void Document::text(double x, double y, std::string_view value, double size, Color color, bool bold) {
    text_codes(x, y, decode(value), size, color, bold);
}
void Document::right(double x, double y, std::string_view value, double size, Color color, bool bold) {
    const auto codes = decode(value);
    text_codes(x - measure(codes, size, bold), y, codes, size, color, bold);
}
double Document::paragraph(double x, double y, double available_width, std::string_view value,
                           double size, Color color, std::size_t maximum_lines) {
    const auto codes = decode(value);
    std::size_t start = 0;
    for (std::size_t row = 0; start < codes.size() && row < maximum_lines; ++row) {
        std::size_t end = start;
        std::size_t break_at = start;
        double used_width = 0;
        while (end < codes.size()) {
            const double advance = glyph(codes[end], false).second.width * size / 1000.0;
            if (used_width + advance > available_width && end != start) break;
            used_width += advance;
            if (codes[end] == ' ') break_at = end;
            // CJK paragraphs can wrap between ideographs. Treating only
            // ASCII spaces as breaks would leave almost-empty mixed lines.
            if ((codes[end] >= 0x2E80U && codes[end] <= 0x9FFFU) ||
                (codes[end] >= 0xF900U && codes[end] <= 0xFAFFU)) break_at = end + 1U;
            ++end;
        }
        if (end < codes.size() && break_at > start) end = break_at;
        std::vector<std::uint32_t> line_codes(codes.begin() + static_cast<std::ptrdiff_t>(start),
                                            codes.begin() + static_cast<std::ptrdiff_t>(end));
        if (row + 1U == maximum_lines && end < codes.size()) {
            while (!line_codes.empty() && measure(line_codes, size, false) + size > available_width)
                line_codes.pop_back();
            line_codes.push_back(0x2026U);
        }
        text_codes(x, y, line_codes, size, color, false);
        y += size * 1.55;
        start = end;
        while (start < codes.size() && codes[start] == ' ') ++start;
    }
    return y;
}
void Document::rect(double x, double y, double w, double h, Color color) {
    pages_.back() += rgb(color) + " rg " + number(x) + " " + number(height - y - h) + " " +
        number(w) + " " + number(h) + " re f\n";
}
void Document::line(double x1, double y1, double x2, double y2, Color color, double thickness) {
    const std::array points{std::pair{x1, y1}, std::pair{x2, y2}};
    polyline(points, color, thickness);
}
void Document::polyline(std::span<const std::pair<double, double>> points, Color color, double thickness) {
    if (points.empty()) return;
    auto& out = pages_.back();
    out += rgb(color) + " RG " + number(thickness) + " w 1 J 1 j\n";
    bool first = true;
    for (const auto& [x, y] : points) {
        out += number(x) + " " + number(height - y) + (first ? " m\n" : " l\n");
        first = false;
    }
    out += "S\n";
}

std::string Document::finish(std::string_view title) {
    std::vector<std::string> objects(2);
    const auto add = [&objects](std::string object) {
        objects.push_back(std::move(object));
        return std::to_string(objects.size()) + " 0 R";
    };
    std::string resources = "<< /Font << ";
    for (std::size_t i = 0; i < used_.size(); ++i) {
        if (used_[i].empty()) continue;
        const auto& asset = font_assets[i];
        const auto program = add(stream_object(
            {reinterpret_cast<const char*>(asset.compressed), asset.compressed_size},
            "/Filter /FlateDecode /Length1 " + std::to_string(asset.original_size)));
        std::string bbox;
        for (std::size_t j = 0; j < 4; ++j) bbox += std::to_string(asset.metrics[j]) + " ";
        const auto descriptor = add("<< /Type /FontDescriptor /FontName /" + std::string(asset.name) +
            " /Flags 32 /FontBBox [" + bbox + "] /ItalicAngle 0 /Ascent " + std::to_string(asset.metrics[4]) +
            " /Descent " + std::to_string(asset.metrics[5]) + " /CapHeight " + std::to_string(asset.metrics[4]) +
            " /StemV 80 /FontFile2 " + program + " >>");
        std::string gids(2, '\0');
        std::string widths;
        std::string cmap = "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
            "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
            "/CMapName /CrossVMAFUnicode def\n/CMapType 2 def\n1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n";
        for (std::size_t j = 0; j < used_[i].size(); ++j) {
            const auto& used = used_[i][j];
            gids.push_back(static_cast<char>(used.glyph.index >> 8U));
            gids.push_back(static_cast<char>(used.glyph.index & 255U));
            widths += std::to_string(used.glyph.width) + " ";
            if (j % 100U == 0) cmap += std::to_string(std::min<std::size_t>(100, used_[i].size() - j)) + " beginbfchar\n";
            cmap += "<" + hex16(static_cast<std::uint16_t>(j + 1U)) + "> <" + unicode_hex(used.codepoint) + ">\n";
            if (j % 100U == 99U || j + 1U == used_[i].size()) cmap += "endbfchar\n";
        }
        cmap += "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n";
        const auto gid_map = add(stream_object(gids));
        const auto unicode = add(stream_object(cmap));
        const auto descendant = add("<< /Type /Font /Subtype /CIDFontType2 /BaseFont /" + std::string(asset.name) +
            " /CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor " + descriptor +
            " /CIDToGIDMap " + gid_map + " /W [1 [" + widths + "]] >>");
        const auto font = add("<< /Type /Font /Subtype /Type0 /BaseFont /" + std::string(asset.name) +
            " /Encoding /Identity-H /DescendantFonts [" + descendant + "] /ToUnicode " + unicode + " >>");
        resources += "/F" + std::to_string(i) + " " + font + " ";
    }
    resources += ">> >>";
    const auto resource_ref = add(resources);
    std::string kids;
    for (const auto& page_content : pages_) {
        const auto contents = add(stream_object(page_content));
        kids += add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + number(width) + " " + number(height) +
            "] /Resources " + resource_ref + " /Contents " + contents + " >>") + " ";
    }
    objects[0] = "<< /Type /Catalog /Pages 2 0 R >>";
    objects[1] = "<< /Type /Pages /Count " + std::to_string(pages_.size()) + " /Kids [" + kids + "] >>";
    std::string title_hex = "FEFF";
    for (auto code : decode(title)) title_hex += unicode_hex(code);
    const auto info = add("<< /Title <" + title_hex + "> /Creator (CrossVMAF Studio) /Producer (CrossVMAF PDF) >>");
    std::string output = "%PDF-1.7\n%\xE2\xE3\xCF\xD3\n";
    std::vector<std::size_t> offsets;
    for (std::size_t i = 0; i < objects.size(); ++i) {
        offsets.push_back(output.size());
        output += std::to_string(i + 1U) + " 0 obj\n" + objects[i] + "\nendobj\n";
    }
    const auto xref = output.size();
    output += "xref\n0 " + std::to_string(objects.size() + 1U) + "\n0000000000 65535 f \n";
    for (auto offset : offsets) {
        const auto value = std::to_string(offset);
        if (value.size() > 10) throw std::runtime_error("PDF exceeds cross-reference limit");
        output += std::string(10U - value.size(), '0') + value + " 00000 n \n";
    }
    return output + "trailer\n<< /Size " + std::to_string(objects.size() + 1U) +
        " /Root 1 0 R /Info " + info + " >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n";
}

} // namespace crossvmaf::studio::pdf
