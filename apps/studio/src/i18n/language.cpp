#include "i18n/language.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace crossvmaf::studio::i18n {
namespace {

constexpr std::array<LanguageInfo, 2U> languages{{
    {Language::simplified_chinese, "zh-CN", "zh_CN", "简体中文"},
    {Language::english, "en", "", "English"},
}};

[[nodiscard]] char lower_ascii(char character) noexcept {
    return character >= 'A' && character <= 'Z'
        ? static_cast<char>(character - 'A' + 'a')
        : character;
}

#ifdef _WIN32

[[nodiscard]] std::vector<std::string> preferred_ui_languages() {
    ULONG language_count = 0U;
    ULONG buffer_length = 0U;
    if (!GetUserPreferredUILanguages(
            MUI_LANGUAGE_NAME, &language_count, nullptr, &buffer_length) ||
        buffer_length == 0U) {
        return {};
    }
    std::vector<wchar_t> buffer(buffer_length);
    if (!GetUserPreferredUILanguages(
            MUI_LANGUAGE_NAME, &language_count, buffer.data(), &buffer_length)) {
        return {};
    }
    // A double-NUL-terminated list of ASCII tags such as "zh-CN".
    std::vector<std::string> tags;
    std::string current;
    for (const wchar_t character : buffer) {
        if (character == L'\0') {
            if (current.empty()) break;
            tags.push_back(std::move(current));
            current.clear();
        } else if (character < 0x80) {
            current.push_back(static_cast<char>(character));
        }
    }
    return tags;
}

#else

[[nodiscard]] std::vector<std::string> preferred_ui_languages() {
    std::vector<std::string> tags;
    // LANGUAGE is a GNU priority list; it only applies with a non-C locale.
    for (const char* variable : {"LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG"}) {
        const char* value = std::getenv(variable);
        if (value == nullptr || *value == '\0') continue;
        const std::string_view list(value);
        std::size_t start = 0U;
        while (start <= list.size()) {
            const std::size_t end = std::min(list.find(':', start), list.size());
            const std::string_view tag = list.substr(start, end - start);
            if (!tag.empty() && tag != "C" && tag != "POSIX") tags.emplace_back(tag);
            start = end + 1U;
        }
    }
    return tags;
}

#endif

}  // namespace

std::span<const LanguageInfo> supported_languages() noexcept {
    return languages;
}

const LanguageInfo& language_info(Language language) noexcept {
    for (const LanguageInfo& info : languages) {
        if (info.language == language) return info;
    }
    return languages.back();
}

std::optional<Language> parse_language_tag(std::string_view tag) noexcept {
    // Primary subtag only: drop region, script, encoding and modifier.
    const std::string_view primary = tag.substr(0U, tag.find_first_of("-_.@"));
    const auto is = [primary](std::string_view expected) noexcept {
        if (primary.size() != expected.size()) return false;
        for (std::size_t index = 0U; index < primary.size(); ++index) {
            if (lower_ascii(primary[index]) != expected[index]) return false;
        }
        return true;
    };
    if (is("zh")) return Language::simplified_chinese;
    if (is("en")) return Language::english;
    return std::nullopt;
}

Language system_language() {
    for (const std::string& tag : preferred_ui_languages()) {
        if (const auto language = parse_language_tag(tag)) return *language;
    }
    return Language::english;
}

Language resolve_language(std::optional<Language> preference) {
    return preference.has_value() ? *preference : system_language();
}

}  // namespace crossvmaf::studio::i18n
