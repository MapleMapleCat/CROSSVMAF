#pragma once

// Studio UI language registry. This is the only list of languages the
// application knows about: settings persistence, the language picker and
// both translation layers (Slint @tr bundles and i18n::tr) are driven from it.
// Adding a language means adding its catalog under apps/studio/i18n and one
// row here; see docs/STUDIO_I18N.md.

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

namespace crossvmaf::studio::i18n {

enum class Language : std::uint8_t {
    // Source language of every msgid; always available without a catalog.
    english,
    simplified_chinese,
};

struct LanguageInfo final {
    Language language;
    // BCP 47 tag persisted in settings.json ("en", "zh-CN").
    std::string_view tag;
    // gettext catalog directory under apps/studio/i18n; empty for the source
    // language. Also the name Slint's bundled translations are selected by.
    std::string_view catalog;
    // Endonym shown in the language picker, never translated.
    std::string_view native_name;
};

// Picker order. The source language is not necessarily first.
[[nodiscard]] std::span<const LanguageInfo> supported_languages() noexcept;
[[nodiscard]] const LanguageInfo& language_info(Language language) noexcept;

// Accepts BCP 47 and POSIX spellings, case-insensitively: "zh-CN", "zh_CN",
// "zh-Hans-SG", "zh", "en-US", "en_GB.UTF-8". Any Chinese variant maps to
// Simplified Chinese, the only Chinese catalog.
[[nodiscard]] std::optional<Language> parse_language_tag(std::string_view tag) noexcept;

// First supported entry of the user's preferred UI languages (Windows MUI
// list, or LANGUAGE/LC_ALL/LC_MESSAGES/LANG elsewhere); English otherwise.
[[nodiscard]] Language system_language();

// An absent preference means "follow the system".
[[nodiscard]] Language resolve_language(std::optional<Language> preference);

}  // namespace crossvmaf::studio::i18n
