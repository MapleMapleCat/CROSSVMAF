#pragma once

// C++ half of Studio's translation layer. UI text produced in C++ (task
// states, error descriptions, report summaries, dialog titles) goes through
// these functions; text written in .slint files uses @tr(...). Both layers
// read the same gettext catalogs in apps/studio/i18n and share one language
// selected through presentation/ui_language.hpp.
//
// Message ids are the English source text. Placeholders follow Slint's @tr
// syntax so a string reads the same in either layer: "{}" takes the next
// argument, "{0}" a positional one, "{n}" the count of a plural message, and
// "{{" / "}}" are literal braces.

#include "i18n/language.hpp"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>

namespace crossvmaf::studio::i18n {

// A translatable literal. The consteval constructor rejects anything but a
// string literal, which keeps every msgid visible to tools/i18n extraction.
class SourceText final {
public:
    template <std::size_t Size>
    consteval SourceText(const char (&text)[Size]) noexcept
        : text_(text, Size - 1U) {}

    [[nodiscard]] constexpr std::string_view view() const noexcept { return text_; }

private:
    std::string_view text_;
};

// Process-wide language for i18n::tr. Safe to read from worker threads.
void set_language(Language language) noexcept;
[[nodiscard]] Language current_language() noexcept;

// Replaces placeholders in an already translated pattern. Placeholders
// without a matching argument are kept verbatim so mistakes stay visible.
[[nodiscard]] std::string format(
    std::string_view pattern,
    std::initializer_list<std::string_view> arguments,
    std::optional<std::uint64_t> count = std::nullopt);

namespace detail {

// Catalog lookups; they fall back to the source text when the active
// language has no translation.
[[nodiscard]] std::string_view translate(
    std::string_view context, std::string_view source) noexcept;
[[nodiscard]] std::string_view translate_plural(
    std::string_view context,
    std::string_view singular,
    std::string_view plural,
    std::uint64_t count) noexcept;

template <typename... Arguments>
concept TextArguments = (std::convertible_to<const Arguments&, std::string_view> && ...);

}  // namespace detail

template <typename... Arguments>
    requires detail::TextArguments<Arguments...>
[[nodiscard]] std::string tr(SourceText source, const Arguments&... arguments) {
    return format(detail::translate({}, source.view()),
                  {std::string_view(arguments)...});
}

// Disambiguates identical English text that needs different translations.
template <typename... Arguments>
    requires detail::TextArguments<Arguments...>
[[nodiscard]] std::string trc(SourceText context, SourceText source,
                              const Arguments&... arguments) {
    return format(detail::translate(context.view(), source.view()),
                  {std::string_view(arguments)...});
}

// Chooses the plural form for count using the catalog's Plural-Forms rule.
template <typename... Arguments>
    requires detail::TextArguments<Arguments...>
[[nodiscard]] std::string trn(SourceText singular, SourceText plural,
                              std::uint64_t count, const Arguments&... arguments) {
    return format(
        detail::translate_plural({}, singular.view(), plural.view(), count),
        {std::string_view(arguments)...}, count);
}

}  // namespace crossvmaf::studio::i18n
