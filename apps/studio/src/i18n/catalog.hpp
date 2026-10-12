#pragma once

// Layout of the translation tables that tools/i18n/studio_i18n.py generates
// from the gettext catalogs at build time. Only translator.cpp includes the
// generated data; nothing else should depend on this header.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace crossvmaf::studio::i18n::catalog {

// gettext allows up to six plural forms (Arabic); most languages use one or two.
inline constexpr std::size_t maximum_plural_forms = 6U;

struct Entry final {
    std::string_view context;
    std::string_view source;
    // Empty for messages without a plural form.
    std::string_view plural_source;
    // forms[0] is the translation of a singular message.
    std::array<std::string_view, maximum_plural_forms> forms;
};

struct Catalog final {
    // Directory name under apps/studio/i18n, for example "zh_CN".
    std::string_view name;
    // The catalog's Plural-Forms expression.
    std::size_t (*plural_form)(std::uint64_t n);
    // Sorted by (context, source, plural_source) as unsigned bytes.
    std::span<const Entry> entries;
};

}  // namespace crossvmaf::studio::i18n::catalog
