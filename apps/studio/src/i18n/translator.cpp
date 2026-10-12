#include "i18n/translator.hpp"

#include "i18n/catalog.hpp"
// Generated from apps/studio/i18n/*/LC_MESSAGES/crossvmaf_studio.po.
#include "i18n/studio_catalog.inc"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <tuple>

namespace crossvmaf::studio::i18n {
namespace {

std::atomic<Language> active_language{Language::english};

[[nodiscard]] const catalog::Catalog* active_catalog() noexcept {
    const std::string_view name = language_info(current_language()).catalog;
    if (name.empty()) return nullptr;
    for (const catalog::Catalog& candidate : catalog::generated_catalogs) {
        if (candidate.name == name) return &candidate;
    }
    return nullptr;
}

[[nodiscard]] const catalog::Entry* find_entry(
    const catalog::Catalog* selected,
    std::string_view context,
    std::string_view source,
    std::string_view plural_source) noexcept {
    if (selected == nullptr) return nullptr;
    const auto key = std::tie(context, source, plural_source);
    const auto entry = std::lower_bound(
        selected->entries.begin(), selected->entries.end(), key,
        [](const catalog::Entry& candidate, const auto& wanted) {
            return std::tie(candidate.context, candidate.source,
                            candidate.plural_source) < wanted;
        });
    if (entry == selected->entries.end() || entry->context != context ||
        entry->source != source || entry->plural_source != plural_source) {
        return nullptr;
    }
    return &*entry;
}

}  // namespace

void set_language(Language language) noexcept {
    active_language.store(language, std::memory_order_relaxed);
}

Language current_language() noexcept {
    return active_language.load(std::memory_order_relaxed);
}

std::string format(
    std::string_view pattern,
    std::initializer_list<std::string_view> arguments,
    std::optional<std::uint64_t> count) {
    const std::string_view* const values = arguments.begin();
    const std::size_t value_count = arguments.size();
    std::string output;
    output.reserve(pattern.size());
    std::size_t next_argument = 0U;
    for (std::size_t position = 0U; position < pattern.size(); ++position) {
        const char character = pattern[position];
        if ((character == '{' || character == '}') &&
            position + 1U < pattern.size() && pattern[position + 1U] == character) {
            output.push_back(character);
            ++position;
            continue;
        }
        if (character != '{') {
            output.push_back(character);
            continue;
        }
        const std::size_t close = pattern.find('}', position);
        if (close == std::string_view::npos) {
            output.append(pattern.substr(position));
            break;
        }
        const std::string_view placeholder =
            pattern.substr(position + 1U, close - position - 1U);
        std::optional<std::string> replacement;
        if (placeholder.empty()) {
            if (next_argument < value_count) {
                replacement = std::string(values[next_argument]);
            }
            ++next_argument;
        } else if (placeholder == "n") {
            if (count.has_value()) replacement = std::to_string(*count);
        } else {
            std::size_t index = 0U;
            const auto [end, error] = std::from_chars(
                placeholder.data(), placeholder.data() + placeholder.size(), index);
            if (error == std::errc{} && end == placeholder.data() + placeholder.size() &&
                index < value_count) {
                replacement = std::string(values[index]);
            }
        }
        if (replacement.has_value()) {
            output.append(*replacement);
        } else {
            output.append(pattern.substr(position, close - position + 1U));
        }
        position = close;
    }
    return output;
}

namespace detail {

std::string_view translate(std::string_view context, std::string_view source) noexcept {
    const catalog::Entry* const entry = find_entry(active_catalog(), context, source, {});
    return entry != nullptr && !entry->forms[0].empty() ? entry->forms[0] : source;
}

std::string_view translate_plural(
    std::string_view context,
    std::string_view singular,
    std::string_view plural,
    std::uint64_t count) noexcept {
    const catalog::Catalog* const selected = active_catalog();
    if (const catalog::Entry* const entry = find_entry(selected, context, singular, plural)) {
        const std::size_t form = selected->plural_form(count);
        if (form < entry->forms.size() && !entry->forms[form].empty()) {
            return entry->forms[form];
        }
    }
    return count == 1U ? singular : plural;
}

}  // namespace detail
}  // namespace crossvmaf::studio::i18n
