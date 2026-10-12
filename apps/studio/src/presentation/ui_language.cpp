#include "presentation/ui_language.hpp"

#include "i18n/translator.hpp"

#include <slint.h>

namespace crossvmaf::studio {

bool applyUiLanguage(i18n::Language language) {
    i18n::set_language(language);
    // The source language has no catalog; "" selects Slint's original strings.
    return slint::select_bundled_translation(i18n::language_info(language).catalog);
}

} // namespace crossvmaf::studio
