#pragma once

// Switches Slint's bundled @tr catalogs and the C++ i18n::tr tables together,
// so text from .slint files and text produced in C++ never disagree.

#include "i18n/language.hpp"

namespace crossvmaf::studio {

// Call after the first Slint component exists: the generated code registers
// its bundled catalogs when a component is created. Slint re-evaluates every
// @tr binding immediately; C++-produced text must be re-rendered by the
// caller. Returns false when the Slint bundle has no catalog for language.
bool applyUiLanguage(i18n::Language language);

} // namespace crossvmaf::studio
