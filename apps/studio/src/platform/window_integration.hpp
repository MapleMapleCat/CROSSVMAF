#pragma once

#include <slint.h>

class StudioApp;

namespace crossvmaf::studio {

// The opaque handle is HWND on Windows and null on other platforms.
[[nodiscard]] void* studioWindowHandle();
void installWindowIntegration(const slint::ComponentHandle<StudioApp>& application);

} // namespace crossvmaf::studio
