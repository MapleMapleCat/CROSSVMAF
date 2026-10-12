#include "platform/window_integration.hpp"
#include "app.h"

#include <slint.h>
#include <algorithm>
#include <chrono>
#include <string>
#include <string_view>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commctrl.h>
#include <windowsx.h>
#endif

namespace crossvmaf::studio {

#ifdef _WIN32

constexpr int title_bar_height = 42;
constexpr int window_button_width = 46;
constexpr int window_button_count = 3;
constexpr int resize_border_width = 6;
constexpr int corner_radius = 8;

void* studioWindowHandle() {
    struct SearchContext final {
        DWORD process_id{};
        HWND window{};
    } context{GetCurrentProcessId(), nullptr};
    EnumWindows(
        [](HWND window, LPARAM parameter) -> BOOL {
            auto* search_context = reinterpret_cast<SearchContext*>(parameter);
            DWORD owner_process_id = 0U;
            GetWindowThreadProcessId(window, &owner_process_id);
            if (owner_process_id != search_context->process_id) return TRUE;
            wchar_t title[256]{};
            if (GetWindowTextW(window, title, 256) > 0 &&
                std::wstring_view(title) == L"CrossVMAF Studio") {
                search_context->window = window;
                return FALSE;
            }
            return TRUE;
        },
        reinterpret_cast<LPARAM>(&context));
    return context.window;
}

namespace {

[[nodiscard]] double windowDpiScale(HWND window) {
    using GetDpiForWindowFunction = UINT(WINAPI*)(HWND);
    static const auto get_dpi_for_window =
        reinterpret_cast<GetDpiForWindowFunction>(GetProcAddress(
            GetModuleHandleW(L"user32.dll"), "GetDpiForWindow"));
    UINT dpi = 96U;
    if (get_dpi_for_window != nullptr) {
        dpi = get_dpi_for_window(window);
    } else if (HDC device_context = GetDC(nullptr)) {
        dpi = static_cast<UINT>(
            GetDeviceCaps(device_context, LOGPIXELSY));
        ReleaseDC(nullptr, device_context);
    }
    return static_cast<double>(dpi) / 96.0;
}

// Clips the frameless window to a roundrect so the desktop shows through
// the corners (Windows 11 style); Slint cannot do this itself because the
// winit popup window is an opaque rectangle. SetWindowRgn edges are not
// anti-aliased, but at this radius the staircase is a single pixel.
// Cleared while maximized: a maximized window fills the screen and stays
// square, matching the hidden border in app.slint.
void updateWindowRegion(HWND window) {
    if (IsZoomed(window)) {
        SetWindowRgn(window, nullptr, TRUE);
        return;
    }
    RECT client_rectangle{};
    GetClientRect(window, &client_rectangle);
    const int diameter = static_cast<int>(
        2.0 * corner_radius * windowDpiScale(window) + 0.5);
    HRGN region = CreateRoundRectRgn(
        0, 0, client_rectangle.right + 1, client_rectangle.bottom + 1,
        diameter, diameter);
    if (region == nullptr) return;
    // On success the system owns the region and deletes the previous one.
    if (SetWindowRgn(window, region, TRUE) == 0) {
        DeleteObject(region);
    }
}

LRESULT CALLBACK captionDragProcedure(
    HWND window,
    UINT message,
    WPARAM word_parameter,
    LPARAM long_parameter,
    UINT_PTR,
    DWORD_PTR) {
    if (message == WM_SETTEXT && long_parameter != 0) {
        // Slint reapplies the title during layout updates, including navigation.
        // Winit retains WS_CAPTION for native window behavior even without a
        // frame, so forwarding an unchanged title can flash the native caption
        // over our custom title bar. Preserve actual title changes.
        const std::wstring_view requested_title(
            reinterpret_cast<const wchar_t*>(long_parameter));
        const int title_length = GetWindowTextLengthW(window);
        if (requested_title.size() == static_cast<std::size_t>(title_length)) {
            std::wstring current_title(requested_title.size() + 1U, L'\0');
            const int copied_length = GetWindowTextW(
                window, current_title.data(), static_cast<int>(current_title.size()));
            current_title.resize(static_cast<std::size_t>(copied_length));
            if (current_title == requested_title) return TRUE;
        }
    }
    if (message == WM_NCACTIVATE) {
        // Same WS_CAPTION leftover as WM_SETTEXT above: forwarding activation
        // changes lets DWM repaint the native frame in (in)active colors,
        // which paints a native caption strip over the top of the window and
        // a 1px border over the custom outer frame from app.slint (the frame
        // "returns" on the next Slint repaint, e.g. pointer hover). Claiming
        // the message never draws the native frame. Winit's focus events are
        // unaffected: Focused(false)/(true) still arrive via
        // WM_KILLFOCUS/WM_SETFOCUS.
        return TRUE;
    }
    if (message == WM_NCPAINT) {
        // WM_NCCALCSIZE maps the client area over the whole window, so there
        // is no non-client area to paint; skipping DefWindowProc avoids the
        // same native frame flash when a non-client repaint is requested
        // (e.g. by SetWindowRgn's redraw flag in updateWindowRegion).
        return 0;
    }
    if (message == WM_SIZE || message == WM_DPICHANGED) {
        const LRESULT result = DefSubclassProc(window, message, word_parameter, long_parameter);
        updateWindowRegion(window);
        return result;
    }
    if (message != WM_NCHITTEST) {
        return DefSubclassProc(window, message, word_parameter, long_parameter);
    }

    const LRESULT default_hit =
        DefSubclassProc(window, message, word_parameter, long_parameter);
    if (default_hit != HTCLIENT) return default_hit;

    const double scale = windowDpiScale(window);
    POINT pointer{GET_X_LPARAM(long_parameter), GET_Y_LPARAM(long_parameter)};
    ScreenToClient(window, &pointer);
    RECT client_rectangle{};
    GetClientRect(window, &client_rectangle);
    const int scaled_border = static_cast<int>(resize_border_width * scale);
    const int title_bottom = static_cast<int>(title_bar_height * scale);
    const int buttons_left = client_rectangle.right - static_cast<int>(
        window_button_width * window_button_count * scale);
    const bool inside_drag_region =
        pointer.y >= scaled_border && pointer.y < title_bottom &&
        pointer.x >= scaled_border && pointer.x < buttons_left;
    return inside_drag_region ? HTCAPTION : default_hit;
}

void installCaptionDrag(const slint::ComponentHandle<StudioApp>& application) {
    if (HWND window = static_cast<HWND>(studioWindowHandle())) {
        DWORD_PTR reference_data{};
        // The deferred retry must never reset a window the user already resized.
        if (GetWindowSubclass(window, captionDragProcedure, 1U, &reference_data)) return;
        if (!SetWindowSubclass(window, captionDragProcedure, 1U, 0U)) return;
        MONITORINFO monitor_info{};
        monitor_info.cbSize = sizeof(monitor_info);
        if (GetMonitorInfoW(MonitorFromWindow(window, MONITOR_DEFAULTTONEAREST), &monitor_info)) {
            const RECT& work = monitor_info.rcWork;
            const int work_width = work.right - work.left;
            const int work_height = work.bottom - work.top;
            // Win32 work areas are physical pixels. Use Slint's actual scale
            // to convert once, including an optional backend scale override.
            const float scale = application->window().scale_factor();
            const int width = std::max(1, std::min(
                static_cast<int>(application->get_initial_width() * scale),
                static_cast<int>(work_width * 0.8)));
            const int height = std::max(1, std::min(
                static_cast<int>(application->get_initial_height() * scale),
                static_cast<int>(work_height * 0.8)));
            application->set_initial_width(static_cast<float>(width) / scale);
            application->set_initial_height(static_cast<float>(height) / scale);
            application->window().set_size(slint::PhysicalSize{{
                static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)}});
            SetWindowPos(window, nullptr,
                         work.left + (work_width - width) / 2,
                         work.top + (work_height - height) / 2,
                         width, height, SWP_NOACTIVATE | SWP_NOZORDER);
        }
        updateWindowRegion(window);
    }
}

} // namespace

#else
void* studioWindowHandle() { return nullptr; }
#endif

void installWindowIntegration(const slint::ComponentHandle<StudioApp>& application) {
#ifdef _WIN32
    installCaptionDrag(application);
    slint::Timer::single_shot(std::chrono::milliseconds(100),
        [weak = slint::ComponentWeakHandle<StudioApp>(application)] {
            if (auto application = weak.lock()) installCaptionDrag(*application);
        });
#else
    (void)application;
#endif
}

} // namespace crossvmaf::studio
