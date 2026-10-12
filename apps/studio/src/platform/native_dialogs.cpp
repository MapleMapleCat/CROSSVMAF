#include "platform/native_dialogs.hpp"
#include "platform/window_integration.hpp"

#include "i18n/translator.hpp"

#include <algorithm>
#include <array>
#include <cwchar>
#include <string>
#include <utility>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <commdlg.h>
#include <shellapi.h>
#endif

namespace crossvmaf::studio {
namespace {

#ifdef _WIN32

[[nodiscard]] std::wstring widenUtf8(std::string_view value) {
    if (value.empty()) return {};
    const int character_count = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        nullptr,
        0);
    if (character_count <= 0) return {};
    std::wstring wide_value(static_cast<std::size_t>(character_count), L'\0');
    MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        value.data(),
        static_cast<int>(value.size()),
        wide_value.data(),
        character_count);
    return wide_value;
}

#endif

} // namespace

Result<void> openDirectory(const std::filesystem::path& directory) {
    std::error_code error;
    std::filesystem::create_directories(directory, error);
    if (error) return Result<void>::failure(
        {ErrorCode::io_error, i18n::tr("Cannot open the report folder")});
#ifdef _WIN32
    const auto result = reinterpret_cast<INT_PTR>(ShellExecuteW(
        static_cast<HWND>(studioWindowHandle()), L"open", directory.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (result > 32) return Result<void>::success();
    return Result<void>::failure(
        {ErrorCode::io_error, i18n::tr("The system could not open the report folder")});
#else
    return Result<void>::failure({ErrorCode::unsupported_format,
        i18n::tr("Opening folders is not supported on this platform yet")});
#endif
}

namespace {

// The paths chosen in the video open dialog, empty when it was cancelled.
[[nodiscard]] std::vector<std::filesystem::path> openVideoDialog(bool multiple) {
#ifdef _WIN32
    std::vector<wchar_t> path_buffer(multiple ? 262144U : 32768U, L'\0');
    // Pairs of NUL-terminated display name and pattern, ending with an extra NUL.
    std::wstring filter = widenUtf8(i18n::tr("Supported videos"));
    filter.push_back(L'\0');
    filter += L"*.mp4;*.m4v;*.mov;*.mkv;*.webm;*.ivf;*.avi;*.asf;*.wmv;*.flv;*.ts;*.m2ts;*.mpg;*.mpeg;*.mjpeg;*.mjpg";
    filter.push_back(L'\0');
    filter += widenUtf8(i18n::tr("All files"));
    filter.push_back(L'\0');
    filter += L"*.*";
    filter.push_back(L'\0');
    filter.push_back(L'\0');
    const std::wstring title = widenUtf8(
        multiple ? i18n::tr("Choose local videos") : i18n::tr("Choose a local video"));
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = static_cast<HWND>(studioWindowHandle());
    dialog.lpstrFilter = filter.c_str();
    dialog.lpstrFile = path_buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(path_buffer.size());
    dialog.lpstrTitle = title.c_str();
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST |
        OFN_EXPLORER | OFN_NOCHANGEDIR | (multiple ? OFN_ALLOWMULTISELECT : 0U);
    if (!GetOpenFileNameW(&dialog)) return {};
    // One file: its full path. Several: the folder, then each file name,
    // each NUL-terminated, ending with an empty string.
    std::vector<std::wstring> parts;
    for (const wchar_t* part = path_buffer.data(); *part != L'\0';
         part += std::wcslen(part) + 1U) {
        parts.emplace_back(part);
    }
    std::vector<std::filesystem::path> paths;
    if (parts.size() == 1U) {
        paths.emplace_back(parts.front());
    } else {
        const std::filesystem::path folder(parts.front());
        for (std::size_t index = 1U; index < parts.size(); ++index) {
            paths.push_back(folder / parts[index]);
        }
    }
    return paths;
#else
    static_cast<void>(multiple);
    return {};
#endif
}

} // namespace

std::optional<std::filesystem::path> chooseVideoFile() {
    auto paths = openVideoDialog(false);
    if (paths.empty()) return std::nullopt;
    return std::move(paths.front());
}

std::vector<std::filesystem::path> chooseVideoFiles() {
    return openVideoDialog(true);
}

std::optional<std::filesystem::path> chooseReportDestination(
    std::string_view suggested_stem,
    std::string_view extension,
    std::string_view format_name) {
#ifdef _WIN32
    std::array<wchar_t, 32768U> path_buffer{};
    const std::wstring suggested_filename =
        widenUtf8(suggested_stem) + L"." + widenUtf8(extension);
    std::copy(
        suggested_filename.begin(),
        suggested_filename.end(),
        path_buffer.begin());

    const std::wstring wide_extension = widenUtf8(extension);
    std::wstring filter = widenUtf8(format_name) + L" (*." +
        wide_extension + L")";
    filter.push_back(L'\0');
    filter += L"*." + wide_extension;
    filter.push_back(L'\0');
    filter += widenUtf8(i18n::tr("All files"));
    filter.push_back(L'\0');
    filter += L"*.*";
    filter.push_back(L'\0');
    filter.push_back(L'\0');
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = static_cast<HWND>(studioWindowHandle());
    dialog.lpstrFilter = filter.c_str();
    dialog.lpstrFile = path_buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(path_buffer.size());
    dialog.lpstrDefExt = wide_extension.c_str();
    const std::wstring title = widenUtf8(i18n::tr("Export CrossVMAF data"));
    dialog.lpstrTitle = title.c_str();
    dialog.Flags = OFN_OVERWRITEPROMPT | OFN_PATHMUSTEXIST |
        OFN_EXPLORER | OFN_NOCHANGEDIR;
    if (!GetSaveFileNameW(&dialog)) return std::nullopt;
    return std::filesystem::path(path_buffer.data());
#else
    static_cast<void>(suggested_stem);
    static_cast<void>(extension);
    static_cast<void>(format_name);
    return std::nullopt;
#endif
}

} // namespace crossvmaf::studio
