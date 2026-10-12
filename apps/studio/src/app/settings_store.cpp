#include "app/settings_store.hpp"

#include <cstdlib>
#include <utility>

namespace crossvmaf::studio {

SettingsStore::SettingsStore(std::filesystem::path path) : path_(std::move(path)) {}

Result<StudioSettings> SettingsStore::load() const {
    return load_settings(path_);
}

Result<void> SettingsStore::save(const StudioSettings& settings) const {
    return save_settings(path_, settings);
}

std::filesystem::path SettingsStore::defaultPath() {
#if defined(_MSC_VER)
    std::size_t size = 0U;
    if (_wgetenv_s(&size, nullptr, 0U, L"APPDATA") == 0 && size > 1U) {
        std::wstring app_data(size, L'\0');
        if (_wgetenv_s(&size, app_data.data(), app_data.size(), L"APPDATA") == 0) {
            app_data.resize(size - 1U);
            return std::filesystem::path(app_data) /
                "CrossVMAF" / "studio-settings.json";
        }
    }
#elif defined(_WIN32)
    if (const char* app_data = std::getenv("APPDATA")) {
        return std::filesystem::path(app_data) /
            "CrossVMAF" / "studio-settings.json";
    }
#else
    if (const char* configuration_home = std::getenv("XDG_CONFIG_HOME")) {
        return std::filesystem::path(configuration_home) /
            "crossvmaf" / "studio-settings.json";
    }
    if (const char* home = std::getenv("HOME")) {
        return std::filesystem::path(home) /
            ".config" / "crossvmaf" / "studio-settings.json";
    }
#endif
    return std::filesystem::current_path() / "studio-settings.json";
}

} // namespace crossvmaf::studio
