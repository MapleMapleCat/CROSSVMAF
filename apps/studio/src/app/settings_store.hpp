#pragma once

#include "app/settings.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>

namespace crossvmaf::studio {

class SettingsStore final {
public:
    [[nodiscard]] static std::filesystem::path defaultPath();
    explicit SettingsStore(std::filesystem::path path = defaultPath());

    // Missing files use defaults; malformed files retain their diagnostic.
    [[nodiscard]] Result<StudioSettings> load() const;
    [[nodiscard]] Result<void> save(const StudioSettings& settings) const;

private:
    std::filesystem::path path_;
};

} // namespace crossvmaf::studio
