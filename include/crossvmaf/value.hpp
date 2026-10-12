#pragma once

#include <cstdint>
#include <string>

namespace crossvmaf {

struct Q32Value {
    std::int64_t raw_q32_32{};

    [[nodiscard]] double value() const noexcept;
    [[nodiscard]] std::string decimal(unsigned places = 9) const;
};

}  // namespace crossvmaf
