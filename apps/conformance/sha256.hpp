#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace crossvmaf::conformance {

[[nodiscard]] std::string sha256_hex(std::span<const std::byte> input);

}  // namespace crossvmaf::conformance
