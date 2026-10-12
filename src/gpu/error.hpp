#pragma once
#include "crossvmaf/error.hpp"
#include <volk.h>
#include <string>
#include <string_view>
#include <utility>

namespace crossvmaf::gpu {

[[nodiscard]] inline Error make_error(std::string_view operation, VkResult result) {
    std::string message(operation);
    message += " failed with VkResult ";
    message += std::to_string(static_cast<int>(result));
    return {result == VK_ERROR_DEVICE_LOST ? ErrorCode::device_lost : ErrorCode::backend_error,
            std::move(message)};
}

[[nodiscard]] inline Result<void> check(std::string_view operation, VkResult result) {
    if (result == VK_SUCCESS) return Result<void>::success();
    return Result<void>::failure(make_error(operation, result));
}

} // namespace crossvmaf::gpu
