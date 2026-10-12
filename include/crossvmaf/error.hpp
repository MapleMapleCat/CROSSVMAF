#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace crossvmaf {

enum class ErrorCode {
    invalid_argument,
    invalid_state,
    unsupported_format,
    unsupported_model,
    no_compatible_device,
    device_self_test_failed,
    device_lost,
    io_error,
    model_error,
    backend_error,
    // The operation ended because cancellation was requested; never used for
    // ordinary backend failures.
    cancelled,
    hardware_decode_unavailable,
};

enum class ErrorInput { none, reference, distorted, both };

struct Error {
    ErrorCode code{};
    std::string message;
    ErrorInput input{ErrorInput::none};
    std::optional<std::uint64_t> frame_index;

    // Add context at ownership boundaries without replacing the original code
    // or a more precise input/frame already supplied by the failing operation.
    [[nodiscard]] Error with_context(
        std::string_view operation, ErrorInput source = ErrorInput::none,
        std::optional<std::uint64_t> frame = std::nullopt) const {
        Error result = *this;
        if (result.input == ErrorInput::none) result.input = source;
        if (!result.frame_index) result.frame_index = frame;
        std::string prefix(operation);
        if (frame && !frame_index) prefix += " [frame " + std::to_string(*frame) + "]";
        result.message = prefix + ": " + message;
        return result;
    }
};

template <typename T>
class Result {
public:
    static Result success(T value) { return Result(std::move(value)); }
    static Result failure(Error error) { return Result(std::move(error)); }

    [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }
    [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] T& value() & { return value_.value(); }
    [[nodiscard]] const T& value() const& { return value_.value(); }
    [[nodiscard]] T&& value() && { return std::move(value_.value()); }
    [[nodiscard]] const Error& error() const& noexcept { return error_; }
    [[nodiscard]] Error&& error() && noexcept { return std::move(error_); }

private:
    explicit Result(T value) : value_(std::move(value)) {}
    explicit Result(Error error) : error_(std::move(error)) {}

    std::optional<T> value_;
    Error error_{};
};

template <>
class Result<void> {
public:
    static Result success() { return Result(true, {}); }
    static Result failure(Error error) { return Result(false, std::move(error)); }

    [[nodiscard]] bool has_value() const noexcept { return success_; }
    [[nodiscard]] explicit operator bool() const noexcept { return success_; }
    [[nodiscard]] const Error& error() const& noexcept { return error_; }
    [[nodiscard]] Error&& error() && noexcept { return std::move(error_); }

private:
    Result(bool success, Error error) : success_(success), error_(std::move(error)) {}

    bool success_{};
    Error error_{};
};

}  // namespace crossvmaf
