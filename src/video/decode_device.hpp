#pragma once
#include "crossvmaf/decode.hpp"
#include <memory>
#include <mutex>

namespace crossvmaf::video {
class DecodeDevice {
  public:
    struct DecoderContext final {
        void *native_context{};
        bool dedicated_queue{};
    };
    virtual ~DecodeDevice() = default;
    [[nodiscard]] virtual StreamDecodeBackend backend() const noexcept = 0;
    [[nodiscard]] virtual void *retain_native_device_context() const noexcept = 0;
    [[nodiscard]] virtual Result<DecoderContext> retain_decoder_context(bool) {
        auto *context = retain_native_device_context();
        if (!context)
            return Result<DecoderContext>::failure(
                {ErrorCode::backend_error, "failed to retain decode device"});
        return Result<DecoderContext>::success({context, false});
    }
    [[nodiscard]] virtual bool unified_video_image_layouts() const noexcept { return false; }
    [[nodiscard]] virtual bool serializes_decode_commands() const noexcept { return false; }
    [[nodiscard]] virtual std::unique_lock<std::mutex> lock_codec_calls() const { return {}; }
    DecodeDeviceInfo info;
};
} // namespace crossvmaf::video
