#pragma once
#include "gpu/device.hpp"
#include "video/backends/vulkan.hpp"
#include "video/decode_device.hpp"
struct AVCodec;
struct AVCodecContext;

namespace crossvmaf::video {
[[nodiscard]] bool backend_built(StreamDecodeBackend backend) noexcept;
[[nodiscard]] StreamDecodeBackend requested_backend(DecodeBackendPreference preference) noexcept;
[[nodiscard]] Result<std::shared_ptr<DecodeDevice>>
open_decode_device(StreamDecodeBackend backend, std::string_view selector, gpu::Device &compute);
[[nodiscard]] const AVCodec *find_backend_codec(StreamDecodeBackend backend, int codec_id);
[[nodiscard]] bool codec_supports_backend(const AVCodec *codec, StreamDecodeBackend backend);
void select_backend_format(AVCodecContext &context, StreamDecodeBackend backend);
[[nodiscard]] int backend_pixel_format(StreamDecodeBackend backend) noexcept;
[[nodiscard]] DecodeStorage backend_storage(StreamDecodeBackend backend) noexcept;
[[nodiscard]] Result<std::shared_ptr<DecodeDevice>>
create_ffmpeg_decode_device(const DecodeDeviceInfo &info);
[[nodiscard]] Result<std::vector<DecodeDeviceInfo>>
enumerate_platform_decode_devices(StreamDecodeBackend backend);
[[nodiscard]] Result<std::vector<DecodeDeviceInfo>> enumerate_nvdec_devices();
} // namespace crossvmaf::video
