#pragma once
#include "video/ffmpeg.hpp"
namespace crossvmaf::video {
[[nodiscard]] Result<AVBufferRef *> open_native_context(AVHWDeviceType type,
                                                        const DecodeDeviceInfo &info,
                                                        AVDictionary *options = nullptr);
[[nodiscard]] Result<AVBufferRef *> open_qsv_context(const DecodeDeviceInfo &info);
[[nodiscard]] Result<AVBufferRef *> open_amf_context(const DecodeDeviceInfo &info);
[[nodiscard]] const AVCodec *find_amf_codec(int codec_id);
// Borrow the native D3D11 texture. The caller must retain the source AMF frame.
[[nodiscard]] Result<void> bind_amf_d3d11_surface(const AVFrame &source, AVFrame &child);
[[nodiscard]] Result<AVBufferRef *> open_nvdec_context(const DecodeDeviceInfo &info);
[[nodiscard]] Result<AVBufferRef *> open_vaapi_context(const DecodeDeviceInfo &info);
[[nodiscard]] const AVCodec *find_qsv_codec(int codec_id);
[[nodiscard]] const AVCodec *find_software_codec(int codec_id);
} // namespace crossvmaf::video
