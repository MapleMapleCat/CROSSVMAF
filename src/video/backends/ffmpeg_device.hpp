#pragma once
#include "video/ffmpeg.hpp"
namespace crossvmaf::video {
[[nodiscard]] Result<AVBufferRef *> open_native_context(AVHWDeviceType type,
                                                        const DecodeDeviceInfo &info,
                                                        AVDictionary *options = nullptr);
[[nodiscard]] Result<AVBufferRef *> open_qsv_context(const DecodeDeviceInfo &info);
[[nodiscard]] Result<AVBufferRef *> open_nvdec_context(const DecodeDeviceInfo &info);
[[nodiscard]] Result<AVBufferRef *> open_vaapi_context(const DecodeDeviceInfo &info);
[[nodiscard]] const AVCodec *find_qsv_codec(int codec_id);
[[nodiscard]] const AVCodec *find_software_codec(int codec_id);
} // namespace crossvmaf::video
