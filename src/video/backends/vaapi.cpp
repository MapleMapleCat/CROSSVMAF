#include "video/backends/ffmpeg_device.hpp"
namespace crossvmaf::video {
Result<AVBufferRef *> open_vaapi_context(const DecodeDeviceInfo &info) {
    return open_native_context(AV_HWDEVICE_TYPE_VAAPI, info);
}
} // namespace crossvmaf::video
