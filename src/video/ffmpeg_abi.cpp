#include "video/ffmpeg_abi.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
}

#include <array>
#include <string>

namespace crossvmaf::video {
namespace {
std::string version_string(std::uint32_t version) {
    return std::to_string(version >> 16U) + "." +
           std::to_string((version >> 8U) & 255U) + "." +
           std::to_string(version & 255U);
}
} // namespace

Result<void> validate_ffmpeg_versions(FfmpegVersions expected, FfmpegVersions runtime) {
    struct Component { const char* name; std::uint32_t expected; std::uint32_t runtime; };
    const std::array components{
        Component{"avutil", expected.avutil, runtime.avutil},
        Component{"avcodec", expected.avcodec, runtime.avcodec},
        Component{"avformat", expected.avformat, runtime.avformat}};
    for (const auto& component : components) {
        if (component.expected != component.runtime) {
            return Result<void>::failure({ErrorCode::backend_error,
                std::string("incompatible FFmpeg ") + component.name + " version: expected " +
                version_string(component.expected) + ", loaded " + version_string(component.runtime)});
        }
    }
    return Result<void>::success();
}

Result<void> require_ffmpeg_abi() {
    // These three scalar C calls are safe before any native object allocation.
    // The DLL is linked for the lifetime of the process; initialization is thread-safe.
    static const auto compatibility = validate_ffmpeg_versions(
        {LIBAVUTIL_VERSION_INT, LIBAVCODEC_VERSION_INT, LIBAVFORMAT_VERSION_INT},
        {avutil_version(), avcodec_version(), avformat_version()});
    return compatibility;
}
} // namespace crossvmaf::video
