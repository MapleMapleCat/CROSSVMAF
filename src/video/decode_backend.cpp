#include "video/decode_backend.hpp"
#include "video/backends/ffmpeg_device.hpp"
#include "video/ffmpeg.hpp"
#include <algorithm>

namespace crossvmaf::video {
bool backend_built(StreamDecodeBackend backend) noexcept {
    switch (backend) {
    case StreamDecodeBackend::software:
        return true;
    case StreamDecodeBackend::vulkan_video:
        return CROSSVMAF_ENABLE_VULKAN_VIDEO != 0;
    case StreamDecodeBackend::qsv:
        return CROSSVMAF_ENABLE_QSV != 0;
    case StreamDecodeBackend::nvdec:
        return CROSSVMAF_ENABLE_NVDEC != 0;
    case StreamDecodeBackend::vaapi:
        return CROSSVMAF_ENABLE_VAAPI != 0;
    case StreamDecodeBackend::amf:
        return CROSSVMAF_ENABLE_AMF != 0;
    }
    return false;
}
StreamDecodeBackend requested_backend(DecodeBackendPreference preference) noexcept {
    switch (preference) {
    case DecodeBackendPreference::vulkan_video:
        return StreamDecodeBackend::vulkan_video;
    case DecodeBackendPreference::qsv:
        return StreamDecodeBackend::qsv;
    case DecodeBackendPreference::nvdec:
        return StreamDecodeBackend::nvdec;
    case DecodeBackendPreference::vaapi:
        return StreamDecodeBackend::vaapi;
    case DecodeBackendPreference::amf:
        return StreamDecodeBackend::amf;
    case DecodeBackendPreference::automatic:
        return StreamDecodeBackend::software;
    }
    return StreamDecodeBackend::software;
}
int backend_pixel_format(StreamDecodeBackend backend) noexcept {
    switch (backend) {
    case StreamDecodeBackend::vulkan_video:
        return AV_PIX_FMT_VULKAN;
    case StreamDecodeBackend::qsv:
        return AV_PIX_FMT_QSV;
    case StreamDecodeBackend::nvdec:
        return AV_PIX_FMT_CUDA;
    case StreamDecodeBackend::vaapi:
        return AV_PIX_FMT_VAAPI;
    case StreamDecodeBackend::amf:
        return AV_PIX_FMT_AMF_SURFACE;
    case StreamDecodeBackend::software:
        return AV_PIX_FMT_NONE;
    }
    return AV_PIX_FMT_NONE;
}
DecodeStorage backend_storage(StreamDecodeBackend backend) noexcept {
    switch (backend) {
    case StreamDecodeBackend::vulkan_video:
        return DecodeStorage::vulkan;
    case StreamDecodeBackend::qsv:
        return DecodeStorage::qsv_surface;
    case StreamDecodeBackend::nvdec:
        return DecodeStorage::cuda;
    case StreamDecodeBackend::vaapi:
        return DecodeStorage::vaapi;
    case StreamDecodeBackend::amf:
        return DecodeStorage::amf_surface;
    case StreamDecodeBackend::software:
        return DecodeStorage::host;
    }
    return DecodeStorage::host;
}
const AVCodec *find_backend_codec(StreamDecodeBackend backend, int codec_id) {
    if (backend == StreamDecodeBackend::qsv)
        return find_qsv_codec(codec_id);
    if (backend == StreamDecodeBackend::amf)
        return find_amf_codec(codec_id);
    if (backend == StreamDecodeBackend::software)
        return find_software_codec(codec_id);
    if (codec_id == AV_CODEC_ID_AV1) {
        return avcodec_find_decoder_by_name(backend == StreamDecodeBackend::software ? "libdav1d"
                                                                                     : "av1");
    }
    return avcodec_find_decoder(static_cast<AVCodecID>(codec_id));
}
bool codec_supports_backend(const AVCodec *codec, StreamDecodeBackend backend) {
    if (!codec)
        return false;
    if (backend == StreamDecodeBackend::software)
        return true;
    for (int index = 0;; ++index) {
        const auto *config = avcodec_get_hw_config(codec, index);
        if (!config)
            return false;
        if (config->pix_fmt == backend_pixel_format(backend) &&
            (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
            return true;
    }
}
template <AVPixelFormat Wanted>
AVPixelFormat strict_format(AVCodecContext *, const AVPixelFormat *formats) {
    if (formats)
        for (; *formats != AV_PIX_FMT_NONE; ++formats)
            if (*formats == Wanted)
                return Wanted;
    return AV_PIX_FMT_NONE;
}
void select_backend_format(AVCodecContext &context, StreamDecodeBackend backend) {
    switch (backend) {
    case StreamDecodeBackend::vulkan_video:
        context.get_format = strict_format<AV_PIX_FMT_VULKAN>;
        break;
    case StreamDecodeBackend::qsv:
        context.get_format = strict_format<AV_PIX_FMT_QSV>;
        break;
    case StreamDecodeBackend::nvdec:
        context.get_format = strict_format<AV_PIX_FMT_CUDA>;
        break;
    case StreamDecodeBackend::vaapi:
        context.get_format = strict_format<AV_PIX_FMT_VAAPI>;
        break;
    case StreamDecodeBackend::amf:
        context.get_format = strict_format<AV_PIX_FMT_AMF_SURFACE>;
        break;
    case StreamDecodeBackend::software:
        break;
    }
}
Result<std::shared_ptr<DecodeDevice>>
open_decode_device(StreamDecodeBackend backend, std::string_view selector, gpu::Device &compute) {
    if (!backend_built(backend))
        return Result<std::shared_ptr<DecodeDevice>>::failure(
            {ErrorCode::hardware_decode_unavailable,
             std::string(to_string(backend)) + " backend was not built"});
    if (backend == StreamDecodeBackend::vulkan_video) {
        if (selector != "same" && selector != compute.device_info().uuid &&
            selector != "vulkan:" + compute.device_info().uuid)
            return Result<std::shared_ptr<DecodeDevice>>::failure(
                {ErrorCode::hardware_decode_unavailable,
                 "Vulkan Video must share the compute device"});
        auto device = prepare_decode_device(compute);
        if (!device || !compute.video_decode_enabled())
            return Result<std::shared_ptr<DecodeDevice>>::failure(
                {ErrorCode::hardware_decode_unavailable,
                 "selected compute device has no Vulkan Video context"});
        device->info = {backend,
                        "vulkan:" + compute.device_info().uuid,
                        compute.device_info().name,
                        compute.device_info().uuid,
                        {}};
        return Result<std::shared_ptr<DecodeDevice>>::success(std::move(device));
    }
    auto devices = backend == StreamDecodeBackend::nvdec
                       ? enumerate_nvdec_devices()
                       : enumerate_platform_decode_devices(backend);
    if (!devices)
        return Result<std::shared_ptr<DecodeDevice>>::failure(devices.error());
    const auto &physical_id = backend == StreamDecodeBackend::nvdec
                                  ? compute.device_info().uuid
                                  : compute.device_info().physical_id;
    for (const auto &info : devices.value()) {
        if ((selector == "same" && !physical_id.empty() && physical_id == info.physical_id) ||
            selector == info.id)
            return create_ffmpeg_decode_device(info);
    }
    return Result<std::shared_ptr<DecodeDevice>>::failure(
        {ErrorCode::hardware_decode_unavailable,
         std::string(to_string(backend)) +
             ": no matching decode device (use --list-decode-devices)"});
}
} // namespace crossvmaf::video

namespace crossvmaf {
std::vector<DecodeBackendInfo> enumerate_decode_backends() {
    std::vector<DecodeBackendInfo> result;
    for (auto backend :
         {StreamDecodeBackend::software, StreamDecodeBackend::vulkan_video,
          StreamDecodeBackend::qsv, StreamDecodeBackend::nvdec, StreamDecodeBackend::vaapi,
          StreamDecodeBackend::amf}) {
        bool platform = true;
#if !defined(_WIN32)
        if (backend == StreamDecodeBackend::amf)
            platform = false;
#endif
#if !defined(__linux__)
        if (backend == StreamDecodeBackend::vaapi)
            platform = false;
#endif
        const bool built = video::backend_built(backend);
        result.push_back({backend, built, platform,
                          !platform ? "unsupported_platform"
                          : !built  ? "not_built"
                                    : "candidate"});
    }
    return result;
}
Result<std::vector<DecodeDeviceInfo>> enumerate_decode_devices(DecodeBackendPreference preference) {
    std::vector<DecodeDeviceInfo> result;
    for (const auto &backend : enumerate_decode_backends()) {
        if (preference != DecodeBackendPreference::automatic &&
            backend.backend != video::requested_backend(preference))
            continue;
        if (!backend.built || !backend.platform_supported ||
            backend.backend == StreamDecodeBackend::software)
            continue;
        if (backend.backend == StreamDecodeBackend::vulkan_video) {
            auto runtime = gpu::Runtime::acquire();
            if (!runtime)
                return Result<std::vector<DecodeDeviceInfo>>::failure(runtime.error());
            auto candidates = runtime.value()->enumerate_candidates();
            if (!candidates)
                return Result<std::vector<DecodeDeviceInfo>>::failure(candidates.error());
            for (auto &candidate : candidates.value())
                if (candidate.supports_vulkan_video)
                    result.push_back({backend.backend,
                                      "vulkan:" + candidate.info.uuid,
                                      candidate.info.name,
                                      candidate.info.uuid,
                                      {}});
        } else {
            auto devices = backend.backend == StreamDecodeBackend::nvdec
                               ? video::enumerate_nvdec_devices()
                               : video::enumerate_platform_decode_devices(backend.backend);
            if (!devices) {
                if (preference != DecodeBackendPreference::automatic)
                    return devices;
                continue;
            }
            for (auto &device : devices.value())
                result.push_back(std::move(device));
        }
    }
    return Result<std::vector<DecodeDeviceInfo>>::success(std::move(result));
}
} // namespace crossvmaf
