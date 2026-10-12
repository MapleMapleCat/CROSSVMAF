#include "video/backends/ffmpeg_device.hpp"
#include "video/decode_backend.hpp"
#if CROSSVMAF_ENABLE_NVDEC
#include <cstdio>
#include <ffnvcodec/dynlink_loader.h>
#endif
namespace crossvmaf::video {
Result<AVBufferRef *> open_nvdec_context(const DecodeDeviceInfo &info) {
    return open_native_context(AV_HWDEVICE_TYPE_CUDA, info);
}
Result<std::vector<DecodeDeviceInfo>> enumerate_nvdec_devices() {
#if CROSSVMAF_ENABLE_NVDEC
    CudaFunctions *functions = nullptr;
    if (cuda_load_functions(&functions, nullptr) < 0 || !functions)
        return Result<std::vector<DecodeDeviceInfo>>::failure(
            {ErrorCode::hardware_decode_unavailable, "NVIDIA CUDA driver could not be loaded"});
    struct Cleanup {
        CudaFunctions *&functions;
        ~Cleanup() { cuda_free_functions(&functions); }
    } cleanup{functions};
    int count{};
    if (functions->cuInit(0) != CUDA_SUCCESS || functions->cuDeviceGetCount(&count) != CUDA_SUCCESS)
        return Result<std::vector<DecodeDeviceInfo>>::failure(
            {ErrorCode::hardware_decode_unavailable, "NVIDIA driver initialization failed"});
    std::vector<DecodeDeviceInfo> result;
    constexpr char hex[] = "0123456789abcdef";
    for (int index = 0; index < count; ++index) {
        CUdevice device{};
        CUuuid uuid{};
        char name[256]{};
        if (!functions->cuDeviceGetUuid || functions->cuDeviceGet(&device, index) != CUDA_SUCCESS ||
            functions->cuDeviceGetUuid(&uuid, device) != CUDA_SUCCESS ||
            functions->cuDeviceGetName(name, sizeof(name), device) != CUDA_SUCCESS)
            continue;
        std::string physical;
        for (std::size_t i = 0; i < sizeof(uuid.bytes); ++i) {
            if (i == 4 || i == 6 || i == 8 || i == 10)
                physical += '-';
            auto value = static_cast<unsigned char>(uuid.bytes[i]);
            physical += hex[value >> 4];
            physical += hex[value & 15];
        }
        result.push_back({StreamDecodeBackend::nvdec, "nvdec:" + physical, name, physical,
                          std::to_string(index)});
    }
    return Result<std::vector<DecodeDeviceInfo>>::success(std::move(result));
#else
    return Result<std::vector<DecodeDeviceInfo>>::failure(
        {ErrorCode::hardware_decode_unavailable, "NVDEC backend was not built"});
#endif
}
} // namespace crossvmaf::video
