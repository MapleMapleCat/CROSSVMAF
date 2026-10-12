#include "video/backends/ffmpeg_device.hpp"
#include "video/decode_backend.hpp"
#include "video/ffmpeg.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#if defined(_WIN32)
#include <dxgi1_2.h>
#include <wrl/client.h>
#elif defined(__linux__)
#include <sys/stat.h>
#include <sys/sysmacros.h>
#endif

namespace crossvmaf::video {
namespace {
class NativeDecodeDevice final : public DecodeDevice {
  public:
    NativeDecodeDevice(AVBufferRef *context, DecodeDeviceInfo descriptor) : context_(context) {
        info = std::move(descriptor);
    }
    ~NativeDecodeDevice() override { av_buffer_unref(&context_); }
    StreamDecodeBackend backend() const noexcept override { return info.backend; }
    void *retain_native_device_context() const noexcept override { return av_buffer_ref(context_); }

  private:
    AVBufferRef *context_{};
};
} // namespace
Result<AVBufferRef *> open_native_context(AVHWDeviceType type, const DecodeDeviceInfo &info,
                                          AVDictionary *options) {
    AVBufferRef *context = nullptr;
    const int error =
        av_hwdevice_ctx_create(&context, type, info.native_selector.c_str(), options, 0);
    if (error < 0)
        return Result<AVBufferRef *>::failure(
            {error == AVERROR(ENOMEM) ? ErrorCode::backend_error : ErrorCode::hardware_decode_unavailable,
             ffmpeg_error_message(error, "opening " + std::string(to_string(info.backend)) +
                                             " device " + info.id)});
    return Result<AVBufferRef *>::success(context);
}
Result<std::shared_ptr<DecodeDevice>> create_ffmpeg_decode_device(const DecodeDeviceInfo &info) {
    const auto compatibility = require_ffmpeg_abi();
    if (!compatibility)
        return Result<std::shared_ptr<DecodeDevice>>::failure(compatibility.error());
    Result<AVBufferRef *> created = Result<AVBufferRef *>::failure(
        {ErrorCode::hardware_decode_unavailable, "unsupported native decode backend"});
    switch (info.backend) {
    case StreamDecodeBackend::qsv:
        created = open_qsv_context(info);
        break;
    case StreamDecodeBackend::nvdec:
        created = open_nvdec_context(info);
        break;
    case StreamDecodeBackend::vaapi:
        created = open_vaapi_context(info);
        break;
    default:
        break;
    }
    if (!created)
        return Result<std::shared_ptr<DecodeDevice>>::failure(created.error());
    auto *context = created.value();
    try {
        return Result<std::shared_ptr<DecodeDevice>>::success(
            std::make_shared<NativeDecodeDevice>(context, info));
    } catch (...) {
        av_buffer_unref(&context);
        throw;
    }
}
Result<std::vector<DecodeDeviceInfo>>
enumerate_platform_decode_devices(StreamDecodeBackend backend) {
    std::vector<DecodeDeviceInfo> result;
#if defined(_WIN32)
    if (backend != StreamDecodeBackend::qsv)
        return Result<std::vector<DecodeDeviceInfo>>::success({});
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        return Result<std::vector<DecodeDeviceInfo>>::failure(
            {ErrorCode::hardware_decode_unavailable, "DXGI adapter enumeration unavailable"});
    for (UINT index = 0;; ++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND)
            break;
        if (!adapter)
            break;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || desc.VendorId != 0x8086 ||
            (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
            continue;
        const auto *bytes = reinterpret_cast<const unsigned char *>(&desc.AdapterLuid);
        std::string id = "luid:";
        constexpr char hex[] = "0123456789abcdef";
        for (std::size_t i = 0; i < sizeof(desc.AdapterLuid); ++i) {
            id += hex[bytes[i] >> 4];
            id += hex[bytes[i] & 15];
        }
        char name[512]{};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name), nullptr, nullptr);
        result.push_back({backend, "qsv:" + id, name, id, std::to_string(index)});
    }
#elif defined(__linux__)
    std::error_code error;
    for (const auto &entry : std::filesystem::directory_iterator("/dev/dri", error)) {
        if (!entry.path().filename().string().starts_with("renderD"))
            continue;
        const auto sys_device =
            std::filesystem::path("/sys/class/drm") / entry.path().filename() / "device";
        unsigned vendor{};
        std::ifstream(sys_device / "vendor") >> std::hex >> vendor;
        if (backend == StreamDecodeBackend::qsv && vendor != 0x8086)
            continue;
        struct stat status{};
        if (::stat(entry.path().c_str(), &status) != 0)
            continue;
        const std::string physical = "drm:" + std::to_string(major(status.st_rdev)) + ":" +
                                     std::to_string(minor(status.st_rdev));
        result.push_back({backend, std::string(to_string(backend)) + ":" + physical,
                          entry.path().string(), physical, entry.path().string()});
    }
#else
    static_cast<void>(backend);
#endif
    return Result<std::vector<DecodeDeviceInfo>>::success(std::move(result));
}
} // namespace crossvmaf::video
