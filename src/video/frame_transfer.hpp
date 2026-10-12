#pragma once
#include "crossvmaf/decode.hpp"
#include "video/backends/vulkan.hpp"
#include "video/transfers/d3d11_vulkan.hpp"
#include <memory>
struct AVFrame;
struct AVBufferRef;
namespace crossvmaf::video {
// One bounded decoder worker owns one transfer context. Returned frames retain
// every native dependency until the input lease has retired its GPU consumers.
class FrameTransfer final {
  public:
    FrameTransfer() = default;
    ~FrameTransfer();
    FrameTransfer(const FrameTransfer &) = delete;
    FrameTransfer &operator=(const FrameTransfer &) = delete;
    [[nodiscard]] Result<AVFrame *> convert(const AVFrame &source, DecodeTransferPolicy policy,
                                            const std::shared_ptr<VulkanDecodeDevice> &destination);
    [[nodiscard]] const std::string &fallback_reason() const noexcept { return fallback_reason_; }
    [[nodiscard]] std::uint64_t last_gpu_copy_bytes() const noexcept {
        return last_gpu_copy_bytes_;
    }
    [[nodiscard]] DecodeTransferPath path() const noexcept { return path_; }

  private:
    [[nodiscard]] Result<AVFrame *> gpu(const AVFrame &source,
                                        const std::shared_ptr<VulkanDecodeDevice> &destination);
    std::unique_ptr<D3d11VulkanTransfer> d3d11_;
    std::shared_ptr<VulkanDecodeDevice> allocation_device_;
    AVBufferRef *frames_{};
    AVBufferRef *qsv_child_frames_{};
    bool selected_{};
    std::string fallback_reason_;
    std::uint64_t last_gpu_copy_bytes_{};
    int source_width_{}, source_height_{}, source_layout_{-1};
    DecodeTransferPath path_{DecodeTransferPath::host};
};
} // namespace crossvmaf::video
