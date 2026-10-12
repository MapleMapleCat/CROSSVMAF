#pragma once
#include "crossvmaf/error.hpp"
#include <cstdint>
#include <memory>
struct AVFrame;
struct AVBufferRef;
namespace crossvmaf::video {
class VulkanDecodeDevice;
class D3d11VulkanTransfer final {
  public:
    D3d11VulkanTransfer();
    ~D3d11VulkanTransfer();
    [[nodiscard]] Result<AVFrame *> transfer(const AVFrame &source, AVBufferRef *destination_frames,
                                             std::shared_ptr<VulkanDecodeDevice> device);
    [[nodiscard]] std::uint64_t last_copy_bytes() const noexcept { return last_copy_bytes_; }

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    AVBufferRef *child_frames_{};
    std::uint64_t last_copy_bytes_{};
};
} // namespace crossvmaf::video
