#if !defined(CROSSVMAF_FRAME_NATIVE)
#include "crossvmaf/frame.hpp"

#include <cstddef>
#include <cstdint>

namespace crossvmaf {

std::uint32_t plane_width(const FrameDescriptor &descriptor, std::size_t plane) noexcept {
    if (plane == 0 || descriptor.pixel_format == PixelFormat::yuv444p) return descriptor.width;
    return descriptor.width / 2U;
}

std::uint32_t plane_height(const FrameDescriptor &descriptor, std::size_t plane) noexcept {
    if (plane == 0 || descriptor.pixel_format != PixelFormat::yuv420p) return descriptor.height;
    return descriptor.height / 2U;
}

namespace video {

Result<void> validate_frame_layout(const FrameDescriptor &descriptor) noexcept {
    if (descriptor.width == 0 || descriptor.height == 0) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "frame dimensions must be non-zero"});
    }

    if (descriptor.bit_depth != 8U && descriptor.bit_depth != 10U && descriptor.bit_depth != 12U &&
        descriptor.bit_depth != 16U) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format, "bit depth must be 8, 10, 12, or 16"});
    }
    if (descriptor.pixel_format == PixelFormat::yuv420p &&
        ((descriptor.width & 1U) != 0U || (descriptor.height & 1U) != 0U)) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format, "YUV420P requires even width and height"});
    }
    if (descriptor.pixel_format == PixelFormat::yuv422p && (descriptor.width & 1U) != 0U) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format, "YUV422P requires even width"});
    }
    return Result<void>::success();
}

Result<void> validate_plane_views(const FrameDescriptor &descriptor,
                                  const FrameView &frame) noexcept {
    const auto descriptor_result = validate_frame_layout(descriptor);
    if (!descriptor_result) return descriptor_result;

    const std::size_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    for (std::size_t plane = 0; plane < frame.planes.size(); ++plane) {
        if (frame.planes[plane].data == nullptr) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument, "all YUV planes must be present"});
        }
        const std::size_t required_stride =
            static_cast<std::size_t>(plane_width(descriptor, plane)) * bytes_per_sample;
        if (frame.planes[plane].stride_bytes < required_stride) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument, "plane stride is smaller than its active row"});
        }
    }
    return Result<void>::success();
}

} // namespace video
} // namespace crossvmaf

#else
#include "video/frame.hpp"
#include "video/ffmpeg.hpp"
extern "C" {
#include <libavutil/hwcontext_vulkan.h>
}
#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace crossvmaf::video {
namespace {
void unlock_vulkan_frame(void *frames_context, void *vulkan_frame) noexcept {
    auto *hardware_frames = static_cast<AVHWFramesContext *>(frames_context);
    auto *frame = static_cast<AVVkFrame *>(vulkan_frame);
    if (hardware_frames == nullptr || frame == nullptr || hardware_frames->hwctx == nullptr) {
        return;
    }
    auto *vulkan_frames = static_cast<AVVulkanFramesContext *>(hardware_frames->hwctx);
    if (vulkan_frames->unlock_frame != nullptr) {
        vulkan_frames->unlock_frame(hardware_frames, frame);
    }
}
[[nodiscard]] bool stride_covers_row(int stride, std::size_t row_bytes) noexcept {
    if (stride == std::numeric_limits<int>::min()) return false;
    return static_cast<std::size_t>(stride < 0 ? -stride : stride) >= row_bytes;
}
[[nodiscard]] Result<void> validate_destination_plane(const WritablePlaneView &destination,
                                                      std::size_t row_bytes,
                                                      std::uint32_t row_count) {
    if (destination.data == nullptr || destination.stride_bytes < row_bytes) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "decoded video destination plane layout is invalid"});
    }

    const std::size_t required_size =
        row_count == 0U
            ? 0U
            : static_cast<std::size_t>(row_count - 1U) * destination.stride_bytes + row_bytes;
    if (destination.size_bytes < required_size) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "decoded video destination plane is too small"});
    }
    return Result<void>::success();
}
[[nodiscard]] Result<void> copy_plane_rows(const AVFrame &source, std::size_t source_plane_index,
                                           const WritablePlaneView &destination,
                                           std::size_t row_bytes, std::uint32_t row_count) {
    const int source_stride = source.linesize[source_plane_index];
    if (source.data[source_plane_index] == nullptr ||
        !stride_covers_row(source_stride, row_bytes)) {
        return Result<void>::failure(
            {ErrorCode::unsupported_format, "decoded video source plane layout is not copyable"});
    }
    const auto destination_validation =
        validate_destination_plane(destination, row_bytes, row_count);
    if (!destination_validation) return destination_validation;

    const bool source_plane_is_tightly_packed =
        source_stride > 0 && static_cast<std::size_t>(source_stride) == row_bytes;
    const bool destination_plane_is_tightly_packed = destination.stride_bytes == row_bytes;
    if (source_plane_is_tightly_packed && destination_plane_is_tightly_packed) {
        const std::size_t contiguous_plane_bytes = row_bytes * static_cast<std::size_t>(row_count);
        std::memcpy(destination.data,
                    reinterpret_cast<const std::byte *>(source.data[source_plane_index]),
                    contiguous_plane_bytes);
        return Result<void>::success();
    }

    for (std::uint32_t row_index = 0U; row_index < row_count; ++row_index) {
        const std::ptrdiff_t source_offset = static_cast<std::ptrdiff_t>(row_index) * source_stride;
        const std::byte *const source_row =
            reinterpret_cast<const std::byte *>(source.data[source_plane_index]) + source_offset;
        std::byte *const destination_row =
            destination.data + static_cast<std::size_t>(row_index) * destination.stride_bytes;
        std::memcpy(destination_row, source_row, row_bytes);
    }
    return Result<void>::success();
}

} // namespace
LockedVulkanDecodedFrame::LockedVulkanDecodedFrame(void *frames_context, void *vulkan_frame,
                                                   UnlockCallback unlock_callback,
                                                   VulkanDecodedFrameAccess access) noexcept
    : frames_context_(frames_context), vulkan_frame_(vulkan_frame),
      unlock_callback_(unlock_callback), access_(access) {
}

LockedVulkanDecodedFrame::~LockedVulkanDecodedFrame() {
    unlock();
}

LockedVulkanDecodedFrame::LockedVulkanDecodedFrame(LockedVulkanDecodedFrame &&other) noexcept
    : frames_context_(std::exchange(other.frames_context_, nullptr)),
      vulkan_frame_(std::exchange(other.vulkan_frame_, nullptr)),
      unlock_callback_(std::exchange(other.unlock_callback_, nullptr)),
      access_(std::exchange(other.access_, {})) {
}

LockedVulkanDecodedFrame &
LockedVulkanDecodedFrame::operator=(LockedVulkanDecodedFrame &&other) noexcept {
    if (this == &other) return *this;
    unlock();
    frames_context_ = std::exchange(other.frames_context_, nullptr);
    vulkan_frame_ = std::exchange(other.vulkan_frame_, nullptr);
    unlock_callback_ = std::exchange(other.unlock_callback_, nullptr);
    access_ = std::exchange(other.access_, {});
    return *this;
}

void LockedVulkanDecodedFrame::unlock() noexcept {
    if (unlock_callback_ != nullptr) {
        unlock_callback_(frames_context_, vulkan_frame_);
    }
    frames_context_ = nullptr;
    vulkan_frame_ = nullptr;
    unlock_callback_ = nullptr;
    access_ = {};
}

Frame::Frame() noexcept = default;
Frame::~Frame() = default;
Frame::Frame(Frame &&) noexcept = default;
Frame &Frame::operator=(Frame &&) noexcept = default;

Frame::Frame(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {
}

const FrameDescriptor &Frame::descriptor() const noexcept {
    static constexpr FrameDescriptor empty_descriptor{};
    return impl_ == nullptr ? empty_descriptor : impl_->descriptor_;
}

VideoTimestamp Frame::timestamp() const noexcept {
    return impl_ == nullptr ? VideoTimestamp{} : impl_->timestamp_;
}

std::chrono::steady_clock::time_point Frame::decode_started_at() const noexcept {
    return impl_ == nullptr ? std::chrono::steady_clock::time_point{} : impl_->decode_started_at_;
}

std::chrono::steady_clock::time_point Frame::decode_completed_at() const noexcept {
    return impl_ == nullptr ? std::chrono::steady_clock::time_point{} : impl_->decode_completed_at_;
}

VideoDecodeBackend Frame::decode_backend() const noexcept {
    return impl_ == nullptr ? VideoDecodeBackend::software : impl_->decode_backend_;
}

HardwareSurfaceFormat Frame::hardware_surface_format() const noexcept {
    return impl_ == nullptr ? HardwareSurfaceFormat::none : impl_->hardware_surface_format_;
}

Result<LockedVulkanDecodedFrame> Frame::lock_vulkan_frame() const {
    if (impl_ == nullptr || impl_->frame_ == nullptr ||
        impl_->frame_->format != AV_PIX_FMT_VULKAN ||
        impl_->hardware_surface_format_ == HardwareSurfaceFormat::none) {
        return Result<LockedVulkanDecodedFrame>::failure(
            {ErrorCode::invalid_state, "decoded frame is not an NV12 or P010 Vulkan frame"});
    }
    AVFrame *const frame = impl_->frame_;
    if (frame->hw_frames_ctx == nullptr || frame->hw_frames_ctx->data == nullptr ||
        frame->data[0] == nullptr) {
        return Result<LockedVulkanDecodedFrame>::failure(
            {ErrorCode::invalid_state, "decoded Vulkan frame is missing its native frame state"});
    }

    auto *const hardware_frames = reinterpret_cast<AVHWFramesContext *>(frame->hw_frames_ctx->data);
    auto *const vulkan_frames = static_cast<AVVulkanFramesContext *>(hardware_frames->hwctx);
    auto *const vulkan_frame = reinterpret_cast<AVVkFrame *>(frame->data[0]);
    if (vulkan_frames == nullptr || vulkan_frame == nullptr ||
        vulkan_frames->lock_frame == nullptr || vulkan_frames->unlock_frame == nullptr) {
        return Result<LockedVulkanDecodedFrame>::failure(
            {ErrorCode::invalid_state, "decoded Vulkan frame has no synchronization callbacks"});
    }

    std::size_t image_count = 0U;
    while (image_count < AV_NUM_DATA_POINTERS &&
           vulkan_frames->format[image_count] != VK_FORMAT_UNDEFINED) {
        ++image_count;
    }
    if (image_count == 0U || image_count > 2U) {
        return Result<LockedVulkanDecodedFrame>::failure(
            {ErrorCode::unsupported_format, "NV12/P010 Vulkan frames must use one or two images"});
    }

    vulkan_frames->lock_frame(hardware_frames, vulkan_frame);
    VulkanDecodedFrameAccess access{
        vulkan_frame->img,
        vulkan_frame->layout,
        vulkan_frame->access,
        vulkan_frame->sem,
        vulkan_frame->sem_value,
        vulkan_frame->queue_family,
        vulkan_frames->format,
        image_count,
        impl_->hardware_surface_format_,
        impl_->descriptor_.width,
        impl_->descriptor_.height,
    };
    return Result<LockedVulkanDecodedFrame>::success(
        LockedVulkanDecodedFrame(hardware_frames, vulkan_frame, unlock_vulkan_frame, access));
}

bool Frame::has_vulkan_storage() const noexcept {
    return impl_ && impl_->frame_ && impl_->frame_->format == AV_PIX_FMT_VULKAN;
}

Result<void> Frame::copy_to(const WritableFrameView &destination) const {
    if (impl_ == nullptr || impl_->frame_ == nullptr) {
        return Result<void>::failure({ErrorCode::invalid_state, "decoded video frame is empty"});
    }
    if (impl_->frame_->hw_frames_ctx != nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan decoded frames must be converted on the GPU"});
    }

    return copy_host_frame(*impl_->frame_, impl_->descriptor_, destination);
}

Result<void> copy_host_frame(const AVFrame& source, const FrameDescriptor& descriptor,
                            const WritableFrameView& destination) {
    const auto valid = validate_frame_layout(descriptor);
    if (!valid) return valid;
    if (source.format == AV_PIX_FMT_NV12 || source.format == AV_PIX_FMT_P010LE) {
        const bool p010 = source.format == AV_PIX_FMT_P010LE;
        const std::size_t sample_bytes = p010 ? 2U : 1U;
        for (std::size_t plane = 0; plane < 3; ++plane) {
            const auto width = plane_width(descriptor, plane);
            const auto height = plane_height(descriptor, plane);
            const auto& output = destination.planes[plane];
            const std::size_t row_bytes = static_cast<std::size_t>(width) * sample_bytes;
            if (!output.data || output.stride_bytes < row_bytes ||
                output.size_bytes < (height == 0 ? 0 : (static_cast<std::size_t>(height) - 1) * output.stride_bytes + row_bytes))
                return Result<void>::failure({ErrorCode::invalid_argument, "destination plane is too small"});
            const std::size_t source_plane = plane == 0 ? 0 : 1;
            const std::size_t step = sample_bytes * (plane == 0 ? 1U : 2U);
            const std::size_t offset = plane == 2 ? sample_bytes : 0U;
            if (!source.data[source_plane] || !stride_covers_row(source.linesize[source_plane], static_cast<std::size_t>(width) * step))
                return Result<void>::failure({ErrorCode::unsupported_format, "invalid semiplanar row stride"});
            for (std::uint32_t y = 0; y < height; ++y) {
                const auto* input = source.data[source_plane] + static_cast<std::ptrdiff_t>(y) * source.linesize[source_plane] + offset;
                auto* target = output.data + static_cast<std::size_t>(y) * output.stride_bytes;
                for (std::uint32_t x = 0; x < width; ++x) {
                    if (p010) {
                        const std::uint16_t value = static_cast<std::uint16_t>((static_cast<unsigned>(input[x * step]) |
                            (static_cast<unsigned>(input[x * step + 1]) << 8U)) >> 6U);
                        std::memcpy(target + static_cast<std::size_t>(x) * 2U, &value, sizeof(value));
                    } else target[x] = static_cast<std::byte>(input[x * step]);
                }
            }
        }
        return Result<void>::success();
    }
    const std::size_t bytes_per_sample = descriptor.bit_depth == 8U ? 1U : 2U;
    for (std::size_t plane_index = 0U; plane_index < 3U; ++plane_index) {
        const std::size_t row_bytes =
            static_cast<std::size_t>(plane_width(descriptor, plane_index)) *
            bytes_per_sample;
        const std::uint32_t row_count = plane_height(descriptor, plane_index);
        const auto plane_copy = copy_plane_rows(
            source, plane_index, destination.planes[plane_index], row_bytes, row_count);
        if (!plane_copy) return plane_copy;
    }
    return Result<void>::success();
}

} // namespace crossvmaf::video
#endif
