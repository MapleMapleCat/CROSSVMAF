#pragma once

#include "gpu/device.hpp"
#include "gpu/buffer.hpp"
#include "video/frame.hpp"
#include "crossvmaf/frame.hpp"
#include "crossvmaf/error.hpp"
#include "gpu/timestamp.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace crossvmaf::conformance {


// Test and conformance utility that owns tightly-packed raw GPU input.
// Production input is owned by SegmentedInputLeasePool and InputStagingRing.
class FrameUploader final {
public:
    [[nodiscard]] static Result<std::unique_ptr<FrameUploader>> create(
        gpu::DeviceView device, FrameDescriptor descriptor,
        std::size_t frame_slot_count = 1U,
        std::size_t reference_luma_history_distance = 2U,
        gpu::TelemetryCollector* telemetry_collector = nullptr);

    FrameUploader(const FrameUploader&) = delete;
    FrameUploader& operator=(const FrameUploader&) = delete;
    ~FrameUploader();

    [[nodiscard]] Result<crossvmaf::video::UploadedFramePair> upload(
        const FrameView& reference, const FrameView& distorted);

private:
    explicit FrameUploader(gpu::DeviceView device, FrameDescriptor descriptor,
                           std::size_t frame_slot_count,
                           std::size_t luma_ring_size,
                           gpu::TelemetryCollector* telemetry_collector) noexcept
        : device_(device), descriptor_(descriptor),
          frame_slot_count_(frame_slot_count), luma_ring_size_(luma_ring_size),
          telemetry_collector_(telemetry_collector) {}

    [[nodiscard]] Result<void> upload_plane(gpu::Buffer& destination,
                                            const PlaneView& source,
                                            std::size_t plane);
    [[nodiscard]] crossvmaf::video::GpuPlaneView make_plane_view(const gpu::Buffer& buffer,
                                               std::size_t plane) const noexcept;

    gpu::DeviceView device_;
    FrameDescriptor descriptor_;
    std::size_t frame_slot_count_{};
    std::size_t luma_ring_size_{};
    gpu::TelemetryCollector* telemetry_collector_{};
    std::uint64_t next_index_{};
    std::vector<std::unique_ptr<gpu::Buffer>> reference_luma_ring_;
    std::vector<std::array<std::unique_ptr<gpu::Buffer>, 2>> reference_chroma_slots_;
    std::vector<std::array<std::unique_ptr<gpu::Buffer>, 3>> distorted_slots_;
    std::vector<std::byte> scratch_;
};

}  // namespace crossvmaf::conformance
