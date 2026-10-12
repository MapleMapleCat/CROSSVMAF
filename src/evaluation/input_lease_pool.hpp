#pragma once
#if defined(CROSSVMAF_EVALUATION_IMPLEMENTATION) || defined(CROSSVMAF_EVALUATION_TESTING)
#include "evaluation/frame_pair_record.hpp"
#include "crossvmaf/error.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/buffer.hpp"
#include "gpu/device.hpp"
#include "gpu/timestamp.hpp"
#include "video/frame.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

namespace crossvmaf::evaluation {

inline constexpr std::size_t input_leases_per_segment = 2U;

enum class SegmentedInputLeaseBank : std::uint8_t {
    none,
    spatial,
    reference_luma,
};

struct SegmentedInputLeaseToken final {
    std::uint64_t frame_index{};
    evaluation::FrameStorageHandle spatial;
    evaluation::FrameStorageHandle reference_luma;

    [[nodiscard]] explicit operator bool() const noexcept {
        return static_cast<bool>(spatial) && static_cast<bool>(reference_luma);
    }
};

struct SegmentedWritableInputLease final {
    SegmentedInputLeaseToken token;
    WritableFrameView reference;
    WritableFrameView distorted;
};

struct SegmentedInputLeaseConsumer final {
    SegmentedInputLeaseBank bank{SegmentedInputLeaseBank::none};
    evaluation::FrameStorageHandle handle;

    [[nodiscard]] explicit operator bool() const noexcept {
        return bank != SegmentedInputLeaseBank::none && static_cast<bool>(handle);
    }
};

struct SegmentedInputLeasePoolStatistics final {
    std::uint64_t reservation_count{};
    std::uint64_t spatial_release_count{};
    std::uint64_t reference_luma_release_count{};
    std::size_t active_spatial_lease_count{};
    std::size_t active_reference_luma_lease_count{};
    std::size_t maximum_active_spatial_lease_count{};
    std::size_t maximum_active_reference_luma_lease_count{};
    std::chrono::nanoseconds spatial_hold_total{};
    std::chrono::nanoseconds reference_luma_hold_total{};
    std::chrono::nanoseconds spatial_hold_maximum{};
    std::chrono::nanoseconds reference_luma_hold_maximum{};
};

// Spatial and reference-luma resources grow independently in two-lease slabs.
// Reservation, publication, and bookkeeping remain confined to the Vulkan
// coordinator thread. A reserved lease's mapped views may be written by the
// staging worker, but the coordinator cannot publish or commit that lease
// until the worker release-publishes contiguous completion. Free handles
// therefore remain caller-owned ordinary vectors.
class SegmentedInputLeasePool final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<SegmentedInputLeasePool>>
    create(gpu::DeviceView device, FrameDescriptor descriptor,
           std::size_t initial_spatial_lease_count, std::size_t reference_luma_history_distance,
           bool vulkan_image_input = false,
           gpu::TelemetryCollector *telemetry_collector = nullptr);

    SegmentedInputLeasePool(const SegmentedInputLeasePool &) = delete;
    SegmentedInputLeasePool &operator=(const SegmentedInputLeasePool &) = delete;
    ~SegmentedInputLeasePool();

    // Empty free vectors report backpressure. Capacity grows one bank at a
    // time only on an explicit caller-thread capacity request.
    [[nodiscard]] Result<std::optional<SegmentedWritableInputLease>>
    try_reserve(std::uint64_t frame_index);

    [[nodiscard]] Result<void> publish(const SegmentedInputLeaseToken &token);
    [[nodiscard]] Result<void> store_hardware_decoded_frame(const SegmentedInputLeaseToken &token,
                                                            evaluation::FramePairSide side,
                                                            video::Frame &frame);
    void abandon(const SegmentedInputLeaseToken &token) noexcept;
    void cancel_producers() noexcept;

    [[nodiscard]] Result<video::UploadedFramePair>
    commit_published(const SegmentedInputLeaseToken &token);
    [[nodiscard]] Result<void> finish_submission(const SegmentedInputLeaseToken &token);
    [[nodiscard]] Result<void> finish_staging_submission(const SegmentedInputLeaseToken &token);

    [[nodiscard]] Result<SegmentedInputLeaseConsumer>
    register_spatial_consumer(const SegmentedInputLeaseToken &token);
    [[nodiscard]] Result<std::vector<SegmentedInputLeaseConsumer>>
    register_consumers(const SegmentedInputLeaseToken &token, bool spatial, bool current_luma,
                       std::optional<std::uint64_t> previous_luma_frame);
    [[nodiscard]] Result<void> retire_consumer(const SegmentedInputLeaseConsumer &consumer);

    [[nodiscard]] Result<video::GpuPlaneView> reference_luma(std::uint64_t frame_index) const;
    void seal_end_of_stream() noexcept;

    [[nodiscard]] Result<void> ensure_capacity(std::size_t spatial_lease_count,
                                               std::size_t reference_luma_lease_count);
    [[nodiscard]] Result<void> ensure_pair_capacity(std::size_t pair_target);

    [[nodiscard]] std::size_t spatial_depth() const noexcept;
    [[nodiscard]] std::size_t reference_luma_depth() const noexcept;
    [[nodiscard]] VkDeviceSize logical_buffer_bytes() const noexcept;
    [[nodiscard]] SegmentedInputLeasePoolStatistics statistics() const noexcept;
    [[nodiscard]] bool input_memory_is_device_local() const noexcept;

  private:
    enum class LeaseState : std::uint8_t {
        free,
        reserved,
        published,
        acquired,
        submitted,
        failed,
    };

    struct SpatialLease;
    struct ReferenceLumaLease;
    struct SpatialSegment;
    struct ReferenceLumaSegment;

    struct FreeLocation final {
        std::uint32_t segment_index{};
        std::uint32_t slot_index{};
    };

    SegmentedInputLeasePool(gpu::DeviceView device, FrameDescriptor descriptor,
                            gpu::TelemetryCollector *telemetry_collector) noexcept;

    [[nodiscard]] Result<std::optional<SegmentedWritableInputLease>>
    reserve_impl(std::uint64_t frame_index);
    [[nodiscard]] Result<void> validate_token(const SegmentedInputLeaseToken &token,
                                              LeaseState expected_state) const;
    [[nodiscard]] Result<void> commit_plane(gpu::Buffer &buffer, VkDeviceSize offset,
                                            VkDeviceSize logical_size, std::size_t plane_index);
    [[nodiscard]] video::GpuPlaneView make_gpu_plane_view(const gpu::Buffer &buffer,
                                                          VkDeviceSize offset,
                                                          VkDeviceSize logical_size,
                                                          std::size_t plane_index) const noexcept;
    [[nodiscard]] video::UploadedFramePair
    make_gpu_frame_pair(const SegmentedInputLeaseToken &token) const noexcept;
    [[nodiscard]] Result<SegmentedInputLeaseConsumer>
    register_consumer(SegmentedInputLeaseBank bank, const evaluation::FrameStorageHandle &handle);
    [[nodiscard]] Result<SegmentedInputLeaseConsumer>
    register_previous_luma_consumer(std::uint64_t frame_index);

    [[nodiscard]] SpatialSegment *find_spatial_segment(std::uint32_t segment_index) const noexcept;
    [[nodiscard]] ReferenceLumaSegment *
    find_reference_luma_segment(std::uint32_t segment_index) const noexcept;
    [[nodiscard]] SpatialLease *
    find_spatial_lease(const evaluation::FrameStorageHandle &handle) const noexcept;
    [[nodiscard]] ReferenceLumaLease *
    find_reference_luma_lease(const evaluation::FrameStorageHandle &handle) const noexcept;

    [[nodiscard]] std::optional<FreeLocation> try_pop_spatial_free() noexcept;
    [[nodiscard]] std::optional<FreeLocation> try_pop_reference_luma_free() noexcept;
    void push_spatial_free(FreeLocation location) noexcept;
    void push_reference_luma_free(FreeLocation location) noexcept;

    [[nodiscard]] std::optional<evaluation::FrameStorageHandle>
    release_spatial_if_unused(SpatialLease &lease) noexcept;
    [[nodiscard]] std::optional<evaluation::FrameStorageHandle>
    release_luma_if_unused(ReferenceLumaLease &lease) noexcept;
    void record_spatial_release(SpatialLease &lease) noexcept;
    void record_luma_release(ReferenceLumaLease &lease) noexcept;
    void mark_invariant_failure() noexcept;
    [[nodiscard]] bool invariant_failed() const noexcept;

    // Growth reserves free-vector capacity before publishing a segment, so
    // every later noexcept release remains allocation-free.
    [[nodiscard]] Result<void> grow_spatial_segment();
    [[nodiscard]] Result<void> grow_reference_luma_segment();

    gpu::DeviceView device_;
    FrameDescriptor descriptor_;
    gpu::TelemetryCollector *telemetry_collector_{};
    bool profiling_enabled_{};
    bool vulkan_image_input_{};
    // Fixed before publishing any lease: later growth must not silently put
    // system-memory planes into a session that bypasses GPU input staging.
    VkMemoryPropertyFlags required_input_memory_{VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT};
    std::array<VkDeviceSize, 3U> plane_logical_sizes_{};
    std::array<VkDeviceSize, 3U> plane_slot_strides_{};
    std::vector<std::unique_ptr<SpatialSegment>> spatial_segments_;
    std::vector<std::unique_ptr<ReferenceLumaSegment>> reference_luma_segments_;
    std::vector<FreeLocation> spatial_free_locations_;
    std::vector<FreeLocation> reference_luma_free_locations_;
    // EvaluationStream::statistics() may read capacity from another thread.
    // These snapshots expose counts without publishing the caller-only vectors.
    std::atomic<std::size_t> spatial_depth_snapshot_{0U};
    std::atomic<std::size_t> reference_luma_depth_snapshot_{0U};
    VkDeviceSize logical_buffer_bytes_{};
    std::size_t reference_luma_history_distance_{};

    std::map<std::uint64_t, evaluation::FrameStorageHandle> luma_history_;
    std::uint64_t next_reservation_index_{};
    bool producer_cancelled_{};
    bool end_of_stream_sealed_{};
    bool input_memory_is_device_local_{};
    bool invariant_failure_{};
    SegmentedInputLeasePoolStatistics statistics_{};
};

} // namespace crossvmaf::evaluation
#endif
