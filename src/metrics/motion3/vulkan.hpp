#pragma once
#include "metrics/motion3/motion3.hpp"
#include <cstdint>
#include "gpu/device.hpp"
#include "video/frame.hpp"
#include "gpu/submission.hpp"
#include "crossvmaf/frame.hpp"
#include "gpu/timestamp.hpp"
#include <chrono>
#include <array>
#include <memory>
#include <optional>
#include <vector>

namespace crossvmaf::metrics::motion3::vulkan {

inline constexpr std::uint32_t workgroup_size = 256U;
inline constexpr std::uint32_t score_ring_size = 8U;
inline constexpr std::uint32_t result_capacity = 2U;
inline constexpr std::uint32_t no_slot = 0xffffffffU;

struct U64Words {
    std::uint32_t low{};
    std::uint32_t high{};
};

struct FilterParameters {
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t previous_stride_bytes{};
    std::uint32_t current_stride_bytes{};
    std::uint32_t bit_depth{};
    std::uint32_t dispatch_groups_x{};
};

struct ReduceParameters {
    std::uint32_t input_count{};
    std::uint32_t dispatch_groups_x{};
};

struct FinalizeParameters {
    U64Words denominator{};
    U64Words maximum{};
    std::uint32_t output_slot{};
};

struct TemporalParameters {
    std::uint32_t mode{};
    std::uint32_t first_slot{};
    std::uint32_t second_slot{no_slot};
    std::uint32_t output_count{};
};

static_assert(sizeof(U64Words) == 8U);
static_assert(sizeof(FilterParameters) == 24U);
static_assert(sizeof(ReduceParameters) == 8U);
static_assert(sizeof(FinalizeParameters) == 20U);
static_assert(sizeof(TemporalParameters) == 16U);

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3::vulkan {

struct TelemetrySnapshot final {
    gpu::GpuOperationHandle operation_handle{};
    gpu::GpuOperationSnapshot operation;
};

class Telemetry final {
  public:
    explicit Telemetry(gpu::TelemetryCollector &collector);

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] gpu::TelemetryCollector &collector() const noexcept;
    [[nodiscard]] gpu::GpuOperationHandle operation_handle() const noexcept;

    void record_submission_latency(std::chrono::nanoseconds duration) noexcept;
    [[nodiscard]] TelemetrySnapshot snapshot() const;

  private:
    gpu::TelemetryCollector *collector_{};
    gpu::GpuOperationHandle operation_{};
};

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3::vulkan {

class Telemetry;
class Resources;
class SharedState;

class Executor final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Executor>> create(gpu::DeviceView device,
                                                                  FrameDescriptor descriptor,
                                                                  Config config,
                                                                  Telemetry *telemetry = nullptr);
    [[nodiscard]] static Result<std::shared_ptr<SharedState>>
    create_shared_state(gpu::DeviceView device);
    [[nodiscard]] static Result<std::unique_ptr<Executor>>
    create_with_shared_state(gpu::DeviceView device, FrameDescriptor descriptor, Config config,
                             std::shared_ptr<SharedState> shared_state,
                             Telemetry *telemetry = nullptr);

    Executor(const Executor &) = delete;
    Executor &operator=(const Executor &) = delete;
    ~Executor();

    [[nodiscard]] Result<std::vector<MotionScore>> push(const video::GpuFrameView &reference,
                                                        const video::GpuPlaneView *previous_luma);
    [[nodiscard]] Result<std::vector<MotionScore>> flush();

    [[nodiscard]] Result<std::optional<VkCommandBuffer>>
    record_push(const video::GpuFrameView &reference, const video::GpuPlaneView *previous_luma);
    [[nodiscard]] Result<std::optional<VkCommandBuffer>> record_flush();
    [[nodiscard]] Result<void> commit_submission(gpu::SubmissionTicket ticket);
    [[nodiscard]] Result<std::vector<MotionScore>>
    retire(gpu::SubmissionTicket ticket, std::chrono::nanoseconds submission_host_elapsed);
    [[nodiscard]] Result<std::vector<MotionScore>> complete_without_submission();
    void cancel() noexcept;

  private:
    enum class PendingOperation : std::uint8_t {
        none,
        push,
        flush,
    };

    explicit Executor(std::unique_ptr<Resources> resources) noexcept;

    [[nodiscard]] Result<std::vector<MotionScore>> submit_pending_for_convenience();
    [[nodiscard]] Result<std::vector<MotionScore>>
    readback(std::chrono::nanoseconds submission_host_elapsed);
    void clear_pending() noexcept;

    std::unique_ptr<Resources> resources_;
    PendingOperation pending_operation_{PendingOperation::none};
    VkCommandBuffer pending_command_buffer_{VK_NULL_HANDLE};
    std::array<std::uint64_t, result_capacity> pending_output_indices_{};
    std::uint32_t pending_output_count_{};
    bool pending_has_command_{};
    bool pending_committed_{};
    gpu::SubmissionTicket pending_ticket_{};
};

} // namespace crossvmaf::metrics::motion3::vulkan

namespace crossvmaf::metrics::motion3 {
// Owns this metric's reusable operation slots and sequence-wide state.
// Evaluation borrows operation slots to record/retire commands, but never
// owns their buffers, pipelines, or streaming batch allocations.
class Vulkan final {
  public:
    static Result<std::unique_ptr<Vulkan>> create(gpu::DeviceView device,
                                                  FrameDescriptor descriptor, Config config,
                                                  vulkan::Telemetry *telemetry);
    ~Vulkan();
    Vulkan(const Vulkan &) = delete;
    Vulkan &operator=(const Vulkan &) = delete;
    Result<void> append_slot();
    vulkan::Executor &slot(std::size_t index) noexcept;

  private:
    class Impl;
    explicit Vulkan(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;
};
} // namespace crossvmaf::metrics::motion3
