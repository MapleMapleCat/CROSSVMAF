#pragma once
#include <cstdint>
#include "crossvmaf/error.hpp"
#include <optional>
#include <string_view>
#include <vector>
#include "gpu/device.hpp"
#include <volk.h>
#include <chrono>
#include <cstddef>
#include <deque>
#include <memory>
#include <span>
#include <thread>

namespace crossvmaf::gpu {

inline constexpr std::uint32_t maximum_streaming_wave_task_count = 64U;

struct StreamingTaskHandle final {
    std::uint32_t slot{};
    std::uint32_t generation{};

    [[nodiscard]] explicit operator bool() const noexcept { return generation != 0U; }
};

static_assert(sizeof(StreamingTaskHandle) == 8U);

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

class StreamingTaskRingCore final {
  public:
    enum class TaskState : std::uint8_t {
        free,
        prefix_pending,
        ready,
        wave,
    };

    struct TaskRecord final {
        std::uint64_t frame_index{};
        std::uint32_t generation{};
        TaskState state{TaskState::free};
    };

    void initialize(std::uint32_t capacity);

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool failed() const noexcept { return failed_; }
    [[nodiscard]] std::uint32_t free_slot_count() const noexcept;
    [[nodiscard]] std::uint32_t prefix_pending_task_count() const noexcept;
    [[nodiscard]] std::uint32_t ready_task_count() const noexcept;

    [[nodiscard]] Result<std::optional<StreamingTaskHandle>> reserve(std::uint64_t frame_index,
                                                                     std::string_view feature_name);
    [[nodiscard]] Result<void> cancel_prefix(StreamingTaskHandle task,
                                             std::string_view feature_name);
    [[nodiscard]] Result<void> publish_ready(StreamingTaskHandle task,
                                             std::string_view feature_name);
    [[nodiscard]] Result<void> snapshot_ready_wave(std::string_view feature_name);
    void mark_wave_recorded() noexcept;
    void complete_retired_wave() noexcept;
    void fail_retired_wave() noexcept;
    void restore_unsubmitted_wave() noexcept;
    void cancel_after_device_drain() noexcept;

    [[nodiscard]] const TaskRecord &task(std::uint32_t slot) const noexcept { return tasks_[slot]; }
    [[nodiscard]] TaskRecord &task(std::uint32_t slot) noexcept { return tasks_[slot]; }
    [[nodiscard]] const std::vector<TaskRecord> &tasks() const noexcept { return tasks_; }
    [[nodiscard]] std::vector<TaskRecord> &tasks() noexcept { return tasks_; }

    [[nodiscard]] const std::vector<StreamingTaskHandle> &pending_wave_tasks() const noexcept {
        return pending_wave_tasks_;
    }
    [[nodiscard]] std::vector<StreamingTaskHandle> &pending_wave_tasks() noexcept {
        return pending_wave_tasks_;
    }

  private:
    std::uint32_t capacity_{};
    bool failed_{};
    std::vector<TaskRecord> tasks_;
    std::vector<StreamingTaskHandle> ready_queue_entries_;
    std::vector<StreamingTaskHandle> pending_wave_tasks_;
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

inline constexpr std::uint64_t submission_wait_timeout_nanoseconds = 10'000'000'000ULL;

struct SubmissionTicket {
    std::uint64_t serial{};

    [[nodiscard]] explicit operator bool() const noexcept { return serial != 0U; }
};

struct SubmissionRetirement {
    SubmissionTicket ticket;
    std::chrono::nanoseconds host_elapsed{};
    // Lane that produced this retirement. Single-lane owners always report 0.
    std::uint32_t lane{};
};

struct SubmissionOwnerStatistics {
    std::uint64_t queue_submit_call_count{};
    std::uint64_t successful_submission_count{};
    std::uint64_t fence_wait_call_count{};
    std::uint64_t submitted_command_buffer_count{};
    std::uint64_t last_attempted_serial{};
    std::uint64_t last_submitted_serial{};
    std::uint64_t last_retired_serial{};
};

struct SubmissionSynchronization final {
    std::span<const VkSemaphore> wait_semaphores;
    std::span<const VkPipelineStageFlags> wait_stage_masks;
    std::span<const VkSemaphore> signal_semaphores;
    // Empty value spans select ordinary binary-semaphore submission. When
    // present, each span must match its semaphore span; binary entries use 0.
    std::span<const std::uint64_t> wait_values;
    std::span<const std::uint64_t> signal_values;
};

// Terminal ownership coordination for Device and wrappers that may destroy
// Vulkan device children. Abandonment intentionally releases the C++ owner
// without invoking its destructor; callers accept the bounded process-lifetime
// leak because destroying a possibly in-use Vulkan object is invalid.
template <typename DeviceResource>
void apply_device_resource_cleanup(std::unique_ptr<DeviceResource> &resource,
                                   DeviceResourceCleanupOutcome cleanup_outcome) noexcept {
    if (cleanup_outcome == DeviceResourceCleanupOutcome::must_abandon_device_resources) {
        static_cast<void>(resource.release());
        return;
    }
    resource.reset();
}

// Submission lanes are independent ordered submission contexts sharing one
// owner thread. Lane i submits to device.queue_for_lane(i % queue_modulus),
// so lanes alias onto one physical queue when the device exposes fewer queues
// than lanes; per-lane fences and FIFO retirement are identical either way.
class SubmissionOwner final {
  public:
    // maximum_pending_submission_count bounds each lane's fence bank.
    // queue_modulus 0 spreads lanes over every queue the device exposes.
    // timing_enabled gates host-side submit/retire clock samples used only by
    // the hotspot profiler.
    [[nodiscard]] static Result<std::unique_ptr<SubmissionOwner>>
    create(DeviceView device, std::size_t maximum_pending_submission_count = 1U,
           std::size_t lane_count = 1U, std::size_t queue_modulus = 0U,
           bool timing_enabled = false);

    SubmissionOwner(const SubmissionOwner &) = delete;
    SubmissionOwner &operator=(const SubmissionOwner &) = delete;
    ~SubmissionOwner();

    [[nodiscard]] Result<SubmissionTicket> submit(std::span<const VkCommandBuffer> command_buffers);
    [[nodiscard]] Result<SubmissionTicket> submit(std::size_t lane_index,
                                                  std::span<const VkCommandBuffer> command_buffers);
    [[nodiscard]] Result<SubmissionTicket> submit(std::size_t lane_index,
                                                  std::span<const VkCommandBuffer> command_buffers,
                                                  const SubmissionSynchronization &synchronization);
    [[nodiscard]] Result<SubmissionRetirement> retire(SubmissionTicket ticket);
    // Retires the largest signaled FIFO prefix of one lane without blocking.
    [[nodiscard]] Result<std::vector<SubmissionRetirement>> poll_prefix(std::size_t lane_index);
    // Retires the largest signaled FIFO prefix of every lane without
    // blocking. Retirements carry their lane.
    [[nodiscard]] Result<std::vector<SubmissionRetirement>> poll_all_lanes();
    // Blocks until at least one lane-front fence is signaled, then retires
    // every signaled FIFO prefix across all lanes.
    [[nodiscard]] Result<std::vector<SubmissionRetirement>> wait_for_any_lane_prefix();
    // Waits for one named submission on one lane and retires that submission
    // together with every earlier submission on the same lane. One physical
    // queue per lane preserves this prefix even when lanes alias.
    [[nodiscard]] Result<std::vector<SubmissionRetirement>>
    wait_for_prefix(std::size_t lane_index, SubmissionTicket ticket);

    // Public cleanup is bounded. A healthy pending submission receives one
    // normal timeout; a sticky owner returns its original failure immediately.
    // Final device-idle fallback is reserved for destruction preparation.
    [[nodiscard]] Result<void> drain();

    // Performs the destructor-only bounded wait and idle fallback explicitly
    // so an aggregate owner can decide whether every device resource is safe
    // to destroy or must instead be abandoned. The result is idempotent and
    // terminal: no further submission is permitted after this call.
    [[nodiscard]] DeviceResourceCleanupOutcome prepare_for_destruction() noexcept;

    [[nodiscard]] bool healthy() const noexcept { return !sticky_error_.has_value(); }
    [[nodiscard]] const SubmissionOwnerStatistics &statistics() const noexcept {
        return statistics_;
    }
    [[nodiscard]] bool last_submit_reached_queue() const noexcept {
        return last_submit_reached_queue_;
    }

  private:
    struct FenceSlot {
        VkFence fence{VK_NULL_HANDLE};
        SubmissionTicket ticket{};
        std::chrono::steady_clock::time_point submission_started_at{};
        bool pending{};
    };

    struct Lane {
        VkQueue queue{VK_NULL_HANDLE};
        std::uint32_t queue_family{};
        std::uint32_t queue_index{};
        std::vector<FenceSlot> fence_slots;
        std::deque<std::size_t> pending_slot_indices;
    };

    explicit SubmissionOwner(DeviceView device) noexcept;

    [[nodiscard]] Result<void> validate_owner_thread() const;
    [[nodiscard]] Result<void> require_healthy() const;
    [[nodiscard]] Result<std::vector<SubmissionRetirement>>
    retire_prefix(std::size_t lane_index, std::size_t inclusive_pending_position);
    [[nodiscard]] Result<std::size_t> find_pending_position(std::size_t lane_index,
                                                            SubmissionTicket ticket) const;
    [[nodiscard]] Result<void> wait_for_fences(std::span<const VkFence> fences, bool wait_for_all,
                                               std::uint64_t timeout_nanoseconds,
                                               bool preserve_existing_failure);
    void mark_prefix_retired(std::size_t lane_index, std::size_t retirement_count,
                             std::vector<SubmissionRetirement> &retirements);
    void store_failure(Error error) noexcept;

    DeviceView device_;
    std::thread::id owner_thread_;
    std::vector<Lane> lanes_;
    SubmissionOwnerStatistics statistics_;
    std::optional<Error> sticky_error_;
    bool last_submit_reached_queue_{};
    bool timing_enabled_{};
    DeviceResourceCleanupOutcome destruction_cleanup_outcome_{
        DeviceResourceCleanupOutcome::safe_to_destroy};
    bool destruction_prepared_{};
};

} // namespace crossvmaf::gpu
