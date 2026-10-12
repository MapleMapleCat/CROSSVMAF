#include "gpu/submission.hpp"
#include <algorithm>
#include <string>
#include "gpu/error.hpp"
#include <array>
#include <cassert>
#include <limits>
#include <utility>

namespace crossvmaf::gpu {
namespace {

[[nodiscard]] Error streaming_task_ring_core_feature_error(std::string_view feature_name,
                                                           std::string_view message) {
    return {ErrorCode::invalid_state, std::string(feature_name) + " " + std::string(message)};
}

} // namespace

void StreamingTaskRingCore::initialize(std::uint32_t capacity) {
    capacity_ = capacity;
    failed_ = false;
    tasks_.assign(capacity, TaskRecord{});
    ready_queue_entries_.clear();
    ready_queue_entries_.reserve(capacity);
    pending_wave_tasks_.clear();
}

std::uint32_t StreamingTaskRingCore::free_slot_count() const noexcept {
    return static_cast<std::uint32_t>(
        std::count_if(tasks_.begin(), tasks_.end(),
                      [](const TaskRecord &task) { return task.state == TaskState::free; }));
}

std::uint32_t StreamingTaskRingCore::prefix_pending_task_count() const noexcept {
    return static_cast<std::uint32_t>(
        std::count_if(tasks_.begin(), tasks_.end(), [](const TaskRecord &task) {
            return task.state == TaskState::prefix_pending;
        }));
}

std::uint32_t StreamingTaskRingCore::ready_task_count() const noexcept {
    return static_cast<std::uint32_t>(ready_queue_entries_.size());
}

Result<std::optional<StreamingTaskHandle>>
StreamingTaskRingCore::reserve(std::uint64_t frame_index, std::string_view feature_name) {
    if (failed_) {
        return Result<std::optional<StreamingTaskHandle>>::failure(
            streaming_task_ring_core_feature_error(feature_name,
                                                   "task ring cannot reserve after failure"));
    }
    for (std::uint32_t slot = 0U; slot < capacity_; ++slot) {
        TaskRecord &task = tasks_[slot];
        if (task.state != TaskState::free) continue;
        ++task.generation;
        if (task.generation == 0U) ++task.generation;
        task.frame_index = frame_index;
        task.state = TaskState::prefix_pending;
        return Result<std::optional<StreamingTaskHandle>>::success(
            StreamingTaskHandle{slot, task.generation});
    }
    return Result<std::optional<StreamingTaskHandle>>::success(std::nullopt);
}

Result<void> StreamingTaskRingCore::cancel_prefix(StreamingTaskHandle task,
                                                  std::string_view feature_name) {
    if (!task || task.slot >= tasks_.size() || tasks_[task.slot].generation != task.generation ||
        tasks_[task.slot].state != TaskState::prefix_pending) {
        return Result<void>::failure(streaming_task_ring_core_feature_error(
            feature_name, "cannot cancel a stale or published prefix task"));
    }
    tasks_[task.slot].frame_index = 0U;
    tasks_[task.slot].state = TaskState::free;
    return Result<void>::success();
}

Result<void> StreamingTaskRingCore::publish_ready(StreamingTaskHandle task,
                                                  std::string_view feature_name) {
    if (failed_ || !task || task.slot >= tasks_.size() ||
        tasks_[task.slot].generation != task.generation ||
        tasks_[task.slot].state != TaskState::prefix_pending || ready_task_count() >= capacity_) {
        return Result<void>::failure(streaming_task_ring_core_feature_error(
            feature_name, "cannot publish a stale or overflowing task"));
    }
    ready_queue_entries_.push_back(task);
    tasks_[task.slot].state = TaskState::ready;
    return Result<void>::success();
}

Result<void> StreamingTaskRingCore::snapshot_ready_wave(std::string_view feature_name) {
    const std::uint32_t snapshot_task_count = ready_task_count();
    if (failed_ || snapshot_task_count == 0U || snapshot_task_count > capacity_) {
        return Result<void>::failure(streaming_task_ring_core_feature_error(
            feature_name, "streaming wave is not ready to record"));
    }

    pending_wave_tasks_.clear();
    pending_wave_tasks_.reserve(snapshot_task_count);
    for (std::uint32_t ready_index = 0U; ready_index < snapshot_task_count; ++ready_index) {
        const StreamingTaskHandle published_task = ready_queue_entries_[ready_index];
        if (!published_task || published_task.slot >= tasks_.size()) {
            return Result<void>::failure(streaming_task_ring_core_feature_error(
                feature_name, "ready queue contains an invalid task"));
        }
        const TaskRecord &task = tasks_[published_task.slot];
        if (task.state != TaskState::ready || task.generation != published_task.generation) {
            return Result<void>::failure(streaming_task_ring_core_feature_error(
                feature_name, "ready queue contains a stale task generation"));
        }
        pending_wave_tasks_.push_back(published_task);
    }
    return Result<void>::success();
}

void StreamingTaskRingCore::mark_wave_recorded() noexcept {
    for (StreamingTaskHandle task : pending_wave_tasks_) {
        tasks_[task.slot].state = TaskState::wave;
    }
}

void StreamingTaskRingCore::complete_retired_wave() noexcept {
    const std::uint32_t retired_task_count = static_cast<std::uint32_t>(pending_wave_tasks_.size());
    for (StreamingTaskHandle completed_task : pending_wave_tasks_) {
        TaskRecord &task = tasks_[completed_task.slot];
        task.frame_index = 0U;
        task.state = TaskState::free;
    }
    ready_queue_entries_.erase(ready_queue_entries_.begin(),
                               ready_queue_entries_.begin() + retired_task_count);
    pending_wave_tasks_.clear();
}

void StreamingTaskRingCore::fail_retired_wave() noexcept {
    for (StreamingTaskHandle task_handle : pending_wave_tasks_) {
        if (task_handle.slot >= tasks_.size()) continue;
        TaskRecord &task = tasks_[task_handle.slot];
        if (task.generation != task_handle.generation) continue;
        task.frame_index = 0U;
        task.state = TaskState::free;
    }
    ready_queue_entries_.clear();
    pending_wave_tasks_.clear();
    failed_ = true;
}

void StreamingTaskRingCore::restore_unsubmitted_wave() noexcept {
    for (StreamingTaskHandle task : pending_wave_tasks_) {
        if (task.slot < tasks_.size() && tasks_[task.slot].generation == task.generation &&
            tasks_[task.slot].state == TaskState::wave) {
            tasks_[task.slot].state = TaskState::ready;
        }
    }
    pending_wave_tasks_.clear();
}

void StreamingTaskRingCore::cancel_after_device_drain() noexcept {
    for (TaskRecord &task : tasks_) {
        task.frame_index = 0U;
        task.state = TaskState::free;
    }
    ready_queue_entries_.clear();
    pending_wave_tasks_.clear();
    failed_ = true;
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {
namespace {

void submission_increment_counter(std::uint64_t &counter) noexcept {
    if (counter != std::numeric_limits<std::uint64_t>::max()) {
        ++counter;
    }
}

void submission_add_to_counter(std::uint64_t &counter, std::uint64_t value) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - counter) {
        counter = std::numeric_limits<std::uint64_t>::max();
        return;
    }
    counter += value;
}

[[nodiscard]] Error submission_invalid_submission_state(std::string message) {
    return {ErrorCode::invalid_state, std::move(message)};
}

class submission_ScopedDeviceQueueLock final {
  public:
    submission_ScopedDeviceQueueLock(DeviceQueueSynchronization synchronization,
                                     std::uint32_t queue_family, std::uint32_t queue_index) noexcept
        : synchronization_(synchronization), queue_family_(queue_family),
          queue_index_(queue_index) {
        if (synchronization_.enabled()) {
            synchronization_.lock(synchronization_.opaque, queue_family_, queue_index_);
            locked_ = true;
        }
    }

    submission_ScopedDeviceQueueLock(const submission_ScopedDeviceQueueLock &) = delete;
    submission_ScopedDeviceQueueLock &operator=(const submission_ScopedDeviceQueueLock &) = delete;

    ~submission_ScopedDeviceQueueLock() {
        if (locked_) {
            synchronization_.unlock(synchronization_.opaque, queue_family_, queue_index_);
        }
    }

  private:
    DeviceQueueSynchronization synchronization_;
    std::uint32_t queue_family_{};
    std::uint32_t queue_index_{};
    bool locked_{};
};

} // namespace

SubmissionOwner::SubmissionOwner(DeviceView device) noexcept
    : device_(device), owner_thread_(std::this_thread::get_id()) {
}

Result<std::unique_ptr<SubmissionOwner>>
SubmissionOwner::create(DeviceView device, std::size_t maximum_pending_submission_count,
                        std::size_t lane_count, std::size_t queue_modulus, bool timing_enabled) {
    if (!device.complete() || maximum_pending_submission_count == 0U || lane_count == 0U) {
        return Result<std::unique_ptr<SubmissionOwner>>::failure(
            {ErrorCode::backend_error,
             "submission owner creation received an incomplete Vulkan device, "
             "an empty fence bank, or zero lanes"});
    }

    auto owner = std::unique_ptr<SubmissionOwner>(new SubmissionOwner(device));
    owner->timing_enabled_ = timing_enabled;
    const std::size_t effective_modulus =
        queue_modulus == 0U
            ? device.lane_queue_count
            : std::min(queue_modulus, static_cast<std::size_t>(device.lane_queue_count));
    owner->lanes_.resize(lane_count);
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (std::size_t lane_index = 0U; lane_index < lane_count; ++lane_index) {
        Lane &lane = owner->lanes_[lane_index];
        lane.queue = device.queue_for_lane(
            effective_modulus == 0U ? lane_index : lane_index % effective_modulus);
        lane.queue_family = device.queue_family;
        lane.queue_index = device.queue_index_for_lane(
            effective_modulus == 0U ? lane_index : lane_index % effective_modulus);
        lane.fence_slots.resize(maximum_pending_submission_count);
        for (FenceSlot &fence_slot : lane.fence_slots) {
            const VkResult result =
                device.table->vkCreateFence(device.device, &fence_info, nullptr, &fence_slot.fence);
            if (result != VK_SUCCESS) {
                return Result<std::unique_ptr<SubmissionOwner>>::failure(
                    make_error("vkCreateFence", result));
            }
        }
    }

    return Result<std::unique_ptr<SubmissionOwner>>::success(std::move(owner));
}

SubmissionOwner::~SubmissionOwner() {
    const DeviceResourceCleanupOutcome cleanup_outcome = prepare_for_destruction();
    if (cleanup_outcome != DeviceResourceCleanupOutcome::safe_to_destroy) {
        return;
    }
    for (Lane &lane : lanes_) {
        for (FenceSlot &fence_slot : lane.fence_slots) {
            if (fence_slot.fence == VK_NULL_HANDLE) continue;
            device_.table->vkDestroyFence(device_.device, fence_slot.fence, nullptr);
            fence_slot.fence = VK_NULL_HANDLE;
        }
    }
}

Result<void> SubmissionOwner::validate_owner_thread() const {
    const bool is_owner_thread = owner_thread_ == std::this_thread::get_id();
    assert(is_owner_thread && "SubmissionOwner must only be driven by its owning thread");
    if (!is_owner_thread) {
        return Result<void>::failure(submission_invalid_submission_state(
            "Vulkan submission owner was called from a non-owner thread"));
    }
    return Result<void>::success();
}

Result<void> SubmissionOwner::require_healthy() const {
    if (destruction_prepared_) {
        return Result<void>::failure(submission_invalid_submission_state(
            "Vulkan submission owner has already prepared for destruction"));
    }
    if (sticky_error_.has_value()) {
        return Result<void>::failure(*sticky_error_);
    }
    return Result<void>::success();
}

void SubmissionOwner::store_failure(Error error) noexcept {
    if (!sticky_error_.has_value()) {
        sticky_error_.emplace(std::move(error));
    }
}

Result<SubmissionTicket> SubmissionOwner::submit(std::span<const VkCommandBuffer> command_buffers) {
    return submit(0U, command_buffers, {});
}

Result<SubmissionTicket> SubmissionOwner::submit(std::size_t lane_index,
                                                 std::span<const VkCommandBuffer> command_buffers) {
    return submit(lane_index, command_buffers, {});
}

Result<SubmissionTicket> SubmissionOwner::submit(std::size_t lane_index,
                                                 std::span<const VkCommandBuffer> command_buffers,
                                                 const SubmissionSynchronization &synchronization) {
    const auto owner_thread_validation = validate_owner_thread();
    if (!owner_thread_validation) {
        return Result<SubmissionTicket>::failure(owner_thread_validation.error());
    }
    last_submit_reached_queue_ = false;
    const auto health_validation = require_healthy();
    if (!health_validation) {
        return Result<SubmissionTicket>::failure(health_validation.error());
    }
    if (lane_index >= lanes_.size()) {
        return Result<SubmissionTicket>::failure(submission_invalid_submission_state(
            "Vulkan submission lane index exceeds the lane count"));
    }
    if (command_buffers.empty()) {
        return Result<SubmissionTicket>::failure(submission_invalid_submission_state(
            "Vulkan submission requires at least one command buffer"));
    }
    if (command_buffers.size() >
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        return Result<SubmissionTicket>::failure(
            {ErrorCode::invalid_argument, "Vulkan submission command-buffer count exceeds uint32"});
    }
    if (std::any_of(
            command_buffers.begin(), command_buffers.end(),
            [](VkCommandBuffer command_buffer) { return command_buffer == VK_NULL_HANDLE; })) {
        return Result<SubmissionTicket>::failure(
            {ErrorCode::invalid_argument, "Vulkan submission received a null command buffer"});
    }
    if (synchronization.wait_semaphores.size() != synchronization.wait_stage_masks.size()) {
        return Result<SubmissionTicket>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan submission wait semaphores and stage masks differ in size"});
    }
    const bool uses_timeline_values =
        !synchronization.wait_values.empty() || !synchronization.signal_values.empty();
    if (uses_timeline_values &&
        (synchronization.wait_values.size() != synchronization.wait_semaphores.size() ||
         synchronization.signal_values.size() != synchronization.signal_semaphores.size())) {
        return Result<SubmissionTicket>::failure(
            {ErrorCode::invalid_argument,
             "Vulkan timeline values must match all wait and signal semaphores"});
    }
    if (synchronization.wait_semaphores.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        synchronization.signal_semaphores.size() >
            static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) ||
        std::any_of(synchronization.wait_semaphores.begin(), synchronization.wait_semaphores.end(),
                    [](VkSemaphore semaphore) { return semaphore == VK_NULL_HANDLE; }) ||
        std::any_of(synchronization.signal_semaphores.begin(),
                    synchronization.signal_semaphores.end(),
                    [](VkSemaphore semaphore) { return semaphore == VK_NULL_HANDLE; })) {
        return Result<SubmissionTicket>::failure(
            {ErrorCode::invalid_argument, "Vulkan submission semaphore configuration is invalid"});
    }

    Lane &lane = lanes_[lane_index];
    const auto available_slot =
        std::find_if(lane.fence_slots.begin(), lane.fence_slots.end(),
                     [](const FenceSlot &fence_slot) { return !fence_slot.pending; });
    if (available_slot == lane.fence_slots.end()) {
        return Result<SubmissionTicket>::failure(
            submission_invalid_submission_state("Vulkan submission fence bank is full"));
    }
    if (statistics_.last_attempted_serial == std::numeric_limits<std::uint64_t>::max()) {
        return Result<SubmissionTicket>::failure(
            submission_invalid_submission_state("Vulkan submission serial space is exhausted"));
    }
    const SubmissionTicket ticket{++statistics_.last_attempted_serial};
    const std::size_t fence_slot_index =
        static_cast<std::size_t>(available_slot - lane.fence_slots.begin());
    FenceSlot &fence_slot = *available_slot;

    VkTimelineSemaphoreSubmitInfo timeline_info{};
    if (uses_timeline_values) {
        timeline_info.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        timeline_info.waitSemaphoreValueCount =
            static_cast<std::uint32_t>(synchronization.wait_values.size());
        timeline_info.pWaitSemaphoreValues = synchronization.wait_values.data();
        timeline_info.signalSemaphoreValueCount =
            static_cast<std::uint32_t>(synchronization.signal_values.size());
        timeline_info.pSignalSemaphoreValues = synchronization.signal_values.data();
    }
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = uses_timeline_values ? &timeline_info : nullptr;
    submit_info.waitSemaphoreCount =
        static_cast<std::uint32_t>(synchronization.wait_semaphores.size());
    submit_info.pWaitSemaphores = synchronization.wait_semaphores.data();
    submit_info.pWaitDstStageMask = synchronization.wait_stage_masks.data();
    submit_info.commandBufferCount = static_cast<std::uint32_t>(command_buffers.size());
    submit_info.pCommandBuffers = command_buffers.data();
    submit_info.signalSemaphoreCount =
        static_cast<std::uint32_t>(synchronization.signal_semaphores.size());
    submit_info.pSignalSemaphores = synchronization.signal_semaphores.data();
    if (timing_enabled_) {
        fence_slot.submission_started_at = std::chrono::steady_clock::now();
    }

    // A signaled fence remains untouched while commands are recorded. Reset is
    // deliberately adjacent to the queue submission that consumes the fence.
    VkResult result = device_.table->vkResetFences(device_.device, 1U, &fence_slot.fence);
    if (result != VK_SUCCESS) {
        store_failure(make_error("vkResetFences", result));
        return Result<SubmissionTicket>::failure(*sticky_error_);
    }

    last_submit_reached_queue_ = true;
    submission_increment_counter(statistics_.queue_submit_call_count);
    {
        submission_ScopedDeviceQueueLock queue_lock(device_.queue_synchronization,
                                                    lane.queue_family, lane.queue_index);
        result = device_.table->vkQueueSubmit(lane.queue, 1U, &submit_info, fence_slot.fence);
    }
    if (result != VK_SUCCESS) {
        if (result == VK_ERROR_DEVICE_LOST) {
            // Device loss does not prove that the queue ignored the submit.
            // Keep the fence and ticket pending so destruction must establish
            // quiescence before any referenced device child is destroyed.
            fence_slot.ticket = ticket;
            fence_slot.pending = true;
            lane.pending_slot_indices.push_back(fence_slot_index);
        }
        store_failure(make_error("vkQueueSubmit", result));
        return Result<SubmissionTicket>::failure(*sticky_error_);
    }

    fence_slot.ticket = ticket;
    fence_slot.pending = true;
    lane.pending_slot_indices.push_back(fence_slot_index);
    statistics_.last_submitted_serial = ticket.serial;
    submission_increment_counter(statistics_.successful_submission_count);
    submission_add_to_counter(statistics_.submitted_command_buffer_count,
                              static_cast<std::uint64_t>(command_buffers.size()));
    return Result<SubmissionTicket>::success(ticket);
}

Result<void> SubmissionOwner::wait_for_fences(std::span<const VkFence> fences, bool wait_for_all,
                                              std::uint64_t timeout_nanoseconds,
                                              bool preserve_existing_failure) {
    if (fences.empty()) return Result<void>::success();
    submission_increment_counter(statistics_.fence_wait_call_count);
    const VkResult result = device_.table->vkWaitForFences(
        device_.device, static_cast<std::uint32_t>(fences.size()), fences.data(),
        wait_for_all ? VK_TRUE : VK_FALSE, timeout_nanoseconds);
    if (result != VK_SUCCESS) {
        if (!preserve_existing_failure || !sticky_error_.has_value()) {
            Error error =
                result == VK_TIMEOUT
                    ? Error{ErrorCode::backend_error, "vkWaitForFences timed out after 10 seconds; "
                                                      "the submission owner is permanently failed"}
                    : make_error("vkWaitForFences", result);
            store_failure(std::move(error));
        }
        return Result<void>::failure(*sticky_error_);
    }

    return Result<void>::success();
}

Result<std::size_t> SubmissionOwner::find_pending_position(std::size_t lane_index,
                                                           SubmissionTicket ticket) const {
    if (lane_index >= lanes_.size()) {
        return Result<std::size_t>::failure(submission_invalid_submission_state(
            "Vulkan retirement lane index exceeds the lane count"));
    }
    if (!ticket) {
        return Result<std::size_t>::failure(submission_invalid_submission_state(
            "Vulkan retirement requires a valid submission ticket"));
    }
    const Lane &lane = lanes_[lane_index];
    for (std::size_t pending_position = 0U; pending_position < lane.pending_slot_indices.size();
         ++pending_position) {
        const FenceSlot &fence_slot = lane.fence_slots[lane.pending_slot_indices[pending_position]];
        if (fence_slot.ticket.serial == ticket.serial) {
            return Result<std::size_t>::success(pending_position);
        }
    }
    return Result<std::size_t>::failure(submission_invalid_submission_state(
        "Vulkan retirement ticket does not name a pending serial"));
}

void SubmissionOwner::mark_prefix_retired(std::size_t lane_index, std::size_t retirement_count,
                                          std::vector<SubmissionRetirement> &retirements) {
    Lane &lane = lanes_[lane_index];
    const auto retirement_time = timing_enabled_ ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
    for (std::size_t retired_count = 0U; retired_count < retirement_count; ++retired_count) {
        const std::size_t fence_slot_index = lane.pending_slot_indices.front();
        lane.pending_slot_indices.pop_front();
        FenceSlot &fence_slot = lane.fence_slots[fence_slot_index];
        retirements.push_back({fence_slot.ticket,
                               timing_enabled_
                                   ? std::chrono::duration_cast<std::chrono::nanoseconds>(
                                         retirement_time - fence_slot.submission_started_at)
                                   : std::chrono::nanoseconds{},
                               static_cast<std::uint32_t>(lane_index)});
        statistics_.last_retired_serial = fence_slot.ticket.serial;
        fence_slot.ticket = {};
        fence_slot.pending = false;
    }
}

Result<std::vector<SubmissionRetirement>>
SubmissionOwner::retire_prefix(std::size_t lane_index, std::size_t inclusive_pending_position) {
    Lane &lane = lanes_[lane_index];
    if (inclusive_pending_position >= lane.pending_slot_indices.size()) {
        return Result<std::vector<SubmissionRetirement>>::failure(
            submission_invalid_submission_state(
                "Vulkan retirement prefix exceeds pending submissions"));
    }
    const FenceSlot &target_slot =
        lane.fence_slots[lane.pending_slot_indices[inclusive_pending_position]];
    const std::array target_fence{target_slot.fence};
    const auto waited =
        wait_for_fences(target_fence, true, submission_wait_timeout_nanoseconds, false);
    if (!waited) {
        return Result<std::vector<SubmissionRetirement>>::failure(waited.error());
    }
    std::vector<SubmissionRetirement> retirements;
    retirements.reserve(inclusive_pending_position + 1U);
    mark_prefix_retired(lane_index, inclusive_pending_position + 1U, retirements);
    return Result<std::vector<SubmissionRetirement>>::success(std::move(retirements));
}

Result<SubmissionRetirement> SubmissionOwner::retire(SubmissionTicket ticket) {
    const auto owner_thread_validation = validate_owner_thread();
    if (!owner_thread_validation) {
        return Result<SubmissionRetirement>::failure(owner_thread_validation.error());
    }
    const auto health_validation = require_healthy();
    if (!health_validation) {
        return Result<SubmissionRetirement>::failure(health_validation.error());
    }
    const auto pending_position = find_pending_position(0U, ticket);
    if (!pending_position) {
        return Result<SubmissionRetirement>::failure(pending_position.error());
    }
    auto retirements = retire_prefix(0U, pending_position.value());
    if (!retirements) {
        return Result<SubmissionRetirement>::failure(retirements.error());
    }
    return Result<SubmissionRetirement>::success(retirements.value().back());
}

Result<std::vector<SubmissionRetirement>> SubmissionOwner::poll_prefix(std::size_t lane_index) {
    const auto owner_thread_validation = validate_owner_thread();
    if (!owner_thread_validation) {
        return Result<std::vector<SubmissionRetirement>>::failure(owner_thread_validation.error());
    }
    const auto health_validation = require_healthy();
    if (!health_validation) {
        return Result<std::vector<SubmissionRetirement>>::failure(health_validation.error());
    }
    if (lane_index >= lanes_.size()) {
        return Result<std::vector<SubmissionRetirement>>::failure(
            submission_invalid_submission_state("Vulkan poll lane index exceeds the lane count"));
    }

    Lane &lane = lanes_[lane_index];
    std::size_t signaled_count = 0U;
    for (const std::size_t fence_slot_index : lane.pending_slot_indices) {
        const VkResult status = device_.table->vkGetFenceStatus(
            device_.device, lane.fence_slots[fence_slot_index].fence);
        if (status == VK_NOT_READY) break;
        if (status != VK_SUCCESS) {
            store_failure(make_error("vkGetFenceStatus", status));
            return Result<std::vector<SubmissionRetirement>>::failure(*sticky_error_);
        }
        ++signaled_count;
    }

    std::vector<SubmissionRetirement> retirements;
    retirements.reserve(signaled_count);
    mark_prefix_retired(lane_index, signaled_count, retirements);
    return Result<std::vector<SubmissionRetirement>>::success(std::move(retirements));
}

Result<std::vector<SubmissionRetirement>> SubmissionOwner::poll_all_lanes() {
    std::vector<SubmissionRetirement> retirements;
    for (std::size_t lane_index = 0U; lane_index < lanes_.size(); ++lane_index) {
        auto lane_retirements = poll_prefix(lane_index);
        if (!lane_retirements) {
            return Result<std::vector<SubmissionRetirement>>::failure(lane_retirements.error());
        }
        retirements.insert(retirements.end(),
                           std::make_move_iterator(lane_retirements.value().begin()),
                           std::make_move_iterator(lane_retirements.value().end()));
    }
    return Result<std::vector<SubmissionRetirement>>::success(std::move(retirements));
}

Result<std::vector<SubmissionRetirement>> SubmissionOwner::wait_for_any_lane_prefix() {
    const auto owner_thread_validation = validate_owner_thread();
    if (!owner_thread_validation) {
        return Result<std::vector<SubmissionRetirement>>::failure(owner_thread_validation.error());
    }
    const auto health_validation = require_healthy();
    if (!health_validation) {
        return Result<std::vector<SubmissionRetirement>>::failure(health_validation.error());
    }

    std::vector<VkFence> lane_front_fences;
    lane_front_fences.reserve(lanes_.size());
    for (const Lane &lane : lanes_) {
        if (lane.pending_slot_indices.empty()) continue;
        lane_front_fences.push_back(lane.fence_slots[lane.pending_slot_indices.front()].fence);
    }
    if (lane_front_fences.empty()) {
        return Result<std::vector<SubmissionRetirement>>::success({});
    }

    const auto waited =
        wait_for_fences(lane_front_fences, false, submission_wait_timeout_nanoseconds, false);
    if (!waited) {
        return Result<std::vector<SubmissionRetirement>>::failure(waited.error());
    }
    return poll_all_lanes();
}

Result<std::vector<SubmissionRetirement>>
SubmissionOwner::wait_for_prefix(std::size_t lane_index, SubmissionTicket ticket) {
    const auto owner_thread_validation = validate_owner_thread();
    if (!owner_thread_validation) {
        return Result<std::vector<SubmissionRetirement>>::failure(owner_thread_validation.error());
    }
    const auto health_validation = require_healthy();
    if (!health_validation) {
        return Result<std::vector<SubmissionRetirement>>::failure(health_validation.error());
    }
    const auto pending_position = find_pending_position(lane_index, ticket);
    if (!pending_position) {
        return Result<std::vector<SubmissionRetirement>>::failure(pending_position.error());
    }
    return retire_prefix(lane_index, pending_position.value());
}

Result<void> SubmissionOwner::drain() {
    const auto owner_thread_validation = validate_owner_thread();
    if (!owner_thread_validation) return owner_thread_validation;

    const auto health_validation = require_healthy();
    if (!health_validation) return health_validation;
    for (std::size_t lane_index = 0U; lane_index < lanes_.size(); ++lane_index) {
        Lane &lane = lanes_[lane_index];
        if (lane.pending_slot_indices.empty()) continue;
        const SubmissionTicket final_ticket =
            lane.fence_slots[lane.pending_slot_indices.back()].ticket;
        const auto retirements = wait_for_prefix(lane_index, final_ticket);
        if (!retirements) return Result<void>::failure(retirements.error());
    }
    return Result<void>::success();
}

DeviceResourceCleanupOutcome SubmissionOwner::prepare_for_destruction() noexcept {
    if (destruction_prepared_) return destruction_cleanup_outcome_;
    destruction_prepared_ = true;

    if (!device_.resource_lifetime->safe_to_destroy()) {
        destruction_cleanup_outcome_ = DeviceResourceCleanupOutcome::must_abandon_device_resources;
        return destruction_cleanup_outcome_;
    }
    bool any_pending = false;
    for (const Lane &lane : lanes_) {
        if (!lane.pending_slot_indices.empty()) {
            any_pending = true;
            break;
        }
    }
    if (!any_pending) return destruction_cleanup_outcome_;

    const bool is_owner_thread = owner_thread_ == std::this_thread::get_id();
    assert(is_owner_thread && "SubmissionOwner must be destroyed by its owning thread");
    if (!is_owner_thread) {
        device_.resource_lifetime->mark_must_abandon_device_resources();
        destruction_cleanup_outcome_ = DeviceResourceCleanupOutcome::must_abandon_device_resources;
        return destruction_cleanup_outcome_;
    }

    try {
        std::vector<VkFence> pending_fences;
        for (const Lane &lane : lanes_) {
            pending_fences.reserve(pending_fences.size() + lane.pending_slot_indices.size());
            for (const std::size_t fence_slot_index : lane.pending_slot_indices) {
                pending_fences.push_back(lane.fence_slots[fence_slot_index].fence);
            }
        }
        const auto waited =
            wait_for_fences(pending_fences, true, submission_wait_timeout_nanoseconds, true);
        if (waited) {
            std::vector<SubmissionRetirement> ignored_retirements;
            for (std::size_t lane_index = 0U; lane_index < lanes_.size(); ++lane_index) {
                mark_prefix_retired(lane_index, lanes_[lane_index].pending_slot_indices.size(),
                                    ignored_retirements);
            }
            return destruction_cleanup_outcome_;
        }
    } catch (...) {
        // Cleanup still gets one allocation-free device-idle proof attempt.
    }

    // This is an exceptional cleanup fallback only. Normal frame processing
    // and healthy destruction never use vkDeviceWaitIdle.
    const VkResult idle_result = device_.table->vkDeviceWaitIdle != nullptr
                                     ? device_.table->vkDeviceWaitIdle(device_.device)
                                     : VK_ERROR_INITIALIZATION_FAILED;
    if (idle_result == VK_SUCCESS || idle_result == VK_ERROR_DEVICE_LOST) {
        if (idle_result == VK_ERROR_DEVICE_LOST && !sticky_error_.has_value()) {
            try {
                store_failure(make_error("vkDeviceWaitIdle", idle_result));
            } catch (...) {
            }
        }
        for (Lane &lane : lanes_) {
            lane.pending_slot_indices.clear();
            for (FenceSlot &fence_slot : lane.fence_slots) {
                fence_slot.pending = false;
            }
        }
        return destruction_cleanup_outcome_;
    }

    // OOM and implementation-specific errors cannot prove that submitted work
    // is quiescent. Permanently poison this Device before any RAII wrapper can
    // destroy a potentially referenced device child.
    device_.resource_lifetime->mark_must_abandon_device_resources();
    destruction_cleanup_outcome_ = DeviceResourceCleanupOutcome::must_abandon_device_resources;
    if (!sticky_error_.has_value()) {
        try {
            store_failure(make_error("vkDeviceWaitIdle", idle_result));
        } catch (...) {
        }
    }
    return destruction_cleanup_outcome_;
}

} // namespace crossvmaf::gpu
