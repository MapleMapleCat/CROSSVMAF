#pragma once
#include "crossvmaf/error.hpp"
#include "gpu/device.hpp"
#include "gpu/submission.hpp"
#include <volk.h>
#include <cstdint>
#include <memory>
#include <span>

namespace crossvmaf::gpu {

class Buffer;

struct BufferMemoryBarrier final {
    const Buffer *buffer{};
    VkAccessFlags source_access{};
    VkAccessFlags destination_access{};
    VkDeviceSize offset{};
    VkDeviceSize size{VK_WHOLE_SIZE};
};

enum class CommandBufferState : std::uint8_t {
    retired,
    recording,
    executable,
    submitted,
    failed,
};

// Pure outer-state rollback used by synchronous feature conveniences. It
// deliberately clears pending state only after CommandContext proves that no
// submitted command can still reference the feature's resources.
class ConveniencePendingState final {
  public:
    [[nodiscard]] bool pending() const noexcept { return phase_ != Phase::idle; }
    [[nodiscard]] bool submission_observed() const noexcept {
        return phase_ == Phase::submission_observed;
    }

    void mark_recorded() noexcept { phase_ = Phase::recorded; }
    void mark_submission_observed() noexcept {
        if (phase_ == Phase::recorded) {
            phase_ = Phase::submission_observed;
        }
    }
    void clear() noexcept { phase_ = Phase::idle; }
    void rollback_after_cancel(CommandBufferState command_state) noexcept {
        if (phase_ == Phase::recorded &&
            can_rollback_uncommitted_operation(false, true, command_state)) {
            phase_ = Phase::idle;
        }
    }

    [[nodiscard]] static constexpr bool
    can_rollback_uncommitted_operation(bool submission_committed, bool has_command,
                                       CommandBufferState command_state) noexcept {
        return !submission_committed &&
               (!has_command || command_state == CommandBufferState::retired);
    }

  private:
    enum class Phase : std::uint8_t {
        idle,
        recorded,
        submission_observed,
    };

    Phase phase_{Phase::idle};
};

class CommandContext final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<CommandContext>> create(DeviceView device);

    CommandContext(const CommandContext &) = delete;
    CommandContext &operator=(const CommandContext &) = delete;
    ~CommandContext();

    [[nodiscard]] Result<VkCommandBuffer> begin();
    [[nodiscard]] Result<VkCommandBuffer> finish();
    [[nodiscard]] Result<void> mark_submitted(SubmissionTicket ticket);
    [[nodiscard]] Result<void> mark_retired(SubmissionTicket ticket);
    [[nodiscard]] Result<SubmissionRetirement> submit_finished_and_wait();
    [[nodiscard]] Result<void> submit_and_wait();
    void cancel() noexcept;

    [[nodiscard]] CommandBufferState state() const noexcept { return state_; }

    void memory_barrier(VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage,
                        VkAccessFlags source_access,
                        VkAccessFlags destination_access) const noexcept;
    void compute_write_to_indirect_read_barrier() const noexcept;
    void buffer_memory_barriers(VkPipelineStageFlags source_stage,
                                VkPipelineStageFlags destination_stage,
                                std::span<const BufferMemoryBarrier> barriers) const noexcept;
    void fill_buffer(const Buffer &buffer, std::uint32_t value = 0U, VkDeviceSize offset = 0U,
                     VkDeviceSize size = VK_WHOLE_SIZE) const noexcept;
    void update_buffer(const Buffer &buffer, std::span<const std::byte> data,
                       VkDeviceSize offset = 0U) const noexcept;
    void copy_buffer(const Buffer &source, const Buffer &destination, VkDeviceSize source_offset,
                     VkDeviceSize destination_offset, VkDeviceSize size) const noexcept;

  private:
    explicit CommandContext(DeviceView device) noexcept : device_(device) {}

    DeviceView device_;
    VkCommandPool command_pool_{VK_NULL_HANDLE};
    VkCommandBuffer command_buffer_{VK_NULL_HANDLE};
    CommandBufferState state_{CommandBufferState::retired};
    SubmissionTicket submitted_ticket_{};
};

class CommandRecordingGuard final {
  public:
    explicit CommandRecordingGuard(CommandContext &commands) noexcept : commands_(&commands) {}

    CommandRecordingGuard(const CommandRecordingGuard &) = delete;
    CommandRecordingGuard &operator=(const CommandRecordingGuard &) = delete;

    ~CommandRecordingGuard() {
        if (commands_ != nullptr) commands_->cancel();
    }

    void release() noexcept { commands_ = nullptr; }

  private:
    CommandContext *commands_{};
};

} // namespace crossvmaf::gpu
