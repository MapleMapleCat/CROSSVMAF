#include "gpu/command.hpp"
#include "gpu/buffer.hpp"
#include "gpu/error.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <utility>

namespace crossvmaf::gpu {

Result<std::unique_ptr<CommandContext>> CommandContext::create(DeviceView device) {
    if (!device.complete()) {
        return Result<std::unique_ptr<CommandContext>>::failure(
            {ErrorCode::backend_error,
             "command context creation received an incomplete Vulkan device"});
    }
    auto context = std::unique_ptr<CommandContext>(new CommandContext(device));
    const VolkDeviceTable &table = *device.table;

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = device.queue_family;
    VkResult result =
        table.vkCreateCommandPool(device.device, &pool_info, nullptr, &context->command_pool_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<CommandContext>>::failure(
            make_error("vkCreateCommandPool", result));
    }

    VkCommandBufferAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation_info.commandPool = context->command_pool_;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = 1;
    result =
        table.vkAllocateCommandBuffers(device.device, &allocation_info, &context->command_buffer_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<CommandContext>>::failure(
            make_error("vkAllocateCommandBuffers", result));
    }

    return Result<std::unique_ptr<CommandContext>>::success(std::move(context));
}

CommandContext::~CommandContext() {
    if (command_pool_ != VK_NULL_HANDLE && device_.resource_lifetime->safe_to_destroy()) {
        device_.table->vkDestroyCommandPool(device_.device, command_pool_, nullptr);
    }
    command_pool_ = VK_NULL_HANDLE;
    command_buffer_ = VK_NULL_HANDLE;
}

Result<VkCommandBuffer> CommandContext::begin() {
    if (state_ != CommandBufferState::retired) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state,
             "Vulkan command buffer is not retired and cannot begin recording"});
    }
    VkResult result = device_.table->vkResetCommandPool(device_.device, command_pool_, 0U);
    if (result != VK_SUCCESS) {
        state_ = CommandBufferState::failed;
        return Result<VkCommandBuffer>::failure(make_error("vkResetCommandPool", result));
    }
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    result = device_.table->vkBeginCommandBuffer(command_buffer_, &begin_info);
    if (result != VK_SUCCESS) {
        state_ = CommandBufferState::failed;
        return Result<VkCommandBuffer>::failure(make_error("vkBeginCommandBuffer", result));
    }
    state_ = CommandBufferState::recording;
    return Result<VkCommandBuffer>::success(command_buffer_);
}

Result<VkCommandBuffer> CommandContext::finish() {
    if (state_ != CommandBufferState::recording) {
        return Result<VkCommandBuffer>::failure(
            {ErrorCode::invalid_state, "no Vulkan command buffer is recording"});
    }
    const VkResult result = device_.table->vkEndCommandBuffer(command_buffer_);
    if (result != VK_SUCCESS) {
        state_ = CommandBufferState::failed;
        return Result<VkCommandBuffer>::failure(make_error("vkEndCommandBuffer", result));
    }
    state_ = CommandBufferState::executable;
    return Result<VkCommandBuffer>::success(command_buffer_);
}

Result<void> CommandContext::mark_submitted(SubmissionTicket ticket) {
    if (state_ != CommandBufferState::executable || !ticket) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan command buffer is not executable at submission"});
    }
    submitted_ticket_ = ticket;
    state_ = CommandBufferState::submitted;
    return Result<void>::success();
}

Result<void> CommandContext::mark_retired(SubmissionTicket ticket) {
    if (state_ != CommandBufferState::submitted || !ticket ||
        ticket.serial != submitted_ticket_.serial) {
        return Result<void>::failure(
            {ErrorCode::invalid_state, "Vulkan command buffer retirement serial does not match"});
    }
    submitted_ticket_ = {};
    state_ = CommandBufferState::retired;
    return Result<void>::success();
}

Result<SubmissionRetirement> CommandContext::submit_finished_and_wait() {
    if (state_ != CommandBufferState::executable) {
        return Result<SubmissionRetirement>::failure(
            {ErrorCode::invalid_state, "Vulkan command buffer must be finished before submission"});
    }

    // Convenience submit-and-wait is off the session poll path. Host elapsed
    // time is required to validate GPU timestamp durations, so timing stays on.
    auto submission_owner = SubmissionOwner::create(device_, 1U, 1U, 0U, true);
    if (!submission_owner) {
        cancel();
        return Result<SubmissionRetirement>::failure(submission_owner.error());
    }
    const std::array command_buffers{command_buffer_};
    const auto submitted = submission_owner.value()->submit(command_buffers);
    if (!submitted) {
        if (submission_owner.value()->last_submit_reached_queue()) {
            state_ = CommandBufferState::failed;
        } else {
            // Fence reset and all earlier failures prove that no queue could
            // have observed this executable command buffer. Reset it so the
            // owning convenience wrapper can clear its outer pending state.
            cancel();
        }
        return Result<SubmissionRetirement>::failure(submitted.error());
    }
    const auto command_marked = mark_submitted(submitted.value());
    if (!command_marked) {
        static_cast<void>(submission_owner.value()->drain());
        state_ = CommandBufferState::failed;
        return Result<SubmissionRetirement>::failure(command_marked.error());
    }

    auto retirement = submission_owner.value()->retire(submitted.value());
    if (!retirement) {
        static_cast<void>(submission_owner.value()->drain());
        state_ = CommandBufferState::failed;
        return Result<SubmissionRetirement>::failure(retirement.error());
    }
    const auto command_retired = mark_retired(submitted.value());
    if (!command_retired) {
        state_ = CommandBufferState::failed;
        return Result<SubmissionRetirement>::failure(command_retired.error());
    }
    return retirement;
}

Result<void> CommandContext::submit_and_wait() {
    if (state_ == CommandBufferState::recording) {
        const auto finished = finish();
        if (!finished) return Result<void>::failure(finished.error());
    }
    const auto retired = submit_finished_and_wait();
    if (!retired) return Result<void>::failure(retired.error());
    return Result<void>::success();
}

void CommandContext::cancel() noexcept {
    if (state_ != CommandBufferState::recording && state_ != CommandBufferState::executable) {
        return;
    }
    if (state_ == CommandBufferState::recording) {
        static_cast<void>(device_.table->vkEndCommandBuffer(command_buffer_));
    }
    const VkResult result = device_.table->vkResetCommandPool(device_.device, command_pool_, 0U);
    state_ = result == VK_SUCCESS ? CommandBufferState::retired : CommandBufferState::failed;
}

void CommandContext::memory_barrier(VkPipelineStageFlags source_stage,
                                    VkPipelineStageFlags destination_stage,
                                    VkAccessFlags source_access,
                                    VkAccessFlags destination_access) const noexcept {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = source_access;
    barrier.dstAccessMask = destination_access;
    device_.table->vkCmdPipelineBarrier(command_buffer_, source_stage, destination_stage, 0, 1,
                                        &barrier, 0, nullptr, 0, nullptr);
}

void CommandContext::compute_write_to_indirect_read_barrier() const noexcept {
    memory_barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                   VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT);
}

void CommandContext::buffer_memory_barriers(
    VkPipelineStageFlags source_stage, VkPipelineStageFlags destination_stage,
    std::span<const BufferMemoryBarrier> barriers) const noexcept {
    for (const BufferMemoryBarrier &description : barriers) {
        if (description.buffer == nullptr) continue;

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = description.source_access;
        barrier.dstAccessMask = description.destination_access;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = description.buffer->handle();
        barrier.offset = description.offset;
        barrier.size = description.size;
        device_.table->vkCmdPipelineBarrier(command_buffer_, source_stage, destination_stage, 0U,
                                            0U, nullptr, 1U, &barrier, 0U, nullptr);
    }
}

void CommandContext::fill_buffer(const Buffer &buffer, std::uint32_t value, VkDeviceSize offset,
                                 VkDeviceSize size) const noexcept {
    device_.table->vkCmdFillBuffer(command_buffer_, buffer.handle(), offset, size, value);
}

void CommandContext::update_buffer(const Buffer &buffer, std::span<const std::byte> data,
                                   VkDeviceSize offset) const noexcept {
    constexpr VkDeviceSize maximum_update_bytes = 65536U;
    VkDeviceSize remaining = static_cast<VkDeviceSize>(data.size());
    VkDeviceSize written = 0U;
    while (remaining >= 4U) {
        VkDeviceSize chunk = std::min(remaining, maximum_update_bytes);
        chunk -= chunk % 4U;
        if (chunk == 0U) break;
        device_.table->vkCmdUpdateBuffer(command_buffer_, buffer.handle(), offset + written, chunk,
                                         data.data() + static_cast<std::size_t>(written));
        written += chunk;
        remaining -= chunk;
    }
}

void CommandContext::copy_buffer(const Buffer &source, const Buffer &destination,
                                 VkDeviceSize source_offset, VkDeviceSize destination_offset,
                                 VkDeviceSize size) const noexcept {
    const VkBufferCopy region{source_offset, destination_offset, size};
    device_.table->vkCmdCopyBuffer(command_buffer_, source.handle(), destination.handle(), 1U,
                                   &region);
}

} // namespace crossvmaf::gpu
