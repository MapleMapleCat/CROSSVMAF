#include "video/vulkan_convert.hpp"
#include "gpu/buffer.hpp"
#include <algorithm>
#include <limits>
#include "gpu/error.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace crossvmaf::video {

Result<HardwareFrameSynchronization>
HardwareFrameSynchronization::create(const video::VulkanDecodedFrameAccess &reference,
                                     const video::VulkanDecodedFrameAccess &distorted,
                                     bool unified_video_image_layouts,
                                     std::span<const VkSemaphore> binary_signals) {
    HardwareFrameSynchronization synchronization;
    synchronization.advance_decoder_timeline_ = !unified_video_image_layouts;
    auto appended = synchronization.append(reference);
    if (appended) appended = synchronization.append(distorted);
    if (!appended) {
        return Result<HardwareFrameSynchronization>::failure(appended.error());
    }
    if (binary_signals.size() > 3U) {
        return Result<HardwareFrameSynchronization>::failure(
            {ErrorCode::invalid_argument,
             "hardware conversion supports at most three feature signals"});
    }
    for (const VkSemaphore semaphore : binary_signals) {
        if (semaphore == VK_NULL_HANDLE) {
            return Result<HardwareFrameSynchronization>::failure(
                {ErrorCode::invalid_argument,
                 "hardware conversion received a null feature semaphore"});
        }
        synchronization.signal_semaphores_[synchronization.signal_count_] = semaphore;
        synchronization.signal_values_[synchronization.signal_count_] = 0U;
        ++synchronization.signal_count_;
    }
    return Result<HardwareFrameSynchronization>::success(synchronization);
}

Result<void> HardwareFrameSynchronization::append(const video::VulkanDecodedFrameAccess &frame) {
    if (frame.image_count == 0U || frame.image_count > 2U || frame.semaphores == nullptr ||
        frame.semaphore_values == nullptr || frame.layouts == nullptr ||
        frame.access_flags == nullptr || frame.queue_families == nullptr) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "hardware conversion requires complete frame synchronization state"});
    }
    for (std::size_t image_index = 0U; image_index < frame.image_count; ++image_index) {
        if (!advance_decoder_timeline_ && frame.layouts[image_index] != VK_IMAGE_LAYOUT_GENERAL) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "wait-only hardware conversion requires FFmpeg GENERAL layouts"});
        }
        const VkSemaphore semaphore = frame.semaphores[image_index];
        const std::uint64_t wait_value = frame.semaphore_values[image_index];
        if (semaphore == VK_NULL_HANDLE ||
            (advance_decoder_timeline_ &&
             wait_value == std::numeric_limits<std::uint64_t>::max())) {
            return Result<void>::failure(
                {ErrorCode::invalid_state,
                 "hardware conversion received an invalid or exhausted timeline"});
        }

        std::size_t synchronization_index = 0U;
        while (synchronization_index < wait_count_ &&
               wait_semaphores_[synchronization_index] != semaphore) {
            ++synchronization_index;
        }
        if (synchronization_index == wait_count_) {
            // Two NV12/P010 sources expose at most four distinct images.
            wait_semaphores_[synchronization_index] = semaphore;
            wait_stages_[synchronization_index] = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
            ++wait_count_;
        }
        wait_values_[synchronization_index] =
            std::max(wait_values_[synchronization_index], wait_value);
        if (advance_decoder_timeline_) {
            signal_semaphores_[synchronization_index] = semaphore;
            signal_values_[synchronization_index] = wait_values_[synchronization_index] + 1U;
            signal_count_ = wait_count_;
        }
    }
    return Result<void>::success();
}

gpu::SubmissionSynchronization HardwareFrameSynchronization::submission() const noexcept {
    return {
        {wait_semaphores_.data(), wait_count_},     {wait_stages_.data(), wait_count_},
        {signal_semaphores_.data(), signal_count_}, {wait_values_.data(), wait_count_},
        {signal_values_.data(), signal_count_},
    };
}

void HardwareFrameSynchronization::mark_submitted(
    video::VulkanDecodedFrameAccess &frame, std::uint32_t compute_queue_family) const noexcept {
    if (!advance_decoder_timeline_) return;
    for (std::size_t image_index = 0U; image_index < frame.image_count; ++image_index) {
        for (std::size_t synchronization_index = 0U; synchronization_index < wait_count_;
             ++synchronization_index) {
            if (wait_semaphores_[synchronization_index] == frame.semaphores[image_index]) {
                frame.semaphore_values[image_index] = signal_values_[synchronization_index];
                break;
            }
        }
        frame.layouts[image_index] = VK_IMAGE_LAYOUT_GENERAL;
        frame.access_flags[image_index] = VK_ACCESS_2_SHADER_READ_BIT;
        if (frame.queue_families[image_index] != VK_QUEUE_FAMILY_IGNORED) {
            frame.queue_families[image_index] = compute_queue_family;
        }
    }
}

} // namespace crossvmaf::video

namespace crossvmaf::video {
namespace {

inline constexpr std::uint32_t vulkan_convert_source_image_binding_count = 4U;
inline constexpr std::uint32_t vulkan_convert_destination_buffer_binding_count = 6U;
inline constexpr std::uint32_t vulkan_convert_total_binding_count =
    vulkan_convert_source_image_binding_count + vulkan_convert_destination_buffer_binding_count;
inline constexpr std::uint32_t vulkan_convert_conversion_local_size_x = 64U;

struct vulkan_convert_ConversionParameters final {
    std::uint32_t luma_width{};
    std::uint32_t luma_height{};
    std::uint32_t luma_stride_words{};
    std::uint32_t chroma_width{};
    std::uint32_t chroma_height{};
    std::uint32_t chroma_stride_words{};
    std::uint32_t input_kind{};
};

static_assert(sizeof(vulkan_convert_ConversionParameters) == 7U * sizeof(std::uint32_t));

[[nodiscard]] std::uint32_t vulkan_convert_divide_rounding_up(std::uint32_t value,
                                                              std::uint32_t divisor) noexcept {
    return value / divisor + (value % divisor == 0U ? 0U : 1U);
}

[[nodiscard]] video::HardwareSurfaceFormat
vulkan_convert_expected_surface_format(const FrameDescriptor &descriptor) noexcept {
    return descriptor.bit_depth == 8U ? video::HardwareSurfaceFormat::nv12
                                      : video::HardwareSurfaceFormat::p010;
}

[[nodiscard]] bool vulkan_convert_source_frame_is_compatible(
    const video::VulkanDecodedFrameAccess &source, const FrameDescriptor &descriptor,
    std::uint32_t compute_queue_family, bool require_general_layout) noexcept {
    if (source.images == nullptr || source.layouts == nullptr || source.access_flags == nullptr ||
        source.semaphores == nullptr || source.semaphore_values == nullptr ||
        source.queue_families == nullptr || source.image_formats == nullptr ||
        source.image_count == 0U || source.image_count > 2U || source.width != descriptor.width ||
        source.height != descriptor.height ||
        source.surface_format != vulkan_convert_expected_surface_format(descriptor)) {
        return false;
    }
    for (std::size_t image_index = 0U; image_index < source.image_count; ++image_index) {
        if (source.images[image_index] == VK_NULL_HANDLE ||
            source.semaphores[image_index] == VK_NULL_HANDLE ||
            source.image_formats[image_index] == VK_FORMAT_UNDEFINED ||
            (require_general_layout && source.layouts[image_index] != VK_IMAGE_LAYOUT_GENERAL)) {
            return false;
        }
        const std::uint32_t owner = source.queue_families[image_index];
        if (owner != VK_QUEUE_FAMILY_IGNORED && owner != compute_queue_family) {
            // FFmpeg normally creates concurrent images when decode and
            // compute use different families. An exclusive foreign owner
            // would require a release barrier on FFmpeg's queue.
            return false;
        }
    }
    return true;
}

struct vulkan_convert_SourceViewDescription final {
    VkImage image{VK_NULL_HANDLE};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageAspectFlags aspect{};
};

[[nodiscard]] std::array<vulkan_convert_SourceViewDescription, 2U>
vulkan_convert_make_source_view_descriptions(const video::VulkanDecodedFrameAccess &source,
                                             const FrameDescriptor &descriptor) noexcept {
    const bool eight_bit = descriptor.bit_depth == 8U;
    const VkFormat luma_format = eight_bit ? VK_FORMAT_R8_UINT : VK_FORMAT_R16_UINT;
    const VkFormat chroma_format = eight_bit ? VK_FORMAT_R8G8_UINT : VK_FORMAT_R16G16_UINT;
    if (source.image_count == 1U) {
        return {{
            {source.images[0U], luma_format, VK_IMAGE_ASPECT_PLANE_0_BIT},
            {source.images[0U], chroma_format, VK_IMAGE_ASPECT_PLANE_1_BIT},
        }};
    }
    return {{
        {source.images[0U], luma_format, VK_IMAGE_ASPECT_COLOR_BIT},
        {source.images[1U], chroma_format, VK_IMAGE_ASPECT_COLOR_BIT},
    }};
}

} // namespace

SemiplanarInputConverter::SemiplanarInputConverter(gpu::DeviceView device,
                                                   FrameDescriptor descriptor) noexcept
    : device_(device), descriptor_(descriptor) {
}

SemiplanarInputConverter::~SemiplanarInputConverter() {
    if (!device_.resource_lifetime->safe_to_destroy()) return;
    for (FrameSlotResources &resources : frame_slots_) {
        destroy_source_views(resources);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyPipeline(device_.device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyPipelineLayout(device_.device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (descriptor_pool_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyDescriptorPool(device_.device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }
    if (descriptor_set_layout_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyDescriptorSetLayout(device_.device, descriptor_set_layout_,
                                                    nullptr);
        descriptor_set_layout_ = VK_NULL_HANDLE;
    }
}

Result<std::unique_ptr<SemiplanarInputConverter>>
SemiplanarInputConverter::create(gpu::DeviceView device, FrameDescriptor descriptor,
                                 std::size_t frame_slot_count,
                                 std::span<const std::uint8_t> shader_spirv) {
    if (!device.complete() || frame_slot_count == 0U ||
        frame_slot_count > std::numeric_limits<std::uint32_t>::max() || shader_spirv.empty() ||
        shader_spirv.size() % sizeof(std::uint32_t) != 0U ||
        descriptor.pixel_format != PixelFormat::yuv420p ||
        (descriptor.bit_depth != 8U && descriptor.bit_depth != 10U)) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            {ErrorCode::invalid_argument,
             "semiplanar conversion requires a valid 8/10-bit YUV420 device and shader"});
    }

    auto converter =
        std::unique_ptr<SemiplanarInputConverter>(new SemiplanarInputConverter(device, descriptor));
    const VolkDeviceTable &table = *device.table;

    std::array<VkDescriptorSetLayoutBinding, vulkan_convert_total_binding_count> layout_bindings{};
    for (std::uint32_t binding_index = 0U; binding_index < vulkan_convert_total_binding_count;
         ++binding_index) {
        VkDescriptorSetLayoutBinding &binding = layout_bindings[binding_index];
        binding.binding = binding_index;
        binding.descriptorType = binding_index < vulkan_convert_source_image_binding_count
                                     ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                     : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1U;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = vulkan_convert_total_binding_count;
    layout_info.pBindings = layout_bindings.data();
    VkResult result = table.vkCreateDescriptorSetLayout(device.device, &layout_info, nullptr,
                                                        &converter->descriptor_set_layout_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            gpu::make_error("vkCreateDescriptorSetLayout", result));
    }

    const std::uint32_t descriptor_set_count = static_cast<std::uint32_t>(frame_slot_count);
    const std::array<VkDescriptorPoolSize, 2U> pool_sizes{{
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         vulkan_convert_source_image_binding_count * descriptor_set_count},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         vulkan_convert_destination_buffer_binding_count * descriptor_set_count},
    }};
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = descriptor_set_count;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    result = table.vkCreateDescriptorPool(device.device, &pool_info, nullptr,
                                          &converter->descriptor_pool_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            gpu::make_error("vkCreateDescriptorPool", result));
    }

    converter->frame_slots_.resize(frame_slot_count);
    std::vector<VkDescriptorSetLayout> set_layouts(frame_slot_count,
                                                   converter->descriptor_set_layout_);
    std::vector<VkDescriptorSet> descriptor_sets(frame_slot_count);
    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = converter->descriptor_pool_;
    allocate_info.descriptorSetCount = descriptor_set_count;
    allocate_info.pSetLayouts = set_layouts.data();
    result = table.vkAllocateDescriptorSets(device.device, &allocate_info, descriptor_sets.data());
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            gpu::make_error("vkAllocateDescriptorSets", result));
    }
    for (std::size_t slot_index = 0U; slot_index < frame_slot_count; ++slot_index) {
        converter->frame_slots_[slot_index].descriptor_set = descriptor_sets[slot_index];
    }

    const VkPushConstantRange push_constant_range{
        VK_SHADER_STAGE_COMPUTE_BIT,
        0U,
        static_cast<std::uint32_t>(sizeof(vulkan_convert_ConversionParameters)),
    };
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1U;
    pipeline_layout_info.pSetLayouts = &converter->descriptor_set_layout_;
    pipeline_layout_info.pushConstantRangeCount = 1U;
    pipeline_layout_info.pPushConstantRanges = &push_constant_range;
    result = table.vkCreatePipelineLayout(device.device, &pipeline_layout_info, nullptr,
                                          &converter->pipeline_layout_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            gpu::make_error("vkCreatePipelineLayout", result));
    }

    std::vector<std::uint32_t> shader_words(shader_spirv.size() / sizeof(std::uint32_t));
    std::memcpy(shader_words.data(), shader_spirv.data(), shader_spirv.size());
    VkShaderModuleCreateInfo shader_module_info{};
    shader_module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_info.codeSize = shader_spirv.size();
    shader_module_info.pCode = shader_words.data();
    VkShaderModule shader_module = VK_NULL_HANDLE;
    result =
        table.vkCreateShaderModule(device.device, &shader_module_info, nullptr, &shader_module);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            gpu::make_error("vkCreateShaderModule", result));
    }

    VkPipelineShaderStageCreateInfo shader_stage{};
    shader_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage.module = shader_module;
    shader_stage.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage;
    pipeline_info.layout = converter->pipeline_layout_;
    result = table.vkCreateComputePipelines(device.device, device.pipeline_cache, 1U,
                                            &pipeline_info, nullptr, &converter->pipeline_);
    table.vkDestroyShaderModule(device.device, shader_module, nullptr);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<SemiplanarInputConverter>>::failure(
            gpu::make_error("vkCreateComputePipelines", result));
    }
    return Result<std::unique_ptr<SemiplanarInputConverter>>::success(std::move(converter));
}

void SemiplanarInputConverter::destroy_source_views(FrameSlotResources &resources) noexcept {
    for (VkImageView &image_view : resources.source_plane_views) {
        if (image_view == VK_NULL_HANDLE) continue;
        device_.table->vkDestroyImageView(device_.device, image_view, nullptr);
        image_view = VK_NULL_HANDLE;
    }
}

void SemiplanarInputConverter::retire_slot(std::size_t frame_slot_index) noexcept {
    if (frame_slot_index >= frame_slots_.size() || !device_.resource_lifetime->safe_to_destroy()) {
        return;
    }
    destroy_source_views(frame_slots_[frame_slot_index]);
}

Result<void>
SemiplanarInputConverter::create_source_views(FrameSlotResources &resources,
                                              const video::VulkanDecodedFrameAccess &reference,
                                              const video::VulkanDecodedFrameAccess &distorted) {
    destroy_source_views(resources);
    const auto reference_views =
        vulkan_convert_make_source_view_descriptions(reference, descriptor_);
    const auto distorted_views =
        vulkan_convert_make_source_view_descriptions(distorted, descriptor_);
    const std::array<vulkan_convert_SourceViewDescription, 4U> view_descriptions{{
        reference_views[0U],
        reference_views[1U],
        distorted_views[0U],
        distorted_views[1U],
    }};

    for (std::size_t view_index = 0U; view_index < view_descriptions.size(); ++view_index) {
        const vulkan_convert_SourceViewDescription &description = view_descriptions[view_index];
        VkImageViewUsageCreateInfo usage_info{};
        usage_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO;
        usage_info.usage = VK_IMAGE_USAGE_STORAGE_BIT;
        VkImageViewCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        create_info.pNext = &usage_info;
        create_info.image = description.image;
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = description.format;
        create_info.subresourceRange.aspectMask = description.aspect;
        create_info.subresourceRange.baseMipLevel = 0U;
        create_info.subresourceRange.levelCount = 1U;
        create_info.subresourceRange.baseArrayLayer = 0U;
        create_info.subresourceRange.layerCount = 1U;
        const VkResult result = device_.table->vkCreateImageView(
            device_.device, &create_info, nullptr, &resources.source_plane_views[view_index]);
        if (result != VK_SUCCESS) {
            destroy_source_views(resources);
            return Result<void>::failure(gpu::make_error("vkCreateImageView", result));
        }
    }
    return Result<void>::success();
}

Result<void> SemiplanarInputConverter::update_descriptor_set(FrameSlotResources &resources,
                                                             const UploadedFramePair &destination) {
    const std::array<const GpuPlaneView *, 6U> destination_planes{{
        &destination.reference.planes[0U],
        &destination.reference.planes[1U],
        &destination.reference.planes[2U],
        &destination.distorted.planes[0U],
        &destination.distorted.planes[1U],
        &destination.distorted.planes[2U],
    }};
    std::array<VkDescriptorImageInfo, 4U> image_infos{};
    std::array<VkDescriptorBufferInfo, 6U> buffer_infos{};
    std::array<VkWriteDescriptorSet, vulkan_convert_total_binding_count> writes{};
    for (std::size_t image_index = 0U; image_index < image_infos.size(); ++image_index) {
        image_infos[image_index].imageView = resources.source_plane_views[image_index];
        image_infos[image_index].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet &write = writes[image_index];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = resources.descriptor_set;
        write.dstBinding = static_cast<std::uint32_t>(image_index);
        write.descriptorCount = 1U;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &image_infos[image_index];
    }
    for (std::size_t buffer_index = 0U; buffer_index < buffer_infos.size(); ++buffer_index) {
        const GpuPlaneView &plane = *destination_planes[buffer_index];
        if (plane.buffer == nullptr ||
            (plane.buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0U ||
            plane.size_bytes == 0U) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "semiplanar conversion destination is not a storage buffer"});
        }
        buffer_infos[buffer_index] = {plane.buffer->handle(), plane.offset_bytes, plane.size_bytes};
        VkWriteDescriptorSet &write =
            writes[vulkan_convert_source_image_binding_count + buffer_index];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = resources.descriptor_set;
        write.dstBinding =
            vulkan_convert_source_image_binding_count + static_cast<std::uint32_t>(buffer_index);
        write.descriptorCount = 1U;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_infos[buffer_index];
    }
    device_.table->vkUpdateDescriptorSets(device_.device, static_cast<std::uint32_t>(writes.size()),
                                          writes.data(), 0U, nullptr);
    return Result<void>::success();
}

void SemiplanarInputConverter::record_source_barriers(
    VkCommandBuffer command_buffer, const video::VulkanDecodedFrameAccess &reference,
    const video::VulkanDecodedFrameAccess &distorted) const noexcept {
    std::array<VkImageMemoryBarrier, 4U> barriers{};
    std::uint32_t barrier_count = 0U;
    const auto append_frame_barriers = [&](const video::VulkanDecodedFrameAccess &source) {
        for (std::size_t image_index = 0U; image_index < source.image_count; ++image_index) {
            VkImageMemoryBarrier &barrier = barriers[barrier_count++];
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = 0U;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            barrier.oldLayout = source.layouts[image_index];
            barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = source.images[image_index];
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = 0U;
            barrier.subresourceRange.levelCount = 1U;
            barrier.subresourceRange.baseArrayLayer = 0U;
            barrier.subresourceRange.layerCount = 1U;
        }
    };
    append_frame_barriers(reference);
    append_frame_barriers(distorted);
    device_.table->vkCmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0U, 0U, nullptr, 0U,
                                        nullptr, barrier_count, barriers.data());
}

Result<void> SemiplanarInputConverter::record(std::size_t frame_slot_index,
                                              VkCommandBuffer command_buffer,
                                              const video::VulkanDecodedFrameAccess &reference,
                                              const video::VulkanDecodedFrameAccess &distorted,
                                              const UploadedFramePair &destination) {
    if (frame_slot_index >= frame_slots_.size() || command_buffer == VK_NULL_HANDLE ||
        !vulkan_convert_source_frame_is_compatible(reference, descriptor_, device_.queue_family,
                                                   device_.unified_video_image_layouts) ||
        !vulkan_convert_source_frame_is_compatible(distorted, descriptor_, device_.queue_family,
                                                   device_.unified_video_image_layouts)) {
        return Result<void>::failure({ErrorCode::unsupported_format,
                                      "semiplanar conversion received incompatible Vulkan frames"});
    }
    FrameSlotResources &resources = frame_slots_[frame_slot_index];
    const auto views_created = create_source_views(resources, reference, distorted);
    if (!views_created) return views_created;
    const auto descriptors_updated = update_descriptor_set(resources, destination);
    if (!descriptors_updated) return descriptors_updated;

    // Production devices use upstream FFmpeg's video layouts. Transitions and
    // timeline feedback remain one protocol; driver capabilities alone cannot
    // enable the optional GENERAL-only contract validated above.
    if (!device_.unified_video_image_layouts) {
        record_source_barriers(command_buffer, reference, distorted);
    }
    const std::uint32_t luma_stride_bytes = destination.reference.planes[0U].stride_bytes;
    const std::uint32_t chroma_stride_bytes = destination.reference.planes[1U].stride_bytes;
    if ((luma_stride_bytes & 3U) != 0U || (chroma_stride_bytes & 3U) != 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_state,
             "semiplanar conversion requires four-byte-aligned plane strides"});
    }
    vulkan_convert_ConversionParameters parameters{
        descriptor_.width,
        descriptor_.height,
        luma_stride_bytes / 4U,
        plane_width(descriptor_, 1U),
        plane_height(descriptor_, 1U),
        chroma_stride_bytes / 4U,
        0U,
    };
    device_.table->vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    device_.table->vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline_layout_, 0U, 1U, &resources.descriptor_set, 0U,
                                           nullptr);
    const std::uint32_t samples_per_word = descriptor_.bit_depth == 8U ? 4U : 2U;
    const std::uint32_t luma_word_count =
        vulkan_convert_divide_rounding_up(descriptor_.width, samples_per_word);
    const std::uint32_t chroma_word_count =
        vulkan_convert_divide_rounding_up(plane_width(descriptor_, 1U), samples_per_word);
    device_.table->vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                      0U, sizeof(parameters), &parameters);
    device_.table->vkCmdDispatch(
        command_buffer,
        vulkan_convert_divide_rounding_up(luma_word_count, vulkan_convert_conversion_local_size_x),
        descriptor_.height, 2U);
    parameters.input_kind = 1U;
    device_.table->vkCmdPushConstants(command_buffer, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                      0U, sizeof(parameters), &parameters);
    device_.table->vkCmdDispatch(command_buffer,
                                 vulkan_convert_divide_rounding_up(
                                     chroma_word_count, vulkan_convert_conversion_local_size_x),
                                 plane_height(descriptor_, 1U), 2U);
    return Result<void>::success();
}

} // namespace crossvmaf::video
