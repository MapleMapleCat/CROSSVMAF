#pragma once
#include "gpu/device.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include "gpu/buffer.hpp"
#include "crossvmaf/error.hpp"
#include <volk.h>
#include <cstddef>
#include <vector>

namespace crossvmaf::gpu {

struct PipelineCacheIdentity {
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint32_t driver_version{};
    std::array<std::uint8_t, VK_UUID_SIZE> compatibility_uuid{};
    std::string_view shader_manifest_sha256;
};

namespace pipeline_cache_internal {

[[nodiscard]] std::string make_pipeline_cache_filename(const PipelineCacheIdentity &identity);

[[nodiscard]] bool
has_compatible_pipeline_cache_header(std::span<const std::uint8_t> cache_data,
                                     const PipelineCacheIdentity &identity) noexcept;

} // namespace pipeline_cache_internal

// Owns the device-wide cache shared by every ComputePipeline. Persistence is
// best-effort: an unavailable directory, invalid file, or driver rejection
// leaves only the in-memory cache and never prevents Vulkan execution. Cache
// files are never redirected to a user-profile cache directory.
class PersistentPipelineCache final {
  public:
    [[nodiscard]] static std::unique_ptr<PersistentPipelineCache>
    create(DeviceView device, std::string_view shader_manifest_sha256 = {});

    PersistentPipelineCache(const PersistentPipelineCache &) = delete;
    PersistentPipelineCache &operator=(const PersistentPipelineCache &) = delete;
    ~PersistentPipelineCache();

    [[nodiscard]] VkPipelineCache handle() const noexcept { return cache_; }

  private:
    PersistentPipelineCache(VkDevice device, const VolkDeviceTable &table, VkPipelineCache cache,
                            std::filesystem::path cache_path) noexcept;

    void persist() noexcept;

    VkDevice device_{VK_NULL_HANDLE};
    const VolkDeviceTable *table_{};
    VkPipelineCache cache_{VK_NULL_HANDLE};
    std::filesystem::path cache_path_;
};

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

struct BufferBinding {
    const Buffer *buffer{};
    VkDeviceSize offset{};
    VkDeviceSize range{VK_WHOLE_SIZE};
};

struct DispatchSize {
    std::uint32_t x{1};
    std::uint32_t y{1};
    std::uint32_t z{1};
};

class ComputePipeline final {
  public:
    [[nodiscard]] static Result<std::unique_ptr<ComputePipeline>>
    create(DeviceView device, std::span<const std::uint8_t> spirv,
           std::uint32_t storage_binding_count, std::uint32_t push_constant_size,
           std::uint32_t descriptor_set_count = 1);

    ComputePipeline(const ComputePipeline &) = delete;
    ComputePipeline &operator=(const ComputePipeline &) = delete;
    ~ComputePipeline();

    [[nodiscard]] Result<void> update_buffers(std::uint32_t descriptor_set_index,
                                              std::span<const BufferBinding> bindings);
    [[nodiscard]] Result<void> record(VkCommandBuffer command_buffer,
                                      std::uint32_t descriptor_set_index,
                                      std::span<const std::byte> push_constants,
                                      DispatchSize dispatch) const;

  private:
    explicit ComputePipeline(DeviceView device) noexcept : device_(device) {}

    void bind_dispatch_resources(VkCommandBuffer command_buffer, std::uint32_t descriptor_set_index,
                                 std::span<const std::byte> push_constants) const noexcept;

    DeviceView device_;
    std::uint32_t storage_binding_count_{};
    std::uint32_t push_constant_size_{};
    VkDeviceSize min_storage_buffer_offset_alignment_{1U};
    VkDeviceSize max_storage_buffer_range_{};
    VkDescriptorSetLayout descriptor_set_layout_{VK_NULL_HANDLE};
    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> descriptor_sets_;
    VkShaderModule shader_module_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline pipeline_{VK_NULL_HANDLE};
};

} // namespace crossvmaf::gpu
