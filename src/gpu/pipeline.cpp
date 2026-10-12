#include "gpu/pipeline.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <system_error>
#include <utility>
#include <vector>
#include "gpu/error.hpp"
#include <limits>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace crossvmaf::gpu {
namespace {

constexpr std::size_t pipeline_cache_pipeline_cache_header_size =
    sizeof(std::uint32_t) * 4U + VK_UUID_SIZE;
constexpr std::uintmax_t pipeline_cache_maximum_pipeline_cache_file_size =
    256ULL * 1024ULL * 1024ULL;

[[nodiscard]] std::uint32_t pipeline_cache_read_uint32(std::span<const std::uint8_t> bytes,
                                                       std::size_t offset) noexcept {
    std::uint32_t value = 0U;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

#if defined(_WIN32)
[[nodiscard]] std::optional<std::filesystem::path>
pipeline_cache_environment_path(const wchar_t *name) {
    wchar_t *value = nullptr;
    std::size_t length = 0U;
    const errno_t status = _wdupenv_s(&value, &length, name);
    if (status != 0 || value == nullptr || value[0] == L'\0') {
        std::free(value);
        return std::nullopt;
    }

    std::filesystem::path result(value);
    std::free(value);
    return result;
}
#else
[[nodiscard]] std::optional<std::filesystem::path>
pipeline_cache_environment_path(const char *name) {
    const char *const value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') return std::nullopt;
    return std::filesystem::path(value);
}
#endif

[[nodiscard]] bool pipeline_cache_trace_pipeline_cache() noexcept {
    static const bool enabled = [] {
#if defined(_WIN32)
        char *value = nullptr;
        std::size_t length = 0U;
        const errno_t status = _dupenv_s(&value, &length, "CROSSVMAF_VULKAN_TRACE");
        const bool result = status == 0 && value != nullptr && std::string_view(value) != "0";
        std::free(value);
        return result;
#else
        const char *const value = std::getenv("CROSSVMAF_VULKAN_TRACE");
        return value != nullptr && std::string_view(value) != "0";
#endif
    }();
    return enabled;
}

void pipeline_cache_trace_cache_event(const char *event, std::size_t byte_count) noexcept {
    if (!pipeline_cache_trace_pipeline_cache()) return;
    std::fprintf(stderr, "[crossvmaf-vulkan] pipeline_cache %s %zu bytes\n", event, byte_count);
}

[[nodiscard]] std::optional<std::filesystem::path> pipeline_cache_configured_cache_directory() {
#if defined(_WIN32)
    if (auto override_directory = pipeline_cache_environment_path(L"CROSSVMAF_VULKAN_CACHE_DIR")) {
        return *override_directory / L"v1";
    }
#else
    if (auto override_directory = pipeline_cache_environment_path("CROSSVMAF_VULKAN_CACHE_DIR")) {
        return *override_directory / "v1";
    }
#endif
    return std::nullopt;
}

[[nodiscard]] std::optional<std::filesystem::path> pipeline_cache_executable_cache_directory() {
#if defined(_WIN32)
    constexpr DWORD maximum_executable_path_length = 32768U;
    std::wstring executable_path(maximum_executable_path_length, L'\0');
    const DWORD character_count =
        ::GetModuleFileNameW(nullptr, executable_path.data(), maximum_executable_path_length);
    if (character_count == 0U || character_count >= maximum_executable_path_length) {
        return std::nullopt;
    }
    executable_path.resize(static_cast<std::size_t>(character_count));
    return std::filesystem::path(executable_path).parent_path() / L"cache" /
           L"vulkan-pipeline-cache" / L"v1";
#else
    std::error_code path_error;
    const std::filesystem::path executable_path =
        std::filesystem::read_symlink("/proc/self/exe", path_error);
    if (path_error || executable_path.empty()) return std::nullopt;
    return executable_path.parent_path() / "cache" / "vulkan-pipeline-cache" / "v1";
#endif
}

[[nodiscard]] std::filesystem::path
pipeline_cache_cache_file_path(const PipelineCacheIdentity &identity) {
    const std::string filename = pipeline_cache_internal::make_pipeline_cache_filename(identity);
    if (const auto configured_directory = pipeline_cache_configured_cache_directory()) {
        return *configured_directory / filename;
    }

    if (const auto executable_directory = pipeline_cache_executable_cache_directory()) {
        return *executable_directory / filename;
    }
    return {};
}

void pipeline_cache_remove_cache_file(const std::filesystem::path &cache_path) noexcept {
    if (cache_path.empty()) return;
    std::error_code ignored_error;
    static_cast<void>(std::filesystem::remove(cache_path, ignored_error));
}

[[nodiscard]] std::vector<std::uint8_t>
pipeline_cache_read_cache_data(const std::filesystem::path &cache_path,
                               const PipelineCacheIdentity &identity) {
    if (cache_path.empty()) return {};

    // Measure and read through one handle so a concurrent atomic replacement
    // cannot pair one file's size with another file's bytes.
    std::ifstream stream(cache_path, std::ios::binary | std::ios::ate);
    if (!stream) return {};
    const std::streamoff stream_size = stream.tellg();
    if (stream_size < 0 ||
        static_cast<std::uintmax_t>(stream_size) < pipeline_cache_pipeline_cache_header_size ||
        static_cast<std::uintmax_t>(stream_size) >
            pipeline_cache_maximum_pipeline_cache_file_size) {
        return {};
    }

    std::vector<std::uint8_t> cache_data(static_cast<std::size_t>(stream_size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(reinterpret_cast<char *>(cache_data.data()),
                     static_cast<std::streamsize>(cache_data.size()))) {
        return {};
    }

    if (!pipeline_cache_internal::has_compatible_pipeline_cache_header(cache_data, identity)) {
        pipeline_cache_remove_cache_file(cache_path);
        return {};
    }
    return cache_data;
}

[[nodiscard]] std::uint64_t pipeline_cache_process_id() noexcept {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(::GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(::getpid());
#endif
}

[[nodiscard]] std::filesystem::path
pipeline_cache_temporary_cache_path(const std::filesystem::path &cache_path) {
    static std::atomic<std::uint64_t> sequence{0U};
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();

    std::ostringstream suffix;
    suffix << ".tmp-" << pipeline_cache_process_id() << '-' << timestamp << '-'
           << sequence.fetch_add(1U, std::memory_order_relaxed);

    std::filesystem::path temporary_path = cache_path;
    temporary_path += std::filesystem::path(suffix.str()).native();
    return temporary_path;
}

[[nodiscard]] bool
pipeline_cache_replace_cache_file(const std::filesystem::path &temporary_path,
                                  const std::filesystem::path &cache_path) noexcept {
#if defined(_WIN32)
    return ::MoveFileExW(temporary_path.c_str(), cache_path.c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != FALSE;
#else
    std::error_code replace_error;
    std::filesystem::rename(temporary_path, cache_path, replace_error);
    return !replace_error;
#endif
}

[[nodiscard]] bool pipeline_cache_write_cache_data(const std::filesystem::path &cache_path,
                                                   std::span<const std::uint8_t> cache_data) {
    if (cache_path.empty() || cache_data.empty() ||
        cache_data.size() > pipeline_cache_maximum_pipeline_cache_file_size) {
        return false;
    }

    std::error_code directory_error;
    std::filesystem::create_directories(cache_path.parent_path(), directory_error);
    if (directory_error) return false;

    const std::filesystem::path temporary_path = pipeline_cache_temporary_cache_path(cache_path);
    std::ofstream stream(temporary_path, std::ios::binary | std::ios::trunc);
    if (!stream) return false;

    stream.write(reinterpret_cast<const char *>(cache_data.data()),
                 static_cast<std::streamsize>(cache_data.size()));
    stream.flush();
    const bool write_succeeded = stream.good();
    stream.close();
    if (!write_succeeded || !pipeline_cache_replace_cache_file(temporary_path, cache_path)) {
        pipeline_cache_remove_cache_file(temporary_path);
        return false;
    }
    return true;
}

[[nodiscard]] PipelineCacheIdentity
pipeline_cache_make_pipeline_cache_identity(VkPhysicalDevice physical_device,
                                            std::string_view shader_manifest_sha256) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    PipelineCacheIdentity identity{};
    identity.vendor_id = properties.vendorID;
    identity.device_id = properties.deviceID;
    identity.driver_version = properties.driverVersion;
    std::copy_n(properties.pipelineCacheUUID, VK_UUID_SIZE, identity.compatibility_uuid.begin());
    identity.shader_manifest_sha256 = shader_manifest_sha256;
    return identity;
}

} // namespace

namespace pipeline_cache_internal {

std::string make_pipeline_cache_filename(const PipelineCacheIdentity &identity) {
    std::ostringstream filename;
    filename << std::hex << std::nouppercase << std::setfill('0') << std::setw(8)
             << identity.vendor_id << '-' << std::setw(8) << identity.device_id << '-'
             << std::setw(8) << identity.driver_version << '-';
    for (const std::uint8_t byte : identity.compatibility_uuid) {
        filename << std::setw(2) << static_cast<unsigned int>(byte);
    }
    filename << '-' << identity.shader_manifest_sha256 << ".bin";
    return filename.str();
}

bool has_compatible_pipeline_cache_header(std::span<const std::uint8_t> cache_data,
                                          const PipelineCacheIdentity &identity) noexcept {
    if (cache_data.size() < pipeline_cache_pipeline_cache_header_size) return false;

    const std::uint32_t header_size = pipeline_cache_read_uint32(cache_data, 0U);
    const std::uint32_t header_version =
        pipeline_cache_read_uint32(cache_data, sizeof(std::uint32_t));
    const std::uint32_t vendor_id =
        pipeline_cache_read_uint32(cache_data, sizeof(std::uint32_t) * 2U);
    const std::uint32_t device_id =
        pipeline_cache_read_uint32(cache_data, sizeof(std::uint32_t) * 3U);

    const bool has_valid_header_size = header_size >= pipeline_cache_pipeline_cache_header_size &&
                                       header_size <= cache_data.size();
    const bool has_supported_version =
        header_version == static_cast<std::uint32_t>(VK_PIPELINE_CACHE_HEADER_VERSION_ONE);
    const bool has_matching_device =
        vendor_id == identity.vendor_id && device_id == identity.device_id;
    const bool has_matching_uuid =
        std::equal(identity.compatibility_uuid.begin(), identity.compatibility_uuid.end(),
                   cache_data.begin() + sizeof(std::uint32_t) * 4U);
    return has_valid_header_size && has_supported_version && has_matching_device &&
           has_matching_uuid;
}

} // namespace pipeline_cache_internal

PersistentPipelineCache::PersistentPipelineCache(VkDevice device, const VolkDeviceTable &table,
                                                 VkPipelineCache cache,
                                                 std::filesystem::path cache_path) noexcept
    : device_(device), table_(&table), cache_(cache), cache_path_(std::move(cache_path)) {
}

std::unique_ptr<PersistentPipelineCache>
PersistentPipelineCache::create(DeviceView device, std::string_view shader_manifest_sha256) {
    if (!device.complete() || device.table->vkCreatePipelineCache == nullptr) {
        return nullptr;
    }

    const PipelineCacheIdentity identity =
        pipeline_cache_make_pipeline_cache_identity(device.physical_device, shader_manifest_sha256);
    std::filesystem::path persistent_path;
    std::vector<std::uint8_t> initial_data;
    try {
        if (!shader_manifest_sha256.empty()) {
            persistent_path = pipeline_cache_cache_file_path(identity);
            initial_data = pipeline_cache_read_cache_data(persistent_path, identity);
            if (!initial_data.empty()) {
                pipeline_cache_trace_cache_event("loaded", initial_data.size());
            }
        }
    } catch (...) {
        persistent_path.clear();
        initial_data.clear();
    }

    VkPipelineCacheCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    create_info.initialDataSize = initial_data.size();
    create_info.pInitialData = initial_data.empty() ? nullptr : initial_data.data();

    VkPipelineCache cache = VK_NULL_HANDLE;
    VkResult result =
        device.table->vkCreatePipelineCache(device.device, &create_info, nullptr, &cache);
    if (result != VK_SUCCESS && !initial_data.empty()) {
        pipeline_cache_remove_cache_file(persistent_path);
        create_info.initialDataSize = 0U;
        create_info.pInitialData = nullptr;
        cache = VK_NULL_HANDLE;
        result = device.table->vkCreatePipelineCache(device.device, &create_info, nullptr, &cache);
    }
    if (result != VK_SUCCESS) return nullptr;

    try {
        return std::unique_ptr<PersistentPipelineCache>(new PersistentPipelineCache(
            device.device, *device.table, cache, std::move(persistent_path)));
    } catch (...) {
        device.table->vkDestroyPipelineCache(device.device, cache, nullptr);
        return nullptr;
    }
}

PersistentPipelineCache::~PersistentPipelineCache() {
    persist();
    if (cache_ != VK_NULL_HANDLE && table_ != nullptr &&
        table_->vkDestroyPipelineCache != nullptr) {
        table_->vkDestroyPipelineCache(device_, cache_, nullptr);
        cache_ = VK_NULL_HANDLE;
    }
}

void PersistentPipelineCache::persist() noexcept {
    if (cache_ == VK_NULL_HANDLE || cache_path_.empty() || table_ == nullptr ||
        table_->vkGetPipelineCacheData == nullptr) {
        return;
    }

    try {
        std::size_t cache_size = 0U;
        VkResult result = table_->vkGetPipelineCacheData(device_, cache_, &cache_size, nullptr);
        if (result != VK_SUCCESS || cache_size == 0U ||
            cache_size > pipeline_cache_maximum_pipeline_cache_file_size) {
            return;
        }

        std::vector<std::uint8_t> cache_data(cache_size);
        result = table_->vkGetPipelineCacheData(device_, cache_, &cache_size, cache_data.data());
        if (result != VK_SUCCESS || cache_size == 0U || cache_size > cache_data.size()) {
            return;
        }

        cache_data.resize(cache_size);
        const bool persisted = pipeline_cache_write_cache_data(cache_path_, cache_data);
        pipeline_cache_trace_cache_event(persisted ? "persisted" : "persistence_failed",
                                         cache_data.size());
    } catch (...) {
        // Pipeline-cache persistence is an optimization, never a requirement.
    }
}

} // namespace crossvmaf::gpu

namespace crossvmaf::gpu {

Result<std::unique_ptr<ComputePipeline>>
ComputePipeline::create(DeviceView device, std::span<const std::uint8_t> spirv,
                        std::uint32_t storage_binding_count, std::uint32_t push_constant_size,
                        std::uint32_t descriptor_set_count) {
    if (!device.complete()) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            {ErrorCode::backend_error,
             "compute pipeline creation received an incomplete Vulkan device"});
    }
    if (spirv.empty() || (spirv.size() % sizeof(std::uint32_t)) != 0U) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            {ErrorCode::invalid_argument,
             "SPIR-V bytecode must be a non-empty multiple of four bytes"});
    }
    if (storage_binding_count == 0U || descriptor_set_count == 0U) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            {ErrorCode::invalid_argument,
             "compute pipelines require storage bindings and descriptor sets"});
    }
    if (push_constant_size != 0U && (push_constant_size % sizeof(std::uint32_t)) != 0U) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            {ErrorCode::invalid_argument,
             "compute push constant size must be a multiple of four bytes"});
    }
    if (storage_binding_count > std::numeric_limits<std::uint32_t>::max() / descriptor_set_count) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            {ErrorCode::invalid_argument, "descriptor count overflows uint32"});
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device.physical_device, &properties);
    if (push_constant_size > properties.limits.maxPushConstantsSize) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            {ErrorCode::backend_error, "shader push constants exceed the Vulkan device limit"});
    }

    auto pipeline = std::unique_ptr<ComputePipeline>(new ComputePipeline(device));
    pipeline->storage_binding_count_ = storage_binding_count;
    pipeline->push_constant_size_ = push_constant_size;
    pipeline->min_storage_buffer_offset_alignment_ =
        std::max<VkDeviceSize>(1U, properties.limits.minStorageBufferOffsetAlignment);
    pipeline->max_storage_buffer_range_ = properties.limits.maxStorageBufferRange;
    const VolkDeviceTable &table = *device.table;

    std::vector<VkDescriptorSetLayoutBinding> layout_bindings(storage_binding_count);
    for (std::uint32_t index = 0; index < storage_binding_count; ++index) {
        auto &binding = layout_bindings[index];
        binding.binding = index;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = storage_binding_count;
    set_layout_info.pBindings = layout_bindings.data();
    VkResult result = table.vkCreateDescriptorSetLayout(device.device, &set_layout_info, nullptr,
                                                        &pipeline->descriptor_set_layout_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            make_error("vkCreateDescriptorSetLayout", result));
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = storage_binding_count * descriptor_set_count;
    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = descriptor_set_count;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    result = table.vkCreateDescriptorPool(device.device, &pool_info, nullptr,
                                          &pipeline->descriptor_pool_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            make_error("vkCreateDescriptorPool", result));
    }

    pipeline->descriptor_sets_.resize(descriptor_set_count);
    std::vector<VkDescriptorSetLayout> set_layouts(descriptor_set_count,
                                                   pipeline->descriptor_set_layout_);
    VkDescriptorSetAllocateInfo set_allocate_info{};
    set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    set_allocate_info.descriptorPool = pipeline->descriptor_pool_;
    set_allocate_info.descriptorSetCount = descriptor_set_count;
    set_allocate_info.pSetLayouts = set_layouts.data();
    result = table.vkAllocateDescriptorSets(device.device, &set_allocate_info,
                                            pipeline->descriptor_sets_.data());
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            make_error("vkAllocateDescriptorSets", result));
    }

    std::vector<std::uint32_t> words(spirv.size() / sizeof(std::uint32_t));
    std::memcpy(words.data(), spirv.data(), spirv.size());
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = spirv.size();
    shader_info.pCode = words.data();
    result =
        table.vkCreateShaderModule(device.device, &shader_info, nullptr, &pipeline->shader_module_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            make_error("vkCreateShaderModule", result));
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_range.offset = 0;
    push_range.size = push_constant_size;
    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &pipeline->descriptor_set_layout_;
    if (push_constant_size != 0U) {
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
    }
    result = table.vkCreatePipelineLayout(device.device, &pipeline_layout_info, nullptr,
                                          &pipeline->pipeline_layout_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            make_error("vkCreatePipelineLayout", result));
    }

    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = pipeline->shader_module_;
    stage_info.pName = "main";
    VkComputePipelineCreateInfo compute_info{};
    compute_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compute_info.stage = stage_info;
    compute_info.layout = pipeline->pipeline_layout_;
    result = table.vkCreateComputePipelines(device.device, device.pipeline_cache, 1, &compute_info,
                                            nullptr, &pipeline->pipeline_);
    if (result != VK_SUCCESS) {
        return Result<std::unique_ptr<ComputePipeline>>::failure(
            make_error("vkCreateComputePipelines", result));
    }
    // A shader module is no longer needed after pipeline creation. Releasing
    // it here avoids retaining a second copy of large integer-only shaders for
    // every extractor instance.
    table.vkDestroyShaderModule(device.device, pipeline->shader_module_, nullptr);
    pipeline->shader_module_ = VK_NULL_HANDLE;
    return Result<std::unique_ptr<ComputePipeline>>::success(std::move(pipeline));
}

ComputePipeline::~ComputePipeline() {
    if (!device_.resource_lifetime->safe_to_destroy()) return;

    if (pipeline_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyPipeline(device_.device, pipeline_, nullptr);
        pipeline_ = VK_NULL_HANDLE;
    }
    if (pipeline_layout_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyPipelineLayout(device_.device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (shader_module_ != VK_NULL_HANDLE) {
        device_.table->vkDestroyShaderModule(device_.device, shader_module_, nullptr);
        shader_module_ = VK_NULL_HANDLE;
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

Result<void> ComputePipeline::update_buffers(std::uint32_t descriptor_set_index,
                                             std::span<const BufferBinding> bindings) {
    if (descriptor_set_index >= descriptor_sets_.size()) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "descriptor set index is out of range"});
    }
    if (bindings.size() != storage_binding_count_) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "buffer binding count does not match the compute pipeline layout"});
    }

    std::vector<VkDescriptorBufferInfo> buffer_infos(bindings.size());
    std::vector<VkWriteDescriptorSet> writes(bindings.size());
    for (std::size_t index = 0; index < bindings.size(); ++index) {
        const BufferBinding &binding = bindings[index];
        if (binding.buffer == nullptr) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument, "compute buffer binding contains a null buffer"});
        }
        if ((binding.buffer->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) == 0U) {
            return Result<void>::failure({ErrorCode::invalid_argument,
                                          "compute buffer binding requires STORAGE_BUFFER usage"});
        }
        if (binding.offset >= binding.buffer->size()) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "compute buffer binding offset must be smaller than the buffer"});
        }
        if ((binding.offset % min_storage_buffer_offset_alignment_) != 0U) {
            return Result<void>::failure({ErrorCode::invalid_argument,
                                          "compute buffer binding offset violates "
                                          "minStorageBufferOffsetAlignment"});
        }
        const VkDeviceSize remaining_buffer_size = binding.buffer->size() - binding.offset;
        if (binding.range != VK_WHOLE_SIZE &&
            (binding.range == 0U || binding.range > remaining_buffer_size)) {
            return Result<void>::failure(
                {ErrorCode::invalid_argument,
                 "compute buffer binding range exceeds the logical buffer size"});
        }
        const VkDeviceSize effective_range =
            binding.range == VK_WHOLE_SIZE ? remaining_buffer_size : binding.range;
        if (effective_range > max_storage_buffer_range_) {
            return Result<void>::failure({ErrorCode::invalid_argument,
                                          "compute buffer binding effective range exceeds "
                                          "maxStorageBufferRange"});
        }
        VkDescriptorBufferInfo &buffer_info = buffer_infos[index];
        buffer_info.buffer = binding.buffer->handle();
        buffer_info.offset = binding.offset;
        buffer_info.range = binding.range;
        VkWriteDescriptorSet &write = writes[index];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = descriptor_sets_[descriptor_set_index];
        write.dstBinding = static_cast<std::uint32_t>(index);
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &buffer_info;
    }
    device_.table->vkUpdateDescriptorSets(device_.device, static_cast<std::uint32_t>(writes.size()),
                                          writes.data(), 0, nullptr);
    return Result<void>::success();
}

Result<void> ComputePipeline::record(VkCommandBuffer command_buffer,
                                     std::uint32_t descriptor_set_index,
                                     std::span<const std::byte> push_constants,
                                     DispatchSize dispatch) const {
    if (command_buffer == VK_NULL_HANDLE || descriptor_set_index >= descriptor_sets_.size() ||
        dispatch.x == 0U || dispatch.y == 0U || dispatch.z == 0U) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument, "invalid compute dispatch arguments"});
    }
    if (push_constants.size() != push_constant_size_) {
        return Result<void>::failure(
            {ErrorCode::invalid_argument,
             "push constant byte count does not match the pipeline layout"});
    }

    bind_dispatch_resources(command_buffer, descriptor_set_index, push_constants);
    device_.table->vkCmdDispatch(command_buffer, dispatch.x, dispatch.y, dispatch.z);
    return Result<void>::success();
}

void ComputePipeline::bind_dispatch_resources(
    VkCommandBuffer command_buffer, std::uint32_t descriptor_set_index,
    std::span<const std::byte> push_constants) const noexcept {
    device_.table->vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    const VkDescriptorSet descriptor_set = descriptor_sets_[descriptor_set_index];
    device_.table->vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline_layout_, 0, 1, &descriptor_set, 0, nullptr);
    if (!push_constants.empty()) {
        device_.table->vkCmdPushConstants(command_buffer, pipeline_layout_,
                                          VK_SHADER_STAGE_COMPUTE_BIT, 0, push_constant_size_,
                                          push_constants.data());
    }
}

} // namespace crossvmaf::gpu
