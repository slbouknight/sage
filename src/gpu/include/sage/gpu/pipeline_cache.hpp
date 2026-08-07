#pragma once

#include <vulkan/vulkan.h>

#include <filesystem>

namespace sage::gpu {

class Device;

// Persisted driver-compiled pipeline blobs. Purely an optimization: a missing,
// corrupt, or foreign cache file is discarded and compilation proceeds normally.
class PipelineCache {
public:
    PipelineCache(const Device& device, std::filesystem::path path);
    ~PipelineCache();

    PipelineCache(const PipelineCache&) = delete;
    PipelineCache& operator=(const PipelineCache&) = delete;
    PipelineCache(PipelineCache&&) = delete;
    PipelineCache& operator=(PipelineCache&&) = delete;

    void save() const;

    [[nodiscard]] VkPipelineCache handle() const { return cache_; }

private:
    const Device& device_;
    std::filesystem::path path_;
    VkPipelineCache cache_ = VK_NULL_HANDLE;
};

}  // namespace sage::gpu