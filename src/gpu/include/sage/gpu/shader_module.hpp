#pragma once

#include <vulkan/vulkan.h>

#include <filesystem>

namespace sage::gpu {

class Device;

// SPIR-V loaded from disk. Short-lived by design: a pipeline consumes the
// handle at creation time, after which the module can be destroyed.
class ShaderModule {
public:
    ShaderModule(const Device& device, const std::filesystem::path& spirv_path);
    ~ShaderModule();

    ShaderModule(const ShaderModule&) = delete;
    ShaderModule& operator=(const ShaderModule&) = delete;
    ShaderModule(ShaderModule&&) = delete;
    ShaderModule& operator=(ShaderModule&&) = delete;

    [[nodiscard]] VkShaderModule handle() const { return module_; }

private:
    const Device& device_;
    VkShaderModule module_ = VK_NULL_HANDLE;
};

}  // namespace sage::gpu