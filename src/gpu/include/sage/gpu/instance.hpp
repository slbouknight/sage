#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace sage::gpu {

class Instance {
public:
    Instance(const std::string& app_name, std::vector<const char*> required_extensions);
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;
    Instance(Instance&&) = delete;
    Instance& operator=(Instance&&) = delete;

    [[nodiscard]] VkInstance handle() const { return instance_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT destroy_debug_messenger_ = nullptr;
};

}  // namespace sage::gpu
