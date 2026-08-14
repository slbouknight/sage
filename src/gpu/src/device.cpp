#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/vk_check.hpp>

#include <algorithm>
#include <vector>

namespace sage::gpu {

Device::Device(const PhysicalDeviceInfo& info) : physical_device_(info.handle) {
    SAGE_VERIFY(info.handle != VK_NULL_HANDLE, "Device requires a selected physical device");
    SAGE_VERIFY(info.queues.complete(), "Device requires complete queue family indices");

    graphics_family_ = *info.queues.graphics;
    present_family_ = *info.queues.present;
    transfer_family_ = *info.queues.transfer;

    std::vector<std::uint32_t> unique_families{graphics_family_, present_family_, transfer_family_};
    std::sort(unique_families.begin(), unique_families.end());
    unique_families.erase(std::unique(unique_families.begin(), unique_families.end()),
                          unique_families.end());

    const float queue_priority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    queue_infos.reserve(unique_families.size());
    for (std::uint32_t family : unique_families) {
        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;
        queue_infos.push_back(queue_info);
    }

    // Features that are only *used* from M2 onward are still enabled here:
    // they can only be requested at device-creation time, so deferring them
    // would mean reopening this code later. See ADR 0005.
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.synchronization2 = VK_TRUE;
    features13.dynamicRendering = VK_TRUE;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;
    features12.timelineSemaphore = VK_TRUE;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    features12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    // Slang's "natural" layout for BDA-accessed structs  packs members C-style,
    //so a float3 can land at offset 12 and straddle a 16-byte boundary.
    // Standard block layout forbids that; scalar layout permits it.
    features12.scalarBlockLayout = VK_TRUE;

    // Slang lowers SV_VertexID to (VertexIndex - BaseVertex) to match HLSL
    // semantics, and reading BaseVertex needs the DrawParameters capability.
    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.pNext = &features12;
    features11.shaderDrawParameters = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;
    features2.features.shaderInt64 = VK_TRUE;

    const std::vector<const char*> device_extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.pNext = &features2;
    create_info.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
    create_info.pQueueCreateInfos = queue_infos.data();
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();

    VK_CHECK(vkCreateDevice(physical_device_, &create_info, nullptr, &device_));

    vkGetDeviceQueue(device_, graphics_family_, 0, &graphics_queue_);
    vkGetDeviceQueue(device_, present_family_, 0, &present_queue_);
    vkGetDeviceQueue(device_, transfer_family_, 0, &transfer_queue_);

    SAGE_LOG_INFO("Logical device created ({} unique queue famil{})", unique_families.size(),
                  unique_families.size() == 1 ? "y" : "ies");
}

Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
}

void Device::wait_idle() const {
    if (device_ != VK_NULL_HANDLE) {
        VK_CHECK(vkDeviceWaitIdle(device_));
    }
}

}  // namespace sage::gpu
