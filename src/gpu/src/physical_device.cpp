#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/physical_device.hpp>
#include <sage/gpu/vk_check.hpp>

#include <algorithm>
#include <cstring>
#include <string>

namespace sage::gpu {

namespace {

constexpr std::uint64_t k_discrete_score = 1000;
constexpr std::uint64_t k_integrated_score = 500;

bool supports_swapchain(VkPhysicalDevice device) {
    std::uint32_t count = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()));

    return std::any_of(
        extensions.begin(), extensions.end(), [](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
        });
}

// Mirrors the feature set Device enables. Checked here so an unsuitable GPU is
// rejected during selection with a clear reason, rather than failing opaquely
// inside vkCreateDevice.
bool supports_required_features(VkPhysicalDevice device, std::string& missing) {
    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.pNext = &features13;

    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.pNext = &features12;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;

    vkGetPhysicalDeviceFeatures2(device, &features2);

    const auto require = [&missing](VkBool32 supported, const char* name) {
        if (supported == VK_FALSE) {
            if (!missing.empty()) {
                missing += ", ";
            }
            missing += name;
        }
    };

    require(features12.timelineSemaphore, "timelineSemaphore");
    require(features13.synchronization2, "synchronization2");
    require(features13.dynamicRendering, "dynamicRendering");
    require(features12.bufferDeviceAddress, "bufferDeviceAddress");
    require(features2.features.shaderInt64, "shaderInt64");
    require(features2.features.samplerAnisotropy, "samplerAnisotropy");
    require(features12.descriptorIndexing, "descriptorIndexing");
    require(features12.shaderSampledImageArrayNonUniformIndexing,
            "shaderSampledImageArrayNonUniformIndexing");
    require(features12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound");
    require(features12.descriptorBindingVariableDescriptorCount,
            "descriptorBindingVariableDescriptorCount");
    require(features12.runtimeDescriptorArray, "runtimeDescriptorArray");
    require(features12.descriptorBindingSampledImageUpdateAfterBind,
            "descriptorBindingSampledImageUpdateAfterBind");
    require(features12.descriptorBindingStorageBufferUpdateAfterBind,
            "descriptorBindingStorageBufferUpdateAfterBind");
    require(features12.descriptorBindingUpdateUnusedWhilePending,
            "descriptorBindingUpdateUnusedWhilePending");
    require(features11.shaderDrawParameters, "shaderDrawParameters");
    require(features12.scalarBlockLayout, "scalarBlockLayout");

    return missing.empty();
}

}  // namespace

QueueFamilyIndices select_queue_families(const std::vector<VkQueueFamilyProperties>& families,
                                         const std::vector<bool>& present_support) {
    QueueFamilyIndices indices;

    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueCount == 0) {
            continue;
        }

        const bool graphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        const bool present = i < present_support.size() && present_support[i];

        // Prefer one family that does both, to avoid queue-family ownership
        // transfers this milestone has no need for.
        if (graphics && present && !indices.graphics.has_value()) {
            indices.graphics = i;
            indices.present = i;
        }
        if (graphics && !indices.graphics.has_value()) {
            indices.graphics = i;
        }
        if (present && !indices.present.has_value()) {
            indices.present = i;
        }
    }

    // A dedicated transfer family (transfer without graphics) is what M3's
    // async staging uploads want. Queue families are fixed at device creation,
    // so it is chosen now even though nothing uses it until then.
    for (std::uint32_t i = 0; i < families.size(); ++i) {
        if (families[i].queueCount == 0) {
            continue;
        }
        const bool transfer = (families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0;
        const bool graphics = (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
        if (transfer && !graphics) {
            indices.transfer = i;
            break;
        }
    }
    if (!indices.transfer.has_value()) {
        indices.transfer = indices.graphics;
    }

    return indices;
}

std::uint64_t device_local_memory_bytes(const VkPhysicalDeviceMemoryProperties& memory) {
    std::uint64_t total = 0;
    for (std::uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        if ((memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total += memory.memoryHeaps[i].size;
        }
    }
    return total;
}

std::uint64_t score_device(const VkPhysicalDeviceProperties& properties,
                           const VkPhysicalDeviceMemoryProperties& memory) {
    std::uint64_t score = 0;
    if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += k_discrete_score;
    } else if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        score += k_integrated_score;
    }
    // Tiebreak on VRAM, in whole GiB so heap size never dominates device type.
    score += device_local_memory_bytes(memory) / (1024ULL * 1024ULL * 1024ULL);
    return score;
}

PhysicalDeviceInfo select_physical_device(VkInstance instance, VkSurfaceKHR surface) {
    std::uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, nullptr));
    SAGE_VERIFY(count > 0, "No Vulkan-capable physical devices found");
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, devices.data()));

    PhysicalDeviceInfo best;
    std::uint64_t best_score = 0;
    bool found = false;

    for (VkPhysicalDevice device : devices) {
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(device, &properties);

        if (properties.apiVersion < VK_API_VERSION_1_3) {
            SAGE_LOG_WARN("Skipping {}: reports Vulkan {}.{}, need 1.3", properties.deviceName,
                          VK_API_VERSION_MAJOR(properties.apiVersion),
                          VK_API_VERSION_MINOR(properties.apiVersion));
            continue;
        }

        if (!supports_swapchain(device)) {
            SAGE_LOG_WARN("Skipping {}: missing {}", properties.deviceName,
                          VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            continue;
        }

        std::string missing;
        if (!supports_required_features(device, missing)) {
            SAGE_LOG_WARN("Skipping {}: missing required features: {}", properties.deviceName,
                          missing);
            continue;
        }

        std::uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

        std::vector<bool> present_support(family_count, false);
        for (std::uint32_t i = 0; i < family_count; ++i) {
            VkBool32 supported = VK_FALSE;
            VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &supported));
            present_support[i] = supported == VK_TRUE;
        }

        const QueueFamilyIndices indices = select_queue_families(families, present_support);
        if (!indices.complete()) {
            SAGE_LOG_WARN("Skipping {}: no graphics+present queue families", properties.deviceName);
            continue;
        }

        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(device, &memory);

        const std::uint64_t score = score_device(properties, memory);
        SAGE_LOG_DEBUG("Candidate {} (score {})", properties.deviceName, score);

        if (!found || score > best_score) {
            best.handle = device;
            best.queues = indices;
            best.properties = properties;
            best_score = score;
            found = true;
        }
    }

    SAGE_VERIFY(found, "No suitable physical device (see the skip reasons logged above)");

    SAGE_LOG_INFO("Selected GPU: {} (Vulkan {}.{}.{})", best.properties.deviceName,
                  VK_API_VERSION_MAJOR(best.properties.apiVersion),
                  VK_API_VERSION_MINOR(best.properties.apiVersion),
                  VK_API_VERSION_PATCH(best.properties.apiVersion));
    SAGE_LOG_INFO("Queue families: graphics={}, present={}, transfer={}", *best.queues.graphics,
                  *best.queues.present, *best.queues.transfer);

    return best;
}

}  // namespace sage::gpu
