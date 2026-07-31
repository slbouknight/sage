#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/vk_check.hpp>

#include <algorithm>
#include <cstring>

namespace sage::gpu {

namespace {

#ifdef SAGE_VULKAN_VALIDATION
constexpr const char* k_validation_layer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                              VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                              const VkDebugUtilsMessengerCallbackDataEXT* data,
                                              void* /*user_data*/) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        SAGE_LOG_ERROR("[vulkan] {}", data->pMessage);
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        SAGE_LOG_WARN("[vulkan] {}", data->pMessage);
    } else {
        SAGE_LOG_DEBUG("[vulkan] {}", data->pMessage);
    }
    // Always VK_FALSE: the callback reports, it never aborts the offending call.
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT make_debug_messenger_info() {
    VkDebugUtilsMessengerCreateInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    // VERBOSE is deliberately excluded: it is dominated by loader chatter
    // (ICD discovery, device list copies) that drowns out real diagnostics.
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debug_callback;
    return info;
}

bool validation_layer_available() {
    std::uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));

    return std::any_of(layers.begin(), layers.end(), [](const VkLayerProperties& layer) {
        return std::strcmp(layer.layerName, k_validation_layer) == 0;
    });
}
#endif

}  // namespace

Instance::Instance(const std::string& app_name, std::vector<const char*> required_extensions) {
    std::uint32_t instance_version = 0;
    VK_CHECK(vkEnumerateInstanceVersion(&instance_version));
    SAGE_VERIFY(instance_version >= VK_API_VERSION_1_3,
                "Vulkan 1.3 or newer is required (loader reports an older version)");

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = app_name.c_str();
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "sage";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;

    std::vector<const char*> layers;

#ifdef SAGE_VULKAN_VALIDATION
    const bool validation = validation_layer_available();
    if (validation) {
        layers.push_back(k_validation_layer);
        required_extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else {
        SAGE_LOG_WARN(
            "Validation layers requested but {} is not available. Is VK_ADD_LAYER_PATH set? "
            "(source the Vulkan SDK's setup-env.sh)",
            k_validation_layer);
    }

    // Chained so the messenger also covers vkCreateInstance/vkDestroyInstance
    // themselves, which the standalone messenger below cannot observe.
    VkDebugUtilsMessengerCreateInfoEXT debug_info = make_debug_messenger_info();
    if (validation) {
        create_info.pNext = &debug_info;
    }
#endif

    create_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(required_extensions.size());
    create_info.ppEnabledExtensionNames = required_extensions.data();

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &instance_));

    for (const char* extension : required_extensions) {
        SAGE_LOG_DEBUG("Enabled instance extension: {}", extension);
    }

#ifdef SAGE_VULKAN_VALIDATION
    if (validation) {
        auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        destroy_debug_messenger_ = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        SAGE_VERIFY(create_messenger != nullptr && destroy_debug_messenger_ != nullptr,
                    "VK_EXT_debug_utils entry points could not be resolved");
        VK_CHECK(create_messenger(instance_, &debug_info, nullptr, &debug_messenger_));
        SAGE_LOG_INFO("Vulkan validation layers enabled");
    }
#endif
}

Instance::~Instance() {
    if (debug_messenger_ != VK_NULL_HANDLE && destroy_debug_messenger_ != nullptr) {
        destroy_debug_messenger_(instance_, debug_messenger_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE) {
        vkDestroyInstance(instance_, nullptr);
    }
}

}  // namespace sage::gpu
