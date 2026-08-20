#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/sampler.hpp>
#include <sage/gpu/vk_check.hpp>

namespace sage::gpu {

Sampler::Sampler(const Device& device) : device_(device) {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(device_.physical_device(), &properties);

    VkSamplerCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    // LINEAR blends the two nearest mip levels rather than snapping to one,
    // which is what stops a seam sliding across a surface as the camera moves.
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    // REPEAT is glTF's default
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.anisotropyEnable = VK_TRUE;
    // Mip selection uses the larger screen-space derivative, so a surface seen
    // edge-on blurs the axis that did not need it. Clamped to the device limit.
    info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    info.compareEnable = VK_FALSE;
    info.minLod = 0.0F;
    // Unclamped, so one sampler serves textures with different mip counts.
    info.maxLod = VK_LOD_CLAMP_NONE;
    info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;

    VK_CHECK(vkCreateSampler(device_.handle(), &info, nullptr, &sampler_));

    SAGE_LOG_INFO("Sampler: linear/linear/linear, repeat, {}x anisotropic",
                  static_cast<int>(properties.limits.maxSamplerAnisotropy));
}

Sampler::~Sampler() {
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(device_.handle(), sampler_, nullptr);
    }
}

}  // namespace sage::gpu