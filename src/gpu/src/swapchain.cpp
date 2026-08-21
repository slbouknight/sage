#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/swapchain.hpp>
#include <sage/gpu/vk_check.hpp>

#include <algorithm>
#include <limits>

namespace sage::gpu {

namespace {

VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& available) {
    // An sRGB attachment format makes the color write encode linear -> sRGB in
    // fixed function. Shading in linear space and presenting the UNORM would
    // hand linear values to a display that expects sRGB, which reads as too
    // dark mid-tones the moment a texture is sampled.
    for (const VkSurfaceFormatKHR& format : available) {
        if (format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return format;
        }
    }
    return available.front();
}

VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& available) {
    for (const VkPresentModeKHR mode : available) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return mode;
        }
    }
    // Always supported by the spec.
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D clamp_extent(VkExtent2D requested, const VkSurfaceCapabilitiesKHR& caps) {
    if (caps.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
        return caps.currentExtent;
    }
    return VkExtent2D{
        std::clamp(requested.width, caps.minImageExtent.width, caps.maxImageExtent.width),
        std::clamp(requested.height, caps.minImageExtent.height, caps.maxImageExtent.height)};
}

}  // namespace

Swapchain::Swapchain(const Device& device, VkSurfaceKHR surface, VkExtent2D extent)
    : device_(device), surface_(surface) {
    create(extent);
}

Swapchain::~Swapchain() {
    destroy();
}

void Swapchain::create(VkExtent2D extent) {
    const VkPhysicalDevice physical = device_.physical_device();

    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface_, &caps));

    std::uint32_t format_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &format_count, nullptr));
    SAGE_VERIFY(format_count > 0, "Surface reports no supported formats");
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    VK_CHECK(
        vkGetPhysicalDeviceSurfaceFormatsKHR(physical, surface_, &format_count, formats.data()));

    std::uint32_t mode_count = 0;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface_, &mode_count, nullptr));
    SAGE_VERIFY(mode_count > 0, "Surface reports no supported present modes");
    std::vector<VkPresentModeKHR> modes(mode_count);
    VK_CHECK(
        vkGetPhysicalDeviceSurfacePresentModesKHR(physical, surface_, &mode_count, modes.data()));

    const VkSurfaceFormatKHR surface_format = choose_surface_format(formats);
    format_ = surface_format.format;
    color_space_ = surface_format.colorSpace;
    present_mode_ = choose_present_mode(modes);
    extent_ = clamp_extent(extent, caps);

    std::uint32_t image_count = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && image_count > caps.maxImageCount) {
        image_count = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface = surface_;
    create_info.minImageCount = image_count;
    create_info.imageFormat = format_;
    create_info.imageColorSpace = color_space_;
    create_info.imageExtent = extent_;
    create_info.imageArrayLayers = 1;
    // TRANSFER_DST for M1's vkCmdClearColorImage; COLOR_ATTACHMENT for the
    // dynamic-rendering path M2 introduces against the same swapchain.
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    create_info.preTransform = caps.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = present_mode_;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    const std::uint32_t families[]{device_.graphics_family(), device_.present_family()};
    if (families[0] != families[1]) {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2;
        create_info.pQueueFamilyIndices = families;
    } else {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    VK_CHECK(vkCreateSwapchainKHR(device_.handle(), &create_info, nullptr, &swapchain_));

    std::uint32_t actual_count = 0;
    VK_CHECK(vkGetSwapchainImagesKHR(device_.handle(), swapchain_, &actual_count, nullptr));
    images_.resize(actual_count);
    VK_CHECK(vkGetSwapchainImagesKHR(device_.handle(), swapchain_, &actual_count, images_.data()));

    image_views_.resize(actual_count);
    for (std::uint32_t i = 0; i < actual_count; ++i) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = images_[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = format_;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_.handle(), &view_info, nullptr, &image_views_[i]));
    }

    render_finished_.resize(actual_count);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (VkSemaphore& semaphore : render_finished_) {
        VK_CHECK(vkCreateSemaphore(device_.handle(), &semaphore_info, nullptr, &semaphore));
    }

    SAGE_LOG_INFO("Swapchain: {}x{}, {} images, present mode {}", extent_.width, extent_.height,
                  actual_count, static_cast<int>(present_mode_));
}

void Swapchain::destroy() {
    for (VkSemaphore semaphore : render_finished_) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_.handle(), semaphore, nullptr);
        }
    }
    render_finished_.clear();
    images_.clear();

    if (swapchain_ != VK_NULL_HANDLE) {
        for (VkImageView view : image_views_) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(device_.handle(), view, nullptr);
            }
        }
        image_views_.clear();
        vkDestroySwapchainKHR(device_.handle(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void Swapchain::recreate(VkExtent2D extent) {
    device_.wait_idle();
    destroy();
    create(extent);
}

AcquiredImage Swapchain::acquire(VkSemaphore image_available) {
    AcquiredImage acquired;
    acquired.result = vkAcquireNextImageKHR(device_.handle(), swapchain_,
                                            std::numeric_limits<std::uint64_t>::max(),
                                            image_available, VK_NULL_HANDLE, &acquired.index);

    if (acquired.result == VK_SUCCESS || acquired.result == VK_SUBOPTIMAL_KHR) {
        acquired.image = images_[acquired.index];
    } else if (acquired.result != VK_ERROR_OUT_OF_DATE_KHR) {
        VK_CHECK(acquired.result);
    }
    return acquired;
}

VkResult Swapchain::present(VkQueue queue, std::uint32_t image_index) {
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &render_finished_[image_index];
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &image_index;

    const VkResult result = vkQueuePresentKHR(queue, &present_info);
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR && result != VK_ERROR_OUT_OF_DATE_KHR) {
        VK_CHECK(result);
    }
    return result;
}

}  // namespace sage::gpu
