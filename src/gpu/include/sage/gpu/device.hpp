#pragma once

#include <sage/gpu/physical_device.hpp>

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sage::gpu {

class Device {
public:
    explicit Device(const PhysicalDeviceInfo& info);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) = delete;
    Device& operator=(Device&&) = delete;

    [[nodiscard]] VkDevice handle() const { return device_; }
    [[nodiscard]] VkPhysicalDevice physical_device() const { return physical_device_; }

    [[nodiscard]] VkQueue graphics_queue() const { return graphics_queue_; }
    [[nodiscard]] VkQueue present_queue() const { return present_queue_; }
    // Created now because queue families are fixed at device creation; unused
    // until M3's transfer-queue staging uploads.
    [[nodiscard]] VkQueue transfer_queue() const { return transfer_queue_; }

    [[nodiscard]] std::uint32_t graphics_family() const { return graphics_family_; }
    [[nodiscard]] std::uint32_t present_family() const { return present_family_; }
    [[nodiscard]] std::uint32_t transfer_family() const { return transfer_family_; }

    void wait_idle() const;

private:
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;

    VkQueue graphics_queue_ = VK_NULL_HANDLE;
    VkQueue present_queue_ = VK_NULL_HANDLE;
    VkQueue transfer_queue_ = VK_NULL_HANDLE;

    std::uint32_t graphics_family_ = 0;
    std::uint32_t present_family_ = 0;
    std::uint32_t transfer_family_ = 0;
};

}  // namespace sage::gpu
