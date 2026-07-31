#pragma once

#include <sage/gpu/allocator.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/frame_pacer.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/surface.hpp>
#include <sage/gpu/swapchain.hpp>
#include <sage/gpu/window.hpp>

namespace sage::app {

class Application {
public:
    Application();
    ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
    Application(Application&&) = delete;
    Application& operator=(Application&&) = delete;

    void run();

private:
    // Returns false when the window is minimized and the frame should be
    // skipped entirely.
    bool recreate_swapchain();
    void record_clear(VkCommandBuffer command_buffer, VkImage image) const;

    gpu::Window window_;
    gpu::Instance instance_;
    gpu::Surface surface_;
    gpu::PhysicalDeviceInfo physical_device_;
    gpu::Device device_;
    gpu::Allocator allocator_;
    gpu::Swapchain swapchain_;
    gpu::FramePacer frame_pacer_;
};

}  // namespace sage::app
