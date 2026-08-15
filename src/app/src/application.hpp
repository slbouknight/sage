#pragma once

#include <sage/core/camera.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/depth_buffer.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/frame_pacer.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/pipeline.hpp>
#include <sage/gpu/pipeline_cache.hpp>
#include <sage/gpu/surface.hpp>
#include <sage/gpu/swapchain.hpp>
#include <sage/gpu/uploader.hpp>
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
    void record_cube(VkCommandBuffer command_buffer, VkImage image, VkImageView image_view,
                     VkExtent2D extent) const;

    core::Camera camera_;
    gpu::Window window_;
    gpu::Instance instance_;
    gpu::Surface surface_;
    gpu::PhysicalDeviceInfo physical_device_;
    gpu::Device device_;
    gpu::Allocator allocator_;
    gpu::BindlessSet bindless_set_;
    gpu::Uploader uploader_;
    gpu::GeometryRegistry geometry_registry_;
    gpu::GeometryRegistry::MeshView cube_;
    gpu::Swapchain swapchain_;
    gpu::DepthBuffer depth_buffer_;
    gpu::PipelineCache pipeline_cache_;
    gpu::GraphicsPipeline pipeline_;
    gpu::FramePacer frame_pacer_;
};

}  // namespace sage::app
