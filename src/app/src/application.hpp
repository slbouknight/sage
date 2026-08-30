#pragma once

#include <sage/core/camera.hpp>
#include <sage/gpu/allocator.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/buffer.hpp>
#include <sage/gpu/depth_buffer.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/frame_pacer.hpp>
#include <sage/gpu/geometry_registry.hpp>
#include <sage/gpu/gltf_loader.hpp>
#include <sage/gpu/imgui_layer.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/material_registry.hpp>
#include <sage/gpu/pipeline.hpp>
#include <sage/gpu/pipeline_cache.hpp>
#include <sage/gpu/sampler.hpp>
#include <sage/gpu/scene.hpp>
#include <sage/gpu/surface.hpp>
#include <sage/gpu/swapchain.hpp>
#include <sage/gpu/texture_registry.hpp>
#include <sage/gpu/uploader.hpp>
#include <sage/gpu/window.hpp>

#include <filesystem>

namespace sage::app {

class Application {
public:
    explicit Application(const std::filesystem::path& model_path);
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
    void record_scene(VkCommandBuffer command_buffer, VkImage image, VkImageView image_view,
                      VkExtent2D extent, std::uint32_t frame_slot) const;
    // Split out of record_scene because the UI draws into the same swapchain
    // image and must get there before it is handed to the presentation engine.
    static void transition_to_present(VkCommandBuffer command_buffer, VkImage image);
    void draw_ui();
    void frame_camera_on(const glm::vec3& bounds_min, const glm::vec3& bounds_max);

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
    gpu::Sampler sampler_;
    gpu::TextureRegistry texture_registry_;
    gpu::MaterialRegistry material_registry_;
    gpu::Buffer frame_buffer_;
    gpu::SceneGraph scene_graph_;
    gpu::Swapchain swapchain_;
    gpu::DepthBuffer depth_buffer_;
    gpu::PipelineCache pipeline_cache_;
    gpu::GraphicsPipeline pipeline_;
    gpu::FramePacer frame_pacer_;
    // Last, so it is destroyed first: its teardown frees Vulkan objects and
    // touches the device, both of which must still be alive.
    gpu::ImGuiLayer imgui_;
};

}  // namespace sage::app
