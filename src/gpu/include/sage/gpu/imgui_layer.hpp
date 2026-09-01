#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sage::gpu {

class Device;
class Instance;
class Window;

// Owns the ImGui context and its GLFW and Vulkan backends.
class ImGuiLayer {
public:
    // `window` must already have installed its own GLFW callbacks: the backend
    // captures whatever is registered as "previous" and chains to it, so
    // constructing this first would silently break scroll and resize handling.
    ImGuiLayer(const Instance& instance, const Device& device, const Window& window,
               VkFormat color_format, std::uint32_t image_count);
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;
    ImGuiLayer(ImGuiLayer&&) = delete;
    ImGuiLayer& operator=(ImGuiLayer&&) = delete;

    // Starts a UI frame. Build windows between this and render().
    static void begin_frame();

    // Records the UI over whatever is already in `target`, in its own rendering
    // scope. Must be called once per begin_frame().
    void render(VkCommandBuffer command_buffer, VkImageView target, VkExtent2D extent) const;

    // True while ImGui is using the pointer or the keys, so the camera can stand down
    // instead instead of fighting a panel drag.
    [[nodiscard]] static bool wants_mouse();
    [[nodiscard]] static bool wants_keyboard();

private:
    const Device& device_;
    // Held by value because ImGui keeps the VkPipelineRenderingCreateInfo it is
    // given, and that struct points at this rather than owning it. A local
    // would dangle the moment the constructor returned.
    VkFormat color_format_ = VK_FORMAT_UNDEFINED;
};

}  // namespace sage::gpu