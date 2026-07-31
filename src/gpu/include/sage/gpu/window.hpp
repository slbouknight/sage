#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

struct GLFWwindow;

namespace sage::gpu {

class Window {
public:
    Window(std::uint32_t width, std::uint32_t height, const std::string& title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;
    Window(Window&&) = delete;
    Window& operator=(Window&&) = delete;

    [[nodiscard]] bool should_close() const;
    [[nodiscard]] VkExtent2D framebuffer_extent() const;

    static void poll_events();
    // Blocks until an event arrives. Used to idle instead of spinning while
    // the window is minimized.
    static void wait_events();

    [[nodiscard]] bool consume_resized();

    [[nodiscard]] GLFWwindow* handle() const { return window_; }

    static std::vector<const char*> required_instance_extensions();

private:
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);

    GLFWwindow* window_ = nullptr;
    bool resized_ = false;
};

}  // namespace sage::gpu
