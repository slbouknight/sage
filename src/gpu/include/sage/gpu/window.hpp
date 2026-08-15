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

    // One frame of input. Sample once per frame, after poll_events().
    struct InputState {
        bool look_active = false;     // right mouse button held
        float cursor_delta_x = 0.0F;  // pixels since last sample
        float cursor_delta_y = 0.0F;
        float scroll_delta = 0.0F;  // ticks accumulated since last sample
        bool forward = false;
        bool back = false;
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    };

    // Consumes accumulated scroll and re-baselines the cursor, soo this must be
    // called exactly once per frame.
    [[nodiscard]] InputState sample_input();

private:
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset);

    GLFWwindow* window_ = nullptr;
    bool resized_ = false;
    bool look_active_ = false;
    double last_cursor_x_ = 0.0;
    double last_cursor_y_ = 0.0;
    float scroll_accumulator_ = 0.0F;
};

}  // namespace sage::gpu
