#pragma once

#include <sage/gpu/drag_gesture.hpp>

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
        bool look_active = false;     // right mouse dragged past the threshold
        float cursor_delta_x = 0.0F;  // pixels since last sample
        float cursor_delta_y = 0.0F;
        float scroll_delta = 0.0F;  // ticks accumulated since last sample
        bool forward = false;
        bool back = false;
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
        // True on the frame a right click is released without having become a
        // camera drag. The two share a button, so one of them has to be decided
        // by how far the cursor moved -- see the note in sample_input.
        bool context_click = false;
        // Where that click happened, in window pixels. Meaningful only when
        // context_click is true.
        float context_click_x = 0.0F;
        float context_click_y = 0.0F;
    };

    // Consumes accumulated scroll and re-baselines the cursor, soo this must be
    // called exactly once per frame.
    [[nodiscard]] InputState sample_input();

private:
    static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
    static void scroll_callback(GLFWwindow* window, double x_offset, double y_offset);

    GLFWwindow* window_ = nullptr;
    bool resized_ = false;
    // Owns the decision about what the right button meant; this class only
    // acts on it by grabbing or releasing the cursor.
    DragGesture right_button_;
    double last_cursor_x_ = 0.0;
    double last_cursor_y_ = 0.0;
    float scroll_accumulator_ = 0.0F;
};

}  // namespace sage::gpu
