#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/window.hpp>

#include <GLFW/glfw3.h>

namespace sage::gpu {

namespace {

void glfw_error_callback(int code, const char* description) {
    SAGE_LOG_ERROR("GLFW error {}: {}", code, description);
}

}  // namespace

void Window::framebuffer_size_callback(GLFWwindow* window, int /*width*/, int /*height*/) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
    if (self != nullptr) {
        self->resized_ = true;
    }
}

Window::Window(std::uint32_t width, std::uint32_t height, const std::string& title) {
    glfwSetErrorCallback(glfw_error_callback);
    SAGE_VERIFY(glfwInit() == GLFW_TRUE, "glfwInit failed");
    SAGE_VERIFY(glfwVulkanSupported() == GLFW_TRUE, "GLFW reports no Vulkan loader/ICD available");

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height), title.c_str(),
                               nullptr, nullptr);
    SAGE_VERIFY(window_ != nullptr, "glfwCreateWindow failed");

    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, &Window::framebuffer_size_callback);
    glfwSetScrollCallback(window_, &Window::scroll_callback);
}

Window::~Window() {
    if (window_ != nullptr) {
        glfwDestroyWindow(window_);
    }
    glfwTerminate();
}

bool Window::should_close() const {
    return glfwWindowShouldClose(window_) == GLFW_TRUE;
}

VkExtent2D Window::framebuffer_extent() const {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    return VkExtent2D{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

void Window::poll_events() {
    glfwPollEvents();
}

void Window::wait_events() {
    glfwWaitEvents();
}

bool Window::consume_resized() {
    const bool was_resized = resized_;
    resized_ = false;
    return was_resized;
}

void Window::scroll_callback(GLFWwindow* window, double /*x_offset*/, double y_offset) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));

    if (self != nullptr) {
        self->scroll_accumulator_ += static_cast<float>(y_offset);
    }
}

Window::InputState Window::sample_input() {
    InputState state;

    const bool right_mouse_down =
        glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;

    if (right_mouse_down && !look_active_) {
        // Entering look mode. Hide and unbound the cursor, the re-read its
        // position: GLFW_CURSOR_DISABLED can report a large jump on the first
        // read afterwards, which would snape the camera around.
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        glfwGetCursorPos(window_, &last_cursor_x_, &last_cursor_y_);
    } else if (!right_mouse_down && look_active_) {
        glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    }
    look_active_ = right_mouse_down;
    state.look_active = look_active_;

    double cursor_x = 0.0;
    double cursor_y = 0.0;
    glfwGetCursorPos(window_, &cursor_x, &cursor_y);

    if (look_active_) {
        state.cursor_delta_x = static_cast<float>(cursor_x - last_cursor_x_);
        state.cursor_delta_y = static_cast<float>(cursor_y - last_cursor_y_);
    }
    last_cursor_x_ = cursor_x;
    last_cursor_y_ = cursor_y;

    state.scroll_delta = scroll_accumulator_;
    scroll_accumulator_ = 0.0F;

    const auto pressed = [this](int key) { return glfwGetKey(window_, key) == GLFW_PRESS; };
    state.forward = pressed(GLFW_KEY_W);
    state.back = pressed(GLFW_KEY_S);
    state.left = pressed(GLFW_KEY_A);
    state.right = pressed(GLFW_KEY_D);
    state.up = pressed(GLFW_KEY_E);
    state.down = pressed(GLFW_KEY_Q);

    return state;
}

std::vector<const char*> Window::required_instance_extensions() {
    std::uint32_t count = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    SAGE_VERIFY(extensions != nullptr, "glfwGetRequiredInstanceExtensions returned null");
    return std::vector<const char*>(extensions, extensions + count);
}

}  // namespace sage::gpu
