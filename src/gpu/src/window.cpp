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

std::vector<const char*> Window::required_instance_extensions() {
    std::uint32_t count = 0;
    const char** extensions = glfwGetRequiredInstanceExtensions(&count);
    SAGE_VERIFY(extensions != nullptr, "glfwGetRequiredInstanceExtensions returned null");
    return std::vector<const char*>(extensions, extensions + count);
}

}  // namespace sage::gpu
