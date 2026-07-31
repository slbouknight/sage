#include <sage/gpu/instance.hpp>
#include <sage/gpu/surface.hpp>
#include <sage/gpu/vk_check.hpp>
#include <sage/gpu/window.hpp>

#include <GLFW/glfw3.h>

namespace sage::gpu {

Surface::Surface(const Instance& instance, const Window& window) : instance_(instance.handle()) {
    VK_CHECK(glfwCreateWindowSurface(instance_, window.handle(), nullptr, &surface_));
}

Surface::~Surface() {
    if (surface_ != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
}

}  // namespace sage::gpu
