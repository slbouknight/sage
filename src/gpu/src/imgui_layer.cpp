#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/imgui_layer.hpp>
#include <sage/gpu/instance.hpp>
#include <sage/gpu/window.hpp>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace sage::gpu {

namespace {
// Descriptors for the font atlas plus any textures panels display. Passing a
// non-zero size makes the backend create and own the pool, which is one fewer
// thing here to size, create and destroy.
constexpr std::uint32_t k_descriptor_pool_size = 64;

}  // namespace

ImGuiLayer::ImGuiLayer(const Instance& instance, const Device& device, const Window& window,
                       VkFormat color_format, std::uint32_t image_count)
    : device_(device), color_format_(color_format) {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    // No imgui.ini. Persisting layout is the editor's business once there is
    // one; until then a stray file appearing in the working directory is a
    // surprise rather than a feature.
    io.IniFilename = nullptr;

    ImGui::StyleColorsDark();

    // install_callbacks = true. The backend chains to whatever GLFW callbacks
    // are already registered rather than replacing them, which is exactly why
    // Window has to be constructed first.
    SAGE_VERIFY(ImGui_ImplGlfw_InitForVulkan(window.handle(), true),
                "ImGui GLFW backend failed to intialize");

    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_format_;

    ImGui_ImplVulkan_InitInfo init{};
    init.ApiVersion = VK_API_VERSION_1_3;
    init.Instance = instance.handle();
    init.PhysicalDevice = device_.physical_device();
    init.Device = device_.handle();
    init.QueueFamily = device_.graphics_family();
    init.Queue = device_.graphics_queue();
    init.DescriptorPoolSize = k_descriptor_pool_size;
    // The backend requires at least two, independently of our frame pacing.
    init.MinImageCount = 2;
    init.ImageCount = image_count;
    init.UseDynamicRendering = true;
    // Since ImGui 1.92 this lives on PipelineInfoMain rather than directly on
    // InitInfo, and the old RenderPass fields are gone. Most examples still
    // online show the previous layout.
    init.PipelineInfoMain.PipelineRenderingCreateInfo = rendering_info;

    SAGE_VERIFY(ImGui_ImplVulkan_Init(&init), "ImGui Vulkan backend failed to initialise");

    SAGE_LOG_INFO("ImGui {} initialised, dynamic rendering, {} swapchain images", IMGUI_VERSION,
                  image_count);
}

void ImGuiLayer::begin_frame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(VkCommandBuffer command_buffer, VkImageView target,
                        VkExtent2D extent) const {
    ImGui::Render();

    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = target;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    // LOAD rather than CLEAR: the scene is already in this attachment and the
    // UI draws over it.
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo rendering{};
    rendering.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering.renderArea.offset = {0, 0};
    rendering.renderArea.extent = extent;
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = 1;
    rendering.pColorAttachments = &color_attachment;
    // No depth attachment: the UI is 2D and must never be occluded by geometry.

    vkCmdBeginRendering(command_buffer, &rendering);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), command_buffer);
    vkCmdEndRendering(command_buffer);
}

bool ImGuiLayer::wants_mouse() {
    return ImGui::GetIO().WantCaptureMouse;
}

bool ImGuiLayer::wants_keyboard() {
    return ImGui::GetIO().WantCaptureKeyboard;
}

ImGuiLayer::~ImGuiLayer() {
    // The backends free Vulkan objects on shutdown, so no frame may still be
    // referencing them.
    device_.wait_idle();

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

}  // namespace sage::gpu