#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <filesystem>

namespace sage::gpu {

class Device;

struct GraphicsPipelineDesc {
    std::filesystem::path spirv_path;
    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;
    VkDescriptorSetLayout set_layout = VK_NULL_HANDLE;
    VkPipelineCache cache = VK_NULL_HANDLE;
};

// Dynamic rendering only: no VkRenderPass, no VkFramebuffer. The color
// attachment format is baked in at creation via VkPipelineRenderingCreateInfo;
// the actual image view is named later by vkCmdBeginRendering.
class GraphicsPipeline {
public:
    GraphicsPipeline(const Device& device, const GraphicsPipelineDesc& desc);
    ~GraphicsPipeline();

    GraphicsPipeline(const GraphicsPipeline&) = delete;
    GraphicsPipeline& operator=(const GraphicsPipeline&) = delete;
    GraphicsPipeline(GraphicsPipeline&&) = delete;
    GraphicsPipeline& operator=(GraphicsPipeline&&) = delete;

    [[nodiscard]] VkPipeline handle() const { return pipeline_; }
    [[nodiscard]] VkPipelineLayout layout() const { return layout_; }

    // A single fixed-size range shared by every pipeline; each shader
    // interprets the bytes as it likes. 128 is the spec's guaranteed minimum
    // for maxPushConstantsSize, so this stays portable.
    static constexpr std::uint32_t k_push_constant_size = 128;

private:
    const Device& device_;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
};

}  // namespace sage::gpu