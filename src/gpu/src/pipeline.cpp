#include <sage/core/log.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/pipeline.hpp>
#include <sage/gpu/shader_module.hpp>
#include <sage/gpu/vk_check.hpp>

#include <array>
#include <cstdint>

namespace sage::gpu {

GraphicsPipeline::GraphicsPipeline(const Device& device, const GraphicsPipelineDesc& desc)
    : device_(device) {
    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_ALL;
    push_range.offset = 0;
    push_range.size = k_push_constant_size;

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &desc.set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    VK_CHECK(vkCreatePipelineLayout(device_.handle(), &layout_info, nullptr, &layout_));

    // One module, two entry points -- selected by pName, not by loading twice
    const ShaderModule shader(device_, desc.spirv_path);

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = shader.handle();
    stages[0].pName = "vertex_main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = shader.handle();
    stages[1].pName = "fragment_main";

    // No vertex buffers: the shader synthesizes positions from SV_VertexID.
    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    // Values are dynamic, but the counts are still baked in here.
    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.depthClampEnable = VK_FALSE;
    raster.rasterizerDiscardEnable = VK_FALSE;
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = desc.cull_backfaces ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    // The values come from vkCmdSetDepthBias; this only decides whether the
    // stage is on at all.
    raster.depthBiasEnable = desc.depth_bias ? VK_TRUE : VK_FALSE;
    raster.lineWidth = 1.0F;

    VkPipelineMultisampleStateCreateInfo multisample{};
    multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisample.sampleShadingEnable = VK_FALSE;

    const bool has_depth = desc.depth_format != VK_FORMAT_UNDEFINED;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable = has_depth ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable = has_depth ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0F;
    depth_stencil.maxDepthBounds = 1.0F;

    const bool writes_ids = desc.id_format != VK_FORMAT_UNDEFINED;
    const std::uint32_t color_attachment_count = desc.depth_only ? 0U : (writes_ids ? 2U : 1U);

    // One entry per colour attachment -- the count must match
    // colorAttachmentCount exactly, independently of whether blending is on.
    //
    // Both entries are identical on purpose. Without the independentBlend
    // device feature, every attachment's blend state must match element for
    // element, and validation rejects the pipeline if they differ. Writing
    // RGBA at a single-component R32_UINT target is harmless -- mask bits for
    // components the format does not have are ignored -- so matching costs
    // nothing, where narrowing the id's mask to R would have cost a device
    // feature request for no benefit.
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable = desc.alpha_blend ? VK_TRUE : VK_FALSE;
    blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    const std::array<VkPipelineColorBlendAttachmentState, 2> blend_attachments{blend_attachment,
                                                                               blend_attachment};

    VkPipelineColorBlendStateCreateInfo color_blend{};
    color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blend.logicOpEnable = VK_FALSE;
    color_blend.attachmentCount = color_attachment_count;
    color_blend.pAttachments = blend_attachments.data();

    constexpr std::array<VkDynamicState, 3> k_all_dynamic_states{
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    // The depth-bias entry is only listed when the pipeline enables bias.
    // Declaring a dynamic state the pipeline never uses is legal but means
    // vkCmdSetDepthBias must still be called before every draw, which would
    // put that requirement on passes that have nothing to do with shadows.
    dynamic_state.dynamicStateCount = desc.depth_bias ? 3U : 2U;
    dynamic_state.pDynamicStates = k_all_dynamic_states.data();

    // This is what stands in for a VkRenderPass.
    const std::array<VkFormat, 2> color_formats{desc.color_format, desc.id_format};

    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = color_attachment_count;
    rendering_info.pColorAttachmentFormats = color_formats.data();
    rendering_info.depthAttachmentFormat = desc.depth_format;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.pNext = &rendering_info;
    // One stage for a depth-only pass: stages[1] names a fragment entry point
    // that a shadow shader deliberately does not have.
    pipeline_info.stageCount = desc.depth_only ? 1U : static_cast<std::uint32_t>(stages.size());
    pipeline_info.pStages = stages.data();
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pColorBlendState = &color_blend;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = layout_;
    pipeline_info.renderPass = VK_NULL_HANDLE;
    pipeline_info.subpass = 0;
    pipeline_info.pDepthStencilState = &depth_stencil;

    VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), desc.cache, 1, &pipeline_info, nullptr,
                                       &pipeline_));

    SAGE_LOG_INFO("Graphics pipeline created");
}

GraphicsPipeline::~GraphicsPipeline() {
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(device_.handle(), pipeline_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device_.handle(), layout_, nullptr);
    }
}
}  // namespace sage::gpu