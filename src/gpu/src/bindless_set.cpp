#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/vk_check.hpp>

#include <array>

namespace sage::gpu {

namespace {

// The two single-image bindings -- the object-id attachment and the HDR colour
// target -- are written identically bar the binding number. Both are
// SAMPLED_IMAGE with no sampler: each is read by integer texel coordinate, at
// exactly one texel per pixel, so there is nothing for a sampler to do.
void write_single_image(VkDevice device, VkDescriptorSet set, std::uint32_t binding,
                        VkImageView view) {
    VkDescriptorImageInfo image_info{};
    image_info.imageView = view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &image_info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

// The same, for the bindings whose view has to travel with a particular
// sampler. `layout` differs between them because one is a depth image.
void write_single_combined_image(VkDevice device, VkDescriptorSet set, std::uint32_t binding,
                                 VkImageView view, VkSampler sampler, VkImageLayout layout) {
    VkDescriptorImageInfo image_info{};
    image_info.sampler = sampler;
    image_info.imageView = view;
    image_info.imageLayout = layout;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.dstArrayElement = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image_info;

    vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
}

}  // namespace

BindlessSet::BindlessSet(const Device& device) : device_(device) {
    VkPhysicalDeviceVulkan12Properties props12{};
    props12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &props12;
    vkGetPhysicalDeviceProperties2(device_.physical_device(), &props2);

    SAGE_VERIFY(props12.maxDescriptorSetUpdateAfterBindStorageBuffers >= k_max_storage_buffers,
                "GPU cannot back the requested bindless storage buffer array");
    SAGE_VERIFY(props12.maxDescriptorSetUpdateAfterBindSampledImages >= k_max_sampled_images,
                "GPU cannot back the requested bindless sampled-image array");

    std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
    bindings[0].binding = k_storage_buffer_binding;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = k_max_storage_buffers;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[1].binding = k_sampled_image_binding;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = k_max_sampled_images;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[2].binding = k_object_id_binding;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    // One: there is a single id attachment, not an array of them.
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[3].binding = k_hdr_color_binding;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[4].binding = k_shadow_map_binding;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[5].binding = k_ldr_color_binding;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[5].descriptorCount = 1;
    bindings[5].stageFlags = VK_SHADER_STAGE_ALL;

    constexpr VkDescriptorBindingFlags k_binding_flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    constexpr std::array<VkDescriptorBindingFlags, 6> k_flags{k_binding_flags, k_binding_flags,
                                                              k_binding_flags, k_binding_flags,
                                                              k_binding_flags, k_binding_flags};

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_info{};
    flags_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    flags_info.bindingCount = static_cast<std::uint32_t>(k_flags.size());
    flags_info.pBindingFlags = k_flags.data();

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.pNext = &flags_info;
    layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_.handle(), &layout_info, nullptr, &layout_));

    const std::array<VkDescriptorPoolSize, 3> pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, k_max_storage_buffers},
        // Plus two for the shadow map and the tonemapped image, which share
        // this descriptor type with the colour-texture array but not its
        // binding.
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k_max_sampled_images + 2},
        // Two: the object-id attachment and the HDR colour target.
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2}};

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();
    VK_CHECK(vkCreateDescriptorPool(device_.handle(), &pool_info, nullptr, &pool_));

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &layout_;
    VK_CHECK(vkAllocateDescriptorSets(device_.handle(), &alloc_info, &set_));

    SAGE_LOG_INFO("Bindless set: {} storage buffers, {} sampled images", k_max_storage_buffers,
                  k_max_sampled_images);
}

void BindlessSet::write_storage_buffer(std::uint32_t index, VkBuffer buffer,
                                       VkDeviceSize size) const {
    SAGE_VERIFY(index < k_max_storage_buffers, "Bindless storage-buffer index out of range");

    VkDescriptorBufferInfo buffer_info{};
    buffer_info.buffer = buffer;
    buffer_info.offset = 0;
    buffer_info.range = size;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = k_storage_buffer_binding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer_info;

    vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
}

void BindlessSet::write_object_id_image(VkImageView view) const {
    write_single_image(device_.handle(), set_, k_object_id_binding, view);
}

void BindlessSet::write_hdr_color_image(VkImageView view) const {
    write_single_image(device_.handle(), set_, k_hdr_color_binding, view);
}

void BindlessSet::write_shadow_map(VkImageView view, VkSampler sampler) const {
    // DEPTH_READ_ONLY rather than SHADER_READ_ONLY: this is a depth image, and
    // the read-only depth layout is what the shadow pass leaves it in.
    write_single_combined_image(device_.handle(), set_, k_shadow_map_binding, view, sampler,
                                VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL);
}

void BindlessSet::write_ldr_color_image(VkImageView view, VkSampler sampler) const {
    write_single_combined_image(device_.handle(), set_, k_ldr_color_binding, view, sampler,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void BindlessSet::write_sampled_image(std::uint32_t index, VkImageView view,
                                      VkSampler sampler) const {
    SAGE_VERIFY(index < k_max_sampled_images, "Bindless sampled-image index out of range");

    VkDescriptorImageInfo image_info{};
    image_info.sampler = sampler;
    image_info.imageView = view;
    image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set_;
    write.dstBinding = k_sampled_image_binding;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image_info;

    vkUpdateDescriptorSets(device_.handle(), 1, &write, 0, nullptr);
}

BindlessSet::~BindlessSet() {
    // Destroying the pool frees every set allocated from it. The set must not
    // be freed separately. The pool was not created with FREE_DESCRIPTOR_SET_BIT
    // so vkFreeDescriptorSets would be invalid.
    if (pool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device_.handle(), pool_, nullptr);
    }
    if (layout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device_.handle(), layout_, nullptr);
    }
}
}  // namespace sage::gpu