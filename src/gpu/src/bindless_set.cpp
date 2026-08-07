#include <sage/core/assert.hpp>
#include <sage/core/log.hpp>
#include <sage/gpu/bindless_set.hpp>
#include <sage/gpu/device.hpp>
#include <sage/gpu/vk_check.hpp>

#include <array>

namespace sage::gpu {

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

    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = k_storage_buffer_binding;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = k_max_storage_buffers;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[1].binding = k_sampled_image_binding;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = k_max_sampled_images;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;

    constexpr VkDescriptorBindingFlags k_binding_flags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    constexpr std::array<VkDescriptorBindingFlags, 2> k_flags{k_binding_flags, k_binding_flags};

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

    const std::array<VkDescriptorPoolSize, 2> pool_sizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, k_max_storage_buffers},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, k_max_sampled_images}};

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